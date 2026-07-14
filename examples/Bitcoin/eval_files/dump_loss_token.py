"""Dump the INTEGRATED loss-prediction module's per-bar confidence.

Unlike the exp4 rater (a separate 4-layer transformer trained post-hoc), the
loss head here lives inside the DiT action head and is trained jointly (Learning
Loss, 1905.03677, ranking objective). This script just runs the trained head's
`predict_loss` over a split and saves the per-bar predicted velocity-loss, in the
SAME npz format as the exp4 rater dumps so it drops into eval_rater_walkforward.

Output: {ts, pred_logloss[(N, H)]}  (the scalar chunk-loss is broadcast across H
so downstream `pred_logloss[:, k-1]` works for any k).

Usage (one GPU per fold):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_loss_token.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold1.yaml \
    --ckpt playground/Checkpoints/v17_btc_lt/fold1/checkpoints/steps_2000_action_model.pt \
    --split test  --output_npz results/v17_lt/v15_btc_fold1.npz --draws 20 \
    framework.action_model.loss_token=true
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import load_state_dict  # noqa: E402
from dump_features import _vlm_forward  # noqa: E402

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args, clip = p.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    # force the loss-token head on (in case the base yaml doesn't set it)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
        ["framework.action_model.loss_token=true"] + normalize_dotlist_args(clip)))
    cfg = apply_config_compat(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device).eval()
    model.action_model.load_state_dict(load_state_dict(args.ckpt), strict=False)
    head = model.action_model
    assert getattr(head, "use_loss_token", False), "checkpoint head has no loss_token module"
    for pm in model.parameters():
        pm.requires_grad_(False)

    ds = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    H = int(cfg.framework.action_model.action_horizon)
    preds: dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"loss_token/{args.split}"):
            lh, mask, _ = _vlm_forward(model, batch)
            actions = torch.tensor(np.stack([s["action"] for s in batch], axis=0),
                                   device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.float32):
                conf = head.predict_loss(lh, actions, encoder_attention_mask=mask,
                                         draws=args.draws)          # (B,)
            conf = conf.cpu().numpy()
            for i, s in enumerate(batch):
                ts = s["base_timestamp"]
                if ts not in preds:
                    preds[ts] = float(conf[i])

    items = sorted(preds.items(), key=lambda kv: kv[0])
    scal = np.array([v for _, v in items], dtype=np.float32)         # (N,)
    out = {"ts": np.array([t for t, _ in items]),
           "pred_logloss": np.repeat(scal[:, None], H, axis=1)}      # (N, H) broadcast
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print(f"[{args.split}] saved {len(items)} bars -> {args.output_npz}  "
          f"(loss range {scal.min():+.3f}..{scal.max():+.3f})")


if __name__ == "__main__":
    main()
