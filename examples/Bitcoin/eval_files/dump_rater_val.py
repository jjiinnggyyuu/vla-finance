"""Dump the trained rater's predicted log velocity-loss on the VALIDATION (Nov)
split, so trade thresholds can be *frozen on val and applied to Dec test*
(textbook walk-forward, no look-ahead). Reuses the saved rater weights (.pt) from
train_rater.py; no retraining.

Usage (one GPU per fold):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_rater_val.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold6.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold6/checkpoints/steps_1000_action_model.pt \
    --rater_pt results/exp4_rater/v15_btc_fold6.pt \
    --output_npz results/exp4_rater/v15_btc_fold6_val.npz --draws 20
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
from train_rater import ConfidenceRater, head_forward  # noqa: E402

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
    p.add_argument("--rater_pt", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args, clip = p.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clip)))
    cfg = apply_config_compat(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device).eval()
    model.action_model.load_state_dict(load_state_dict(args.ckpt), strict=False)
    head = model.action_model
    for pm in model.parameters():
        pm.requires_grad_(False)

    ckpt = torch.load(args.rater_pt, map_location=device)
    rater = ConfidenceRater(int(ckpt["dim"])).to(device).eval()
    rater.load_state_dict(ckpt["state_dict"])
    print(f"[val] rater dim={ckpt['dim']} best_epoch={ckpt.get('best_epoch')} "
          f"val_corr={ckpt.get('val_corr'):+.3f}")

    ds = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    preds: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="rater/val"):
            lh, mask, _ = _vlm_forward(model, batch)
            actions = torch.tensor(np.stack([s["action"] for s in batch], axis=0),
                                   device=device, dtype=torch.float32)
            B, H, _ = actions.shape
            acc = torch.zeros(B, H, device=device)
            for _ in range(args.draws):
                with torch.autocast("cuda", dtype=torch.float32):
                    mo, _ = head_forward(head, lh, actions, mask)
                acc += rater(mo.float())
            acc = (acc / args.draws).cpu().numpy()
            for i, s in enumerate(batch):
                ts = s["base_timestamp"]
                if ts not in preds:
                    preds[ts] = acc[i]

    items = sorted(preds.items(), key=lambda kv: kv[0])
    out = {"ts": np.array([t for t, _ in items]),
           "pred_logloss": np.stack([v for _, v in items]).astype(np.float32)}
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print(f"[val] saved {len(items)} bars x {out['pred_logloss'].shape[1]} -> {args.output_npz}")


if __name__ == "__main__":
    main()
