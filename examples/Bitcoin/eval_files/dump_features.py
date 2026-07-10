"""Experiment 2 -- dump frozen VL features + predictions for a confidence probe.

For a frozen DiT+BTC checkpoint, on a given split (validation OR test), we save
per bar:
  * feature      : masked mean-pool of the VLM last_hidden  -> (N, H)
  * sample_rets_k: K sampled k-step returns (same as exp1)   -> (N, K)
  * true_ret_k, lc, ts

Experiment 2 then trains a tiny head  feature -> P(direction correct)  on the
VALIDATION split (out-of-sample for the DiT, so its error pattern is
representative), and evaluates the resulting confidence on the TEST split via
risk_coverage.py. The DiT stays frozen -> no gradient reaches the backbone
(satisfies the "13th-token gradient must not flow" rule by construction).

Efficiency: the VLM forward is deterministic, so we run it ONCE per bar to get
both the feature and last_hidden, then sample the action head K times reusing
last_hidden (~Kx cheaper than calling the full predict_action K times).

Usage (one split per call; run val AND test):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_features.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold6.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold6/checkpoints/steps_1000_action_model.pt \
    --split validation --output_npz results/exp2_features/v15_btc_fold6_val.npz --num_samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import (  # noqa: E402
    load_state_dict,
    restore_prices,
    restore_prices_delta,
)

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import (
    normalize_dotlist_args,
    resize_images,
)
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

K_VALUES = [1, 3, 6, 12]
BTC = 0
SL = slice(BTC * 4, BTC * 4 + 4)


def _vlm_forward(model, examples):
    """Replicate the VLM half of QwenGR00T.predict_action (run ONCE per bar).
    Returns (last_hidden [B,L,H], mask [B,L] bool|None, state|None)."""
    if not isinstance(examples, list):
        examples = [examples]
    batch_images = [to_pil_preserve(e["image"]) for e in examples]
    instructions = [e["lang"] for e in examples]
    state = [e["state"] for e in examples] if "state" in examples[0] else None

    sz = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    if sz:
        batch_images = resize_images(batch_images, target_size=sz)

    qi = model.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
    mask = qi.get("attention_mask", None)
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.qwen_vl_interface(**qi, output_attentions=False,
                                      output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]                       # [B, L, H]
    if state is not None:
        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
    return last_hidden, mask, state


def _pool(last_hidden, mask):
    """Masked mean-pool over the sequence -> (B, H) fixed feature."""
    x = last_hidden.float()
    if mask is None:
        return x.mean(dim=1)
    m = mask.float().unsqueeze(-1)                                # [B,L,1]
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _restore(delta_or_lr, lc_a, target_mode, B, H):
    if target_mode == "delta":
        return restore_prices_delta(delta_or_lr, lc_a)
    lc_rep = np.repeat(lc_a, H)
    return restore_prices(delta_or_lr.reshape(-1, 4), lc_rep).reshape(B, H, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", required=True, choices=["validation", "test"])
    p.add_argument("--output_npz", required=True)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args, clip = p.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clip)))
    cfg = apply_config_compat(cfg)
    target_mode = str(getattr(cfg.datasets.vla_data, "target_mode", "log_ratio"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device).eval()
    model.action_model.load_state_dict(load_state_dict(args.ckpt), strict=False)
    print(f"[feat] {args.split}  ckpt={args.ckpt}  K={args.num_samples}")

    ds = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    K = args.num_samples
    records: dict[str, dict] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"feat/{args.split}"):
            target = np.stack([s["action"] for s in batch], axis=0)
            last_close = np.stack([np.atleast_1d(s["last_close"]) for s in batch]).astype(np.float32)
            asset_std = np.stack([np.atleast_1d(s["asset_std"]) for s in batch]).astype(np.float32)
            B, H, _ = target.shape
            std_a = asset_std[:, BTC][:, None, None]
            lc_a = last_close[:, BTC]

            # VLM forward ONCE -> feature + last_hidden for K head samples.
            last_hidden, mask, state = _vlm_forward(model, batch)
            feat = _pool(last_hidden, mask).detach().cpu().numpy().astype(np.float32)  # (B, Hdim)

            true_prices = _restore(target[:, :, SL] * std_a, lc_a, target_mode, B, H)

            sample_prices = np.empty((B, K, H, 4), dtype=np.float32)
            with torch.autocast("cuda", dtype=torch.float32):
                for j in range(K):
                    pred = model.action_model.predict_action(
                        last_hidden, state, encoder_attention_mask=mask
                    )                                              # (B,H,4N) tensor
                    pred = pred.detach().cpu().numpy()
                    sample_prices[:, j] = _restore(pred[:, :, SL] * std_a, lc_a, target_mode, B, H)

            for i in range(B):
                ts = batch[i]["base_timestamp"]
                if ts in records:
                    continue
                lc = float(lc_a[i])
                rec = {"lc": lc, "feature": feat[i]}
                for k in K_VALUES:
                    ci = k - 1
                    rec[f"true_ret_k{k}"] = float(true_prices[i, ci, 3]) / lc - 1.0
                    rec[f"sample_rets_k{k}"] = (sample_prices[i, :, ci, 3] / lc - 1.0).astype(np.float32)
                records[ts] = rec

    items = sorted(records.items(), key=lambda kv: kv[0])
    N = len(items)
    out = {
        "ts": np.array([ts for ts, _ in items]),
        "lc": np.array([r["lc"] for _, r in items], dtype=np.float32),
        "feature": np.stack([r["feature"] for _, r in items]),         # (N, Hdim)
    }
    for k in K_VALUES:
        out[f"true_ret_k{k}"] = np.array([r[f"true_ret_k{k}"] for _, r in items], dtype=np.float32)
        out[f"sample_rets_k{k}"] = np.stack([r[f"sample_rets_k{k}"] for _, r in items])

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)

    m6 = out["sample_rets_k6"].mean(axis=1)
    t6 = out["true_ret_k6"]
    da6 = float(np.mean((m6 > 0) == (t6 > 0)))
    print(f"[feat] saved {N} bars, feature dim {out['feature'].shape[1]}, "
          f"DA_k6={da6:.4f} -> {args.output_npz}")


if __name__ == "__main__":
    main()
