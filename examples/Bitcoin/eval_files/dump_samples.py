"""Experiment 1 -- per-bar sample dump for confidence estimation.

Runs the selected DiT+BTC checkpoint on the test split and, at each bar, calls
`predict_action` K times. Flow-matching starts from fresh N(0,1) noise each call,
so the K predictions differ; their spread is the sample-variance confidence
signal. We save, per bar (time-ordered), the K sampled k-step returns, the true
k-step return, and the price -- exactly the npz that risk_coverage.py consumes.

Un-normalization / price reconstruction reuse eval_bitcoin_close_mae so the
numbers match the 4-axis pipeline (BTC = asset 0, x asset_std, restore prices).

Single-process (one GPU per fold). Pin the GPU with CUDA_VISIBLE_DEVICES; run the
6 folds across GPUs 4-7 with run_dump_samples.sh.

Usage:
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_samples.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold6.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold6/checkpoints/steps_1000_action_model.pt \
    --output_npz results/exp1_samples/v15_btc_fold6.npz --num_samples 10
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

K_VALUES = [1, 3, 6, 12]
BTC = 0                      # BTC is asset 0; metrics are BTC-only
SL = slice(BTC * 4, BTC * 4 + 4)


def _restore(delta_or_lr, lc_a, target_mode, B, H):
    """(B,H,4) raw-delta/log-ratio + (B,) last_close -> (B,H,4) absolute prices."""
    if target_mode == "delta":
        return restore_prices_delta(delta_or_lr, lc_a)
    lc_rep = np.repeat(lc_a, H)
    return restore_prices(delta_or_lr.reshape(-1, 4), lc_rep).reshape(B, H, 4)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt", required=True, help="selected .pt checkpoint")
    p.add_argument("--output_npz", required=True)
    p.add_argument("--num_samples", type=int, default=10, help="K noise draws per bar")
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
    print(f"[dump] loaded {args.ckpt}  (target_mode={target_mode}, K={args.num_samples})")

    ds = get_vla_dataset(cfg.datasets.vla_data, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    K = args.num_samples
    records: dict[str, dict] = {}     # base_timestamp -> per-bar record (dedup by ts)

    with torch.no_grad():
        for batch in tqdm(loader, desc="dump"):
            target = np.stack([s["action"] for s in batch], axis=0)                  # (B,H,4N) normalized
            last_close = np.stack([np.atleast_1d(s["last_close"]) for s in batch]).astype(np.float32)  # (B,N)
            asset_std = np.stack([np.atleast_1d(s["asset_std"]) for s in batch]).astype(np.float32)    # (B,N)
            B, H, _ = target.shape
            std_a = asset_std[:, BTC][:, None, None]      # (B,1,1)
            lc_a = last_close[:, BTC]                      # (B,)

            # True prices (same for all K samples).
            true_prices = _restore(target[:, :, SL] * std_a, lc_a, target_mode, B, H)  # (B,H,4)

            # K sampled predictions. Each predict_action call re-samples N(0,1)
            # noise -> a different denoised trajectory.
            sample_prices = np.empty((B, K, H, 4), dtype=np.float32)
            for j in range(K):
                pred = model.predict_action(batch)["normalized_actions"]           # (B,H,4N)
                sample_prices[:, j] = _restore(pred[:, :, SL] * std_a, lc_a, target_mode, B, H)

            for i in range(B):
                ts = batch[i]["base_timestamp"]
                if ts in records:
                    continue
                lc = float(lc_a[i])
                rec = {"lc": lc}
                for k in K_VALUES:
                    close_idx = k - 1                      # k-step close = horizon k
                    rec[f"true_ret_k{k}"] = float(true_prices[i, close_idx, 3]) / lc - 1.0
                    # K sampled k-step returns at this bar.
                    rec[f"sample_rets_k{k}"] = (
                        sample_prices[i, :, close_idx, 3] / lc - 1.0
                    ).astype(np.float32)                   # (K,)
                records[ts] = rec

    # Time-order and pack into arrays.
    items = sorted(records.items(), key=lambda kv: kv[0])
    N = len(items)
    out = {
        "ts": np.array([ts for ts, _ in items]),
        "lc": np.array([r["lc"] for _, r in items], dtype=np.float32),
    }
    for k in K_VALUES:
        out[f"true_ret_k{k}"] = np.array([r[f"true_ret_k{k}"] for _, r in items], dtype=np.float32)
        out[f"sample_rets_k{k}"] = np.stack([r[f"sample_rets_k{k}"] for _, r in items])  # (N,K)

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)

    # Quick provenance print: base DA at k=6 from the sample mean (sanity vs JSON).
    m6 = out["sample_rets_k6"].mean(axis=1)
    t6 = out["true_ret_k6"]
    da6 = float(np.mean((m6 > 0) == (t6 > 0)))
    print(f"[dump] saved {N} bars x K={K} -> {args.output_npz}")
    print(f"[dump] base DA_k6 (sample mean) = {da6:.4f}  "
          f"(compare to the JSON's da_k6 for this fold)")


if __name__ == "__main__":
    main()
