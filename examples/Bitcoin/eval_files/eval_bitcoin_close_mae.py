"""Evaluate Bitcoin GR00T checkpoints.

Metrics (following Kronos paper conventions):
  - close_mae              : Mean Absolute Error on restored close prices
  - price_ic / price_rankic: Per-sample Pearson/Spearman correlation between
                             predicted and true OHLC series over 12 horizons,
                             averaged over 4 channels then over all samples.
  - return_ic / return_rankic: Cross-sectional Pearson/Spearman correlation
                             between predicted and true H-step returns across
                             all samples  (r = close_{t+H} / last_close - 1).

The model predicts log-ratio OHLC targets:
    pred = log(future_ohlc / last_input_close)

This script restores close prices and computes the metrics above.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_state_dict(path: str) -> dict:
    ckpt_path = Path(path)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(ckpt_path))
    return torch.load(str(ckpt_path), map_location="cpu")


def restore_prices(log_ratio: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    """log-ratio → absolute price.  log_ratio: (B*H, 4),  last_close: (B*H,)"""
    return np.exp(log_ratio) * last_close[:, None]


def _safe_corr(x: np.ndarray, y: np.ndarray):
    """Return (pearson_r, spearman_r) or (0,0) if degenerate."""
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 0.0
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return float(r), float(rho)


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, dataloader, output_csv: str | None = None) -> dict:
    """Run inference and compute all metrics."""
    total_abs_error = 0.0
    total_count = 0

    # Price Series IC/RankIC (per-sample correlations, then averaged)
    price_ics: list[float] = []
    price_rankics: list[float] = []

    # Return IC/RankIC (cross-sectional, collected across batches)
    all_pred_returns: list[float] = []
    all_true_returns: list[float] = []

    csv_rows: list[dict] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        # ---- inference ------------------------------------------------
        pred = model.predict_action(batch)["normalized_actions"]  # (B, 12, 4) log-ratio
        target = np.stack([s["action"] for s in batch], axis=0)  # (B, 12, 4) log-ratio
        last_close = np.asarray([s["last_close"] for s in batch], dtype=np.float32)  # (B,)

        B, H, C = pred.shape  # C == 4  (O,H,L,C)

        # ---- restore absolute prices ----------------------------------
        lc_rep = np.repeat(last_close, H)  # (B*H,)
        pred_prices = restore_prices(pred.reshape(-1, C), lc_rep).reshape(B, H, C)   # (B,12,4)
        true_prices = restore_prices(target.reshape(-1, C), lc_rep).reshape(B, H, C)

        # ================================================================
        # 1. Close MAE
        # ================================================================
        pred_close = pred_prices[:, :, 3]   # (B,12)
        true_close = true_prices[:, :, 3]
        abs_err = np.abs(pred_close - true_close)
        total_abs_error += float(abs_err.sum())
        total_count += int(abs_err.size)

        # ================================================================
        # 2. Price Series IC / RankIC  (Kronos paper §4)
        #    For each sample:  correlate pred[i, :, c] vs true[i, :, c]
        #    over the 12-step horizon, for each channel c, then average.
        # ================================================================
        for i in range(B):
            ic_per_ch, rankic_per_ch = [], []
            for c in range(C):
                r, rho = _safe_corr(pred_prices[i, :, c], true_prices[i, :, c])
                ic_per_ch.append(r)
                rankic_per_ch.append(rho)
            price_ics.append(float(np.mean(ic_per_ch)))
            price_rankics.append(float(np.mean(rankic_per_ch)))

        # ================================================================
        # 3. Return IC / RankIC  (Kronos paper §4)
        #    r_hat = pred_close[:, -1] / last_close - 1
        #    r_true = true_close[:, -1] / last_close - 1
        #    (cross-sectional, collected over all samples)
        # ================================================================
        for i in range(B):
            lc = float(last_close[i])
            all_pred_returns.append(float(pred_prices[i, -1, 3]) / lc - 1.0)
            all_true_returns.append(float(true_prices[i, -1, 3]) / lc - 1.0)

        # ================================================================
        # 4. Optional CSV output
        # ================================================================
        if output_csv:
            base_timestamps   = [s["base_timestamp"]    for s in batch]
            future_timestamps = [s["future_timestamps"] for s in batch]
            for b in range(B):
                lc = float(last_close[b])
                for h in range(H):
                    csv_rows.append({
                        "base_timestamp":   base_timestamps[b],
                        "target_timestamp": future_timestamps[b][h],
                        "horizon":          h + 1,
                        "last_close":       lc,
                        "pred_open":        round(float(pred_prices[b, h, 0]), 2),
                        "pred_high":        round(float(pred_prices[b, h, 1]), 2),
                        "pred_low":         round(float(pred_prices[b, h, 2]), 2),
                        "pred_close":       round(float(pred_prices[b, h, 3]), 2),
                        "true_open":        round(float(true_prices[b, h, 0]), 2),
                        "true_high":        round(float(true_prices[b, h, 1]), 2),
                        "true_low":         round(float(true_prices[b, h, 2]), 2),
                        "true_close":       round(float(true_prices[b, h, 3]), 2),
                    })

    # ---- save CSV --------------------------------------------------------
    if output_csv and csv_rows:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved predictions to {out_path}")

    # ---- aggregate metrics -----------------------------------------------
    close_mae = total_abs_error / max(total_count, 1)

    # price series IC / RankIC: mean over all samples
    price_ic     = float(np.mean(price_ics))     if price_ics     else 0.0
    price_rankic = float(np.mean(price_rankics)) if price_rankics else 0.0

    # return IC / RankIC: cross-sectional over all samples
    arr_pr = np.asarray(all_pred_returns)
    arr_tr = np.asarray(all_true_returns)
    return_ic, return_rankic = _safe_corr(arr_pr, arr_tr)

    return {
        "close_mae":      close_mae,
        "num_close_pts":  total_count,
        "price_ic":       price_ic,
        "price_rankic":   price_rankic,
        "return_ic":      return_ic,
        "return_rankic":  return_rankic,
        "num_samples":    len(price_ics),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="examples/Bitcoin/train_files/starvla_train_bitcoin.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to *_action_model.pt or *.safetensors checkpoint")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None, help="Path to save per-sample OHLC predictions as CSV")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    if args.limit_samples is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_framework(cfg)
    state_dict = load_state_dict(args.ckpt)
    missing, unexpected = model.action_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    metrics = evaluate(model, dataloader, output_csv=args.output_csv)
    metrics.update({
        "split":      args.split,
        "checkpoint": args.ckpt,
    })

    # ---- pretty print ---------------------------------------------------
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Split          : {metrics['split']}")
    print(f"  Samples        : {metrics['num_samples']}")
    print(f"  Close MAE      : {metrics['close_mae']:.4f}")
    print(f"  Price IC       : {metrics['price_ic']:.6f}")
    print(f"  Price RankIC   : {metrics['price_rankic']:.6f}")
    print(f"  Return IC      : {metrics['return_ic']:.6f}")
    print(f"  Return RankIC  : {metrics['return_rankic']:.6f}")
    print("=" * 55)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
