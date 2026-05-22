"""Evaluate Bitcoin GR00T checkpoints with one metric: close MAE.

The model predicts log-ratio OHLC targets:

    pred = log(future_ohlc / last_input_close)

This script restores close prices and computes mean absolute error over all
samples and all 12 forecast horizons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def load_state_dict(path: str) -> dict:
    ckpt_path = Path(path)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(ckpt_path))
    return torch.load(str(ckpt_path), map_location="cpu")


def restore_prices(log_ratio: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    return np.exp(log_ratio) * last_close[:, None]


@torch.inference_mode()
def evaluate_close_mae(model, dataloader, output_csv: str = None) -> dict:
    total_abs_error = 0.0
    total_count = 0
    csv_rows = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pred = model.predict_action(batch)["normalized_actions"]  # [B, 12, 4], log-ratio scale
        target = np.stack([sample["action"] for sample in batch], axis=0)
        last_close = np.asarray([sample["last_close"] for sample in batch], dtype=np.float32)

        # 실제 가격으로 복원 (B, 12, 4)
        pred_prices  = restore_prices(pred.reshape(-1, 4),   np.repeat(last_close, pred.shape[1])).reshape(-1, 12, 4)
        true_prices  = restore_prices(target.reshape(-1, 4), np.repeat(last_close, target.shape[1])).reshape(-1, 12, 4)

        pred_close = pred_prices[:, :, 3]
        true_close = true_prices[:, :, 3]
        abs_error = np.abs(pred_close - true_close)

        total_abs_error += float(abs_error.sum())
        total_count += int(abs_error.size)

        # CSV 저장용 행 수집
        if output_csv:
            base_timestamps   = [sample["base_timestamp"]    for sample in batch]
            future_timestamps = [sample["future_timestamps"] for sample in batch]
            for b in range(pred_prices.shape[0]):
                lc = float(last_close[b])
                for h in range(pred_prices.shape[1]):
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

    # CSV 저장
    if output_csv and csv_rows:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved predictions to {output_path}")

    close_mae = total_abs_error / max(total_count, 1)
    return {
        "metric": "close_mae",
        "close_mae": close_mae,
        "num_close_points": total_count,
    }


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

    metrics = evaluate_close_mae(model, dataloader, output_csv=args.output_csv)
    metrics.update(
        {
            "split": args.split,
            "checkpoint": args.ckpt,
            "num_samples": len(dataset),
        }
    )
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
