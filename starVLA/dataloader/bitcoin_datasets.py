"""Bitcoin OHLCVA dataset for StarVLA/GR00T experiments.

This dataset maps market windows to the sample contract expected by QwenGR00T:

    {"image": [PIL.Image], "lang": str, "action": np.ndarray[T, 4]}

The action target is future OHLC represented as log ratios against the last
input close by default. Use `denormalize_ohlc` below to convert predictions
back to prices.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


DEFAULT_COLUMNS = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}


def collate_fn(batch):
    return batch


def _cfg_get(cfg, key, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _load_rows(csv_path: Path, columns: dict[str, str]) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = [columns[k] for k in ("open", "high", "low", "close", "volume")]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for raw in reader:
            timestamp = raw.get(columns.get("timestamp", "timestamp"), "")
            row = {
                "timestamp": timestamp,
                "timestamp_dt": _parse_timestamp(timestamp),
                "open": float(raw[columns["open"]]),
                "high": float(raw[columns["high"]]),
                "low": float(raw[columns["low"]]),
                "close": float(raw[columns["close"]]),
                "volume": float(raw[columns["volume"]]),
                "amount": float(raw[columns["amount"]]) if columns.get("amount") in raw and raw[columns["amount"]] else 0.0,
            }
            rows.append(row)
    return rows


def _parse_timestamp(value: str) -> datetime:
    value = str(value).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _parse_date_bound(value: str | None, *, is_end: bool = False) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 10:
        dt = datetime.fromisoformat(text)
        return datetime.combine(dt.date(), time.max if is_end else time.min)
    return _parse_timestamp(text)


def _safe_log_ratio(values: np.ndarray, base: float) -> np.ndarray:
    eps = 1e-12
    return np.log(np.maximum(values, eps) / max(float(base), eps)).astype(np.float32)


def denormalize_ohlc(prediction: np.ndarray, last_close: float) -> np.ndarray:
    """Convert log-ratio OHLC predictions back to absolute prices."""
    return np.exp(prediction) * float(last_close)


class BitcoinOhlcDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        input_window: int = 80,
        prediction_horizon: int = 12,
        stride: int = 1,
        image_size: tuple[int, int] = (448, 448),
        columns: dict[str, str] | None = None,
        target_mode: str = "log_ratio",
        text_precision: int = 2,
        max_text_candles: int | None = 80,
        split_start: str | None = None,
        split_end: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.input_window = int(input_window)
        self.prediction_horizon = int(prediction_horizon)
        self.stride = int(stride)
        self.image_size = tuple(image_size)
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}
        self.target_mode = target_mode
        self.text_precision = int(text_precision)
        self.max_text_candles = max_text_candles
        self.split_start = _parse_date_bound(split_start)
        self.split_end = _parse_date_bound(split_end, is_end=True)

        if self.target_mode != "log_ratio":
            raise ValueError("Only target_mode='log_ratio' is implemented for the first Bitcoin prototype.")

        self.rows = _load_rows(self.csv_path, self.columns)
        min_len = self.input_window + self.prediction_horizon
        if len(self.rows) < min_len:
            raise ValueError(f"Need at least {min_len} rows, got {len(self.rows)} from {self.csv_path}")

        self.starts = [
            start
            for start in range(0, len(self.rows) - min_len + 1, self.stride)
            if self._is_in_split(start)
        ]

    def _is_in_split(self, start: int) -> bool:
        end = start + self.input_window + self.prediction_horizon
        sample_start = self.rows[start]["timestamp_dt"]
        sample_end = self.rows[end - 1]["timestamp_dt"]
        if self.split_start is not None and sample_start < self.split_start:
            return False
        if self.split_end is not None and sample_end > self.split_end:
            return False
        return True

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict:
        start = self.starts[index]
        split = start + self.input_window
        end = split + self.prediction_horizon
        past = self.rows[start:split]
        future = self.rows[split:end]

        last_close = past[-1]["close"]
        image = self._render_chart(past)
        text = self._format_text(past)
        action = self._make_target(future, last_close)

        return {
            "image": [image],
            "lang": text,
            "action": action,
            "last_close": np.array(last_close, dtype=np.float32),
            "base_timestamp": past[-1]["timestamp"],        # 마지막 입력 캔들 시각 (t0)
            "future_timestamps": [r["timestamp"] for r in future],  # 예측 대상 시각 12개
        }

    def _make_target(self, future: list[dict], last_close: float) -> np.ndarray:
        prices = np.array([[r["open"], r["high"], r["low"], r["close"]] for r in future], dtype=np.float32)
        return _safe_log_ratio(prices, last_close)

    def _format_text(self, past: list[dict]) -> str:
        dt = past[-1]["timestamp_dt"]
        last_close = past[-1]["close"]

        lines = [
            "Bitcoin (BTC/USDT) hourly market data.",
            f"Time: {dt.strftime('%A')} {dt.hour:02d}:00 UTC, {dt.strftime('%B')} {dt.year}",
            f"Current price: ${last_close:,.2f}",
            "",
            "Recent 10 candles (open / high / low / close):",
        ]

        recent = past[-10:]
        n = len(recent)
        for i, row in enumerate(recent):
            offset = n - i  # counts down: 10, 9, ..., 1
            lines.append(
                f"t-{offset:<2}: O={row['open']:.2f} H={row['high']:.2f}"
                f" L={row['low']:.2f} C={row['close']:.2f}"
            )

        lines.append("")
        lines.append(f"Predict the next {self.prediction_horizon} hourly BTC OHLC candles.")

        return "\n".join(lines)

    def _render_chart(self, past: list[dict]) -> Image.Image:
        width, height = self.image_size
        margin_left, margin_right, margin_top, margin_bottom = 44, 12, 18, 32
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        highs = np.array([r["high"] for r in past], dtype=np.float32)
        lows = np.array([r["low"] for r in past], dtype=np.float32)
        p_min, p_max = float(lows.min()), float(highs.max())
        pad = max((p_max - p_min) * 0.04, 1e-6)
        p_min -= pad
        p_max += pad

        def y(price: float) -> float:
            return margin_top + (p_max - price) / (p_max - p_min) * plot_h

        draw.rectangle([margin_left, margin_top, width - margin_right, height - margin_bottom], outline=(215, 220, 225))
        for frac in (0.25, 0.5, 0.75):
            yy = margin_top + plot_h * frac
            draw.line([margin_left, yy, width - margin_right, yy], fill=(235, 238, 240))

        step = plot_w / max(len(past), 1)
        body_w = max(2, int(step * 0.55))
        for i, row in enumerate(past):
            x = margin_left + step * (i + 0.5)
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = (20, 145, 85) if c >= o else (210, 65, 65)
            draw.line([x, y(h), x, y(l)], fill=color, width=1)
            y_o, y_c = y(o), y(c)
            top, bottom = min(y_o, y_c), max(y_o, y_c)
            if math.isclose(top, bottom):
                bottom = top + 1
            draw.rectangle([x - body_w / 2, top, x + body_w / 2, bottom], fill=color, outline=color)

        return img


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> BitcoinOhlcDataset:
    data_root = Path(_cfg_get(data_cfg, "data_root_dir"))
    csv_file = _cfg_get(data_cfg, "csv_file", "btc_1h.csv")
    csv_path = data_root / csv_file

    image_size = _cfg_get(data_cfg, "image_size", [448, 448])
    columns = dict(_cfg_get(data_cfg, "columns", DEFAULT_COLUMNS))
    splits = _cfg_get(data_cfg, "splits", {})
    split_cfg = splits.get(mode, {}) if hasattr(splits, "get") else {}
    return BitcoinOhlcDataset(
        csv_path=csv_path,
        input_window=_cfg_get(data_cfg, "input_window", 80),
        prediction_horizon=_cfg_get(data_cfg, "prediction_horizon", 12),
        stride=_cfg_get(data_cfg, "stride", 1),
        image_size=(int(image_size[0]), int(image_size[1])),
        columns=columns,
        target_mode=_cfg_get(data_cfg, "target_mode", "log_ratio"),
        text_precision=_cfg_get(data_cfg, "text_precision", 2),
        max_text_candles=_cfg_get(data_cfg, "max_text_candles", 80),
        split_start=_cfg_get(split_cfg, "start", None),
        split_end=_cfg_get(split_cfg, "end", None),
    )


def build_bitcoin_dataloader(cfg, mode: str = "train") -> DataLoader:
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=mode)
    num_workers = int(_cfg_get(cfg.datasets.vla_data, "num_workers", 4))
    loader_kwargs = {
        "batch_size": cfg.datasets.vla_data.per_device_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": True,
        "shuffle": mode == "train",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)
