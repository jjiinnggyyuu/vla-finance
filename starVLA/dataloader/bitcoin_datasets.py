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
        split_start: str | None = None,
        split_end: str | None = None,
        eth_csv_path: Path | None = None,
        use_image: bool = True,
        num_text_candles: int = 10,
        use_volume_in_text: bool = False,
        use_roc_in_text: bool = False,
        timeframe_label: str = "hourly",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.input_window = int(input_window)
        self.prediction_horizon = int(prediction_horizon)
        self.stride = int(stride)
        self.image_size = tuple(image_size)
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}
        self.target_mode = target_mode
        self.text_precision = int(text_precision)
        self.split_start = _parse_date_bound(split_start)
        self.split_end = _parse_date_bound(split_end, is_end=True)
        self.use_image = bool(use_image)
        self.num_text_candles = int(num_text_candles)
        self.use_volume_in_text = bool(use_volume_in_text)
        self.use_roc_in_text = bool(use_roc_in_text)
        self.timeframe_label = str(timeframe_label)

        if self.target_mode not in ("log_ratio", "delta"):
            raise ValueError(f"Unsupported target_mode: {self.target_mode!r}. Choose 'log_ratio' or 'delta'.")

        self.rows = _load_rows(self.csv_path, self.columns)
        min_len = self.input_window + self.prediction_horizon
        if len(self.rows) < min_len:
            raise ValueError(f"Need at least {min_len} rows, got {len(self.rows)} from {self.csv_path}")

        self.starts = [
            start
            for start in range(0, len(self.rows) - min_len + 1, self.stride)
            if self._is_in_split(start)
        ]

        # Optional ETH context: load and index by timestamp string for O(1) lookup
        self.eth_rows: dict[str, dict] = {}
        if eth_csv_path is not None:
            eth_path = Path(eth_csv_path)
            if eth_path.exists():
                eth_list = _load_rows(eth_path, self.columns)
                self.eth_rows = {r["timestamp"]: r for r in eth_list}
                print(f"[Dataset] Loaded {len(self.eth_rows)} ETH rows from {eth_path}")
            else:
                print(f"[Dataset] Warning: eth_csv_path={eth_path} not found. ETH context disabled.")

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

        # ETH context: past candles (text) + future candles (aux target)
        eth_past: list[dict] | None = None
        eth_action = None
        if self.eth_rows:
            # Past: last num_text_candles for text prompt
            recent_btc = past[-self.num_text_candles:]
            eth_past_candles = [self.eth_rows.get(r["timestamp"]) for r in recent_btc]
            if all(c is not None for c in eth_past_candles):
                eth_past = eth_past_candles  # type: ignore[assignment]

            # Future: 12 candles as auxiliary prediction target
            eth_future_candles = [self.eth_rows.get(r["timestamp"]) for r in future]
            if all(c is not None for c in eth_future_candles):
                # Use ETH's own last-known close as the normalisation base
                eth_last_close_row = self.eth_rows.get(past[-1]["timestamp"])
                if eth_last_close_row is not None:
                    eth_last_close = eth_last_close_row["close"]
                    eth_action = self._make_target(
                        eth_future_candles,  # type: ignore[arg-type]
                        eth_last_close,
                    )

        text = self._format_text(past, eth_past=eth_past)
        action = self._make_target(future, last_close)

        sample = {
            "image": [image],
            "lang": text,
            "action": action,
            "last_close": np.array(last_close, dtype=np.float32),
            "base_timestamp": past[-1]["timestamp"],
            "future_timestamps": [r["timestamp"] for r in future],
        }
        if eth_action is not None:
            sample["eth_action"] = eth_action
        return sample

    def _make_target(self, future: list[dict], last_close: float) -> np.ndarray:
        prices = np.array([[r["open"], r["high"], r["low"], r["close"]] for r in future], dtype=np.float32)
        if self.target_mode == "log_ratio":
            return _safe_log_ratio(prices, last_close)
        # delta: each candle normalized by previous candle's close
        T = len(future)
        delta = np.zeros_like(prices)
        prev_close = last_close
        for t in range(T):
            delta[t] = _safe_log_ratio(prices[t:t+1], prev_close)[0]
            prev_close = float(prices[t, 3])
        return delta

    def _format_text(self, past: list[dict], eth_past: list[dict] | None = None) -> str:
        dt = past[-1]["timestamp_dt"]
        last_close = past[-1]["close"]

        def _roc_str(candles, label):
            """t-12 기준으로 1,3,6,12시간 후 변화율.
            candles[-12]이 base, candles[-11/-9/-6/-1]이 1/3/6/12h 후."""
            if len(candles) < 12:
                return ""
            base = candles[-12]["close"]
            if base <= 0:
                return ""
            offsets = {1: -11, 3: -9, 6: -6, 12: -1}
            parts = []
            for h, idx in offsets.items():
                v = (candles[idx]["close"] - base) / base * 100
                sign = "+" if v >= 0 else ""
                parts.append(f"{h}h={sign}{v:.2f}%")
            return f"{label} change (from t-12): " + "  ".join(parts)

        n_candles = self.num_text_candles
        vol_header = " / volume" if self.use_volume_in_text else ""

        tf = self.timeframe_label
        lines = [
            f"Bitcoin (BTC/USDT) {tf} market data.",
            f"Time: {dt.strftime('%A')} {dt.hour:02d}:{dt.minute:02d} UTC, {dt.strftime('%B')} {dt.year}",
            f"Current price: ${last_close:,.2f}",
        ]
        if self.use_roc_in_text:
            btc_roc = _roc_str(past, "BTC")
            if btc_roc:
                lines.append(btc_roc)
        lines += [
            "",
            f"Recent {n_candles} candles (open / high / low / close{vol_header}):",
        ]

        recent = past[-n_candles:]
        n = len(recent)
        for i, row in enumerate(recent):
            offset = n - i
            candle = (f"t-{offset:<2}: O={row['open']:.2f} H={row['high']:.2f}"
                      f" L={row['low']:.2f} C={row['close']:.2f}")
            if self.use_volume_in_text:
                candle += f" V={row['volume']:.2f}"
            lines.append(candle)

        # ETH context section (only when ETH data is available)
        if eth_past is not None:
            eth_last_close = eth_past[-1]["close"]
            lines.append("")
            lines.append(f"Correlated asset ETH/USDT (current: ${eth_last_close:,.2f}):")
            if self.use_roc_in_text:
                eth_roc = _roc_str(eth_past, "ETH")
                if eth_roc:
                    lines.append(eth_roc)
            n_eth = len(eth_past)
            for i, row in enumerate(eth_past):
                offset = n_eth - i
                candle = (f"t-{offset:<2}: O={row['open']:.2f} H={row['high']:.2f}"
                          f" L={row['low']:.2f} C={row['close']:.2f}")
                if self.use_volume_in_text:
                    candle += f" V={row['volume']:.2f}"
                lines.append(candle)

        lines.append("")
        if eth_past is not None:
            lines.append(
                f"Predict the next {self.prediction_horizon} {tf} BTC OHLC candles"
                " based on the BTC and ETH context above."
            )
        else:
            lines.append(f"Predict the next {self.prediction_horizon} {tf} BTC OHLC candles.")

        return "\n".join(lines)

    def _render_chart(self, past: list[dict]) -> Image.Image:
        if not self.use_image:
            return Image.new("RGB", self.image_size, (128, 128, 128))
        width, height = self.image_size
        margin_left, margin_right, margin_top, margin_bottom = 44, 12, 18, 32
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # ── Volume area: bottom 22% of plot, 4px gap above ───────────
        vol_ratio   = 0.22
        vol_gap     = 4
        price_plot_h = int(plot_h * (1.0 - vol_ratio))
        vol_plot_h   = plot_h - price_plot_h - vol_gap
        vol_area_top = margin_top + price_plot_h + vol_gap
        vol_area_bot = margin_top + plot_h

        closes = np.array([r["close"] for r in past], dtype=np.float32)
        highs = np.array([r["high"] for r in past], dtype=np.float32)
        lows = np.array([r["low"] for r in past], dtype=np.float32)
        volumes = np.array([r["volume"] for r in past], dtype=np.float32)
        n = len(past)

        # ── Price range ───────────────────────────────────────────────
        p_min, p_max = float(lows.min()), float(highs.max())
        pad = max((p_max - p_min) * 0.04, 1e-6)
        p_min -= pad
        p_max += pad

        step = plot_w / max(n, 1)

        def y(price: float) -> float:
            return margin_top + (p_max - price) / (p_max - p_min) * price_plot_h

        def x_of(i: int) -> float:
            return margin_left + step * (i + 0.5)

        # ── Grid & border (price area) ───────────────────────────────
        draw.rectangle([margin_left, margin_top, width - margin_right, margin_top + price_plot_h], outline=(215, 220, 225))
        for frac in (0.25, 0.5, 0.75):
            yy = margin_top + price_plot_h * frac
            draw.line([margin_left, yy, width - margin_right, yy], fill=(235, 238, 240))

        # ── Candles ──────────────────────────────────────────────────
        body_w = max(2, int(step * 0.55))
        for i, row in enumerate(past):
            xc = x_of(i)
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = (20, 145, 85) if c >= o else (210, 65, 65)
            draw.line([xc, y(h), xc, y(l)], fill=color, width=1)
            y_o, y_c = y(o), y(c)
            top, bottom = min(y_o, y_c), max(y_o, y_c)
            if math.isclose(top, bottom):
                bottom = top + 1
            draw.rectangle([xc - body_w / 2, top, xc + body_w / 2, bottom], fill=color, outline=color)

        # ── Volume bars ───────────────────────────────────────────────
        vol_max = float(volumes.max()) if volumes.max() > 0 else 1.0
        for i, row in enumerate(past):
            xc = x_of(i)
            bar_h = int((row["volume"] / vol_max) * vol_plot_h)
            bar_h = max(bar_h, 1)
            bar_top = vol_area_bot - bar_h
            color = (20, 145, 85) if row["close"] >= row["open"] else (210, 65, 65)
            vol_color = tuple(int(c * 0.6 + 255 * 0.4) for c in color)  # 투명도 효과
            draw.rectangle([xc - body_w / 2, bar_top, xc + body_w / 2, vol_area_bot], fill=vol_color)

        # ── Volume area border ────────────────────────────────────────
        draw.rectangle([margin_left, vol_area_top, width - margin_right, vol_area_bot], outline=(215, 220, 225))

        return img


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> BitcoinOhlcDataset:
    data_root = Path(_cfg_get(data_cfg, "data_root_dir"))
    csv_file = _cfg_get(data_cfg, "csv_file", "btc_1h.csv")
    csv_path = data_root / csv_file

    # Optional ETH correlated asset context
    eth_csv_file = _cfg_get(data_cfg, "eth_csv_file", None)
    eth_csv_path = (data_root / eth_csv_file) if eth_csv_file else None

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
        split_start=_cfg_get(split_cfg, "start", None),
        split_end=_cfg_get(split_cfg, "end", None),
        eth_csv_path=eth_csv_path,
        use_image=_cfg_get(data_cfg, "use_image", True),
        # `num_text_candles` is the canonical key; `max_text_candles` is a legacy
        # alias kept for older YAMLs. num_text_candles wins when both are present
        # (e.g. v11 sets num=40, max=10 → 40 candles).
        num_text_candles=(
            _cfg_get(data_cfg, "num_text_candles", None)
            if _cfg_get(data_cfg, "num_text_candles", None) is not None
            else _cfg_get(data_cfg, "max_text_candles", 10)
        ),
        use_volume_in_text=_cfg_get(data_cfg, "use_volume_in_text", False),
        use_roc_in_text=_cfg_get(data_cfg, "use_roc_in_text", False),
        timeframe_label=_cfg_get(data_cfg, "timeframe_label", "hourly"),
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
