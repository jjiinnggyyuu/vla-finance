"""Multi-asset OHLCVA dataset for StarVLA/GR00T experiments.

This dataset maps market windows to the sample contract expected by QwenGR00T:

    {"image": [PIL.Image], "lang": str, "action": np.ndarray[T, 4*N]}

The action target is the future OHLC of *N assets* concatenated along the last
axis, represented as per-step delta log-returns and normalised per asset by a
fixed standard deviation computed on the training split. The first asset in the
list is the *primary* asset (drives the sliding windows and the chart image);
the remaining assets are co-predicted (folded into the same action vector) and
shown as correlated-asset context in the text prompt.

Config (datasets.vla_data):

    assets:
      - {name: BTC, csv: crypto_data/btc_1h.csv, weight: 0.6}
      - {name: ETH, csv: crypto_data/eth_1h.csv, weight: 0.2}
      - {name: XRP, csv: crypto_data/xrp_1h.csv, weight: 0.2}

A single-asset list reproduces the BTC-only setup. Use `denormalize_ohlc` /
the per-sample `asset_std` + `last_close` to convert predictions back to prices.
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


def _compute_asset_delta_std(rows: list[dict], start_dt, end_dt) -> float:
    """Per-asset scalar std of per-step delta log-returns over a date range.

    delta[t] = log(ohlc[t] / close[t-1]) for o/h/l/c, pooled across channels.
    Used to normalise targets to ~unit variance (matches the N(0,1) flow-matching
    noise scale and balances assets of different volatility). Computed on the
    *training* range only to avoid look-ahead leakage; the same value is reused
    for validation/test/eval.
    """
    deltas: list[float] = []
    for t in range(1, len(rows)):
        ts = rows[t]["timestamp_dt"]
        if start_dt is not None and ts < start_dt:
            continue
        if end_dt is not None and ts > end_dt:
            continue
        prev_close = rows[t - 1]["close"]
        if prev_close <= 0:
            continue
        for k in ("open", "high", "low", "close"):
            v = rows[t][k]
            if v > 0:
                deltas.append(math.log(v / prev_close))
    if not deltas:
        return 1.0
    std = float(np.std(np.asarray(deltas, dtype=np.float64)))
    return std if std > 1e-8 else 1.0


class BitcoinOhlcDataset(Dataset):
    def __init__(
        self,
        assets: list[dict],
        input_window: int = 80,
        prediction_horizon: int = 12,
        stride: int = 1,
        image_size: tuple[int, int] = (448, 448),
        columns: dict[str, str] | None = None,
        target_mode: str = "log_ratio",
        text_precision: int = 2,
        split_start: str | None = None,
        split_end: str | None = None,
        use_image: bool = True,
        num_text_candles: int = 10,
        use_volume_in_text: bool = False,
        use_roc_in_text: bool = False,
        input_relative: bool = False,
        timeframe_label: str = "hourly",
    ) -> None:
        """
        Args:
            assets: ordered list of dicts, each with keys:
                - "name" (str): display name, e.g. "BTC"
                - "rows" (list[dict]): loaded OHLCV rows (from _load_rows)
                - "std"  (float): normalisation std (1.0 disables normalisation)
                - "weight" (float): loss weight (consumed by the action head, kept
                   here only for bookkeeping / passthrough)
              assets[0] is the primary asset and drives the sliding windows.
        """
        if not assets:
            raise ValueError("`assets` must contain at least one asset.")
        self.assets = assets
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
        # Input normalisation: render candle OHLC as % change from the asset's
        # current price (anchor), making the text scale-invariant across time
        # (BTC $19k vs $100k look identical) — addresses distribution shift.
        self.input_relative = bool(input_relative)
        self.timeframe_label = str(timeframe_label)

        if self.target_mode not in ("log_ratio", "delta"):
            raise ValueError(f"Unsupported target_mode: {self.target_mode!r}. Choose 'log_ratio' or 'delta'.")

        # Primary asset drives the windows; the rest are looked up by timestamp.
        self.primary = self.assets[0]
        self.rows = self.primary["rows"]
        # Per-asset O(1) timestamp -> row index lookup for the co-predicted assets.
        for a in self.assets:
            a["index"] = {r["timestamp"]: r for r in a["rows"]}

        min_len = self.input_window + self.prediction_horizon
        if len(self.rows) < min_len:
            raise ValueError(f"Need at least {min_len} primary rows, got {len(self.rows)}")

        # A primary row is "aligned" if every co-predicted asset has that timestamp.
        # A window is valid only if every row inside it is aligned (so all assets
        # share an identical timestamp set across the whole window).
        other_indices = [a["index"] for a in self.assets[1:]]
        aligned = np.fromiter(
            (all(r["timestamp"] in idx for idx in other_indices) for r in self.rows),
            dtype=np.int64,
            count=len(self.rows),
        )
        cum = np.concatenate([[0], np.cumsum(aligned)])  # prefix sum for O(1) window check

        self.starts = []
        for start in range(0, len(self.rows) - min_len + 1, self.stride):
            end = start + min_len
            if cum[end] - cum[start] != min_len:
                continue  # some asset is missing a timestamp in this window
            if self._is_in_split(start):
                self.starts.append(start)

        names = ", ".join(f"{a['name']}(std={a['std']:.4g})" for a in self.assets)
        print(f"[Dataset] assets=[{names}]  samples={len(self.starts)}")

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

    def _aligned_window(self, asset: dict, primary_past: list[dict], primary_future: list[dict]):
        """Return (past, future) candle lists for `asset` aligned to the primary
        timestamps. The primary asset returns its own rows directly."""
        if asset is self.primary:
            return primary_past, primary_future
        idx = asset["index"]
        past = [idx[r["timestamp"]] for r in primary_past]
        future = [idx[r["timestamp"]] for r in primary_future]
        return past, future

    def __getitem__(self, index: int) -> dict:
        start = self.starts[index]
        split = start + self.input_window
        end = split + self.prediction_horizon
        primary_past = self.rows[start:split]
        primary_future = self.rows[split:end]

        image = self._render_chart(primary_past)

        actions = []          # per-asset (H, 4) normalised delta
        last_closes = []      # per-asset last close (for price restoration)
        stds = []             # per-asset normalisation std
        asset_pasts = []      # (name, past_candles) for text
        for a in self.assets:
            a_past, a_future = self._aligned_window(a, primary_past, primary_future)
            a_last_close = a_past[-1]["close"]
            a_delta = self._make_target(a_future, a_last_close)   # (H, 4) raw delta
            a_std = float(a["std"])
            actions.append(a_delta / a_std)                       # normalise
            last_closes.append(a_last_close)
            stds.append(a_std)
            asset_pasts.append((a["name"], a_past[-self.num_text_candles:]))

        action = np.concatenate(actions, axis=1).astype(np.float32)  # (H, 4*N)
        text = self._format_text(asset_pasts)

        return {
            "image": [image],
            "lang": text,
            "action": action,
            "last_close": np.asarray(last_closes, dtype=np.float32),   # (N,)
            "asset_std": np.asarray(stds, dtype=np.float32),           # (N,)
            "asset_names": [a["name"] for a in self.assets],
            "base_timestamp": primary_past[-1]["timestamp"],
            "future_timestamps": [r["timestamp"] for r in primary_future],
        }

    def _make_target(self, future: list[dict], last_close: float) -> np.ndarray:
        prices = np.array([[r["open"], r["high"], r["low"], r["close"]] for r in future], dtype=np.float32)
        if self.target_mode == "log_ratio":
            return _safe_log_ratio(prices, last_close)
        # delta: each candle normalized by previous candle's close
        T = len(future)
        delta = np.zeros_like(prices)
        prev_close = last_close
        for t in range(T):
            delta[t] = _safe_log_ratio(prices[t:t + 1], prev_close)[0]
            prev_close = float(prices[t, 3])
        return delta

    def _format_text(self, asset_pasts: list[tuple[str, list[dict]]]) -> str:
        """Render the multi-asset prompt. asset_pasts[0] is the primary asset."""
        primary_name, primary_past = asset_pasts[0]
        dt = primary_past[-1]["timestamp_dt"]
        last_close = primary_past[-1]["close"]

        def _roc_str(candles, label):
            """Change vs t-12 base at 1/3/6/12 steps ahead."""
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

        vol_header = " / volume(V) + quote-amount(A,$M)" if self.use_volume_in_text else ""
        tf = self.timeframe_label

        def _render_candles(candles, anchor=None):
            n = len(candles)
            out = []
            if self.input_relative:
                base = float(anchor) if anchor else float(candles[-1]["close"])
                def _p(v):
                    r = (float(v) / base - 1.0) * 100.0 if base > 0 else 0.0
                    return f"{'+' if r >= 0 else ''}{r:.2f}%"
                for i, row in enumerate(candles):
                    offset = n - i
                    candle = (f"t-{offset:<2}: O={_p(row['open'])} H={_p(row['high'])}"
                              f" L={_p(row['low'])} C={_p(row['close'])}")
                    if self.use_volume_in_text:
                        candle += f" V={row['volume']:.2f} A={row['amount']/1e6:.2f}"
                    out.append(candle)
                return out
            for i, row in enumerate(candles):
                offset = n - i
                candle = (f"t-{offset:<2}: O={row['open']:.2f} H={row['high']:.2f}"
                          f" L={row['low']:.2f} C={row['close']:.2f}")
                if self.use_volume_in_text:
                    candle += f" V={row['volume']:.2f}"
                out.append(candle)
            return out

        lines = [
            f"{primary_name} {tf} market data.",
            f"Time: {dt.strftime('%A')} {dt.hour:02d}:{dt.minute:02d} UTC, {dt.strftime('%B')} {dt.year}",
            f"Current price: ${last_close:,.2f}",
        ]
        if self.use_roc_in_text:
            roc = _roc_str(primary_past, primary_name)
            if roc:
                lines.append(roc)
        _fmt_note = " as % vs current price" if self.input_relative else ""
        lines += ["", f"Recent {self.num_text_candles} candles (open / high / low / close{vol_header}){_fmt_note}:"]
        lines += _render_candles(primary_past, last_close)

        # Co-predicted / correlated assets
        for name, past in asset_pasts[1:]:
            lines.append("")
            lines.append(f"Correlated asset {name} (current: ${past[-1]['close']:,.2f}):")
            if self.use_roc_in_text:
                roc = _roc_str(past, name)
                if roc:
                    lines.append(roc)
            lines += _render_candles(past, past[-1]["close"])

        lines.append("")
        if len(asset_pasts) > 1:
            names = ", ".join(n for n, _ in asset_pasts)
            lines.append(
                f"Predict the next {self.prediction_horizon} {tf} OHLC candles for each of: {names}."
            )
        else:
            lines.append(f"Predict the next {self.prediction_horizon} {tf} {primary_name} OHLC candles.")

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
        vol_ratio = 0.22
        vol_gap = 4
        price_plot_h = int(plot_h * (1.0 - vol_ratio))
        vol_plot_h = plot_h - price_plot_h - vol_gap
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
            vol_color = tuple(int(c * 0.6 + 255 * 0.4) for c in color)
            draw.rectangle([xc - body_w / 2, bar_top, xc + body_w / 2, vol_area_bot], fill=vol_color)

        # ── Volume area border ────────────────────────────────────────
        draw.rectangle([margin_left, vol_area_top, width - margin_right, vol_area_bot], outline=(215, 220, 225))

        return img


def _resolve_assets(data_cfg, data_root: Path, columns: dict) -> list[dict]:
    """Build the ordered asset list from config, loading rows for each.

    Supports the new `assets:` list and a legacy fallback to `csv_file`
    (single primary asset). Each returned dict has name/rows/weight; `std`
    is filled in by the caller after the training range is known.
    """
    assets_cfg = _cfg_get(data_cfg, "assets", None)
    resolved = []
    if assets_cfg:
        for a in assets_cfg:
            name = _cfg_get(a, "name", "ASSET")
            csv_rel = _cfg_get(a, "csv", None)
            weight = float(_cfg_get(a, "weight", 1.0))
            if csv_rel is None:
                raise ValueError(f"asset {name!r} is missing a `csv` path")
            rows = _load_rows(data_root / csv_rel, columns)
            resolved.append({"name": name, "rows": rows, "weight": weight})
    else:
        # Legacy single-asset fallback (BTC-only).
        csv_file = _cfg_get(data_cfg, "csv_file", "btc_1h.csv")
        rows = _load_rows(data_root / csv_file, columns)
        resolved.append({"name": _cfg_get(data_cfg, "asset_name", "BTC"), "rows": rows, "weight": 1.0})
    return resolved


def get_vla_dataset(data_cfg, mode: str = "train", **kwargs) -> BitcoinOhlcDataset:
    data_root = Path(_cfg_get(data_cfg, "data_root_dir"))
    columns = dict(_cfg_get(data_cfg, "columns", DEFAULT_COLUMNS))
    image_size = _cfg_get(data_cfg, "image_size", [448, 448])
    splits = _cfg_get(data_cfg, "splits", {})
    split_cfg = splits.get(mode, {}) if hasattr(splits, "get") else {}

    assets = _resolve_assets(data_cfg, data_root, columns)

    # Per-asset normalisation std from the *training* range (no look-ahead).
    # Reused identically for train/val/test/eval so restoration is consistent.
    normalize = bool(_cfg_get(data_cfg, "normalize_targets", True))
    train_cfg = splits.get("train", {}) if hasattr(splits, "get") else {}
    train_start = _parse_date_bound(_cfg_get(train_cfg, "start", None))
    train_end = _parse_date_bound(_cfg_get(train_cfg, "end", None), is_end=True)
    for a in assets:
        a["std"] = _compute_asset_delta_std(a["rows"], train_start, train_end) if normalize else 1.0

    return BitcoinOhlcDataset(
        assets=assets,
        input_window=_cfg_get(data_cfg, "input_window", 80),
        prediction_horizon=_cfg_get(data_cfg, "prediction_horizon", 12),
        stride=_cfg_get(data_cfg, "stride", 1),
        image_size=(int(image_size[0]), int(image_size[1])),
        columns=columns,
        target_mode=_cfg_get(data_cfg, "target_mode", "log_ratio"),
        text_precision=_cfg_get(data_cfg, "text_precision", 2),
        split_start=_cfg_get(split_cfg, "start", None),
        split_end=_cfg_get(split_cfg, "end", None),
        use_image=_cfg_get(data_cfg, "use_image", True),
        # `num_text_candles` is the canonical key; `max_text_candles` is a legacy
        # alias kept for older YAMLs. num_text_candles wins when both are present.
        num_text_candles=(
            _cfg_get(data_cfg, "num_text_candles", None)
            if _cfg_get(data_cfg, "num_text_candles", None) is not None
            else _cfg_get(data_cfg, "max_text_candles", 10)
        ),
        use_volume_in_text=_cfg_get(data_cfg, "use_volume_in_text", False),
        use_roc_in_text=_cfg_get(data_cfg, "use_roc_in_text", False),
        input_relative=_cfg_get(data_cfg, "input_relative", False),
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
