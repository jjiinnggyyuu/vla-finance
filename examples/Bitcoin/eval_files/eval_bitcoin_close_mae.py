"""Evaluate Bitcoin checkpoints.

Metrics:
  - close_mae      : Mean Absolute Error on restored close prices.
  - price_ic       : Per-sample Pearson correlation between predicted and true
                     OHLC price sequences over the 12-step horizon, averaged
                     over 4 channels then over all samples. Measures "does the
                     predicted 12-step price trajectory have the right shape".
  - price_rankic   : Same as price_ic but using Spearman (rank) correlation.
  - da_k           : Directional accuracy at horizon k (k=1,3,6,12).
  - aer_k          : 1-month cumulative excess return of a long-short strategy vs
                     buy-and-hold BTC benchmark (geometric compounding).
                     Slippage deducted per trade via --slippage (default 0.1%).
  - sortino_k      : Raw Sortino ratio (mean / downside_std, no annualization).
                     Only penalises downside volatility. Relative comparison only.
  - mdd_k          : Maximum Drawdown of the strategy equity curve.
  - long/short/watch_ratio_k : Fraction of samples predicted long / short / no-trade.
  - win_rate_k     : Fraction of actual trades (excl. watch) that ended in profit.
  - profit_factor_k: Gross profit / gross loss over actual trades (>1 = profitable).

NOTE: stride=1 means windows overlap heavily; absolute AER/Sortino numbers are
      inflated vs a true non-overlapping backtest. Use for relative model comparison.
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


def restore_prices_delta(delta: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    """delta log-ratios → absolute prices via sequential reconstruction.
    delta: (B, H, 4),  last_close: (B,)
    """
    B, H, C = delta.shape
    prices = np.zeros_like(delta)
    for b in range(B):
        prev_close = float(last_close[b])
        for t in range(H):
            prices[b, t] = np.exp(delta[b, t]) * prev_close
            prev_close = float(prices[b, t, 3])  # update with current close
    return prices


# ---------------------------------------------------------------------------
# Trading strategies: map a predicted k-step return -> position in [-1, +1]
# ---------------------------------------------------------------------------
#   long_short : +1 / -1 / 0 with a confidence gate θ (|pred| must exceed θ)
#   long_only  : +1 / 0     (never shorts; spot-realistic, dodges short bias)
#   magnitude  : clip(pred / scale, -1, +1)  (size ∝ predicted move)
# Slippage is charged on |position| inside the backtest, so fractional positions
# pay proportionally less cost.

# Position functions take (sample, k) so context-aware strategies (e.g. a
# trend filter using the price series) can look beyond the single prediction.
def _ls_pos(theta: float):
    """Long-short with confidence gate θ."""
    return lambda s, k: 1.0 if s["pred_ret"][k] > theta else (-1.0 if s["pred_ret"][k] < -theta else 0.0)


def _lo_pos(theta: float):
    """Long-only with confidence gate θ (no shorting)."""
    return lambda s, k: 1.0 if s["pred_ret"][k] > theta else 0.0


def _mag_pos(scale: float):
    """Magnitude-scaled position: bet size proportional to predicted move."""
    sc = scale if scale > 1e-12 else 1e-12
    return lambda s, k: float(np.clip(s["pred_ret"][k] / sc, -1.0, 1.0))


def _downside_pos(theta: float):
    """Downside-protection overlay: default 100% long (capture B&H upside),
    step aside to cash only when the model strongly predicts a drop (< -θ)."""
    return lambda s, k: 1.0 if s["pred_ret"][k] > -theta else 0.0


def _trend_pos(theta: float):
    """Trend filter: long only when the model is bullish (>θ) AND price is in an
    uptrend (above its trailing SMA); else cash. Sidesteps bear regimes."""
    return lambda s, k: 1.0 if (s["pred_ret"][k] > theta and s.get("trend_up", True)) else 0.0


def _pct(theta: float) -> str:
    return f"{theta * 100:.2f}%"


def _safe_corr(x: np.ndarray, y: np.ndarray):
    """Return (pearson_r, spearman_r) or (0,0) if degenerate."""
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 0.0
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return float(r), float(rho)


def _plot_equity_curves(equity_curves: dict, plot_dir: str, k_values: list[int]) -> None:
    """k별 누적 수익 곡선(전략 vs BTC 홀드)을 2x2 서브플롯 PNG로 저장."""
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이 파일로만 저장
    import matplotlib.pyplot as plt

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, k in zip(axes.flat, k_values):
        equity_strat, equity_bench = equity_curves[k]
        ax.plot(equity_strat, label="Strategy", color="C0", linewidth=1.5)
        ax.plot(equity_bench, label="BTC Hold", color="C1", linewidth=1.2, alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        final_s = equity_strat[-1] if len(equity_strat) else 1.0
        final_b = equity_bench[-1] if len(equity_bench) else 1.0
        ax.set_title(f"k={k}  (strat x{final_s:.3f}  vs  hold x{final_b:.3f})")
        ax.set_xlabel("time step (hour)")
        ax.set_ylabel("cumulative equity (start=1.0)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "equity_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved equity curves to {out_path}")


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, dataloader, slippage: float = 0.001, target_mode: str = "log_ratio", output_csv: str | None = None, plot_dir: str | None = None, asset_index: int = 0) -> dict:
    """우리 VLA 모델로 추론 → (역정규화) → 가격 복원 → 공통 지표 계산.

    멀티자산: 모델은 (B, H, 4*N) 정규화 delta를 출력한다. 평가는 BTC(=primary,
    asset_index=0)만 한다 — 자산 4채널을 떼어, 해당 자산 std로 역정규화한 뒤
    가격을 복원한다. last_close/asset_std는 샘플당 (N,) 배열로 들어온다.

    예측 수집만 담당하고, 실제 지표 계산은 compute_metrics_from_prices()에
    위임한다. 이렇게 분리해야 Kronos 베이스라인도 동일한 지표 함수를 공유한다.
    """
    pred_prices_list: list[np.ndarray] = []  # 각 (12, 4)
    true_prices_list: list[np.ndarray] = []
    last_close_list: list[float] = []
    ts_list: list[str] = []
    future_ts_list: list[list[str]] = []

    a = asset_index
    sl = slice(a * 4, a * 4 + 4)   # this asset's 4 OHLC channels in the 4N vector

    for batch in tqdm(dataloader, desc="Evaluating"):
        pred = model.predict_action(batch)["normalized_actions"]  # (B, H, 4*N)
        target = np.stack([s["action"] for s in batch], axis=0)   # (B, H, 4*N)
        # Per-asset last_close / std (samples carry (N,) arrays).
        last_close = np.stack([np.atleast_1d(s["last_close"]) for s in batch]).astype(np.float32)  # (B, N)
        asset_std  = np.stack([np.atleast_1d(s["asset_std"])  for s in batch]).astype(np.float32)  # (B, N)

        B, H, _ = pred.shape
        # Slice this asset, then un-normalize delta back to raw scale (× std).
        std_a = asset_std[:, a][:, None, None]            # (B,1,1)
        pred_a = pred[:, :, sl] * std_a                    # (B, H, 4) raw delta
        true_a = target[:, :, sl] * std_a
        lc_a = last_close[:, a]                            # (B,)

        if target_mode == "delta":
            pred_prices = restore_prices_delta(pred_a, lc_a)
            true_prices = restore_prices_delta(true_a, lc_a)
        else:
            lc_rep = np.repeat(lc_a, H)
            pred_prices = restore_prices(pred_a.reshape(-1, 4), lc_rep).reshape(B, H, 4)
            true_prices = restore_prices(true_a.reshape(-1, 4), lc_rep).reshape(B, H, 4)

        for i in range(B):
            pred_prices_list.append(pred_prices[i])
            true_prices_list.append(true_prices[i])
            last_close_list.append(float(lc_a[i]))
            ts_list.append(batch[i]["base_timestamp"])
            future_ts_list.append(batch[i]["future_timestamps"])

    return compute_metrics_from_prices(
        pred_prices_list, true_prices_list, last_close_list, ts_list, future_ts_list,
        slippage=slippage, output_csv=output_csv, plot_dir=plot_dir,
    )


def compute_metrics_from_prices(
    pred_prices_list: list[np.ndarray],
    true_prices_list: list[np.ndarray],
    last_close_list: list[float],
    ts_list: list[str],
    future_ts_list: list[list] | None = None,
    slippage: float = 0.001,
    output_csv: str | None = None,
    plot_dir: str | None = None,
) -> dict:
    """복원된 절대 가격(pred/true)으로부터 모든 지표를 계산하는 공통 코어.

    우리 모델과 Kronos가 공유한다. 입력은 이미 실제 가격 단위의 (H, 4) 배열 리스트.

    Args:
        pred_prices_list: 샘플별 예측 OHLC, 각 (H, 4)
        true_prices_list: 샘플별 실제 OHLC, 각 (H, 4)
        last_close_list:  샘플별 입력 윈도우 마지막 close (수익률 base)
        ts_list:          샘플별 base_timestamp (정렬 키)
        future_ts_list:   샘플별 미래 타임스탬프 (CSV 출력용, 옵션)
    """
    K_VALUES = [1, 3, 6, 12]
    total_abs_error = 0.0
    total_count = 0
    price_ics: list[float] = []
    price_rankics: list[float] = []
    all_samples: list[dict] = []
    csv_rows: list[dict] = []

    for idx in range(len(pred_prices_list)):
        pred_prices = pred_prices_list[idx]   # (H, 4)
        true_prices = true_prices_list[idx]   # (H, 4)
        H, C = pred_prices.shape
        lc = float(last_close_list[idx])      # 수익률 base = 입력 마지막 close

        # Close MAE
        pred_close = pred_prices[:, 3]
        true_close = true_prices[:, 3]
        total_abs_error += float(np.abs(pred_close - true_close).sum())
        total_count += int(pred_close.size)

        # Price Series IC / RankIC (trajectory)
        ic_per_ch, rankic_per_ch = [], []
        for c in range(C):
            r, rho = _safe_corr(pred_prices[:, c], true_prices[:, c])
            ic_per_ch.append(r)
            rankic_per_ch.append(rho)
        price_ics.append(float(np.mean(ic_per_ch)))
        price_rankics.append(float(np.mean(rankic_per_ch)))

        # 트레이딩 시뮬레이션용 샘플 (현재가 base 대비 수익률)
        sample = {
            "ts": ts_list[idx],
            "lc": lc,                     # current price (for trend-filter SMA)
            "pred_ret": {k: float(pred_prices[k-1, 3]) / lc - 1.0 for k in K_VALUES},
            "true_ret": {k: float(true_prices[k-1, 3]) / lc - 1.0 for k in K_VALUES},
        }
        all_samples.append(sample)

        # CSV 출력
        if output_csv:
            fts = future_ts_list[idx] if future_ts_list else [""] * H
            for h in range(H):
                csv_rows.append({
                    "base_timestamp":   ts_list[idx],
                    "target_timestamp": fts[h],
                    "horizon":          h + 1,
                    "last_close":       lc,
                    "pred_open":  round(float(pred_prices[h, 0]), 2),
                    "pred_high":  round(float(pred_prices[h, 1]), 2),
                    "pred_low":   round(float(pred_prices[h, 2]), 2),
                    "pred_close": round(float(pred_prices[h, 3]), 2),
                    "true_open":  round(float(true_prices[h, 0]), 2),
                    "true_high":  round(float(true_prices[h, 1]), 2),
                    "true_low":   round(float(true_prices[h, 2]), 2),
                    "true_close": round(float(true_prices[h, 3]), 2),
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

    # ---- 타임스탬프 정렬 -----------------------------------------------
    all_samples.sort(key=lambda x: x["ts"])
    N = len(all_samples)

    # Trend filter context: uptrend if current price > trailing SMA of current
    # prices. Samples are consecutive (stride-1) so lc[] is the price series.
    # Uses only past/current prices (no look-ahead).
    _MA_WIN = 24
    _lc = [s["lc"] for s in all_samples]
    for i, s in enumerate(all_samples):
        lo = max(0, i - _MA_WIN + 1)
        sma = sum(_lc[lo:i + 1]) / (i - lo + 1)
        s["trend_up"] = bool(s["lc"] > sma)

    # ---- aggregate metrics -----------------------------------------------
    close_mae    = total_abs_error / max(total_count, 1)
    price_ic     = float(np.mean(price_ics))     if price_ics     else 0.0
    price_rankic = float(np.mean(price_rankics)) if price_rankics else 0.0

    def _direction_stats(samples, k, slippage):
        """Prediction-level (strategy-independent) stats: DA + long/short/watch ratios."""
        N = len(samples)
        dir_list = [1.0 if (s["pred_ret"][k] > 0) == (s["true_ret"][k] > 0) else 0.0
                    for s in samples]
        da = float(np.mean(dir_list)) if dir_list else 0.0

        n_long  = sum(1 for s in samples if s["pred_ret"][k] >  slippage)
        n_short = sum(1 for s in samples if s["pred_ret"][k] < -slippage)
        n_watch = N - n_long - n_short

        n_long_ns  = sum(1 for s in samples if s["pred_ret"][k] > 0.0)
        n_short_ns = sum(1 for s in samples if s["pred_ret"][k] < 0.0)
        ns = n_long_ns + n_short_ns
        return {
            "da": da,
            "long_ratio": n_long / N, "short_ratio": n_short / N, "watch_ratio": n_watch / N,
            "long_ratio_ns": (n_long_ns / ns) if ns else 0.5,
            "short_ratio_ns": (n_short_ns / ns) if ns else 0.5,
        }

    def _run_strategy(samples, k, pos_fn, slippage):
        """k-slot overlapping backtest for one position rule `pos_fn(pred_ret) -> [-1,1]`.

        Capital is split into k slots of 1/k each; slot `offset` trades at
        t=offset, offset+k, ... (non-overlapping within a slot) and compounds
        independently. Total equity = forward-filled sum of slots. Returns the
        principal return (ret), excess vs buy-and-hold (aer), MDD, win rate,
        trade count, and the equity curves.
        """
        N = len(samples)
        strat_equity_t = np.full(N, np.nan)
        bench_equity_t = np.full(N, np.nan)
        trade_pnls = []
        n_trades = 0

        for offset in range(k):
            cap_s = 1.0 / k
            cap_b = 1.0 / k
            for t in range(offset, N, k):
                s = samples[t]
                true_ret = s["true_ret"][k]
                position = pos_fn(s, k)                           # in [-1, +1]
                slot_ret = position * true_ret - abs(position) * slippage
                cap_s *= (1.0 + slot_ret)
                cap_b *= (1.0 + true_ret)
                strat_equity_t[t] = cap_s
                bench_equity_t[t] = cap_b
                if position != 0.0:
                    trade_pnls.append(slot_ret)
                    n_trades += 1

        def _forward_fill_sum(equity_t, k, init_each):
            slot_series = []
            for offset in range(k):
                idxs = list(range(offset, N, k))
                vals = equity_t[idxs]
                full = np.empty(N)
                cur = init_each
                j = 0
                for t in range(N):
                    if j < len(idxs) and idxs[j] == t:
                        cur = vals[j]
                        j += 1
                    full[t] = cur
                slot_series.append(full)
            return np.sum(slot_series, axis=0)

        equity_strat = np.concatenate([[1.0], _forward_fill_sum(strat_equity_t, k, 1.0 / k)])
        # Buy-and-hold: PURE hold from the first to the last price over the eval
        # window (k-independent). The old k-slot bench made B&H vary with k, which
        # is unnatural (holding doesn't depend on the trading horizon).
        _lc0 = float(samples[0]["lc"])
        equity_bench = np.concatenate([[1.0], np.array([float(s["lc"]) / _lc0 for s in samples])])

        ret_s = float(equity_strat[-1] - 1.0)   # principal return (원금대비 누적)
        ret_b = float(equity_bench[-1] - 1.0)   # buy-and-hold (k-independent, pure hold)
        aer = ret_s - ret_b

        run_max = np.maximum.accumulate(equity_strat)
        mdd = float(np.min((equity_strat - run_max) / run_max)) if len(equity_strat) else 0.0

        port_ret = np.diff(equity_strat) / equity_strat[:-1]
        mean_step = float(np.mean(port_ret)) if len(port_ret) else 0.0
        downside = port_ret[port_ret < 0]
        ds_step = float(np.std(downside)) if len(downside) > 1 else 0.0
        sortino = float(mean_step / ds_step) if ds_step > 1e-12 else 0.0

        pnls = np.asarray(trade_pnls)
        if len(pnls) > 0:
            wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
            win_rate = float(len(wins) / len(pnls))
            gp = float(wins.sum()); gl = float(-losses.sum())
            profit_factor = float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 1e-12 else 0.0)
        else:
            win_rate = 0.0; profit_factor = 0.0

        return {
            "ret": ret_s, "aer": aer, "bench_ret": ret_b, "mdd": mdd,
            "sortino": sortino, "win_rate": win_rate, "profit_factor": profit_factor,
            "num_trades": n_trades,
            "equity_strat": equity_strat, "equity_bench": equity_bench,
        }

    metrics = {
        "close_mae":     close_mae,
        "price_ic":      price_ic,
        "price_rankic":  price_rankic,
        "slippage":      slippage,
        "num_samples":   N,
        "num_close_pts": total_count,
    }

    # Confidence-gate thresholds for the long-short / long-only sweeps.
    # The first (= slippage) reproduces the original "trade whenever it beats cost" rule.
    THETAS = [slippage, 0.003, 0.005, 0.01]
    strategies_out = {"long_short": {}, "long_only": {}, "magnitude": {},
                      "downside": {}, "trend": {}}

    equity_curves = {}  # k -> (equity_strat, equity_bench)  시각화용 (primary strategy)
    for k in K_VALUES:
        # ── Prediction-level direction stats (strategy-independent) ──
        d = _direction_stats(all_samples, k, slippage)
        metrics[f"da_k{k}"]             = d["da"]
        metrics[f"long_ratio_k{k}"]     = d["long_ratio"]
        metrics[f"short_ratio_k{k}"]    = d["short_ratio"]
        metrics[f"watch_ratio_k{k}"]    = d["watch_ratio"]
        metrics[f"long_ratio_ns_k{k}"]  = d["long_ratio_ns"]
        metrics[f"short_ratio_ns_k{k}"] = d["short_ratio_ns"]

        # ── Primary strategy = long-short @ θ=slippage → legacy flat keys ──
        prim = _run_strategy(all_samples, k, _ls_pos(slippage), slippage)
        metrics[f"aer_k{k}"]           = prim["aer"]
        metrics[f"ret_k{k}"]           = prim["ret"]
        metrics[f"bench_ret_k{k}"]     = prim["bench_ret"]
        metrics[f"mdd_k{k}"]           = prim["mdd"]
        metrics[f"sortino_k{k}"]       = prim["sortino"]
        metrics[f"win_rate_k{k}"]      = prim["win_rate"]
        metrics[f"profit_factor_k{k}"] = prim["profit_factor"]
        equity_curves[k] = (prim["equity_strat"], prim["equity_bench"])

        # ── Strategy sweep: long-short & long-only over θ, plus magnitude-scaled ──
        for th in THETAS:
            for name, pos in (("long_short", _ls_pos(th)), ("long_only", _lo_pos(th)),
                              ("downside", _downside_pos(th)), ("trend", _trend_pos(th))):
                r = _run_strategy(all_samples, k, pos, slippage)
                bucket = strategies_out[name].setdefault(_pct(th), {})
                bucket[f"ret_k{k}"]        = r["ret"]
                bucket[f"aer_k{k}"]        = r["aer"]
                bucket[f"mdd_k{k}"]        = r["mdd"]
                bucket[f"num_trades_k{k}"] = r["num_trades"]

        # magnitude-scaled: scale = std of predicted k-step return over the eval set
        scale = float(np.std([s["pred_ret"][k] for s in all_samples]))
        rm = _run_strategy(all_samples, k, _mag_pos(scale), slippage)
        strategies_out["magnitude"][f"ret_k{k}"]        = rm["ret"]
        strategies_out["magnitude"][f"aer_k{k}"]        = rm["aer"]
        strategies_out["magnitude"][f"mdd_k{k}"]        = rm["mdd"]
        strategies_out["magnitude"][f"num_trades_k{k}"] = rm["num_trades"]
        strategies_out["magnitude"][f"scale_k{k}"]      = scale

    metrics["strategies"] = strategies_out

    # 누적 수익 곡선 PNG 저장
    if plot_dir:
        _plot_equity_curves(equity_curves, plot_dir, K_VALUES)

    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="examples/Bitcoin/train_files/starvla_train_bitcoin.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--slippage", type=float, default=0.001,
                        help="Round-trip slippage per trade (e.g. 0.001 = 0.1%%, Binance futures taker x2)")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--plot_dir", default=None,
                        help="Directory to save equity curve PNG (omit to skip plotting)")
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

    target_mode = str(getattr(cfg.datasets.vla_data, "target_mode", "log_ratio"))
    metrics = evaluate(model, dataloader, slippage=args.slippage, target_mode=target_mode, output_csv=args.output_csv, plot_dir=args.plot_dir)
    metrics.update({"split": args.split, "checkpoint": args.ckpt})

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Split    : {metrics['split']}")
    print(f"  Samples  : {metrics['num_samples']}")
    print(f"  Slippage : {metrics['slippage']*100:.3f}% per trade")
    print(f"  ---")
    print(f"  Close MAE : {metrics['close_mae']:.4f}")
    print(f"  --- Trajectory IC/RankIC (per-sample, 12-step price shape) ---")
    print(f"  {'IC':12} {metrics['price_ic']:>8.4f}")
    print(f"  {'RankIC':12} {metrics['price_rankic']:>8.4f}")
    print(f"  --- Investment Simulation (long-short, slippage={metrics['slippage']*100:.3f}%) ---")
    print(f"  {'':12} {'k=1':>8} {'k=3':>8} {'k=6':>8} {'k=12':>8}")
    print(f"  {'DA':12} {metrics['da_k1']:>8.4f} {metrics['da_k3']:>8.4f} {metrics['da_k6']:>8.4f} {metrics['da_k12']:>8.4f}")
    print(f"  {'Long':12} {metrics['long_ratio_k1']:>8.4f} {metrics['long_ratio_k3']:>8.4f} {metrics['long_ratio_k6']:>8.4f} {metrics['long_ratio_k12']:>8.4f}")
    print(f"  {'Short':12} {metrics['short_ratio_k1']:>8.4f} {metrics['short_ratio_k3']:>8.4f} {metrics['short_ratio_k6']:>8.4f} {metrics['short_ratio_k12']:>8.4f}")
    print(f"  {'Watch':12} {metrics['watch_ratio_k1']:>8.4f} {metrics['watch_ratio_k3']:>8.4f} {metrics['watch_ratio_k6']:>8.4f} {metrics['watch_ratio_k12']:>8.4f}")
    print(f"  --- Long/Short ratio (no slippage, sign only) ---")
    print(f"  {'Long(ns)':12} {metrics['long_ratio_ns_k1']:>8.4f} {metrics['long_ratio_ns_k3']:>8.4f} {metrics['long_ratio_ns_k6']:>8.4f} {metrics['long_ratio_ns_k12']:>8.4f}")
    print(f"  {'Short(ns)':12} {metrics['short_ratio_ns_k1']:>8.4f} {metrics['short_ratio_ns_k3']:>8.4f} {metrics['short_ratio_ns_k6']:>8.4f} {metrics['short_ratio_ns_k12']:>8.4f}")
    print(f"  {'Return':12} {metrics['ret_k1']:>8.4f} {metrics['ret_k3']:>8.4f} {metrics['ret_k6']:>8.4f} {metrics['ret_k12']:>8.4f}")
    print(f"  {'BTC-hold':12} {metrics['bench_ret_k1']:>8.4f} {metrics['bench_ret_k3']:>8.4f} {metrics['bench_ret_k6']:>8.4f} {metrics['bench_ret_k12']:>8.4f}")
    print(f"  {'AER':12} {metrics['aer_k1']:>8.4f} {metrics['aer_k3']:>8.4f} {metrics['aer_k6']:>8.4f} {metrics['aer_k12']:>8.4f}")
    print(f"  {'Sortino':12} {metrics['sortino_k1']:>8.4f} {metrics['sortino_k3']:>8.4f} {metrics['sortino_k6']:>8.4f} {metrics['sortino_k12']:>8.4f}")
    print(f"  {'MDD':12} {metrics['mdd_k1']*100:>7.2f}% {metrics['mdd_k3']*100:>7.2f}% {metrics['mdd_k6']*100:>7.2f}% {metrics['mdd_k12']*100:>7.2f}%")
    print(f"  {'Win rate':12} {metrics['win_rate_k1']:>8.4f} {metrics['win_rate_k3']:>8.4f} {metrics['win_rate_k6']:>8.4f} {metrics['win_rate_k12']:>8.4f}")
    print(f"  {'Profit fac':12} {metrics['profit_factor_k1']:>8.4f} {metrics['profit_factor_k3']:>8.4f} {metrics['profit_factor_k6']:>8.4f} {metrics['profit_factor_k12']:>8.4f}")
    print("=" * 70)

    # ── Trading-strategy comparison (ret = principal return, aer = excess vs B&H) ──
    K_VALUES = [1, 3, 6, 12]
    strat = metrics["strategies"]

    def _strategy_rows():
        """Yield (label, dict-of-per-k-metrics) for every strategy variant, in order."""
        for th_label, d in strat["long_short"].items():
            yield (f"long_short @{th_label}", d)
        for th_label, d in strat["long_only"].items():
            yield (f"long_only  @{th_label}", d)
        yield ("magnitude", strat["magnitude"])

    for metric in ("ret", "aer"):
        title = "principal return (ret)" if metric == "ret" else "excess return (AER)"
        print(f"\n  --- Trading strategies: {title}  [k=1/3/6/12] ---")
        print(f"  {'strategy':20} {'k=1':>8} {'k=3':>8} {'k=6':>8} {'k=12':>8}")
        for label, d in _strategy_rows():
            vals = "".join(f"{d.get(f'{metric}_k{k}', float('nan')):>+9.3f}" for k in K_VALUES)
            print(f"  {label:20}{vals}")
        if metric == "ret":
            bench = "".join(f"{metrics[f'bench_ret_k{k}']:>+9.3f}" for k in K_VALUES)
            print(f"  {'BTC buy&hold':20}{bench}")
    # trade counts at k=6 (how much each gate trims activity)
    print(f"\n  --- num_trades @k=6 (gating trims trades) ---")
    for label, d in _strategy_rows():
        print(f"  {label:20} {d.get('num_trades_k6', 0):>8}")
    print("=" * 70)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {out_path}")

        # Tidy long-format strategy CSV next to the JSON (one row per strategy×k).
        strat_csv = out_path.with_name(out_path.stem + "_strategies.csv")
        rows = []
        for label, d in _strategy_rows():
            for k in K_VALUES:
                rows.append({
                    "strategy": label, "k": k,
                    "ret": d.get(f"ret_k{k}"), "aer": d.get(f"aer_k{k}"),
                    "mdd": d.get(f"mdd_k{k}"), "num_trades": d.get(f"num_trades_k{k}"),
                    "bench_ret": metrics[f"bench_ret_k{k}"],
                })
        with strat_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved strategy summary to {strat_csv}")


if __name__ == "__main__":
    main()
