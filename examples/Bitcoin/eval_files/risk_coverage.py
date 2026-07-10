"""Risk-coverage evaluator + sanity harness (Stage 0 of the confidence-gating plan).

Every uncertainty experiment (sample-variance, sigma-head, error-head, VFD)
produces one thing: a per-bar *confidence* score. This module turns such a score
into a verdict, in the SAME DA / ret / AER units as the 4-axis ablation, so
results are directly comparable.

    For each fold: keep the top-X% most-confident bars, trade only those, and
    measure DA / ret / AER at each coverage X. Then average the 6 folds (method
    A), and also plot each fold's curve to check the gain is consistent (not one
    lucky fold -- the same regime-memorization trap we caught before).

    ret/AER use the exact k-slot overlapping backtest of eval_bitcoin_close_mae:
    a non-selected (or |pred|<=theta) bar simply takes position 0 (cash), so the
    confidence gate is just another gate inside the existing backtest. AER is vs
    pure buy-and-hold (start->end price, k-independent).

Before trusting the yardstick we validate the *evaluator itself* with two checks
whose answers we already know (label-shuffle must stay flat; cheating-inject must
slope up steeply). Run `python risk_coverage.py` to self-validate on synthetic
data (no GPU). Real experiments call `evaluate_experiment([fold1.npz, ...], k=6)`.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _dir_correct(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Directional hit, matching the eval convention `(pred>0)==(true>0)`."""
    return (pred > 0) == (true > 0)


def _trend_up(lc: np.ndarray, win: int = 24) -> np.ndarray:
    """Uptrend flag: current price > trailing SMA (past/current only, no look-ahead)."""
    lc = np.asarray(lc, dtype=np.float64)
    N = len(lc)
    out = np.ones(N, dtype=bool)
    for i in range(N):
        lo = max(0, i - win + 1)
        out[i] = lc[i] > lc[lo:i + 1].mean()
    return out


# The five trading strategies, mirroring eval_bitcoin_close_mae. Each returns a
# per-bar base position in [-1,1] (BEFORE the confidence gate zeroes out
# non-selected bars). `theta` is the per-setup confidence gate on |pred|.
STRATEGIES = ("long_short", "long_only", "magnitude", "downside", "trend")


def strategy_positions(name: str, pred_ret: np.ndarray, lc: np.ndarray | None,
                       theta: float) -> np.ndarray:
    p = np.asarray(pred_ret, dtype=np.float64)
    if name == "long_short":       # long if >θ, short if <-θ, else cash
        return np.where(p > theta, 1.0, np.where(p < -theta, -1.0, 0.0))
    if name == "long_only":        # long if >θ else cash
        return np.where(p > theta, 1.0, 0.0)
    if name == "downside":         # default long; cash only on strong drop (<-θ)
        return np.where(p > -theta, 1.0, 0.0)
    if name == "trend":            # long if bullish AND price uptrend
        tu = _trend_up(lc) if lc is not None else np.ones(len(p), dtype=bool)
        return np.where((p > theta) & tu, 1.0, 0.0)
    if name == "magnitude":        # bet size ∝ predicted move (continuous)
        scale = p.std() if p.std() > 1e-12 else 1e-12
        return np.clip(p / scale, -1.0, 1.0)
    raise ValueError(f"unknown strategy: {name}")


# --------------------------------------------------------------------------- #
#  k-slot overlapping backtest (compounded), matching eval_bitcoin_close_mae.
# --------------------------------------------------------------------------- #
def _ffill_sum(equity_t: np.ndarray, k: int, init_each: float, N: int) -> np.ndarray:
    slot_series = []
    for offset in range(k):
        idxs = list(range(offset, N, k))
        full = np.empty(N)
        cur = init_each
        j = 0
        for t in range(N):
            if j < len(idxs) and idxs[j] == t:
                cur = equity_t[idxs[j]]
                j += 1
            full[t] = cur
        slot_series.append(full)
    return np.sum(slot_series, axis=0)


def _kslot_backtest(true_ret, position, k, slippage, lc=None):
    """Compounded k-slot backtest. `position`: (N,) in [-1,1], 0 = cash.
    Returns (ret, aer_or_None, n_trades). AER vs pure buy-and-hold (needs lc)."""
    N = len(true_ret)
    eq_t = np.full(N, np.nan)
    n_trades = 0
    for offset in range(k):
        cap = 1.0 / k
        for t in range(offset, N, k):
            pos = float(position[t])
            slot_ret = pos * true_ret[t] - abs(pos) * slippage
            cap *= (1.0 + slot_ret)
            eq_t[t] = cap
            if pos != 0.0:
                n_trades += 1
    equity = np.concatenate([[1.0], _ffill_sum(eq_t, k, 1.0 / k, N)])
    ret = float(equity[-1] - 1.0)
    aer = None
    if lc is not None:
        bench = float(lc[-1] / lc[0] - 1.0)   # pure hold, k-independent
        aer = ret - bench
    return ret, aer, n_trades


# --------------------------------------------------------------------------- #
#  Risk-coverage curve for ONE fold.
# --------------------------------------------------------------------------- #
def risk_coverage_curve(
    pred_ret: np.ndarray,
    true_ret: np.ndarray,
    confidence: np.ndarray,
    k: int = 6,
    lc: np.ndarray | None = None,
    theta: float = 0.0,
    coverages=DEFAULT_COVERAGES,
    slippage: float = 0.001,
    strategy: str = "long_short",
    min_n: int = 10,
) -> list[dict]:
    """One fold's risk-coverage curve for one trading strategy.

    Args:
        pred_ret/true_ret: (N,) predicted / realized k-step return, TIME ORDER.
        confidence:        (N,) higher = more confident.
        k:                 holding horizon (slots) for the backtest.
        lc:                (N,) current price series -> pure-hold B&H for AER,
                           and the trend filter's SMA.
        theta:             |pred|>theta gate (0 = confidence gate only).
        strategy:          one of STRATEGIES.

    Returns per-coverage {coverage, n_selected, n_trades, da, ret, aer}. At
    coverage 1.0 this is the baseline (confidence gate off). `n_trades` is the
    actual number of executed trades (position != 0) -- so you can see how the
    confidence gate trims trading for each strategy.
    """
    pred_ret = np.asarray(pred_ret, dtype=np.float64)
    true_ret = np.asarray(true_ret, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    N = len(pred_ret)
    assert len(true_ret) == N == len(confidence), "pred/true/confidence length mismatch"

    rank = np.empty(N, dtype=np.int64)
    rank[np.argsort(-confidence, kind="stable")] = np.arange(N)  # 0 = most confident
    base_pos = strategy_positions(strategy, pred_ret, lc, theta)   # before gating

    rows = []
    for c in coverages:
        m = min(N, max(min_n, int(round(c * N))))
        selected = rank < m                                   # top-m confident bars
        da = float(np.mean(_dir_correct(pred_ret[selected], true_ret[selected])))

        # Confidence gate: keep the base position only on selected bars, else cash.
        position = np.where(selected, base_pos, 0.0)

        ret, aer, n_tr = _kslot_backtest(true_ret, position, k, slippage, lc)
        rows.append({"coverage": float(c), "n_selected": int(m), "n_trades": int(n_tr),
                     "da": da, "ret": ret, "aer": aer})
    return rows


# --------------------------------------------------------------------------- #
#  Confidence from raw samples (experiment 1 = sample variance).
# --------------------------------------------------------------------------- #
def confidence_from_samples(sample_rets: np.ndarray, method: str = "neg_std") -> np.ndarray:
    """(N, K) K sampled k-step returns per bar -> (N,) confidence (higher=surer).

    neg_std       : -std over K samples (tight samples = confident).
    neg_range     : -(max-min) over K.
    dir_agreement : |fraction long - 0.5|*2  (all samples agree on direction).
    """
    s = np.asarray(sample_rets, dtype=np.float64)
    if method == "neg_std":
        return -s.std(axis=1)
    if method == "neg_range":
        return -(s.max(axis=1) - s.min(axis=1))
    if method == "dir_agreement":
        frac_long = (s > 0).mean(axis=1)
        return np.abs(frac_long - 0.5) * 2.0
    raise ValueError(f"unknown confidence method: {method}")


# --------------------------------------------------------------------------- #
#  Average across folds (method A: per-fold curves, then average).
# --------------------------------------------------------------------------- #
def average_folds(fold_rows: list[list[dict]]) -> list[dict]:
    """fold_rows: list of per-fold `rows`. Returns per-coverage cross-fold mean."""
    ncov = len(fold_rows[0])
    out = []
    for i in range(ncov):
        cov = fold_rows[0][i]["coverage"]
        das = np.array([fr[i]["da"] for fr in fold_rows])
        rets = np.array([fr[i]["ret"] for fr in fold_rows])
        aers = np.array([fr[i]["aer"] for fr in fold_rows if fr[i]["aer"] is not None])
        trades = np.array([fr[i]["n_trades"] for fr in fold_rows])
        out.append({
            "coverage": cov,
            "da": float(das.mean()), "da_std": float(das.std()),
            "ret": float(rets.mean()), "ret_std": float(rets.std()),
            "aer": (float(aers.mean()) if len(aers) else None),
            "n_trades": float(trades.mean()), "n_trades_total": int(trades.sum()),
            "n_folds": len(fold_rows),
        })
    return out


def curve_slope(rows: list[dict], metric: str = "da") -> float:
    """Least-squares slope of `metric` vs selectivity (1 - coverage). Robust to
    the noisy extreme-coverage tail (the failure the harness caught first run)."""
    x = np.array([1.0 - r["coverage"] for r in rows])
    y = np.array([r[metric] for r in rows])
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return 0.0 if denom < 1e-12 else float(np.sum(x * (y - y.mean())) / denom)


def _ascii_curve(rows, metric="da", width=24):
    rows = sorted(rows, key=lambda r: -r["coverage"])
    vals = [r[metric] for r in rows]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    out = []
    for r in rows:
        v = r[metric]
        fill = int(round((v - lo) / span * width))
        out.append(f"  cov {r['coverage']:4.1f}  {v:+.3f}  |{'#' * fill}{'.' * (width - fill)}|")
    return "\n".join(out)


def print_experiment(per_fold: list[list[dict]], avg: list[dict], metric_focus="da"):
    """Per-fold consistency (DA curves) + the 6-fold-average DA/ret/AER table."""
    print("\n" + "=" * 70)
    print("PER-FOLD consistency  (DA vs coverage) -- all should trend the same way")
    print("=" * 70)
    for fi, rows in enumerate(per_fold, 1):
        base = next(r for r in rows if r["coverage"] == 1.0)[metric_focus]
        gain = curve_slope(rows, metric_focus)
        print(f"fold{fi}  base {metric_focus} {base:+.3f}   slope {gain:+.3f}")
        print(_ascii_curve(rows, metric_focus))
    print("\n" + "=" * 74)
    print("6-FOLD AVERAGE  (baseline = coverage 1.0)")
    print("=" * 74)
    print(f"{'cov':>5} {'DA':>7} {'ret':>9} {'AER':>9} {'trades/fold':>12}   (+/- std)")
    for r in sorted(avg, key=lambda r: -r["coverage"]):
        aer = f"{r['aer']:+.3f}" if r["aer"] is not None else "   --"
        print(f"{r['coverage']:5.1f} {r['da']:+.3f} {r['ret']:+.4f} {aer:>9} "
              f"{r['n_trades']:12.0f}   (DA+/-{r['da_std']:.3f}, ret+/-{r['ret_std']:.3f})")
    print("=" * 74)


# --------------------------------------------------------------------------- #
#  Top-level: evaluate one experiment across folds.
# --------------------------------------------------------------------------- #
def load_fold(path: str, k: int):
    """Load a per-fold dump. Expected npz keys (time-ordered):
       sample_rets_k{k} (N,K) raw sampled k-step returns, true_ret_k{k} (N,), lc (N,).
    Falls back to pred_ret_k{k}/confidence_k{k} if raw samples are absent."""
    d = np.load(path)
    true_ret = d[f"true_ret_k{k}"]
    lc = d["lc"] if "lc" in d else None
    samples = d[f"sample_rets_k{k}"] if f"sample_rets_k{k}" in d else None
    pred_ret = d[f"pred_ret_k{k}"] if f"pred_ret_k{k}" in d else None
    return {"true_ret": true_ret, "lc": lc, "sample_rets": samples, "pred_ret": pred_ret}


def _fold_inputs(fold_paths, k, confidence_method):
    """Load every fold once -> list of (pred_ret, true_ret, confidence, lc)."""
    inputs = []
    for p in fold_paths:
        f = load_fold(p, k)
        if f["sample_rets"] is not None:
            pred_ret = f["sample_rets"].mean(axis=1)
            confidence = confidence_from_samples(f["sample_rets"], confidence_method)
        else:
            pred_ret = f["pred_ret"]
            confidence = np.load(p)[f"confidence_k{k}"]
        inputs.append((pred_ret, f["true_ret"], confidence, f["lc"]))
    return inputs


def evaluate_experiment(
    fold_paths: list[str],
    k: int = 6,
    theta: float = 0.0,
    confidence_method: str = "neg_std",
    strategy: str = "long_only",
    slippage: float = 0.001,
    coverages=DEFAULT_COVERAGES,
    verbose: bool = True,
):
    """Method-A evaluation for ONE strategy: per-fold curves + 6-fold average."""
    inputs = _fold_inputs(fold_paths, k, confidence_method)
    per_fold = [
        risk_coverage_curve(pr, tr, cf, k=k, lc=lc, theta=theta,
                            coverages=coverages, slippage=slippage, strategy=strategy)
        for (pr, tr, cf, lc) in inputs
    ]
    avg = average_folds(per_fold)
    if verbose:
        print(f"\n########  strategy = {strategy}  (k={k}, theta={theta})  ########")
        print_experiment(per_fold, avg)
    return {"per_fold": per_fold, "average": avg}


def evaluate_strategies(
    fold_paths: list[str],
    k: int = 6,
    theta: float = 0.0,
    confidence_method: str = "neg_std",
    strategies=STRATEGIES,
    slippage: float = 0.001,
    coverages=DEFAULT_COVERAGES,
    verbose: bool = True,
):
    """Compare ALL strategies side by side, with per-strategy trade counts.

    Prints, per strategy, the 6-fold-average DA/ret/AER AND executed trades at
    each coverage -- so you see which strategy keeps trading under the confidence
    gate and which one starves itself of trades.
    """
    inputs = _fold_inputs(fold_paths, k, confidence_method)
    results = {}
    for strat in strategies:
        per_fold = [
            risk_coverage_curve(pr, tr, cf, k=k, lc=lc, theta=theta,
                                coverages=coverages, slippage=slippage, strategy=strat)
            for (pr, tr, cf, lc) in inputs
        ]
        results[strat] = average_folds(per_fold)

    if verbose:
        for metric in ("n_trades", "aer", "da"):
            label = {"n_trades": "TRADES / fold (executed)", "aer": "AER (6-fold avg)",
                     "da": "DA (6-fold avg)"}[metric]
            print("\n" + "=" * 78)
            print(f"{label}   [rows = strategy, cols = coverage]   k={k}, theta={theta}")
            print("=" * 78)
            covs = sorted({r["coverage"] for r in next(iter(results.values()))}, reverse=True)
            print(f"{'strategy':>11} " + " ".join(f"{c:>7.1f}" for c in covs))
            for strat, avg in results.items():
                by = {r["coverage"]: r for r in avg}
                cells = []
                for c in covs:
                    v = by[c][metric]
                    if metric == "n_trades":
                        cells.append(f"{v:7.0f}")
                    elif v is None:
                        cells.append("     --")
                    else:
                        cells.append(f"{v:+7.3f}")
                print(f"{strat:>11} " + " ".join(cells))
            print("=" * 78)
    return results


# --------------------------------------------------------------------------- #
#  Sanity harness: validate the evaluator on answers we already know.
# --------------------------------------------------------------------------- #
def sanity_check(pred_ret, true_ret, *, slippage=0.001, steep_thresh=0.15,
                 flat_thresh=0.03, n_shuffles=25, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    pred_ret = np.asarray(pred_ret, dtype=np.float64)
    true_ret = np.asarray(true_ret, dtype=np.float64)

    # Check A: cheating injection (oracle confidence) -> MUST slope up.
    oracle_conf = -np.abs(pred_ret - true_ret)
    rows_cheat = risk_coverage_curve(pred_ret, true_ret, oracle_conf, slippage=slippage)
    slope_cheat = curve_slope(rows_cheat, "da")
    cheat_ok = slope_cheat >= steep_thresh

    # Check B: label shuffle, averaged over permutations -> MUST stay flat.
    magnitude_conf = np.abs(pred_ret)
    slopes, da_fulls = [], []
    for _ in range(n_shuffles):
        rows_s = risk_coverage_curve(pred_ret, rng.permutation(true_ret),
                                     magnitude_conf, slippage=slippage)
        slopes.append(curve_slope(rows_s, "da"))
        da_fulls.append(next(r for r in rows_s if r["coverage"] == 1.0)["da"])
    slope_shuf, slope_std = float(np.mean(slopes)), float(np.std(slopes))
    da_full_shuf = float(np.mean(da_fulls))
    flat_ok = abs(slope_shuf) <= flat_thresh
    coin_ok = abs(da_full_shuf - 0.5) <= 0.05

    passed = cheat_ok and flat_ok and coin_ok
    if verbose:
        print("=" * 66)
        print("STAGE 0  --  risk-coverage evaluator sanity harness")
        print("=" * 66)
        print(f"[A] cheating-inject   slope(DA) = {slope_cheat:+.3f}  "
              f"(need >= {steep_thresh})   -> {'PASS' if cheat_ok else 'FAIL'}")
        print(_ascii_curve(rows_cheat, "da"))
        print(f"\n[B] label-shuffle x{n_shuffles}  slope(DA) = {slope_shuf:+.3f} "
              f"(+/-{slope_std:.3f})  (need |.| <= {flat_thresh}) "
              f"-> {'PASS' if flat_ok else 'FAIL'}")
        print(f"    label-shuffle     DA@full  = {da_full_shuf:.3f}  (need ~0.5)"
              f"               -> {'PASS' if coin_ok else 'FAIL'}")
        print("-" * 66)
        print(f"OVERALL: {'PASS -- evaluator is honest' if passed else 'FAIL -- do NOT trust curves'}")
        print("=" * 66)
    return {"passed": passed, "cheat_slope": slope_cheat,
            "shuffle_slope": slope_shuf, "shuffle_da_full": da_full_shuf}


def _synthetic_fold(n=600, skill=0.30, vol=0.02, seed=0):
    """Synthetic fold: skillful preds + a random-walk price series (for AER)."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0.0, vol, n)
    pred = skill * true + rng.normal(0.0, vol, n) * np.sqrt(max(1e-6, 1 - skill**2))
    lc = 100.0 * np.exp(np.cumsum(true))          # price path implied by returns
    return pred, true, lc


if __name__ == "__main__":
    import sys
    # (1) Validate the evaluator itself (no GPU).
    pred, true, _ = _synthetic_fold()
    print(f"synthetic: N={len(pred)}, base DA = {np.mean(_dir_correct(pred, true)):.3f}\n")
    res = sanity_check(pred, true)

    # (2) Demo: all 5 strategies across 6 synthetic folds, with trade counts.
    #     Build synthetic per-fold dumps in-memory and drive evaluate_strategies.
    folds = [_synthetic_fold(seed=fi, skill=0.15 + 0.05 * fi) for fi in range(6)]

    def _fake_inputs():
        """Turn synthetic folds into (pred, true, confidence, lc) tuples, mimicking
        experiment-1 sample dumps (confidence = -std over 8 fake samples)."""
        rng = np.random.default_rng(0)
        out = []
        for p, t, lc in folds:
            samples = p[:, None] + rng.normal(0, 0.005, (len(p), 8))
            out.append((samples.mean(1), t, -samples.std(1), lc))
        return out

    print("\n[demo] all 5 strategies x 6 synthetic folds -- executed trade counts:")
    fake = _fake_inputs()
    results = {}
    for strat in STRATEGIES:
        per_fold = [risk_coverage_curve(pr, tr, cf, k=6, lc=lc, strategy=strat)
                    for (pr, tr, cf, lc) in fake]
        results[strat] = average_folds(per_fold)
    covs = list(DEFAULT_COVERAGES)
    for metric, label in (("n_trades", "TRADES/fold"), ("aer", "AER"), ("da", "DA")):
        print("\n" + "-" * 78)
        print(f"{label}   [rows=strategy, cols=coverage]")
        print(f"{'strategy':>11} " + " ".join(f"{c:>7.1f}" for c in covs))
        for strat, avg in results.items():
            by = {r["coverage"]: r for r in avg}
            cells = []
            for c in covs:
                v = by[c][metric]
                cells.append(f"{v:7.0f}" if metric == "n_trades"
                             else ("     --" if v is None else f"{v:+7.3f}"))
            print(f"{strat:>11} " + " ".join(cells))

    sys.exit(0 if res["passed"] else 1)
