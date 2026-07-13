"""Deployable (walk-forward) evaluation of the integrated rater.

The headline eval used per-fold quantile thresholds computed on the TEST set --
mild look-ahead, not live-deployable. Here the thresholds are chosen on VALIDATION
(Nov) and applied unchanged to TEST (Dec). Two variants:

  A) val-fixed quantile: keep the chosen quantiles (|pred| top-50%, loss bot-30%),
     but read the ABSOLUTE cutoff values off the val distribution, apply to test.
  B) val-optimized cutoff (user's proposal): grid-search (pred_q, vloss_q) to
     MAXIMIZE val AER, then apply that fold's best absolute cutoffs to test.

Both are fully real-time: at Dec inference the cutoff numbers are already known
from Nov. Compares against the (look-ahead) in-sample number and the baseline.

Needs the rater's val predictions -> results/exp4_rater/v15_btc_fold{f}_val.npz
(dump_rater_val.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_coverage import _kslot_backtest  # noqa: E402

FEAT = "results/exp2_features"
RATER = "results/exp4_rater"


def _load(fold, split, k):
    tag = "vali" if split == "val" else "test"
    fe = np.load(f"{FEAT}/v15_btc_fold{fold}_{tag}.npz")
    suf = "_val" if split == "val" else ""
    r = np.load(f"{RATER}/v15_btc_fold{fold}{suf}.npz")
    assert (fe["ts"] == r["ts"]).all(), f"ts mismatch fold{fold} {split}"
    return {
        "pred": fe[f"sample_rets_k{k}"][:, 0],
        "true": fe[f"true_ret_k{k}"],
        "lc": fe["lc"],
        "loss": r["pred_logloss"][:, k - 1].astype(np.float32),
    }


def _gate_backtest(d, pth, vth, k):
    """Apply ABSOLUTE cutoffs pth (|pred|>=) and vth (loss<=). long-only."""
    p = d["pred"]
    gate = (np.abs(p) >= pth) & (d["loss"] <= vth)
    pos = np.where(gate & (p > 0), 1.0, 0.0)
    ret, aer, n = _kslot_backtest(d["true"], pos, k, 0.001, d["lc"])
    da = (((p[gate] > 0) == (d["true"][gate] > 0)).mean() if gate.sum() else np.nan)
    return da, ret, aer, int(gate.sum())


def variant_A(k, pq=0.5, vq=0.3):
    """Fixed quantiles; absolute cutoffs read off VAL, applied to TEST."""
    das, rets, aers, covs = [], [], [], []
    for f in range(1, 7):
        v, t = _load(f, "val", k), _load(f, "test", k)
        pth = np.quantile(np.abs(v["pred"]), pq)
        vth = np.quantile(v["loss"], vq)
        da, ret, aer, n = _gate_backtest(t, pth, vth, k)
        das.append(da); rets.append(ret); aers.append(aer); covs.append(n / len(t["pred"]))
    return np.nanmean(das), np.nanmean(rets), np.nanmean(aers), np.nanmean(covs)


def variant_B(k, grid_p, grid_v, select="aer"):
    """Grid-search (pred_q,vloss_q) maximizing VAL <select>; apply best to TEST."""
    das, rets, aers, covs, chosen = [], [], [], [], []
    for f in range(1, 7):
        v, t = _load(f, "val", k), _load(f, "test", k)
        best, best_key = None, -1e9
        for pq in grid_p:
            for vq in grid_v:
                pth = np.quantile(np.abs(v["pred"]), pq)
                vth = np.quantile(v["loss"], vq)
                da_v, ret_v, aer_v, n_v = _gate_backtest(v, pth, vth, k)
                if n_v < 15:            # need a few val trades to be meaningful
                    continue
                key = {"aer": aer_v, "ret": ret_v, "da": da_v}[select]
                if key > best_key:
                    best_key, best = key, (pq, vq, pth, vth)
        if best is None:                # fallback: trade all long
            best = (0.0, 1.0, 0.0, np.inf)
        pq, vq, pth, vth = best
        da, ret, aer, n = _gate_backtest(t, pth, vth, k)
        das.append(da); rets.append(ret); aers.append(aer)
        covs.append(n / len(t["pred"])); chosen.append((f, pq, vq))
    return np.nanmean(das), np.nanmean(rets), np.nanmean(aers), np.nanmean(covs), chosen


def baseline(k):
    das, rets, aers = [], [], []
    for f in range(1, 7):
        t = _load(f, "test", k)
        p = t["pred"]; pos = np.where(p > 0, 1.0, 0.0)
        ret, aer, n = _kslot_backtest(t["true"], pos, k, 0.001, t["lc"])
        das.append(((p > 0) == (t["true"] > 0)).mean()); rets.append(ret); aers.append(aer)
    return np.mean(das), np.mean(rets), np.mean(aers)


def main(k):
    gp = [0.3, 0.4, 0.5, 0.6, 0.7]
    gv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    print(f"\n{'='*76}\nWALK-FORWARD rater eval (k={k}) -- thresholds from VAL, applied to TEST\n{'='*76}")
    bd, br, ba = baseline(k)
    print(f"{'method':>34} {'DA':>7} {'ret':>9} {'AER':>9} {'cov':>7}")
    print(f"{'baseline (trade all)':>34} {bd:.3f} {br:+.4f} {ba:+.4f} {'100%':>7}")
    da, r, a, c = variant_A(k)
    print(f"{'A: val-fixed quantile (50/30)':>34} {da:.3f} {r:+.4f} {a:+.4f} {c*100:6.1f}%")
    for sel in ("aer", "ret"):
        da, r, a, c, ch = variant_B(k, gp, gv, sel)
        print(f"{f'B: val-optimized (max val {sel})':>34} {da:.3f} {r:+.4f} {a:+.4f} {c*100:6.1f}%")
        picks = " ".join(f"f{f}:{int(pq*100)}/{int(vq*100)}" for f, pq, vq in ch)
        print(f"{'   picked pred%/loss%:':>34} {picks}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    main(a.k)
