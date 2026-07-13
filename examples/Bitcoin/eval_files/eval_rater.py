"""Evaluate the INTEGRATED (transformer, attention) confidence rater on TEST.

train_rater.py dumped per-bar per-candle predicted log velocity-loss on Dec test
(results/exp4_rater/v15_btc_fold{f}.npz, key pred_logloss (N,12)). Here we ask the
two questions that decide whether the integrated rater beat the post-hoc probe:

  1. TEST corr(predicted log-loss, oracle log-loss) per fold and avg
     -- post-hoc probe got 0.08; integrated val corr was ~0.53. Does it transfer?
  2. Two-gate backtest (|pred| large AND predicted-loss low) using the RATER's
     predicted loss -- does it reproduce the ORACLE two-gate DA (~0.566 @k=12)?

No look-ahead: thresholds are per-fold quantiles of that fold's own test bars
(same convention as the pre-check). Oracle uses ground-truth velocity loss (upper
bound); rater uses only inference-available features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_coverage import _kslot_backtest  # noqa: E402

FEAT = "results/exp2_features"
VELL = "results/exp4_velloss"
RATER = "results/exp4_rater"


def _load(fold, k):
    fe = np.load(f"{FEAT}/v15_btc_fold{fold}_test.npz")
    vl = np.load(f"{VELL}/v15_btc_fold{fold}.npz")
    r = np.load(f"{RATER}/v15_btc_fold{fold}.npz")
    assert (fe["ts"] == vl["ts"]).all() and (fe["ts"] == r["ts"]).all()
    return {
        "pred": fe[f"sample_rets_k{k}"][:, 0],
        "true": fe[f"true_ret_k{k}"],
        "lc": fe["lc"],
        "oracle_ll": np.log(vl["vel_loss"][:, k - 1].astype(np.float32) + 1e-6),
        "rater_ll": r["pred_logloss"][:, k - 1].astype(np.float32),
    }


def rater_quality(k):
    cs = []
    for f in range(1, 7):
        d = _load(f, k)
        cs.append(np.corrcoef(d["rater_ll"], d["oracle_ll"])[0, 1])
    return float(np.mean(cs)), cs


def two_gate(k, pred_q, vloss_q, src, mode="lo"):
    """src in {rater, oracle}. Gate = |pred|>=pred_q-qtile AND loss<=vloss_q-qtile."""
    das, rets, aers, ntr = [], [], [], []
    for f in range(1, 7):
        d = _load(f, k)
        conf = d["rater_ll"] if src == "rater" else d["oracle_ll"]  # lower = confident
        p = d["pred"]
        pth = np.quantile(np.abs(p), pred_q)
        vth = np.quantile(conf, vloss_q)
        gate = (np.abs(p) >= pth) & (conf <= vth)
        pos = (np.where(gate & (p > 0), 1.0, 0.0) if mode == "lo"
               else np.where(gate, np.sign(p), 0.0))
        ret, aer, n = _kslot_backtest(d["true"], pos, k, 0.001, d["lc"])
        das.append(((p[gate] > 0) == (d["true"][gate] > 0)).mean() if gate.sum() else 0.5)
        rets.append(ret); aers.append(aer); ntr.append(n)
    return np.mean(das), np.std(das), np.mean(rets), np.mean(aers), np.mean(ntr), das


def main(k):
    print(f"\n{'='*74}\nINTEGRATED RATER eval (k={k})   [post-hoc probe was corr 0.08]\n{'='*74}")
    c, cs = rater_quality(k)
    print(f"TEST corr(rater loss, oracle loss) = {c:+.3f}   "
          f"fold {[f'{x:+.2f}' for x in cs]}")

    print(f"\n{'gate':>24} {'src':>7} {'DA':>13} {'ret':>9} {'AER':>8} {'trades':>7}")
    for pq, vq in [(0.5, 0.5), (0.5, 0.3)]:
        for src in ("oracle", "rater"):
            d, ds, r, a, n, _ = two_gate(k, pq, vq, src)
            tag = f"pred>{int((1-pq)*100)}% loss<{int(vq*100)}%"
            print(f"{tag:>24} {src:>7} {d:.3f}±{ds:.3f} {r:+.4f} {a:+.3f} {n:>7.0f}")
    d, ds, r, a, n, _ = two_gate(k, 0.0, 1.0, "oracle")
    print(f"{'baseline (all long)':>24} {'--':>7} {d:.3f}±{ds:.3f} {r:+.4f} {a:+.3f} {n:>7.0f}")

    # per-fold DA for the headline gate (cherry-picking check)
    print(f"\nper-fold DA (pred>50% loss<30%, rater):")
    _, _, _, _, _, das = two_gate(k, 0.5, 0.3, "rater")
    _, _, _, _, _, dab = two_gate(k, 0.0, 1.0, "oracle")
    for i in range(6):
        print(f"  fold{i+1}: baseline {dab[i]:.3f} -> gated {das[i]:.3f}  "
              f"({das[i]-dab[i]:+.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    main(a.k)
