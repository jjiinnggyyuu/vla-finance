"""Experiment 3 -- learned ERROR head + two-gate trading (post-hoc, frozen DiT+BTC).

The senior's idea: predict the model's own error, then trade only when predicted
error is small AND the predicted move is large (feedback #5). Unlike experiment 2
(predict direction-correctness, val AUC ~0.51 = unlearnable), error magnitude IS
learnable from the frozen features (val AUC ~0.60).

Pipeline (reuses exp2's frozen-feature probe machinery):
  target  = "low error"  (|pred - true| below the fold-median), a binary label.
  probe   = standardize -> PCA -> logistic, trained on VALIDATION, applied to TEST.
  gate A  = confidence (P low error) top-X%       [risk-coverage coverage]
  gate B  = |pred| > theta  (large predicted move) [risk-coverage theta]
  two-gate = A and B  == feedback #5.

Everything is 6-fold averaged and scored with the Stage-0 validated risk_coverage.
Run: python examples/Bitcoin/eval_files/exp3_error.py [--k 6] [--n_pca 64] [--theta 0.005]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp2_confidence as E  # noqa: E402  (reuse _load / _pca / _train_probe)
from risk_coverage import average_folds, curve_slope, risk_coverage_curve  # noqa: E402

E._train_probe.__defaults__ = (150, 0.05, 1e-2, 0)
assert E.PRED_MODE == "single", "experiment 3 uses the faithful single prediction"


def _low_error_label(pred, true):
    """1 if this bar's |error| is below the fold median (a 'precise' bar)."""
    err = np.abs(pred - true)
    return (err < np.median(err)).astype(np.float32)


def _error_confidence(fold, k, n_pca, shuffle=False, seed=0):
    """Train P(low error) probe on validation, return test confidence."""
    val = E._load(fold, "vali", k)
    test = E._load(fold, "test", k)
    y = _low_error_label(val["pred"], val["true"])
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    mu, sd, comp = E._pca_fit(val["feat"], n_pca)
    predict, _ = E._train_probe(E._pca_apply(val["feat"], mu, sd, comp), y, seed=seed)
    return predict(E._pca_apply(test["feat"], mu, sd, comp))


def _avg_curve(k, n_pca, theta, strategy, shuffle=False):
    per = []
    for f in range(1, 7):
        t = E._load(f, "test", k)
        conf = _error_confidence(f, k, n_pca, shuffle=shuffle, seed=f)
        per.append(risk_coverage_curve(t["pred"], t["true"], conf, k=k, lc=t["lc"],
                                       theta=theta, strategy=strategy))
    return average_folds(per)


def _print_table(title, avg):
    print(f"\n--- {title}  (DA slope {curve_slope(avg,'da'):+.3f}) ---")
    print(f"{'cov':>5} {'DA':>7} {'ret':>9} {'AER':>9} {'trades':>7}")
    for r in sorted(avg, key=lambda r: -r["coverage"]):
        aer = f"{r['aer']:+.3f}" if r["aer"] is not None else "  --"
        print(f"{r['coverage']:>5.1f} {r['da']:+.3f} {r['ret']:+.4f} {aer:>9} {r['n_trades']:7.0f}")


def evaluate(k=6, n_pca=64, theta=0.005, strategy="long_only", n_perm=20):
    print(f"\n{'='*72}\nEXPERIMENT 3 -- error head + two-gate  "
          f"(k={k}, PCA={n_pca}, strategy={strategy})\n{'='*72}")

    # (1) confidence gate only (theta=0): does 'low predicted error' help?
    a0 = _avg_curve(k, n_pca, 0.0, strategy)
    _print_table("gate A only: P(low error), theta=0", a0)

    # (2) two-gate (feedback #5): low error AND |pred|>theta
    a2 = _avg_curve(k, n_pca, theta, strategy)
    _print_table(f"TWO-GATE: P(low error) AND |pred|>{theta}", a2)

    # (3) permutation significance of the two-gate DA slope
    real = curve_slope(a2, "da")
    null = np.array([curve_slope(_avg_curve(k, n_pca, theta, strategy, shuffle=True), "da")
                     for _ in range(n_perm)])
    p = float((null >= real).mean())
    print(f"\nTWO-GATE DA slope {real:+.3f}  vs null {null.mean():+.3f}±{null.std():.3f}"
          f"  p={p:.3f}  {'SIGNIFICANT' if p < 0.05 else 'not sig'}")

    # (4) best AER vs the no-gate baseline
    base = next(r for r in a2 if r["coverage"] == 1.0)
    best = max(a2, key=lambda r: (r["aer"] if r["aer"] is not None else -9))
    print(f"baseline AER (cov 1.0) {base['aer']:+.3f}  ->  best AER {best['aer']:+.3f} "
          f"@ cov {best['coverage']:.1f}  (trades {best['n_trades']:.0f}/fold)")
    return {"gateA": a0, "twogate": a2, "p": p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_pca", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.005)
    ap.add_argument("--strategy", default="long_only")
    a = ap.parse_args()
    evaluate(k=a.k, n_pca=a.n_pca, theta=a.theta, strategy=a.strategy)
