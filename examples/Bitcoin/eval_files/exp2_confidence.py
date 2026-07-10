"""Experiment 2 -- learned aleatoric confidence probe (post-hoc, frozen DiT+BTC).

Trains a small linear probe on frozen VL features to predict P(direction correct),
then gates trading by that confidence and scores with the (Stage-0 validated)
risk-coverage evaluator. Everything is per-fold and out-of-sample:

  train the probe on VALIDATION features (Nov, out-of-sample for the DiT) ->
  predict confidence on TEST features (Dec) -> risk-coverage -> average 6 folds.

Overfitting guard (2560-dim features, only 629 val bars): standardize -> PCA to
~32 dims -> logistic regression with weight decay. A label-shuffle control (train
on shuffled val labels) MUST give a flat test curve; if it doesn't, the probe is
memorizing noise and no result can be trusted.

Compares the learned confidence against experiment-1's sample-std confidence on
the same test bars.

Run: python examples/Bitcoin/eval_files/exp2_confidence.py [--n_pca 32] [--k 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_coverage import (  # noqa: E402
    average_folds, confidence_from_samples, curve_slope, risk_coverage_curve,
)

FEAT_DIR = "results/exp2_features"


# Prediction mode. "single" = ONE predict_action draw (experiment 2 as planned:
# one forward pass + MLP confidence). "mean" = K-sample average (an ensemble, a
# different experiment). Default: single, per the corrected plan.
PRED_MODE = "single"


def _load(fold: int, split: str, k: int):
    d = np.load(f"{FEAT_DIR}/v15_btc_fold{fold}_{split}.npz")
    s = d[f"sample_rets_k{k}"]
    pred = s[:, 0] if PRED_MODE == "single" else s.mean(axis=1)   # ONE draw (single)
    std_conf = confidence_from_samples(s, "neg_std")     # exp1 signal (uses all K)
    return {"feat": d["feature"].astype(np.float32), "pred": pred,
            "true": d[f"true_ret_k{k}"], "lc": d["lc"], "std_conf": std_conf}


def _pca_fit(X, n):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    return mu, sd, Vt[:n]                                  # components (n, D)


def _pca_apply(X, mu, sd, comp):
    return ((X - mu) / sd) @ comp.T                        # (N, n)


def _train_probe(Xtr, ytr, epochs=400, lr=0.05, wd=1e-2, seed=0):
    """Logistic regression on PCA features (standardized), weight-decay regularized."""
    torch.manual_seed(seed)
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr.astype(np.float32))
    pmu, psd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - pmu) / psd
    lin = nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(lin(Xn).squeeze(-1), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr_acc = float(((torch.sigmoid(lin(Xn).squeeze(-1)) > 0.5).float() == y).float().mean())

    def predict(Xnew):
        Xt = (torch.tensor(Xnew, dtype=torch.float32) - pmu) / psd
        with torch.no_grad():
            return torch.sigmoid(lin(Xt).squeeze(-1)).numpy()
    return predict, tr_acc


def _confidence_for_fold(fold, k, n_pca, shuffle=False, seed=0):
    """Train probe on val, return (test_confidence, diagnostics)."""
    val = _load(fold, "vali", k)
    test = _load(fold, "test", k)
    ylab = ((val["pred"] > 0) == (val["true"] > 0)).astype(np.float32)
    if shuffle:
        ylab = np.random.default_rng(seed).permutation(ylab)

    mu, sd, comp = _pca_fit(val["feat"], n_pca)
    Xv = _pca_apply(val["feat"], mu, sd, comp)
    Xt = _pca_apply(test["feat"], mu, sd, comp)
    predict, tr_acc = _train_probe(Xv, ylab, seed=seed)
    conf = predict(Xt)

    # test-side diagnostic: does confidence rank actual correctness?
    correct = ((test["pred"] > 0) == (test["true"] > 0)).astype(np.float32)
    c = float(np.corrcoef(conf, correct)[0, 1]) if conf.std() > 1e-9 else 0.0
    return conf, {"train_acc": tr_acc, "test_corr_conf_correct": c,
                  "conf_std": float(conf.std())}


def evaluate(k=6, n_pca=32, theta=0.0, strategy="long_only"):
    folds = range(1, 7)
    learned, shuffled, exp1 = [], [], []
    print(f"\n{'='*70}\nEXPERIMENT 2 -- learned confidence probe  "
          f"(k={k}, PCA={n_pca}, strategy={strategy}, theta={theta})\n{'='*70}")
    print(f"{'fold':>4} {'train_acc':>10} {'corr(conf,correct)':>19} {'conf_std':>9}")
    for f in folds:
        t = _load(f, "test", k)
        conf, diag = _confidence_for_fold(f, k, n_pca)
        print(f"{f:>4} {diag['train_acc']:>10.3f} {diag['test_corr_conf_correct']:>19.3f}"
              f" {diag['conf_std']:>9.4f}")
        learned.append(risk_coverage_curve(t["pred"], t["true"], conf, k=k, lc=t["lc"],
                                           theta=theta, strategy=strategy))
        # label-shuffle control (train on shuffled val labels)
        sconf, _ = _confidence_for_fold(f, k, n_pca, shuffle=True, seed=f)
        shuffled.append(risk_coverage_curve(t["pred"], t["true"], sconf, k=k, lc=t["lc"],
                                            theta=theta, strategy=strategy))
        # exp1 sample-std confidence on the same test bars
        exp1.append(risk_coverage_curve(t["pred"], t["true"], t["std_conf"], k=k, lc=t["lc"],
                                        theta=theta, strategy=strategy))

    def _table(name, per_fold):
        avg = average_folds(per_fold)
        slope = curve_slope(avg, "da")
        print(f"\n--- {name}  (DA slope = {slope:+.3f})  6-fold avg ---")
        print(f"{'cov':>5} {'DA':>7} {'ret':>9} {'AER':>9} {'trades':>8}")
        for r in sorted(avg, key=lambda r: -r["coverage"]):
            aer = f"{r['aer']:+.3f}" if r["aer"] is not None else "   --"
            print(f"{r['coverage']:5.1f} {r['da']:+.3f} {r['ret']:+.4f} {aer:>9} {r['n_trades']:8.0f}")
        return avg, slope

    print("\n" + "#" * 70)
    la, ls = _table("LEARNED confidence (experiment 2)", learned)
    sa, ss = _table("SHUFFLE control (must be flat)", shuffled)
    ea, es = _table("SAMPLE-STD (experiment 1, same bars)", exp1)
    print("\n" + "=" * 70)
    print(f"DA slope   learned {ls:+.3f}   shuffle {ss:+.3f}   exp1-std {es:+.3f}")
    print("VERDICT:", "learned confidence helps" if ls > 0.02 and ls > ss + 0.02
          else "no usable learned signal (learned ~ shuffle/flat)")
    print("=" * 70)
    return {"learned": la, "shuffle": sa, "exp1": ea}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_pca", type=int, default=32)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--strategy", default="long_only")
    a = ap.parse_args()
    evaluate(k=a.k, n_pca=a.n_pca, theta=a.theta, strategy=a.strategy)
