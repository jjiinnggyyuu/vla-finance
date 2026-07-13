"""Experiment 4 -- AsyncVLA-style velocity-loss rater (the senior's 13th token, post-hoc).

The pre-check showed the *directly computed* per-candle velocity loss, combined
with |pred|, gives a real DA signal (0.566 @k=12). But computing it needs ground
truth, so it can't run live. Here we train a small **rater** to *predict* the
per-candle velocity loss from inference-available features (the frozen VLM
features), so it works in real time -- exactly the AsyncVLA confidence rater /
Learning-Loss module, but predicting our validated target (velocity loss, not the
final-action MSE that AsyncVLA uses, which is exp3's flat-bar trap).

Post-hoc / 2-stage (like AsyncVLA): the DiT+BTC is frozen. The rater trains on the
VALIDATION split (Nov, out-of-sample for the DiT) to regress log velocity loss,
and is evaluated on TEST (Dec). Trade gate = predicted-low-loss AND large |pred|.

Success = does the *predicted* loss reproduce the oracle (directly-computed) two-gate
DA (~0.566)? If yes, the signal is usable live and the integrated 13th-token
training is warranted.

Inputs:
  results/exp2_features/v15_btc_fold{f}_{vali,test}.npz  (feature, sample_rets, ...)
  results/exp4_velloss/v15_btc_fold{f}_{val,''}.npz       (vel_loss, per candle)
Run: python examples/Bitcoin/eval_files/exp4_rater.py [--k 6] [--n_pca 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp2_confidence as E  # reuse _pca_fit/_pca_apply and _load  # noqa: E402
from risk_coverage import _kslot_backtest  # noqa: E402

E.PRED_MODE = "single"
FEAT = "results/exp2_features"
VELL = "results/exp4_velloss"


def _load(fold, split, k):
    """Return feature, pred(single), true, lc, velocity-loss(candle k), all time-aligned."""
    fe = np.load(f"{FEAT}/v15_btc_fold{fold}_{'vali' if split=='val' else 'test'}.npz")
    vsuf = "_val" if split == "val" else ""
    vl = np.load(f"{VELL}/v15_btc_fold{fold}{vsuf}.npz")
    assert (fe["ts"] == vl["ts"]).all(), f"ts mismatch fold{fold} {split}"
    return {
        "feat": fe["feature"].astype(np.float32),
        "pred": fe[f"sample_rets_k{k}"][:, 0],
        "true": fe[f"true_ret_k{k}"],
        "lc": fe["lc"],
        "vloss": vl["vel_loss"][:, k - 1].astype(np.float32),   # candle-k velocity loss
    }


def _train_regressor(X, y, epochs=300, lr=0.05, wd=1e-2, seed=0):
    """Ridge-regularized linear regression (torch) on PCA features -> log loss."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    xmu, xsd = Xt.mean(0), Xt.std(0) + 1e-6
    ymu, ysd = yt.mean(), yt.std() + 1e-6
    Xn = (Xt - xmu) / xsd
    yn = (yt - ymu) / ysd
    lin = nn.Linear(Xn.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(lin(Xn).squeeze(-1), yn)
        loss.backward(); opt.step()

    def predict(Xnew):
        Xtn = (torch.tensor(Xnew, dtype=torch.float32) - xmu) / xsd
        with torch.no_grad():
            return (lin(Xtn).squeeze(-1) * ysd + ymu).numpy()
    return predict


def rater_confidence(fold, k, n_pca, use_oracle=False):
    """Train rater on val, predict per-bar velocity loss on test.
    Returns (predicted_loss_test, oracle_loss_test)."""
    val, test = _load(fold, "val", k), _load(fold, "test", k)
    oracle = test["vloss"]
    if use_oracle:
        return oracle, oracle
    mu, sd, comp = E._pca_fit(val["feat"], n_pca)
    Xv = E._pca_apply(val["feat"], mu, sd, comp)
    Xt = E._pca_apply(test["feat"], mu, sd, comp)
    predict = _train_regressor(Xv, np.log(val["vloss"] + 1e-6))
    pred_logloss = predict(Xt)
    return pred_logloss, np.log(oracle + 1e-6)


def two_gate_backtest(k, n_pca, pred_q, vloss_q, use_oracle, mode="lo"):
    """Gate: |pred|>=fold-pred_q-quantile AND predicted-loss<=fold-vloss_q-quantile."""
    das, rets, aers, ntr = [], [], [], []
    for f in range(1, 7):
        t = _load(f, "test", k)
        conf_loss, _ = rater_confidence(f, k, n_pca, use_oracle)   # lower = more confident
        p = t["pred"]
        pth = np.quantile(np.abs(p), pred_q)
        vth = np.quantile(conf_loss, vloss_q)
        gate = (np.abs(p) >= pth) & (conf_loss <= vth)
        pos = np.where(gate & (p > 0), 1.0, 0.0) if mode == "lo" else np.where(gate, np.sign(p), 0.0)
        ret, aer, n = _kslot_backtest(t["true"], pos, k, 0.001, t["lc"])
        das.append(((p[gate] > 0) == (t["true"][gate] > 0)).mean() if gate.sum() else 0.5)
        rets.append(ret); aers.append(aer); ntr.append(n)
    return np.mean(das), np.std(das), np.mean(rets), np.mean(aers), np.mean(ntr)


def rater_quality(k, n_pca):
    """How well does the rater predict the velocity loss on test?"""
    cs, os_ = [], []
    for f in range(1, 7):
        pl, ol = rater_confidence(f, k, n_pca, use_oracle=False)
        cs.append(np.corrcoef(pl, ol)[0, 1])
    return np.mean(cs), cs


def main(k=6, n_pca=64):
    print(f"\n{'='*72}\nEXPERIMENT 4 -- velocity-loss rater (k={k}, PCA={n_pca})\n{'='*72}")
    c, cs = rater_quality(k, n_pca)
    print(f"rater quality: corr(predicted loss, oracle loss) on test = {c:+.3f}  "
          f"fold {[f'{x:+.2f}' for x in cs]}")
    print(f"\n{'gate':>26} {'src':>7} {'DA':>13} {'ret':>8} {'AER':>8} {'trades':>7}")
    for pq, vq in [(0.5, 0.5), (0.5, 0.3)]:
        for src, oracle in [("oracle", True), ("rater", False)]:
            d, ds, r, a, n = two_gate_backtest(k, n_pca, pq, vq, oracle)
            tag = f"pred>{int((1-pq)*100)}% loss<{int(vq*100)}%"
            print(f"{tag:>26} {src:>7} {d:.3f}±{ds:.3f} {r:+.4f} {a:+.3f} {n:>7.0f}")
    # baseline
    d, ds, r, a, n = two_gate_backtest(k, n_pca, 0.0, 1.0, True)
    print(f"{'baseline (all long)':>26} {'--':>7} {d:.3f}±{ds:.3f} {r:+.4f} {a:+.3f} {n:>7.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_pca", type=int, default=64)
    a = ap.parse_args()
    main(k=a.k, n_pca=a.n_pca)
