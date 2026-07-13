"""Unified WALK-FORWARD evaluation of exp1-4 on the SAME footing.

Every experiment reduces to a per-bar confidence signal (higher = trade), produced
out-of-sample on both val (Nov) and test (Dec). Trade thresholds are chosen on VAL
and applied to TEST -- fully deployable, no look-ahead. Two-gate for all:
    trade if  |pred| >= P-cutoff  AND  confidence >= C-cutoff
Cutoffs come from val quantiles (variant A = fixed 50%/top-30%; variant B = grid
search maximizing val AER). long-only, k in {6,12}, 6-fold average.

Confidence per experiment:
  exp1  -std of the 10 predict_action samples            (no training)
  exp2  P(direction correct), logistic probe on PCA(feat) (inner-CV on val -> OOS)
  exp3  P(low |error|),        logistic probe on PCA(feat) (inner-CV on val -> OOS)
  exp4  -rater predicted log velocity-loss                (rater, val+test dumped)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp2_confidence as E  # _pca_fit/_pca_apply/_train_probe  # noqa: E402
from risk_coverage import _kslot_backtest  # noqa: E402

FEAT = "results/exp2_features"
RATER = "results/exp4_rater"


def _base(fold, split, k):
    tag = "vali" if split == "val" else "test"
    d = np.load(f"{FEAT}/v15_btc_fold{fold}_{tag}.npz")
    return {
        "pred": d[f"sample_rets_k{k}"][:, 0],
        "samples": d[f"sample_rets_k{k}"],
        "true": d[f"true_ret_k{k}"],
        "lc": d["lc"],
        "feat": d["feature"].astype(np.float32),
        "ts": d["ts"],
    }


# ---- confidence signals (higher = more confident), OOS on val and test -------
def conf_exp1(v, t, k):
    return -v["samples"].std(axis=1), -t["samples"].std(axis=1)


def _oof_val_probe(feat, y, n_pca, folds=5):
    """Out-of-fold val confidence via inner CV (no leakage for threshold-setting)."""
    N = len(y)
    oof = np.zeros(N)
    idx = np.arange(N)
    bins = idx % folds
    for fb in range(folds):
        tr, te = bins != fb, bins == fb
        if te.sum() == 0 or tr.sum() < 20:
            continue
        mu, sd, comp = E._pca_fit(feat[tr], n_pca)
        pred, _ = E._train_probe(E._pca_apply(feat[tr], mu, sd, comp), y[tr], seed=fb)
        oof[te] = pred(E._pca_apply(feat[te], mu, sd, comp))
    return oof


def _full_probe(feat_tr, y_tr, feat_te, n_pca):
    mu, sd, comp = E._pca_fit(feat_tr, n_pca)
    pred, _ = E._train_probe(E._pca_apply(feat_tr, mu, sd, comp), y_tr, seed=0)
    return pred(E._pca_apply(feat_te, mu, sd, comp))


def conf_exp2(v, t, k, n_pca=32):
    y = ((v["pred"] > 0) == (v["true"] > 0)).astype(np.float32)   # direction correct
    cv = _oof_val_probe(v["feat"], y, n_pca)
    ct = _full_probe(v["feat"], y, t["feat"], n_pca)
    return cv, ct


def conf_exp3(v, t, k, n_pca=64):
    ev = np.abs(v["pred"] - v["true"]); et = np.abs(t["pred"] - t["true"])  # noqa: F841
    y = (ev <= np.median(ev)).astype(np.float32)                 # low-error label
    cv = _oof_val_probe(v["feat"], y, n_pca)
    ct = _full_probe(v["feat"], y, t["feat"], n_pca)
    return cv, ct


def conf_exp4(v, t, k):
    lv = np.load(f"{RATER}/v15_btc_fold{v['fold']}_val.npz")
    lt = np.load(f"{RATER}/v15_btc_fold{t['fold']}.npz")
    assert (lv["ts"] == v["ts"]).all() and (lt["ts"] == t["ts"]).all()
    return -lv["pred_logloss"][:, k - 1], -lt["pred_logloss"][:, k - 1]


CONF = {"exp1": conf_exp1, "exp2": conf_exp2, "exp3": conf_exp3, "exp4": conf_exp4}


# ---- gating + backtest -------------------------------------------------------
def _gate_bt(d, pth, cth, conf, k):
    p = d["pred"]
    gate = (np.abs(p) >= pth) & (conf >= cth)
    pos = np.where(gate & (p > 0), 1.0, 0.0)
    ret, aer, n = _kslot_backtest(d["true"], pos, k, 0.001, d["lc"])
    da = (((p[gate] > 0) == (d["true"][gate] > 0)).mean() if gate.sum() else np.nan)
    return da, ret, aer, int(gate.sum())


def variant_A(exp, k, pq=0.5, cq=0.7):
    das, rets, aers, covs = [], [], [], []
    for f in range(1, 7):
        v = _base(f, "val", k); t = _base(f, "test", k); v["fold"] = t["fold"] = f
        cv, ct = CONF[exp](v, t, k)
        pth = np.quantile(np.abs(v["pred"]), pq); cth = np.quantile(cv, cq)
        da, ret, aer, n = _gate_bt(t, pth, cth, ct, k)
        das.append(da); rets.append(ret); aers.append(aer); covs.append(n / len(t["pred"]))
    return np.nanmean(das), np.nanmean(rets), np.nanmean(aers), np.nanmean(covs)


def variant_B(exp, k, gp, gc):
    das, rets, aers, covs = [], [], [], []
    for f in range(1, 7):
        v = _base(f, "val", k); t = _base(f, "test", k); v["fold"] = t["fold"] = f
        cv, ct = CONF[exp](v, t, k)
        best, bk = None, -1e9
        for pq in gp:
            for cq in gc:
                pth = np.quantile(np.abs(v["pred"]), pq); cth = np.quantile(cv, cq)
                _, _, aer_v, n_v = _gate_bt(v, pth, cth, cv, k)
                if n_v < 15:
                    continue
                if aer_v > bk:
                    bk, best = aer_v, (pth, cth)
        if best is None:
            best = (0.0, -1e9)
        da, ret, aer, n = _gate_bt(t, best[0], best[1], ct, k)
        das.append(da); rets.append(ret); aers.append(aer); covs.append(n / len(t["pred"]))
    return np.nanmean(das), np.nanmean(rets), np.nanmean(aers), np.nanmean(covs)


def baseline(k):
    das, rets, aers = [], [], []
    for f in range(1, 7):
        t = _base(f, "test", k); p = t["pred"]; pos = np.where(p > 0, 1.0, 0.0)
        ret, aer, n = _kslot_backtest(t["true"], pos, k, 0.001, t["lc"])
        das.append(((p > 0) == (t["true"] > 0)).mean()); rets.append(ret); aers.append(aer)
    return np.mean(das), np.mean(rets), np.mean(aers)


def main(k):
    gp = [0.3, 0.4, 0.5, 0.6, 0.7]
    gc = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n{'='*80}\nWALK-FORWARD, all experiments, two-gate (k={k})  -- thresholds from VAL\n{'='*80}")
    bd, br, ba = baseline(k)
    print(f"{'method':>34} {'DA':>7} {'ret':>9} {'AER':>9} {'cov':>7}")
    print(f"{'baseline (trade all)':>34} {bd:.3f} {br:+.4f} {ba:+.4f} {'100%':>7}  <- AER to beat")
    for exp in ["exp1", "exp2", "exp3", "exp4"]:
        da, r, a, c = variant_A(exp, k)
        print(f"{exp+' A: val-fixed 50/top30':>34} {da:.3f} {r:+.4f} {a:+.4f} {c*100:6.1f}%")
        da, r, a, c = variant_B(exp, k, gp, gc)
        print(f"{exp+' B: val-optimized':>34} {da:.3f} {r:+.4f} {a:+.4f} {c*100:6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    main(a.k)
