"""여러 fold의 평가 JSON을 읽어서 지표 평균/표준편차를 출력.

v15 OFT/OFT+DCT 비교, Kronos 베이스라인 등 6-fold 결과 집계에 공용으로 사용.

사용 예:
  python examples/Bitcoin/eval_files/aggregate_folds.py results/kronos_baseline/fold*.json
  python examples/Bitcoin/eval_files/aggregate_folds.py results/v15_oft/fold*.json --label "OFT"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 집계할 스칼라 지표 (fold마다 다른 값이 나오는 것들)
SCALAR_KEYS = ["close_mae", "price_ic", "price_rankic"]
K_VALUES = [1, 3, 6, 12]
PER_K_KEYS = ["da", "aer", "ret", "bench_ret", "sortino", "mdd",
              "win_rate", "profit_factor",
              "long_ratio", "short_ratio", "watch_ratio",
              "long_ratio_ns", "short_ratio_ns"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    """inf/nan을 제외하고 평균/표준편차. 전부 걸러지면 (nan, nan)."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="fold별 평가 JSON 경로들")
    parser.add_argument("--label", default="", help="출력 제목용 라벨")
    parser.add_argument("--output_json", default=None, help="집계 결과 저장 경로")
    args = parser.parse_args()

    results = []
    for fp in args.json_files:
        with open(fp) as f:
            results.append(json.load(f))
    n_folds = len(results)

    agg = {}

    # 셀: "mean±std" 문자열 (칸 너비 CW로 고정)
    CW = 15  # 각 k 열 너비
    NW = 16  # metric 이름 열 너비

    def cell(m, s):
        return f"{m:+.3f} ± {s:.3f}"

    def hline(ch="-"):
        print("+" + "-" * NW + ("+" + "-" * CW) * len(K_VALUES) + "+")

    title = f"{n_folds}-FOLD AVERAGE"
    if args.label:
        title += f"  [{args.label}]"
    print("\n" + title)
    print(f"Folds: {', '.join(Path(fp).stem for fp in args.json_files)}")

    # ── 스칼라 지표 표 ── (값이 커서 별도 너비 SW)
    SW = 24
    print("+" + "-" * NW + "+" + "-" * SW + "+")
    print("|" + "metric".center(NW) + "|" + "mean ± std".center(SW) + "|")
    print("+" + "-" * NW + "+" + "-" * SW + "+")
    for key in SCALAR_KEYS:
        vals = [r.get(key) for r in results]
        m, s = _mean_std(vals)
        agg[key] = {"mean": m, "std": s}
        print("|" + f" {key}".ljust(NW) + "|" + f"{m:.4f} ± {s:.4f}".center(SW) + "|")
    print("+" + "-" * NW + "+" + "-" * SW + "+")

    # ── per-k 지표 표 ──
    print()
    hline()
    header = "|" + "metric".center(NW) + "".join("|" + f"k={k}".center(CW) for k in K_VALUES) + "|"
    print(header)
    hline()
    for base in PER_K_KEYS:
        row = "|" + f" {base}".ljust(NW)
        for k in K_VALUES:
            key = f"{base}_k{k}"
            vals = [r.get(key) for r in results]
            m, s = _mean_std(vals)
            agg[key] = {"mean": m, "std": s}
            row += "|" + cell(m, s).center(CW)
        print(row + "|")
    hline()

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_folds": n_folds, "label": args.label, "metrics": agg}, indent=2))
        print(f"Saved aggregate to {out}")


if __name__ == "__main__":
    main()
