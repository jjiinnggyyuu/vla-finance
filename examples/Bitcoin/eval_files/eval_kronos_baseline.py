"""Kronos 베이스라인 평가 — BTC 단일 자산.

우리 VLA 모델과 *완전히 동일한* 데이터셋/split/지표로 Kronos를 평가한다.
공정 비교의 핵심:
  - 입력 윈도우: 우리 bitcoin_datasets 와 동일 (input_window 캔들)
  - 예측 호라이즌: 동일 (prediction_horizon 스텝)
  - 평가 split: 동일 (--config_yaml + --split 로 동일 fold 지정)
  - 지표: eval_bitcoin_close_mae.compute_metrics_from_prices() 재사용

Kronos는 출력이 이미 실제 가격 단위(역정규화 완료)라 별도 복원이 필요 없다.

사용 예:
  python examples/Bitcoin/eval_files/eval_kronos_baseline.py \
    --config_yaml examples/Bitcoin/train_files/v15_dct_ablation/starvla_train_bitcoin_v15_fold1_oft.yaml \
    --split test --sample_count 5 \
    --plot_dir results/kronos/fold1/ --output_json results/kronos/fold1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# Kronos 레포 경로 등록
KRONOS_ROOT = Path(__file__).resolve().parents[3] / "reference" / "Kronos"
sys.path.insert(0, str(KRONOS_ROOT))

from starVLA.dataloader.bitcoin_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

# 우리 지표 코어 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import compute_metrics_from_prices

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def build_kronos_predictor(tokenizer_id: str, model_id: str, max_context: int, device: str):
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    return predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True,
                        help="우리 모델과 동일한 데이터셋/split을 정의한 YAML")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--sample_count", type=int, default=5,
                        help="Kronos 확률적 샘플링 횟수 (평균내서 안정화)")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tokenizer_id", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--model_id", default="NeoQuasar/Kronos-small")
    parser.add_argument("--max_context", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--plot_dir", default=None)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    # 우리 데이터셋 로드 (raw 캔들 접근용)
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    rows = dataset.rows
    starts = dataset.starts
    input_window = dataset.input_window
    horizon = dataset.prediction_horizon

    if args.limit_samples is not None:
        starts = starts[: args.limit_samples]

    print(f"Kronos baseline | split={args.split} | samples={len(starts)} "
          f"| input_window={input_window} | horizon={horizon}")

    predictor = build_kronos_predictor(
        args.tokenizer_id, args.model_id, args.max_context, args.device
    )

    pred_prices_list: list[np.ndarray] = []
    true_prices_list: list[np.ndarray] = []
    last_close_list: list[float] = []
    ts_list: list[str] = []
    future_ts_list: list[list] = []

    for start in tqdm(starts, desc="Kronos predicting"):
        split = start + input_window
        end = split + horizon
        past = rows[start:split]      # 입력 윈도우
        future = rows[split:end]      # 정답 미래

        # Kronos 입력 DataFrame
        x_df = pd.DataFrame({
            "open":   [r["open"]   for r in past],
            "high":   [r["high"]   for r in past],
            "low":    [r["low"]    for r in past],
            "close":  [r["close"]  for r in past],
            "volume": [r["volume"] for r in past],
            "amount": [r["amount"] for r in past],
        })
        x_timestamp = pd.Series([r["timestamp_dt"] for r in past])
        y_timestamp = pd.Series([r["timestamp_dt"] for r in future])

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=args.T,
            top_p=args.top_p,
            sample_count=args.sample_count,
            verbose=False,
        )

        # OHLC 4채널만 (B 동일 단위, 실제 가격)
        pred_prices = pred_df[["open", "high", "low", "close"]].values.astype(np.float32)  # (H, 4)
        true_prices = np.array(
            [[r["open"], r["high"], r["low"], r["close"]] for r in future], dtype=np.float32
        )  # (H, 4)

        pred_prices_list.append(pred_prices)
        true_prices_list.append(true_prices)
        last_close_list.append(float(past[-1]["close"]))
        ts_list.append(past[-1]["timestamp"])
        future_ts_list.append([r["timestamp"] for r in future])

    metrics = compute_metrics_from_prices(
        pred_prices_list, true_prices_list, last_close_list, ts_list, future_ts_list,
        slippage=args.slippage, output_csv=args.output_csv, plot_dir=args.plot_dir,
    )
    metrics.update({"split": args.split, "model": args.model_id, "sample_count": args.sample_count})

    print("\n" + "=" * 70)
    print("  KRONOS BASELINE RESULTS")
    print("=" * 70)
    print(f"  Split    : {metrics['split']}")
    print(f"  Model    : {metrics['model']}  (sample_count={metrics['sample_count']})")
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
    print(f"  {'AER':12} {metrics['aer_k1']:>8.4f} {metrics['aer_k3']:>8.4f} {metrics['aer_k6']:>8.4f} {metrics['aer_k12']:>8.4f}")
    print(f"  {'Sortino':12} {metrics['sortino_k1']:>8.4f} {metrics['sortino_k3']:>8.4f} {metrics['sortino_k6']:>8.4f} {metrics['sortino_k12']:>8.4f}")
    print(f"  {'MDD':12} {metrics['mdd_k1']*100:>7.2f}% {metrics['mdd_k3']*100:>7.2f}% {metrics['mdd_k6']*100:>7.2f}% {metrics['mdd_k12']*100:>7.2f}%")
    print(f"  {'Win rate':12} {metrics['win_rate_k1']:>8.4f} {metrics['win_rate_k3']:>8.4f} {metrics['win_rate_k6']:>8.4f} {metrics['win_rate_k12']:>8.4f}")
    print(f"  {'Profit fac':12} {metrics['profit_factor_k1']:>8.4f} {metrics['profit_factor_k3']:>8.4f} {metrics['profit_factor_k6']:>8.4f} {metrics['profit_factor_k12']:>8.4f}")
    print("=" * 70)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
