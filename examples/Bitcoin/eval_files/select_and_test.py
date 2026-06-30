"""Model selection + test 평가 자동화 (단일 fold/run).

흐름:
  1. validation split으로 모든 체크포인트 평가 → 각자 da_k6 기록
  2. da_k6 최고 체크포인트 선택 (model selection)
  3. 선택된 체크포인트로 test split 평가 → JSON + equity curve 저장

멀티 GPU 분산 추론 지원 (accelerate launch):
  - 각 GPU에 모델 통째로 올림(DDP 래핑 X) → 데이터만 DistributedSampler로 분할
  - 각 GPU가 자기 몫 추론 → gather_object 로 전체 예측을 모든 rank가 공유
  - 모든 rank가 동일한 gathered 데이터로 같은 best 체크포인트를 결정(결정론적)
  - 지표 계산/파일 저장은 main process만

사용 예:
  # 단일 GPU
  python examples/Bitcoin/eval_files/select_and_test.py --config_yaml ... --ckpt_dir ...
  # 멀티 GPU (fold 하나를 GPU 2개로 분산)
  accelerate launch --num_processes 2 examples/Bitcoin/eval_files/select_and_test.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import (  # noqa: E402
    compute_metrics_from_prices,
    load_state_dict,
    restore_prices,
    restore_prices_delta,
)

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# Model selection 기준: 1순위 da_k6, 동점 시 2순위 aer_k6.
#   da_k6  — 6시간 호라이즌 방향 정확도 (v9_delta에서 가장 신뢰도 높았던 지표).
#   aer_k6 — 동점 tie-break. 같은 호라이즌(k=6)의 초과수익(전략−BTC홀드).
SELECTION_METRIC = "da_k6"
SELECTION_TIEBREAK = "aer_k6"


def build_loader(cfg, split, batch_size, num_workers, accelerator):
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=split)
    sampler = None
    if accelerator.num_processes > 1:
        # shuffle=False: 순서 보존(어차피 compute_metrics가 ts로 재정렬).
        #   균등 분할 위해 padding 발생 → gather 후 ts로 중복 제거.
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler,
    )


def collect_predictions(model, dataloader, target_mode, accelerator):
    """각 GPU에서 자기 shard 추론 → gather_object로 전체 예측을 모든 rank가 공유.

    반환: (preds, trues, last_closes, ts, future_ts) — ts 기준 중복 제거됨.
    """
    local = []  # 이 rank가 처리한 샘플들의 dict 리스트
    a = 0          # evaluate only BTC (primary asset = asset 0)
    sl = slice(a * 4, a * 4 + 4)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating",
                          disable=not accelerator.is_local_main_process):
            pred = model.predict_action(batch)["normalized_actions"]  # (B, H, 4*N)
            target     = np.stack([s["action"] for s in batch], axis=0)            # (B, H, 4*N)
            last_close = np.stack([np.atleast_1d(s["last_close"]) for s in batch]).astype(np.float32)  # (B, N)
            asset_std  = np.stack([np.atleast_1d(s["asset_std"])  for s in batch]).astype(np.float32)  # (B, N)

            B, H, _ = pred.shape
            # Slice BTC, un-normalize delta back to raw scale (× std), restore prices.
            std_a = asset_std[:, a][:, None, None]    # (B,1,1)
            pred_a = pred[:, :, sl] * std_a            # (B, H, 4) raw delta
            true_a = target[:, :, sl] * std_a
            lc_a = last_close[:, a]                    # (B,)

            if target_mode == "delta":
                pred_prices = restore_prices_delta(pred_a, lc_a)
                true_prices = restore_prices_delta(true_a, lc_a)
            else:
                lc_rep = np.repeat(lc_a, H)
                pred_prices = restore_prices(pred_a.reshape(-1, 4), lc_rep).reshape(B, H, 4)
                true_prices = restore_prices(true_a.reshape(-1, 4), lc_rep).reshape(B, H, 4)

            for i in range(B):
                local.append({
                    "pred":  pred_prices[i].astype(np.float32),
                    "true":  true_prices[i].astype(np.float32),
                    "lc":    float(lc_a[i]),
                    "ts":    batch[i]["base_timestamp"],
                    "fts":   batch[i]["future_timestamps"],
                })

    # 멀티 GPU: 각 rank 리스트를 모든 rank가 공유 (gather_object는 모든 rank에 전체 반환).
    if accelerator.num_processes > 1:
        gathered = gather_object(local)
    else:
        gathered = local

    # DistributedSampler padding으로 생긴 중복을 ts 기준으로 제거.
    seen = set()
    preds, trues, lcs, ts, fts = [], [], [], [], []
    for s in gathered:
        if s["ts"] in seen:
            continue
        seen.add(s["ts"])
        preds.append(s["pred"]); trues.append(s["true"]); lcs.append(s["lc"])
        ts.append(s["ts"]);      fts.append(s["fts"])
    return preds, trues, lcs, ts, fts


def evaluate_ckpt(model, loader, target_mode, accelerator, slippage, plot_dir=None):
    """체크포인트 1개 평가 → 지표 dict. 모든 rank가 동일 결과(결정론적)."""
    preds, trues, lcs, ts, fts = collect_predictions(model, loader, target_mode, accelerator)
    return compute_metrics_from_prices(
        preds, trues, lcs, ts, fts, slippage=slippage, plot_dir=plot_dir,
    )


def main() -> None:
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--ckpt_dir",    required=True)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--slippage",    type=float, default=0.001)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--plot_dir",    default=None)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    target_mode = str(getattr(cfg.datasets.vla_data, "target_mode", "log_ratio"))

    # 모델을 각 GPU에 통째로 올림 (DDP 래핑 X → predict_action 직접 호출 가능).
    model = build_framework(cfg).to(accelerator.device).eval()

    val_loader  = build_loader(cfg, "validation", args.batch_size, args.num_workers, accelerator)
    test_loader = build_loader(cfg, "test",       args.batch_size, args.num_workers, accelerator)

    ckpts = sorted(Path(args.ckpt_dir).glob("steps_*_action_model.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {args.ckpt_dir}")

    if is_main:
        print(f"=== Model selection on validation ({SELECTION_METRIC}, tie-break {SELECTION_TIEBREAK}) ===")

    best_ckpt = None
    best_key  = (-float("inf"), -float("inf"))  # (da_k6, aer_k6)
    val_scores = {}
    for ckpt in ckpts:
        sd = load_state_dict(str(ckpt))
        model.action_model.load_state_dict(sd, strict=False)

        # 모든 rank가 동일 gathered 데이터로 동일 지표 계산 → 동일 best 결정.
        m = evaluate_ckpt(model, val_loader, target_mode, accelerator, args.slippage)
        score    = float(m[SELECTION_METRIC])
        tiebreak = float(m[SELECTION_TIEBREAK])
        val_scores[ckpt.name] = {k: float(v) for k, v in m.items()
                                 if isinstance(v, (int, float))}
        key    = (score, tiebreak)
        marker = ""
        if key > best_key:
            best_key, best_ckpt = key, ckpt
            marker = " <- best"
        if is_main:
            print(f"  {ckpt.name:32} {SELECTION_METRIC}={score:.4f}  {SELECTION_TIEBREAK}={tiebreak:.4f}{marker}")

    if is_main:
        print(f"\nSelected: {best_ckpt.name}  "
              f"({SELECTION_METRIC}={best_key[0]:.4f}, {SELECTION_TIEBREAK}={best_key[1]:.4f})")
        print(f"\n=== Final test evaluation ===")

    # 선택된 체크포인트로 test 평가 (plot은 main만).
    sd = load_state_dict(str(best_ckpt))
    model.action_model.load_state_dict(sd, strict=False)
    metrics = evaluate_ckpt(model, test_loader, target_mode, accelerator,
                            args.slippage, plot_dir=(args.plot_dir if is_main else None))

    if not is_main:
        return

    metrics.update({
        "split":              "test",
        "selected_ckpt":      best_ckpt.name,
        "selection_metric":   SELECTION_METRIC,
        "selection_tiebreak": SELECTION_TIEBREAK,
        f"val_{SELECTION_METRIC}":   best_key[0],
        f"val_{SELECTION_TIEBREAK}": best_key[1],
        "val_scores":         val_scores,
    })
    print(f"  test {SELECTION_METRIC}={metrics[SELECTION_METRIC]:.4f}  "
          f"price_ic={metrics['price_ic']:.4f}  aer_k6={metrics['aer_k6']:.4f}")

    if args.output_json:
        import csv
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Saved test metrics to {out}")

        csv_path    = out.with_name(out.stem + "_val_ckpts.csv")
        metric_keys = sorted({k for v in val_scores.values() for k in v})
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["checkpoint", "selected", *metric_keys])
            for name in sorted(val_scores, key=lambda n: int(n.split("_")[1])):
                row = val_scores[name]
                w.writerow([name, "*" if name == best_ckpt.name else "",
                            *[row.get(k, "") for k in metric_keys]])
        print(f"Saved per-checkpoint validation metrics to {csv_path}")


if __name__ == "__main__":
    main()
