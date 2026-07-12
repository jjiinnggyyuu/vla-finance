"""Experiment 4 pre-check (B) -- dump per-bar, per-candle flow-matching VELOCITY loss.

The AsyncVLA confidence rater is trained to *predict* the per-candle flow-matching
loss. Before training that rater (expensive), we compute the loss *directly*
(Diff-DAgger style) on the frozen DiT+BTC and test whether it is a useful trade
signal -- specifically the two-gate "low velocity loss AND large |pred|".

For each test bar we replicate the head's training-loss computation over K random
(noise, t) draws and average:
    velocity = actions - noise
    per_candle_loss = mean_ch( (pred_velocity - velocity)^2 * asset_weight )   (H,)
This is the exact quantity the AsyncVLA pseudo-label is built from, but computed,
not predicted. Saved as vel_loss (N, H) per fold, aligned by timestamp.

Usage (one GPU per fold):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_velocity_loss.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold6.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold6/checkpoints/steps_1000_action_model.pt \
    --output_npz results/exp4_velloss/v15_btc_fold6.npz --num_draws 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import load_state_dict  # noqa: E402
from dump_features import _vlm_forward  # reuse the VLM-forward replication  # noqa: E402

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


@torch.inference_mode()
def per_candle_velocity_loss(head, vl_embs, actions, mask, num_draws):
    """Replicate the head's flow-matching loss, per candle, averaged over draws.
    Returns (B, H) tensor of per-candle velocity loss."""
    B, H, _ = actions.shape
    device = actions.device
    aw = head.asset_channel_weights.to(torch.float32)
    acc = torch.zeros(B, H, device=device, dtype=torch.float32)

    for _ in range(num_draws):
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = head.sample_time(B, device=device, dtype=actions.dtype)[:, None, None]
        noisy = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_disc = (t[:, 0, 0] * head.num_timestep_buckets).long()

        af = head.action_encoder(noisy, t_disc)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(af.shape[1], device=device)
            af = af + head.position_embedding(pos_ids).unsqueeze(0)
        ft = head.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa = torch.cat((ft, af), dim=1)

        model_output = head.model(
            hidden_states=sa, encoder_hidden_states=vl_embs,
            encoder_attention_mask=mask, timestep=t_disc,
            return_all_hidden_states=False,
        )
        pred = head.action_decoder(model_output)
        pred_actions = pred[:, -H:]
        # per-candle: mean over the 4 OHLC channels (asset-weighted), like the scalar loss
        pc = (((pred_actions - velocity) ** 2) * aw).mean(dim=2)   # (B, H)
        acc += pc.float()
    return acc / num_draws


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--num_draws", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args, clip = p.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clip)))
    cfg = apply_config_compat(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device).eval()
    model.action_model.load_state_dict(load_state_dict(args.ckpt), strict=False)
    head = model.action_model
    print(f"[velloss] ckpt={args.ckpt}  draws={args.num_draws}")

    ds = get_vla_dataset(cfg.datasets.vla_data, mode="test")
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    records: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="velloss"):
            last_hidden, mask, _ = _vlm_forward(model, batch)
            actions = torch.tensor(
                np.stack([s["action"] for s in batch], axis=0),
                device=device, dtype=torch.float32,
            )  # (B, H, 4) normalized target (BTC only)
            with torch.autocast("cuda", dtype=torch.float32):
                vl = per_candle_velocity_loss(head, last_hidden, actions, mask, args.num_draws)
            vl = vl.detach().cpu().numpy()  # (B, H)
            for i, s in enumerate(batch):
                ts = s["base_timestamp"]
                if ts not in records:
                    records[ts] = vl[i]

    items = sorted(records.items(), key=lambda kv: kv[0])
    out = {
        "ts": np.array([ts for ts, _ in items]),
        "vel_loss": np.stack([v for _, v in items]).astype(np.float32),  # (N, H)
    }
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    vl = out["vel_loss"]
    print(f"[velloss] saved {len(items)} bars x {vl.shape[1]} candles -> {args.output_npz}")
    print(f"[velloss] candle-6 loss: mean {vl[:,5].mean():.4f}  range [{vl[:,5].min():.4f},{vl[:,5].max():.4f}]")


if __name__ == "__main__":
    main()
