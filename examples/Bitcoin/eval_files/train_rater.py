"""Experiment 4 (integrated) -- AsyncVLA-style confidence rater, GPU training.

The post-hoc probe (feature -> velocity loss) failed to predict the loss
(corr 0.08). This is the *integrated* version: a small transformer rater reads
the DiT trunk's per-candle representation (which attended to the VL context) and
predicts the per-candle flow-matching velocity loss. Attention over the 12 candle
tokens may extract what the frozen-feature probe could not.

2-stage (AsyncVLA): the DiT+BTC stays FROZEN; only the rater trains -> no gradient
reaches the backbone (senior's rule). Target = per-candle velocity loss of the
current (noise, t) draw, detached. We regress log-loss with MSE.

Trains on the model's TRAIN split (Jan-Oct, ~7,229 bars -- enough for a small
transformer), then dumps per-bar per-candle predicted loss on TEST (K draws
averaged), for the two-gate evaluation in exp4_rater-style scoring.

Usage (one GPU per fold):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/train_rater.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold6.yaml \
    --ckpt playground/Checkpoints/v15_btc_1h/fold6/checkpoints/steps_1000_action_model.pt \
    --output_npz results/exp4_rater/v15_btc_fold6.npz --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bitcoin_close_mae import load_state_dict  # noqa: E402
from dump_features import _vlm_forward  # noqa: E402

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


class ConfidenceRater(nn.Module):
    """Transformer over the 12 candle representations -> per-candle predicted loss.
    AsyncVLA-style (attention lets candles + context inform each rating)."""

    def __init__(self, dim, n_layers=4, n_heads=4, drop=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=drop, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):                       # x: (B, H, dim)
        return self.head(self.encoder(x)).squeeze(-1)   # (B, H) predicted log-loss


def head_forward(head, vl_embs, actions, mask):
    """Replicate the DiT head's training pass for ONE (noise, t) draw.
    Returns (model_output_action[B,H,D], per_candle_velocity_loss[B,H])."""
    B, H, _ = actions.shape
    device = actions.device
    noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
    t = head.sample_time(B, device=device, dtype=actions.dtype)[:, None, None]
    noisy = (1 - t) * noise + t * actions
    velocity = actions - noise
    t_disc = (t[:, 0, 0] * head.num_timestep_buckets).long()

    af = head.action_encoder(noisy, t_disc)
    if head.config.add_pos_embed:
        pos = torch.arange(af.shape[1], device=device)
        af = af + head.position_embedding(pos).unsqueeze(0)
    ft = head.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
    sa = torch.cat((ft, af), dim=1)
    model_output = head.model(
        hidden_states=sa, encoder_hidden_states=vl_embs,
        encoder_attention_mask=mask, timestep=t_disc, return_all_hidden_states=False,
    )                                            # (B, seq, D)
    mo_act = model_output[:, -H:, :]             # (B, H, D) per-candle repr
    pred_actions = head.action_decoder(model_output)[:, -H:]
    aw = head.asset_channel_weights.to(pred_actions.dtype)
    per_candle_loss = (((pred_actions - velocity) ** 2) * aw).mean(dim=2)  # (B, H)
    return mo_act, per_candle_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_npz", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--test_draws", type=int, default=20)
    p.add_argument("--val_draws", type=int, default=8)
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
    for pm in model.parameters():
        pm.requires_grad_(False)                 # freeze DiT + VLM (2-stage)

    D = int(head.model.config.output_dim)
    rater = ConfidenceRater(D).to(device).train()
    opt = torch.optim.AdamW(rater.parameters(), lr=args.lr, weight_decay=1e-2)
    print(f"[rater] dim={D}  epochs={args.epochs}  ckpt={args.ckpt}")

    def batch_actions(batch):
        return torch.tensor(np.stack([s["action"] for s in batch], axis=0),
                            device=device, dtype=torch.float32)

    def collect(loader, k_draws, desc):
        """Per-bar per-candle (predicted log-loss, actual log-loss), K draws averaged."""
        preds, trues, tslist = {}, {}, []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                lh, mask, _ = _vlm_forward(model, batch)
                actions = batch_actions(batch)
                B, H, _ = actions.shape
                pacc = torch.zeros(B, H, device=device)
                tacc = torch.zeros(B, H, device=device)
                for _ in range(k_draws):
                    with torch.autocast("cuda", dtype=torch.float32):
                        mo, vl = head_forward(head, lh, actions, mask)
                    pacc += rater(mo.float())
                    tacc += torch.log(vl.float() + 1e-6)
                pacc = (pacc / k_draws).cpu().numpy()
                tacc = (tacc / k_draws).cpu().numpy()
                for i, s in enumerate(batch):
                    ts = s["base_timestamp"]
                    if ts not in preds:
                        preds[ts] = pacc[i]; trues[ts] = tacc[i]; tslist.append(ts)
        tslist.sort()
        P = np.stack([preds[t] for t in tslist]); T = np.stack([trues[t] for t in tslist])
        return np.array(tslist), P, T

    def val_score():
        rater.eval()
        _, P, T = collect(val_loader, args.val_draws, "rater/val")
        # mean over candles of corr(predicted, actual) across bars -- higher = better
        cs = [np.corrcoef(P[:, c], T[:, c])[0, 1] for c in range(P.shape[1])]
        rater.train()
        return float(np.nanmean(cs))

    # ── data ──────────────────────────────────────────────────────────────
    train_ds = get_vla_dataset(cfg.datasets.vla_data, mode="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                              num_workers=args.num_workers, shuffle=True)
    val_ds = get_vla_dataset(cfg.datasets.vla_data, mode="validation")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                            num_workers=args.num_workers, shuffle=False)
    test_ds = get_vla_dataset(cfg.datasets.vla_data, mode="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                             num_workers=args.num_workers, shuffle=False)

    # ── train on TRAIN split, select best epoch by VAL correlation ────────
    import copy
    best_val, best_state, best_ep = -1e9, None, -1
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for batch in tqdm(train_loader, desc=f"rater ep{ep}"):
            with torch.no_grad():
                lh, mask, _ = _vlm_forward(model, batch)
                actions = batch_actions(batch)
                with torch.autocast("cuda", dtype=torch.float32):
                    mo, vloss = head_forward(head, lh, actions, mask)
            pred = rater(mo.float().detach())
            target = torch.log(vloss.float().detach() + 1e-6)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vc = val_score()
        star = ""
        if vc > best_val:
            best_val, best_ep = vc, ep
            best_state = copy.deepcopy(rater.state_dict())
            star = " <- best"
        print(f"[rater] epoch {ep}: train MSE {tot/max(nb,1):.4f}  val corr {vc:+.3f}{star}", flush=True)

    # ── load best rater, dump TEST predictions + save weights ─────────────
    rater.load_state_dict(best_state)
    rater.eval()
    print(f"[rater] selected epoch {best_ep} (val corr {best_val:+.3f})")
    ts, P, _ = collect(test_loader, args.test_draws, "rater/test")
    out = {"ts": ts, "pred_logloss": P.astype(np.float32)}
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    torch.save({"state_dict": best_state, "best_epoch": best_ep, "val_corr": best_val,
                "dim": D}, str(Path(args.output_npz).with_suffix(".pt")))
    print(f"[rater] saved {len(ts)} bars x {P.shape[1]} candles -> {args.output_npz}")
    print(f"[rater] weights -> {Path(args.output_npz).with_suffix('.pt')}")


if __name__ == "__main__":
    main()
