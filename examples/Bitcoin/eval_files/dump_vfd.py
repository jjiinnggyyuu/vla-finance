"""Dump per-bar Velocity-Field Disagreement (VFD) confidence.

VFD (arXiv 2606.18043): epistemic uncertainty of a flow-matching VLA = how much
two independently-trained ensemble members' velocity fields disagree along the
generation ODE. Higher disagreement = model doesn't know this bar = less trust.

We run ONE shared VLM forward (both members use the same frozen Qwen3-VL) and TWO
action heads (member1=v15 seed42, member2=v18 seed1). During the 4-step Euler
generation we accumulate  kappa_s * ||v1 - v2||^2  at each step (kappa_s = s/(1-s),
weighting cleaner late steps more, exactly as Eq.7), averaged over `draws` initial
noises. Output matches the rater-dump format so it drops into eval_rater_walkforward
(store VFD as `pred_logloss`; higher = worse, so gating loss<=vth keeps low-VFD).

Usage (one GPU per fold):
  CUDA_VISIBLE_DEVICES=4 python examples/Bitcoin/eval_files/dump_vfd.py \
    --config_yaml examples/Bitcoin/train_files/v15_btc_1h/fold1.yaml \
    --ckpt1 playground/Checkpoints/v15_btc_1h/fold1/checkpoints/steps_2000_action_model.pt \
    --ckpt2 playground/Checkpoints/v18_btc_vfd_m2/fold1/checkpoints/steps_2000_action_model.pt \
    --split test --output_npz results/v18_vfd/v15_btc_fold1.npz --draws 8
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
from dump_features import _vlm_forward  # noqa: E402

from starVLA.dataloader.bitcoin_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def _velocity(head, vl_embs, x, t_disc, mask):
    """One DiT forward -> predicted velocity (B, H, action_dim) at state x, step t_disc."""
    B = x.shape[0]
    tt = torch.full((B,), t_disc, device=x.device)
    af = head.action_encoder(x, tt)
    if head.config.add_pos_embed:
        pos = torch.arange(af.shape[1], dtype=torch.long, device=x.device)
        af = af + head.position_embedding(pos).unsqueeze(0)
    ft = head.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
    sa = torch.cat((ft, af), dim=1)
    mo = head.model(hidden_states=sa, encoder_hidden_states=vl_embs,
                    encoder_attention_mask=mask, timestep=tt)
    return head.action_decoder(mo)[:, -head.action_horizon:]


@torch.no_grad()
def _vfd(head1, head2, vl_embs, mask, draws):
    """VFD score per bar (Eq.7): sum_s kappa_s ||v1-v2||^2 over the 4-step gen, avg over draws."""
    B = vl_embs.shape[0]
    H, D = head1.action_horizon, head1.action_dim
    n = head1.num_inference_timesteps
    dt = 1.0 / n
    acc = torch.zeros(B, device=vl_embs.device, dtype=torch.float32)
    for _ in range(draws):
        x = torch.randn(B, H, D, device=vl_embs.device, dtype=vl_embs.dtype)
        for step in range(n):
            s = step / float(n)                          # flow time in [0,1)
            t_disc = int(s * head1.num_timestep_buckets)
            kappa = s / (1.0 - s)                          # Eq.7 weight (0 at s=0)
            v1 = _velocity(head1, vl_embs, x, t_disc, mask)
            v2 = _velocity(head2, vl_embs, x, t_disc, mask)
            acc += kappa * ((v1 - v2) ** 2).float().mean(dim=(1, 2))
            x = x + dt * 0.5 * (v1 + v2)                  # integrate with the ensemble mean
    return acc / draws


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--ckpt1", required=True, help="member 1 action_model (v15)")
    p.add_argument("--ckpt2", required=True, help="member 2 action_model (v18 seed1)")
    p.add_argument("--output_npz", required=True)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    args, clip = p.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clip)))
    cfg = apply_config_compat(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(cfg).to(device).eval()       # shared frozen VLM + head1
    head1 = model.action_model
    head1.load_state_dict(load_state_dict(args.ckpt1), strict=False)
    head2 = get_action_model(cfg).to(device).eval()      # second head, same arch
    head2.load_state_dict(load_state_dict(args.ckpt2), strict=False)
    for pm in list(model.parameters()) + list(head2.parameters()):
        pm.requires_grad_(False)

    ds = get_vla_dataset(cfg.datasets.vla_data, mode=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, shuffle=False)

    H = int(cfg.framework.action_model.action_horizon)
    preds: dict[str, float] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"vfd/{args.split}"):
            lh, mask, _ = _vlm_forward(model, batch)
            with torch.autocast("cuda", dtype=torch.float32):
                vfd = _vfd(head1, head2, lh, mask, args.draws)     # (B,)
            vfd = vfd.cpu().numpy()
            for i, s in enumerate(batch):
                ts = s["base_timestamp"]
                if ts not in preds:
                    preds[ts] = float(vfd[i])

    items = sorted(preds.items(), key=lambda kv: kv[0])
    scal = np.array([v for _, v in items], dtype=np.float32)
    out = {"ts": np.array([t for t, _ in items]),
           "pred_logloss": np.repeat(scal[:, None], H, axis=1)}    # (N,H) broadcast
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **out)
    print(f"[{args.split}] saved {len(items)} bars -> {args.output_npz}  "
          f"(VFD range {scal.min():.4f}..{scal.max():.4f})")


if __name__ == "__main__":
    main()
