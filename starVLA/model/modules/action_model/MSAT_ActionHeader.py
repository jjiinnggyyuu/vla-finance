"""MSAT flow-matching action head (RLDX-1's Multi-Stream Action Transformer).

Drop-in alternative to the GR00T DiT head: same flow-matching interface
(forward(vl_embs, actions) -> loss; predict_action -> 4-step Euler), same
per-asset weighting / DCT / read-off support, but the single-stream cross-
attention DiT is replaced by MSAT's multi-stream joint self-attention
(VL + state/action streams; physics stream disabled for the finance domain).

Config (framework.action_model):
    action_model_type: MSAT-B        # base head/dim preset
    diffusion_model_cfg:
      cross_attention_dim: <vl_dim>  # VLM hidden, set at runtime by the framework
      depth_multi_stream: 4          # double-stream (VL+SA joint) blocks
      depth_single_stream: 8         # single-stream (Flux-style) blocks
      output_dim: 1024
      dropout: 0.2
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import ActionEncoder
from starVLA.model.modules.action_model.msat_head import MSAT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


# Base MSAT presets (inner_dim = num_attention_heads * attention_head_dim = sa_dim).
MSATConfig = {
    "MSAT-B": {"num_attention_heads": 8, "attention_head_dim": 64},   # inner = 512
    "MSAT-L": {"num_attention_heads": 16, "attention_head_dim": 64},  # inner = 1024
}


class FlowmatchingMSATHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        head_cfg = MSATConfig[config.action_model_type]
        heads = head_cfg["num_attention_heads"]
        head_dim = head_cfg["attention_head_dim"]
        self.input_embedding_dim = heads * head_dim   # inner_dim = sa_dim

        dcfg = dict(config.diffusion_model_cfg)
        output_dim = int(dcfg.get("output_dim", 1024))
        vl_dim = int(dcfg["cross_attention_dim"])      # VLM hidden (set by framework)

        # MSAT with VL + SA streams only (physics disabled). RoPE positional
        # embeddings are applied internally, so no external position embedding.
        # RoPE max length must cover VL tokens (image+text, ~1.5k) + SA tokens.
        # MSAT's default (512) is far too small for a VLM sequence, so bump it.
        max_seq_len = int(dcfg.get("action_model_max_seq_len", 4096))
        self.model = MSAT(
            num_attention_heads=heads,
            attention_head_dim=head_dim,
            output_dim=output_dim,
            depth_multi_stream=int(dcfg.get("depth_multi_stream", 4)),
            depth_single_stream=int(dcfg.get("depth_single_stream", 8)),
            dropout=float(dcfg.get("dropout", 0.0)),
            sa_dim=self.input_embedding_dim,
            vl_dim=vl_dim,
            action_model_max_seq_len=max_seq_len,
            use_swiglu=True,
            positional_embeddings="rope_vl_sa",
            use_physics=False,
        )
        self.msat_output_dim = output_dim

        # ── VL compression (RLDX "cognition tokens") ──────────────────────
        #   MSAT does joint self-attention over VL+SA, which is O((N_vl)^2) and
        #   OOMs on a full ~1k-token VLM sequence. RLDX compresses the VL stream
        #   to a small fixed set of learnable "cognition" query tokens via one
        #   cross-attention layer (Perceiver/Q-Former style). num_cog_tokens=0
        #   disables it (full VL passed through).
        self.num_cog_tokens = int(dcfg.get("num_cog_tokens", 64))
        if self.num_cog_tokens > 0:
            self.cog_queries = nn.Parameter(0.02 * torch.randn(1, self.num_cog_tokens, vl_dim))
            self.cog_attn = nn.MultiheadAttention(vl_dim, num_heads=8, batch_first=True)
            self.cog_norm = nn.LayerNorm(vl_dim)

        self.action_horizon = int(config.action_horizon)
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(config.state_dim, self.hidden_size, self.input_embedding_dim)
            if config.state_dim else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim, hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(output_dim, self.hidden_size, self.action_dim)

        # Read-off aux heads (v14-style), same as the DiT head.
        self.num_readoff_assets = int(getattr(config, "num_readoff_assets", 0))
        if self.num_readoff_assets > 0:
            self.readoff_decoders = nn.ModuleList([
                MLP(output_dim, self.hidden_size, 4) for _ in range(self.num_readoff_assets)
            ])

        # Per-asset (per-channel) loss weights, normalised to mean 1.
        asset_weights = list(getattr(config, "asset_weights", None) or [])
        ohlc = 4
        n_assets = self.action_dim // ohlc
        if asset_weights and len(asset_weights) == n_assets:
            ch = np.repeat(np.asarray(asset_weights, dtype=np.float32), ohlc)
        else:
            ch = np.ones(self.action_dim, dtype=np.float32)
        ch = ch * (self.action_dim / float(ch.sum()))
        self.register_buffer("asset_channel_weights", torch.tensor(ch).view(1, 1, -1))

        # Learnable planning/query tokens prepended to the action sequence.
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return self.config.noise_s * (1 - sample)

    def _compress_vl(self, vl_embs, mask):
        """Compress a long VL sequence to num_cog_tokens via cross-attention.
        Returns (compressed_vl, None) — the compressed stream needs no mask."""
        if self.num_cog_tokens <= 0:
            return vl_embs, mask
        B = vl_embs.shape[0]
        q = self.cog_queries.expand(B, -1, -1).to(vl_embs.dtype)
        # key_padding_mask: True = ignore. VLM mask is 1=visible, so invert.
        kpm = (~mask.bool()) if mask is not None else None
        out, _ = self.cog_attn(q, vl_embs, vl_embs, key_padding_mask=kpm, need_weights=False)
        return self.cog_norm(out), None      # (B, num_cog, vl_dim), no mask

    def _msat(self, sa_embs, vl_embs, t_disc, mask):
        vl_c, mask_c = self._compress_vl(vl_embs, mask)
        out = self.model(
            hidden_states=sa_embs, encoder_hidden_states=vl_c,
            timestep=t_disc, encoder_attention_mask=mask_c,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None, readoff_targets=None):
        device = vl_embs.device
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
        noisy = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy, t_disc)          # (B, H, inner)
        state_features = self.state_encoder(state) if state is not None else None
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self._msat(sa_embs, vl_embs, t_disc, encoder_attention_mask)
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        aw = self.asset_channel_weights.to(pred_actions.dtype)
        loss = (((pred_actions - velocity) ** 2) * aw).mean()

        dct_weight = float(getattr(self.full_config.framework, "dct_loss_weight", 0.0))
        if dct_weight > 0.0:
            loss = loss + self._freq_loss(noise + pred_actions, actions)

        if self.num_readoff_assets > 0 and readoff_targets is not None:
            ro_w = float(getattr(self.full_config.framework, "eth_loss_weight", 0.5))
            for i, dec in enumerate(self.readoff_decoders):
                pred_i = dec(model_output)[:, -actions.shape[1]:]
                loss = loss + ro_w * F.l1_loss(pred_i, readoff_targets[:, :, i * 4:(i + 1) * 4])
        return loss

    def _freq_loss(self, pred, target):
        fw = self.full_config.framework
        weight = float(getattr(fw, "dct_loss_weight", 0.0))
        if weight == 0.0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        pred = pred.float(); target = target.float()
        B, T, D = pred.shape
        if (not hasattr(self, "_dct_matrix") or self._dct_matrix.shape[0] != T
                or self._dct_matrix.device != pred.device):
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            dct_m[0, :] *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            self._dct_matrix = dct_m
        low_w = float(getattr(fw, "dct_low_freq_weight", 1.0))
        high_w = float(getattr(fw, "dct_high_freq_weight", 3.0))
        split = float(getattr(fw, "dct_freq_split", 0.5))
        sim = str(getattr(fw, "dct_similarity_type", "mse"))
        split_idx = max(1, int(T * split))
        fwt = torch.ones(T, device=pred.device, dtype=pred.dtype)
        fwt[:split_idx] = low_w; fwt[split_idx:] = high_w
        fwt = fwt.view(1, T, 1)
        aw = self.asset_channel_weights.to(pred.dtype)
        w = fwt * aw
        pred_dct = torch.matmul(pred.permute(0, 2, 1), self._dct_matrix.t()).permute(0, 2, 1)
        tgt_dct = torch.matmul(target.permute(0, 2, 1), self._dct_matrix.t()).permute(0, 2, 1)
        if sim == "mse":
            loss = ((pred_dct - tgt_dct) ** 2 * w).mean()
        elif sim == "mae":
            loss = ((pred_dct - tgt_dct).abs() * w).mean()
        else:
            pn = F.normalize(pred_dct, dim=-1); tn = F.normalize(tgt_dct, dim=-1)
            loss = ((1.0 - (pn * tn).sum(-1, keepdim=True)) * fwt).mean()
        return weight * loss

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None):
        B = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(B, self.action_horizon, self.action_dim, dtype=vl_embs.dtype, device=device)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self.state_encoder(state) if state is not None else None
        for i in range(num_steps):
            t_cont = i / float(num_steps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((B,), t_disc, device=device)
            action_features = self.action_encoder(actions, t_tensor)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            model_output = self._msat(sa_embs, vl_embs, t_tensor, encoder_attention_mask)
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    return FlowmatchingMSATHead(full_config=config)
