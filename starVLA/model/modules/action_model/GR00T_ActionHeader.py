# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT


from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # ------------------------------------------------------------------
        # DiT architecture selection
        #   action_model_type: "DiT-B" | "DiT-L"
        #     DiT-B → input_embedding_dim=768,  heads=12, head_dim=64
        #     DiT-L → input_embedding_dim=1536, heads=32, head_dim=48
        #   diffusion_model_cfg overrides/extends the base DiT shape.
        #   In particular, diffusion_model_cfg.cross_attention_dim MUST be
        #   set by the framework to match the VLM hidden size BEFORE calling
        #   get_action_model(), e.g.:
        #       cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim
        #           = vlm.model.config.hidden_size
        # ------------------------------------------------------------------
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        # ------------------------------------------------------------------
        # Action horizon (chunk length sent to the DiT)
        #   Single source of truth: `action_horizon` (e.g. 8).
        #   Legacy YAMLs that only provide `future_action_window_size` are
        #   normalised to `action_horizon` upstream by
        #   `share_tools.apply_config_compat`, so this code never touches
        #   the legacy alias.
        # ------------------------------------------------------------------
        self.action_horizon = int(config.action_horizon)

        # ------------------------------------------------------------------
        # Action / state dimensions
        #   action_dim: DoF of the robot action (e.g. 7 for 6-DoF + gripper)
        #   state_dim:  proprioception dimension; set to 0/None to disable
        #               the state_encoder branch entirely.
        # ------------------------------------------------------------------
        self.action_dim = config.action_dim

        # ------------------------------------------------------------------
        # Inference denoising steps
        #   num_inference_timesteps: Euler steps during predict_action().
        #   Typically 4–10; fewer = faster but less accurate.
        # ------------------------------------------------------------------
        self.num_inference_timesteps = config.num_inference_timesteps

        # ------------------------------------------------------------------
        # hidden_size: intermediate MLP width for state_encoder / action_decoder.
        #   Decoupled from input_embedding_dim so you can use a smaller hidden
        #   for the MLP without changing the DiT latent size.
        # ------------------------------------------------------------------
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        # The action vector concatenates N assets' OHLC: action_dim = 4 * N.
        # All assets are co-predicted through the same diffusion process (no
        # separate read-off head). The legacy single-asset ETH read-off decoder
        # has been removed — correlated assets now live inside `action_dim`.
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # ── Read-off aux heads (v14-style) ────────────────────────────────
        #   num_readoff_assets > 0 → BTC is diffusion-generated (action_decoder)
        #   while each aux asset (ETH, XRP, ...) gets its own MLP that regresses
        #   OHLC directly from the DiT output (L1), used only as a regulariser
        #   and never run at inference. Zero in the default action-vector mode.
        self.num_readoff_assets = int(getattr(config, "num_readoff_assets", 0))
        if self.num_readoff_assets > 0:
            self.readoff_decoders = nn.ModuleList([
                MLP(input_dim=self.model.config.output_dim,
                    hidden_dim=self.hidden_size, output_dim=4)
                for _ in range(self.num_readoff_assets)
            ])

        # ── Loss-prediction module (Learning Loss, 1905.03677) ────────────
        #   A small head that predicts the flow-matching velocity loss of the
        #   generated 12-candle chunk -- one scalar per bar, used downstream as
        #   a confidence gate (higher predicted loss = less confident).
        #   • reads the DiT's action-token output features (mean-pooled)
        #   • target = per-sample velocity loss, DETACHED (stop-gradient), so the
        #     module imitates the loss without steering the target/backbone
        #   • trained with a PAIRWISE RANKING loss (not MSE): the paper shows MSE
        #     fails because the loss scale drifts during training; ranking keeps
        #     only the order, which is all our trade gate needs.
        #   loss_token_detach=True (default) fully isolates the price predictor;
        #   set False to co-adapt (gradient flows into the DiT/backbone).
        self.use_loss_token = bool(getattr(config, "loss_token", False))
        if self.use_loss_token:
            self.loss_token_weight = float(getattr(config, "loss_token_weight", 0.1))
            self.loss_token_margin = float(getattr(config, "loss_token_margin", 1.0))
            self.loss_token_detach = bool(getattr(config, "loss_token_detach", True))
            self.loss_head = MLP(
                input_dim=self.model.config.output_dim,
                hidden_dim=self.hidden_size,
                output_dim=1,
            )

        # ── Per-asset loss weights (BTC-priority) ─────────────────────────
        #   `asset_weights` (one per asset, e.g. [0.6, 0.2, 0.2]) is expanded to
        #   a per-channel vector (each asset spans OHLC=4 channels) and applied
        #   to BOTH the flow-matching MSE and the DCT loss so BTC stays the
        #   priority in time- and frequency-domain alike. Normalised to mean 1
        #   so the overall loss scale is invariant to the absolute weights
        #   (only the relative weighting matters). Defaults to equal weights.
        asset_weights = list(getattr(config, "asset_weights", None) or [])
        ohlc = 4
        n_assets = self.action_dim // ohlc
        if asset_weights and len(asset_weights) == n_assets:
            ch = np.repeat(np.asarray(asset_weights, dtype=np.float32), ohlc)
        else:
            ch = np.ones(self.action_dim, dtype=np.float32)
        ch = ch * (self.action_dim / float(ch.sum()))  # normalise → mean 1.0
        self.register_buffer(
            "asset_channel_weights", torch.tensor(ch, dtype=torch.float32).view(1, 1, -1)
        )

        # ------------------------------------------------------------------
        # future_tokens: learnable query tokens prepended before the action
        #   sequence so the DiT has dedicated "planning" slots.
        #   num_target_vision_tokens controls how many such tokens are added.
        # ------------------------------------------------------------------
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Positional embedding over the action sequence
        #   add_pos_embed: whether to add sinusoidal-style learned PE
        #   max_seq_len:   max supported action sequence length
        # ------------------------------------------------------------------
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Flow-matching noise schedule (Beta distribution)
        #   noise_beta_alpha / noise_beta_beta: Beta(α, β) shape params.
        #   noise_s: upper-clip of the sampled value so t ∈ [0, noise_s].
        #   num_timestep_buckets: discretise continuous t into N buckets for
        #     the timestep encoder inside DiT.
        # ------------------------------------------------------------------
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None,
        encoder_attention_mask=None, readoff_targets: torch.Tensor = None,
    ):
        """
        vl_embs:  shape (B, seq_length, feature_dim)
        actions:  shape (B, action_horizon, action_dim)  — diffusion target
                  (BTC OHLC in read-off mode, or all assets in action-vector mode)
        readoff_targets: shape (B, action_horizon, 4*num_readoff) or None — aux
                  assets (ETH, XRP, ...) regressed via read-off heads (v14-style).
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Flow-matching loss over all assets, weighted per channel (BTC-priority).
        aw = self.asset_channel_weights.to(pred_actions.dtype)
        loss = (((pred_actions - velocity) ** 2) * aw).mean()

        # ── DCT (frequency-domain) auxiliary loss (VLANeXt-style) ──────────
        #   디퓨전은 velocity를 예측하므로, 깨끗한 액션(x_start)을 복원해서 적용.
        #   우리 규약: noisy = (1-t)*noise + t*actions, velocity = actions - noise.
        #   따라서 예측 액션 = noise + pred  (pred ≈ velocity 이므로).
        #   자산 가중(aw)을 MSE와 동일하게 곱해 BTC 우선을 주파수 도메인에도 적용.
        dct_weight = float(getattr(self.full_config.framework, "dct_loss_weight", 0.0))
        if dct_weight > 0.0:
            pred_action_recon = noise + pred_actions   # velocity → 액션 복원
            loss = loss + self._freq_loss(pred_action_recon, actions)

        # ── Read-off aux loss (v14-style): L1 regression of ETH/XRP from the
        #    DiT output. readoff_weight = framework.eth_loss_weight (default 0.5).
        if self.num_readoff_assets > 0 and readoff_targets is not None:
            ro_w = float(getattr(self.full_config.framework, "eth_loss_weight", 0.5))
            for i, dec in enumerate(self.readoff_decoders):
                pred_i = dec(model_output)[:, -actions.shape[1]:]      # (B, horizon, 4)
                tgt_i = readoff_targets[:, :, i * 4:(i + 1) * 4]
                loss = loss + ro_w * F.l1_loss(pred_i, tgt_i)

        # ── Loss-prediction ranking loss (Learning Loss, 1905.03677) ───────
        if self.use_loss_token:
            # per-sample flow-matching velocity loss = detached ground-truth
            # target for the module (same channel weighting as the main loss).
            per_sample_vloss = (((pred_actions - velocity) ** 2) * aw).mean(dim=(1, 2)).detach()
            feat = model_output[:, -actions.shape[1]:]         # DiT action-token features
            if self.loss_token_detach:
                feat = feat.detach()                           # protect price predictor/backbone
            lhat = self.loss_head(feat.mean(dim=1)).squeeze(-1)  # (B,) predicted loss
            loss = loss + self.loss_token_weight * self._rank_loss(lhat, per_sample_vloss)
        return loss

    def _rank_loss(self, lhat: torch.Tensor, ltrue: torch.Tensor) -> torch.Tensor:
        """Pairwise margin-ranking loss (Yoo & Kweon Eq. 2). Compares each item i
        against its mirror B-1-i; penalises predicted pairs whose order disagrees
        with the true-loss order. Scale-free — only the ranking is learned."""
        B = lhat.shape[0]
        if B % 2 == 1:                       # need an even number of items to pair
            lhat, ltrue = lhat[:B - 1], ltrue[:B - 1]
            B -= 1
        if B < 2:
            return lhat.new_zeros(())
        dp = (lhat - lhat.flip(0))[:B // 2]         # predicted-loss differences
        dt = (ltrue - ltrue.flip(0))[:B // 2]       # true-loss differences (detached)
        sign = 2.0 * torch.sign(torch.clamp(dt, min=0)) - 1.0   # +1 if l_i>l_j else -1
        return F.relu(self.loss_token_margin - sign * dp).mean()

    def _freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """DCT-II 주파수 도메인 보조 손실 (VLANeXt 구현, OFT와 동일).

        시간축으로 DCT-II 변환 후 저주파/고주파에 다른 가중치를 줘 MSE/MAE/Cosine.
        dct_loss_weight=0이면 0 텐서 반환. 반환값은 weight가 이미 곱해진 상태.
        """
        fw = self.full_config.framework
        weight = float(getattr(fw, "dct_loss_weight", 0.0))
        if weight == 0.0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        pred = pred.float()
        target = target.float()
        B, T, D = pred.shape

        if (not hasattr(self, "_dct_matrix")
                or self._dct_matrix.shape[0] != T
                or self._dct_matrix.device != pred.device):
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            dct_m[0, :]  *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            self._dct_matrix = dct_m

        low_w    = float(getattr(fw, "dct_low_freq_weight",  1.0))
        high_w   = float(getattr(fw, "dct_high_freq_weight", 3.0))
        split    = float(getattr(fw, "dct_freq_split",       0.5))
        sim_type = str(getattr(fw,   "dct_similarity_type",  "mse"))

        split_idx = max(1, int(T * split))
        freq_weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        freq_weights[:split_idx] = low_w
        freq_weights[split_idx:] = high_w
        freq_weights = freq_weights.view(1, T, 1)

        # Combine frequency weights (1,T,1) with per-asset channel weights (1,1,D)
        # so DCT honours both the low/high-freq split and the BTC-priority weighting.
        aw = self.asset_channel_weights.to(pred.dtype)   # (1,1,D)
        weights = freq_weights * aw                       # broadcast → (1,T,D)

        pred_dct   = torch.matmul(pred.permute(0, 2, 1),   self._dct_matrix.t()).permute(0, 2, 1)
        target_dct = torch.matmul(target.permute(0, 2, 1), self._dct_matrix.t()).permute(0, 2, 1)

        if sim_type == "mse":
            loss = ((pred_dct - target_dct) ** 2 * weights).mean()
        elif sim_type == "mae":
            loss = ((pred_dct - target_dct).abs() * weights).mean()
        elif sim_type == "cosine":
            pred_norm   = F.normalize(pred_dct,   dim=-1)
            target_norm = F.normalize(target_dct, dim=-1)
            cos_dist = 1.0 - (pred_norm * target_norm).sum(dim=-1, keepdim=True)
            loss = (cos_dist * freq_weights).mean()
        else:
            raise ValueError(f"Unknown dct_similarity_type: {sim_type!r}")

        return weight * loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @torch.no_grad()
    def predict_loss(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        draws: int = 20,
    ) -> torch.Tensor:
        """Per-bar predicted velocity-loss (confidence signal) from the loss head.

        Replicates the training forward pass `draws` times (fresh noise / time each)
        and averages the loss-head output -> shape (B,). Higher = the model expects
        a larger flow-matching loss = less confident. Mirrors the exp4 rater dump
        protocol so it drops into the same walk-forward eval pipeline.
        """
        assert self.use_loss_token, "predict_loss requires framework.action_model.loss_token=true"
        device = vl_embs.device
        B = actions.shape[0]
        state_features = self.state_encoder(state) if state is not None else None
        acc = torch.zeros(B, device=device, dtype=torch.float32)
        for _ in range(draws):
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            t = self.sample_time(B, device=actions.device, dtype=actions.dtype)[:, None, None]
            noisy = (1 - t) * noise + t * actions
            t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy, t_disc)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_disc,
            )
            feat = model_output[:, -actions.shape[1]:].mean(dim=1)
            acc += self.loss_head(feat).squeeze(-1).float()
        return acc / draws

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
