"""Stage 5 — the stochastic generator (Milestone 2).

A conditional **rectified-flow** (flow-matching) generator for the next frame,
conditioned on the steering core's motion output. The key design choice is that
the flow is trained on the **residual relative to the motion-warped last frame**:

    warped = warp(last_context_frame, predicted_shift)     # steer: the readout
    target_residual r = true_next_frame - warped           # content only
    x_t = (1-t) * noise + t * r        (rectified-flow interpolation)
    r_hat = flow_net(x_t, t, condition)                    # velocity integral
    next  = warped + r_hat

Why this and not "denoise the whole frame": the Stage 4 probe (§7.12) shows that a
generator given an explicit direction signal still renders *no* shift when the
motion has to come out of a free pixel residual, while it follows the shift
decisively when the motion is an explicit warp. Keeping the warp **outside** the
learned residual makes the steering path structural: the residual, being aligned
with the target, only has to invent the newly revealed content (and undo the warp
would require the model to deliberately re-align, which its aligned target does not
reward).

The generator is genuinely stochastic: sampling starts from a random seed, so the
same context and clicks give different (plausible) new content — most where
something new is revealed, least during steady walking (where the residual target
is nearly zero).

Interface (same as the steering core, plus ``seed``)::

    predict(context, action=None, features=None, seed=None, return_shift=False)

``seed=None`` draws a fresh random seed each call (true stochastic sampling).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvBlock
from steer import FEAT_DIM, SteeringModel, warp_translate

# x_t(3) + warped(3) + dx(1) + t(1). The *un-warped* context frames are
# deliberately NOT fed to the denoiser: given the frame two steps back it can
# reconstruct the pre-warp content and cancel the steering (measured: the
# closed-loop image shift collapsed to ~2% of the real turn). The warped frame is
# the only "previous frame" it sees, so the motion has to be carried through it.
N_FIXED_CH = 8


class SelfAttnBlock(nn.Module):
    """Single-head-less spatial self-attention at a low resolution.

    A conv stack with a measured ~27 px receptive field cannot relate the two sides
    of a corridor or the vanishing point to the near wall, which is what "is this
    frame structurally consistent?" needs.  At the bottleneck (12x16 at 96x72) full
    attention over all 192 tokens is cheap.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.norm(x).reshape(B, C, H * W).transpose(1, 2)
        a, _ = self.attn(t, t, t, need_weights=False)
        a = a.transpose(1, 2).reshape(B, C, H, W)
        return x + torch.tanh(self.scale) * a


class VelocityUNet(nn.Module):
    """U-Net predicting the flow velocity, conditioned by channels + FiLM."""

    def __init__(self, in_ch: int, cond_dim: int, out_ch: int = 3, base: int = 32,
                 mlp_width: int = 128, attn: bool = False, attn_heads: int = 4):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
        # attention runs one level coarser than the bottleneck: at 96x72 that is a
        # 9x12 grid, which is still fully global and costs 4x less memory than the
        # bottleneck itself (the 18x24 attention matrix at batch 64 needs ~190 MB).
        self.attn = SelfAttnBlock(base * 4, attn_heads) if attn else None
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False),
        )
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1, bias=False),
        )
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)
        self._film_chs = [base * 4, base * 2, base]
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_width), nn.ReLU(),
            nn.Linear(mlp_width, 2 * sum(self._film_chs)),
        )

    def _film(self, x, g, b):
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, cond):
        c = self.film_mlp(cond)
        gamma, beta = torch.chunk(c, 2, dim=1)
        c3, c2, c1 = self._film_chs
        g3, b3 = gamma[:, :c3], beta[:, :c3]
        g2, b2 = gamma[:, c3:c3 + c2], beta[:, c3:c3 + c2]
        g1, b1 = gamma[:, c3 + c2:], beta[:, c3 + c2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        if self.attn is not None:
            h, w = e3.shape[-2:]
            e3 = F.interpolate(self.attn(F.avg_pool2d(e3, 2)), size=(h, w),
                               mode="nearest")
        e3 = self._film(e3, g3, b3)
        d2 = self._film(self.dec2(torch.cat([self.up2(e3), e2], dim=1)), g2, b2)
        d1 = self._film(self.dec1(torch.cat([self.up1(d2), e1], dim=1)), g1, b1)
        return self.out(d1)


def flow_input(x_t, warped, context, dx, t):
    """Assemble the denoiser's input channels (see ``N_FIXED_CH``)."""
    B, _, H, W = x_t.shape
    tch = t.view(B, 1, 1, 1).expand(B, 1, H, W)
    dxch = (dx / 8.0).view(B, 1, 1, 1).expand(B, 1, H, W)
    return torch.cat([x_t, warped, dxch, tch], dim=1)


class FlowSteerModel(nn.Module):
    """Steering core (frozen readout + warp) + stochastic residual flow."""

    def __init__(self, K: int = 4, H: int = 10, base: int = 32, steps: int = 4,
                 cond_dim: int = 128, core_mode: str = "warp_resid",
                 freeze_core: bool = True, feat_dim: int = FEAT_DIM,
                 noise_scale: float = 0.08, attn: bool = False,
                 net_base: int | None = None,
                 noise_level_max: float = 0.0, level_buckets: int = 0,
                 ctx_dropout: float = 0.0, level_dim: int = 32):
        super().__init__()
        self.K = K
        self.H = H
        self.steps = steps
        # The residual being generated is *small* (a fraction of a frame's change),
        # so the flow must start from noise of that scale: with unit noise the
        # few-step sampler's velocity error leaks a ~5% high-frequency residue into
        # every frame, which compounds over a rollout (measured: 3.7x real sharpness
        # after 60 steps and a collapsed phase-correlation shift).
        self.noise_scale = float(noise_scale)
        self.core = SteeringModel(K=K, H=H, base=base, mode=core_mode)
        # the conditioning vector is [click features, the readout's H increments, dx]
        self.attn = bool(attn)
        # ``base`` sizes the steering core (fixed by its checkpoint); ``net_base``
        # sizes the denoiser, so the generator can grow without touching the core.
        self.net_base = int(net_base or base)
        # Stage 5c, GameNGen Sec 3.2.1 (additive; both default to off, so every
        # existing checkpoint loads and behaves bit-identically):
        #   * noise_level_max > 0 gives the model a learned embedding of the noise
        #     bucket its (corrupted) frame conditioning sits in, so training can
        #     corrupt the context at a random level and inference can ask for level 0;
        #   * ctx_dropout > 0 adds a learned null token for the frame conditioning,
        #     so classifier-free guidance can be applied at inference.
        self.noise_level_max = float(noise_level_max or 0.0)
        self.level_buckets = int(level_buckets or 0)
        self.ctx_dropout = float(ctx_dropout or 0.0)
        self.level_dim = int(level_dim)
        self.level_embed = (nn.Embedding(self.level_buckets + 1, self.level_dim)
                            if (self.noise_level_max > 0 and self.level_buckets > 0)
                            else None)
        self.null_embed = (nn.Parameter(torch.zeros(self.level_dim))
                           if self.ctx_dropout > 0 else None)
        extra = (self.level_dim if self.level_embed is not None else 0) \
            + (self.level_dim if self.null_embed is not None else 0)
        self.cond_dim = K * feat_dim + H + 1 + extra
        self.net = VelocityUNet(N_FIXED_CH, self.cond_dim, base=self.net_base,
                                mlp_width=cond_dim, attn=self.attn)
        self.freeze_core = freeze_core
        if freeze_core:
            for p in self.core.parameters():
                p.requires_grad_(False)

    # -- conditioning ------------------------------------------------------ #

    def condition(self, context: torch.Tensor, features: torch.Tensor,
                  level: torch.Tensor | None = None,
                  dropped: torch.Tensor | None = None):
        """``level``/``dropped`` default to the clean, not-dropped case (level 0)."""
        with torch.no_grad() if self.freeze_core else torch.enable_grad():
            inc, sign_logit = self.core.readout(context, features)
            dx = inc[:, 0]
            warped = warp_translate(context[:, -1], dx)
        cond = torch.cat([features.reshape(features.shape[0], -1), inc,
                          (dx / 8.0).unsqueeze(1)], dim=1)
        B = cond.shape[0]
        if self.level_embed is not None:
            lv = (torch.zeros(B, dtype=torch.long, device=cond.device)
                  if level is None else level.to(cond.device).long())
            cond = torch.cat([cond, self.level_embed(lv)], dim=1)
        if self.null_embed is not None:
            d = (torch.zeros(B, device=cond.device, dtype=cond.dtype)
                 if dropped is None else dropped.to(cond.device, cond.dtype))
            cond = torch.cat([cond, d[:, None] * self.null_embed[None, :]], dim=1)
        return warped, dx, inc, sign_logit, cond

    def velocity(self, x_t, t, warped, context, dx, cond):
        return self.net(flow_input(x_t, warped, context, dx, t), cond)

    # -- inference --------------------------------------------------------- #

    def predict(self, context: torch.Tensor, action: torch.Tensor | None = None,
                features: torch.Tensor | None = None, seed: int | None = None,
                return_shift: bool = False, noise: torch.Tensor | None = None,
                level: int | None = None, guidance: float = 1.0):
        from steer import features_from_context_action
        if features is None:
            if action is None:
                raise ValueError("FlowSteerModel needs `features` or `action`")
            features = features_from_context_action(action)
        context = context.float()
        features = features.float()
        B = context.shape[0]
        # Inference always conditions on the *clean* frame conditioning: the noise
        # level is 0 (the level embedding's clean bucket) unless a caller says
        # otherwise. Old checkpoints have no level embedding, so this is a no-op for
        # them and they behave exactly as before.
        lv = None
        if self.level_embed is not None:
            lv = torch.full((B,), int(level or 0), dtype=torch.long, device=context.device)
        warped, dx, inc, sign_logit, cond = self.condition(context, features, level=lv)
        if noise is not None:
            # Explicit unit-variance noise in the warped frame's shape. Additive:
            # allows a caller to supply temporally correlated noise; when omitted the
            # behaviour is exactly as before (per-call fresh, or seed-deterministic).
            x = self.noise_scale * noise.to(warped.device, warped.dtype)
        elif seed is None:
            x = self.noise_scale * torch.randn_like(warped)
        else:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            x = self.noise_scale * torch.randn(warped.shape, generator=g).to(
                warped.device, warped.dtype)
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((B,), i * dt, device=warped.device)
            v = self.velocity(x, t, warped, context, dx, cond)
            if self.null_embed is not None and guidance != 1.0:
                # classifier-free guidance on the frame conditioning only
                # (GameNGen Sec 3.3): drop the warped frame and extrapolate.
                zeros = torch.zeros_like(warped)
                _, _, _, _, cond_u = self.condition(context, features, level=lv,
                                                    dropped=torch.ones(B, device=context.device))
                v_u = self.velocity(x, t, zeros, context, dx, cond_u)
                v = v_u + guidance * (v - v_u)
            x = x + dt * v
        out = torch.clamp(warped + x, -1.0, 1.0)
        if return_shift:
            return out, {"dx": dx, "inc": inc, "sign_logit": sign_logit,
                         "warped": warped, "residual": x}
        return out
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((B,), i * dt, device=warped.device)
            v = self.velocity(x, t, warped, context, dx, cond)
            x = x + dt * v
        out = torch.clamp(warped + x, -1.0, 1.0)
        if return_shift:
            return out, {"dx": dx, "inc": inc, "sign_logit": sign_logit,
                         "warped": warped, "residual": x}
        return out

    def forward(self, *a, **k):
        return self.predict(*a, **k)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def revealed_mask(dx: torch.Tensor, H: int, W: int, edge_weight: float = 0.0,
                  min_cols: int = 3):
    """Spatial weight that is high where the warp had to invent content.

    ``warp_translate`` uses ``padding_mode="border"``, so a content shift of ``dx`` px
    repeats the border into the revealed strip: ``ceil(|dx|)`` columns at the leading
    edge (the left edge for a rightward shift).  Those columns are exactly where the
    real frame contains something the warp cannot have — the next doorway coming into
    view.  Weighting the flow loss there (and, to a smaller extent, in a few columns
    beyond) directs capacity at the new content instead of at re-fitting the part of
    the frame the warp already reproduced.
    """
    B = dx.shape[0]
    w = torch.ones(B, 1, H, W, device=dx.device, dtype=torch.float32)
    if edge_weight <= 0:
        return w
    for i in range(B):
        n = int(max(min_cols, abs(float(dx[i])) + 1.5))
        n = min(n, W // 2)
        if float(dx[i]) >= 0:
            w[i, :, :, :n] = 1.0 + edge_weight
        else:
            w[i, :, :, W - n:] = 1.0 + edge_weight
    # normalise so the overall loss scale does not change with the weight
    return w / w.mean()


def flow_training_loss(model: FlowSteerModel, context, features, target,
                       noise_aug: float = 0.0, rng=None, edge_weight: float = 0.0,
                       noise_level_max: float = 0.0, level_buckets: int = 0,
                       level_clean_frac: float = 0.25, ctx_dropout: float = 0.0,
                       corrupt_context: bool = False):
    """One rectified-flow training step's loss (residual parameterisation).

    ``target`` is the true next frame [B,3,H,W] in [-1,1]. Returns
    ``(loss, parts)`` where parts carries the readout-supervision hand-off.
    ``edge_weight`` up-weights the newly revealed strip (see :func:`revealed_mask`).

    Stage 5c adds GameNGen's noise-augmented frame conditioning (Sec 3.2.1): the
    **frame-valued conditioning the denoiser sees** — the warped last frame, built from
    the last context frame — is corrupted with Gaussian noise whose level is drawn per
    sample from ``level_buckets`` buckets up to ``noise_level_max`` (with
    ``level_clean_frac`` of the batch left exactly clean), and the bucket reaches the
    model through the learned level embedding. The **target stays clean** and the
    residual is defined relative to the *corrupted* warp the model is actually given,
    so the model must learn to remove the noise rather than copy it. Inference asks for
    level 0, which is exactly the clean case.

    Divergence from GameNGen worth naming: their *action* conditioning is a separate
    clean input, while ours is a frozen readout applied to the context frames, so
    corrupting the context would also corrupt the motion estimate. We therefore keep
    the motion path (``dx``/``inc``) clean and corrupt only the frame channel — the
    same structure (clean action, corrupted frame context) — unless
    ``corrupt_context=True`` is passed. With ``noise_level_max=0`` (the default) this
    function is exactly the old behaviour, bit for bit.
    """
    B = context.shape[0]
    lv = dropped = None
    noisy_last = None
    if noise_level_max > 0 and level_buckets > 0:
        lv = torch.randint(0, level_buckets + 1, (B,), device=context.device)
        level = noise_level_max * lv.to(context.dtype) / level_buckets
        clean = torch.rand(B, device=context.device) < float(level_clean_frac)
        level = torch.where(clean, torch.zeros_like(level), level)
        lv = torch.where(clean, torch.zeros_like(lv), lv)
        if corrupt_context:
            context = context + level.view(B, 1, 1, 1, 1) * torch.randn_like(context)
        else:
            noisy_last = (context[:, -1]
                          + level.view(B, 1, 1, 1) * torch.randn_like(context[:, -1]))
    if ctx_dropout > 0:
        dropped = (torch.rand(B, device=context.device) < float(ctx_dropout)
                   ).to(context.dtype)
    warped, dx, inc, sign_logit, cond = model.condition(context, features, level=lv,
                                                        dropped=dropped)
    if noisy_last is not None:
        # the denoiser's frame channel is built from the NOISY last frame
        warped = warp_translate(noisy_last, dx)
    r = target - warped
    t = torch.rand(B, device=r.device)
    t_ = t.view(B, 1, 1, 1)
    x0 = model.noise_scale * torch.randn_like(r)
    x_t = (1.0 - t_) * x0 + t_ * r
    ctx_den = context
    if noise_aug > 0:
        ctx_den = context + noise_aug * torch.randn_like(context)
    warped_in = warped
    if dropped is not None:
        warped_in = warped * (1.0 - dropped).view(B, 1, 1, 1)
    v = model.velocity(x_t, t, warped_in, ctx_den, dx, cond)
    v_target = r - x0
    # MSE, not L1: the residual being generated is heavy-tailed (near zero on most
    # steps, large at reveals), and an L1 velocity loss drives the sampler to the
    # *median* residual — ~0 — which collapses the generator to the motion-warped
    # last frame (measured: |residual| 0.02 against a real per-step change of 0.2,
    # i.e. no new content and no diversity).
    if edge_weight > 0:
        m = revealed_mask(dx, r.shape[2], r.shape[3], edge_weight)
        l_flow = ((v - v_target).pow(2) * m).mean()
    else:
        l_flow = (v - v_target).pow(2).mean()
    return l_flow, {"l_flow": float(l_flow.detach())}


def save_flow_checkpoint(path: str, model: FlowSteerModel, config: dict,
                         val_loss: float, step: int, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "kind": "flow",
        "model_config": {
            "K": model.K, "H": model.H, "steps": model.steps,
            "base": config.get("base", 32), "cond_dim": config.get("cond_dim", 128),
            "attn": bool(config.get("attn", False)),
            "net_base": int(config.get("net_base") or config.get("base", 32)),
            "core_mode": config.get("core_mode", "warp_resid"),
            "freeze_core": config.get("freeze_core", True),
            "noise_scale": config.get("noise_scale", 0.08),
            "noise_level_max": float(config.get("noise_level_max") or 0.0),
            "level_buckets": int(config.get("level_buckets") or 0),
            "ctx_dropout": float(config.get("ctx_dropout") or 0.0),
            "level_dim": int(config.get("level_dim", 32)),
        },
        "config": dict(config),
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_flow_checkpoint(path: str, device: str = "cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FlowSteerModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
