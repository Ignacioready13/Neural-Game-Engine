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


class VelocityUNet(nn.Module):
    """Small U-Net predicting the flow velocity, conditioned by channels + FiLM."""

    def __init__(self, in_ch: int, cond_dim: int, out_ch: int = 3, base: int = 32,
                 mlp_width: int = 128):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
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
        e3 = self._film(self.enc3(self.down2(e2)), g3, b3)
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
                 noise_scale: float = 0.08):
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
        self.net = VelocityUNet(N_FIXED_CH, K * feat_dim + H + 1, base=base,
                                mlp_width=cond_dim)
        self.freeze_core = freeze_core
        if freeze_core:
            for p in self.core.parameters():
                p.requires_grad_(False)

    # -- conditioning ------------------------------------------------------ #

    def condition(self, context: torch.Tensor, features: torch.Tensor):
        with torch.no_grad() if self.freeze_core else torch.enable_grad():
            inc, sign_logit = self.core.readout(context, features)
            dx = inc[:, 0]
            warped = warp_translate(context[:, -1], dx)
        cond = torch.cat([features.reshape(features.shape[0], -1), inc,
                          (dx / 8.0).unsqueeze(1)], dim=1)
        return warped, dx, inc, sign_logit, cond

    def velocity(self, x_t, t, warped, context, dx, cond):
        return self.net(flow_input(x_t, warped, context, dx, t), cond)

    # -- inference --------------------------------------------------------- #

    def predict(self, context: torch.Tensor, action: torch.Tensor | None = None,
                features: torch.Tensor | None = None, seed: int | None = None,
                return_shift: bool = False):
        from steer import features_from_context_action
        if features is None:
            if action is None:
                raise ValueError("FlowSteerModel needs `features` or `action`")
            features = features_from_context_action(action)
        context = context.float()
        features = features.float()
        warped, dx, inc, sign_logit, cond = self.condition(context, features)
        B = context.shape[0]
        if seed is None:
            x = self.noise_scale * torch.randn_like(warped)
        else:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
            x = self.noise_scale * torch.randn(warped.shape, generator=g).to(
                warped.device, warped.dtype)
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

def flow_training_loss(model: FlowSteerModel, context, features, target,
                       noise_aug: float = 0.0, rng=None):
    """One rectified-flow training step's loss (residual parameterisation).

    ``target`` is the true next frame [B,3,H,W] in [-1,1]. Returns
    ``(loss, parts)`` where parts carries the readout-supervision hand-off.
    """
    warped, dx, inc, sign_logit, cond = model.condition(context, features)
    r = target - warped
    B = r.shape[0]
    t = torch.rand(B, device=r.device)
    t_ = t.view(B, 1, 1, 1)
    x0 = model.noise_scale * torch.randn_like(r)
    x_t = (1.0 - t_) * x0 + t_ * r
    ctx_in = context
    if noise_aug > 0:
        ctx_in = context + noise_aug * torch.randn_like(context)
    v = model.velocity(x_t, t, warped, ctx_in, dx, cond)
    v_target = r - x0
    # MSE, not L1: the residual being generated is heavy-tailed (near zero on most
    # steps, large at reveals), and an L1 velocity loss drives the sampler to the
    # *median* residual — ~0 — which collapses the generator to the motion-warped
    # last frame (measured: |residual| 0.02 against a real per-step change of 0.2,
    # i.e. no new content and no diversity).
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
            "core_mode": config.get("core_mode", "warp_resid"),
            "freeze_core": config.get("freeze_core", True),
            "noise_scale": config.get("noise_scale", 0.08),
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
