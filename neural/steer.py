"""Stage 5 — the steering core (Milestone 1).

The Stage 4 conclusion (report §7.11-§7.12, §9) is that a pixel-residual decoder
cannot render the small global view shift a click produces, but a *structured*
output can: predict the ego-motion explicitly and warp the last frame by it. This
module supplies the three pieces of that core, plus the composite model:

  * ``click_features`` — the click-derived inputs (``actionX``,
    ``actionHasClicked``, an explicit ``deltaActionX`` and frames-since-click).
    All are derived from the click history alone, so they satisfy N1.
  * ``warp_translate`` — a differentiable horizontal/vertical translation of an
    image by sub-pixel amounts (``grid_sample``), the structured motion op.
  * ``DirectionReadout`` — a small encoder over the context frames + click
    features that predicts the *per-step horizontal image shift* for the next H
    steps. During training only, it is supervised by the phase-correlation
    shift table (``shift_table.py``) and by an auxiliary direction-sign head;
    at inference it is fed only frames + click signals (N1/N2).
  * ``SteeringModel`` — readout -> warp -> *constrained* residual. The residual is
    computed at quarter resolution and gated by a learned scalar initialised small,
    so it cannot shadow the warp (the Stage 4 P6 failure mode).

Interface (superset of the Stage 4 one, so the harness stays compatible)::

    predict(context, action, features=None, seed=None) -> next_frame
        context  : [B, K, 3, H, W] float in [-1, 1]
        action   : [B, K, 2] float [actionX, actionHasClicked]
        features : [B, K, F] float; derived from `action` when omitted
        next     : [B, 3, H, W] float in [-1, 1]

`seed` is ignored (deterministic core); it exists so the harness can call every
model the same way.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvBlock

FEAT_DIM = 4
SINCE_CLIP = 24          # frames-since-click is clipped here and scaled to [0, 1]


# --------------------------------------------------------------------------- #
# Click-derived features (N1: click signals only)
# --------------------------------------------------------------------------- #

def click_features(actions: torch.Tensor) -> torch.Tensor:
    """Per-step click features from an action history (oldest -> newest).

    ``actions``: [B, T, 2] float ``[actionX, actionHasClicked]``.

    Note the schema semantics: ``actionHasClicked`` is a **latch** ("≥1 click has
    been accepted as of this step"), not a per-click event — so "frames since the
    last click" cannot come from it. It is instead read off ``actionX``: the held
    signal only changes when a *new* click is accepted (the never-clicked default
    mode holds ``actionX == 0``), so a change in ``actionX`` is the click event.

    Returns [B, T, ``FEAT_DIM``] with, at each step,

      0. ``actionX``                       (held click position, [-1, 1])
      1. ``actionHasClicked``              (0/1 latch)
      2. ``deltaActionX`` / 2              (change vs the previous step)
      3. frames-since-last-click / 24      (0 at a click, clipped at 24)
    """
    ax = actions[..., 0]
    ahc = actions[..., 1]
    B, T = ax.shape
    d = torch.zeros_like(ax)
    change = torch.zeros_like(ax, dtype=torch.bool)
    if T > 1:
        d[:, 1:] = ax[:, 1:] - ax[:, :-1]
        change[:, 1:] = (ax[:, 1:] != ax[:, :-1])
    since = ax.new_zeros(B, T)
    cnt = torch.full((B,), float(SINCE_CLIP), device=ax.device, dtype=ax.dtype)
    for t in range(T):
        cnt = torch.where(change[:, t], torch.zeros_like(cnt),
                          torch.clamp(cnt + 1.0, max=float(SINCE_CLIP)))
        since[:, t] = cnt
    return torch.stack([ax, ahc, d / 2.0, since / float(SINCE_CLIP)], dim=-1)


def features_from_context_action(action: torch.Tensor) -> torch.Tensor:
    """Fallback features when only the K context actions are known: same as
    ``click_features`` but with an unknown (clipped) history before the window."""
    return click_features(action)


# --------------------------------------------------------------------------- #
# Structured motion: differentiable translation
# --------------------------------------------------------------------------- #

def warp_translate(img: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor | None = None,
                   mode: str = "bilinear") -> torch.Tensor:
    """Shift the *content* of ``img`` by (dx, dy) pixels.

    ``img`` [B, 3, H, W] in [-1, 1]; ``dx``/``dy`` [B] in pixels, with the
    ``metrics.estimate_shift_x`` convention: positive dx = content moves **right**
    = the camera turned left. ``padding_mode="border"`` so newly exposed edges
    repeat the border instead of going black.
    """
    B, _, H, W = img.shape
    if dy is None:
        dy = torch.zeros_like(dx)
    theta = torch.zeros(B, 2, 3, device=img.device, dtype=torch.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # sampling at (o - shift) moves content by +shift
    theta[:, 0, 2] = -(2.0 * dx.to(torch.float32) / W)
    theta[:, 1, 2] = -(2.0 * dy.to(torch.float32) / H)
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    return F.grid_sample(img.float(), grid, mode=mode, padding_mode="border",
                         align_corners=False)


# --------------------------------------------------------------------------- #
# The direction / ego-motion readout
# --------------------------------------------------------------------------- #

class DirectionReadout(nn.Module):
    """Context frames + click features -> per-step horizontal shift increments.

    A small convolutional encoder (the baseline's encoder shape) over the K
    context frames, global-average-pooled, concatenated with an MLP embedding of
    the K click-feature vectors, then a head that emits H signed per-step shifts
    (plus a direction-sign logit used only to sharpen training).
    """

    def __init__(self, K: int, H: int, feat_dim: int = FEAT_DIM, base: int = 32,
                 cond_dim: int = 128, spatial_click: bool = True):
        super().__init__()
        self.K = K
        self.H = H
        self.spatial_click = spatial_click
        self.enc1 = ConvBlock(K * (5 if spatial_click else 3), base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
        self.feat_mlp = nn.Sequential(
            nn.Linear(K * feat_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(base * 4 + 64, cond_dim), nn.ReLU(),
        )
        self.inc_head = nn.Linear(cond_dim, H)     # per-step shifts
        self.sign_head = nn.Linear(cond_dim, 1)    # auxiliary direction logit

    @staticmethod
    def _click_column(x: torch.Tensor, W: int, sigma_frac: float = 0.06) -> torch.Tensor:
        """Gaussian column map centred on the click's screen position, [B, T, W]."""
        cols = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W)
        centre = (x * 0.5 + 0.5) * (W - 1)                     # actionX -> column
        return torch.exp(-((cols - centre.unsqueeze(-1)) / (sigma_frac * W)) ** 2)

    def forward(self, context: torch.Tensor, features: torch.Tensor):
        B, K = context.shape[0], context.shape[1]
        H, W = context.shape[3], context.shape[4]
        if self.spatial_click:
            # Give the encoder the click *position* in image space. Without it a
            # global-pooled trunk has to multiply a scalar click value by the door
            # layout to find "which door is nearest to the click" — the comparison
            # the direction decision needs. The Gaussian column makes it local.
            col = self._click_column(features[..., 0], W)      # [B,K,W]
            colch = col[:, :, None, :].expand(B, K, H, W)
            dch = (features[..., 2] * 2.0)[:, :, None, None].expand(B, K, H, W)
            x = torch.cat([context, colch[:, :, None], dch[:, :, None]],
                          dim=2).reshape(B, K * 5, H, W)
        else:
            x = context.reshape(B, K * 3, H, W)
        e = self.enc3(self.down2(self.enc2(self.down1(self.enc1(x)))))
        v = e.mean((2, 3))
        f = self.feat_mlp(features.reshape(B, -1))
        h = self.trunk(torch.cat([v, f], dim=1))
        inc = self.inc_head(h)
        return inc, self.sign_head(h).squeeze(1)


# --------------------------------------------------------------------------- #
# The steering model
# --------------------------------------------------------------------------- #

class SteeringModel(nn.Module):
    """Milestone 1: readout -> structured warp -> gated low-resolution residual.

    ``mode`` selects the M1 variants:
      ``"warp_resid"`` — warp + constrained residual (the default);
      ``"warp_only"``  — warp only (the P6 decisive variant, as a control).
    """

    def __init__(self, K: int = 4, H: int = 10, base: int = 32, feat_dim: int = FEAT_DIM,
                 mode: str = "warp_resid", resid_base: int = 24, resid_down: int = 4,
                 align_corners: bool = False, spatial_click: bool = True):
        super().__init__()
        self.K = K
        self.H = H
        self.mode = mode
        self.resid_down = resid_down
        self.align_corners = align_corners
        self.readout = DirectionReadout(K, H, feat_dim=feat_dim, base=base,
                                        spatial_click=spatial_click)
        self.use_resid = mode == "warp_resid"
        if self.use_resid:
            self.res = nn.Sequential(
                nn.Conv2d(3 + 1, resid_base, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(resid_base, resid_base, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(resid_base, 3, 3, padding=1),
            )
            # small gate: the warp carries the motion, the residual only sharpens
            self.res_gate = nn.Parameter(torch.tensor(-2.0))

    # -- inference -------------------------------------------------------- #

    def predict(self, context: torch.Tensor, action: torch.Tensor | None = None,
                features: torch.Tensor | None = None, seed: int | None = None,
                return_shift: bool = False):
        if features is None:
            if action is None:
                raise ValueError("SteeringModel needs either `features` or `action`")
            features = features_from_context_action(action)
        context = context.float()
        features = features.float()
        inc, sign_logit = self.readout(context, features)
        dx = inc[:, 0]                       # next-step horizontal content shift (px)
        last = context[:, -1]
        warped = warp_translate(last, dx)
        if self.use_resid:
            small = F.avg_pool2d(warped, self.resid_down)
            r_in = torch.cat([small, (dx / 8.0).view(-1, 1, 1, 1).expand(
                -1, 1, small.shape[2], small.shape[3])], dim=1)
            r = self.res(r_in)
            r = F.interpolate(r, size=warped.shape[-2:], mode="bilinear",
                              align_corners=self.align_corners)
            out = warped + torch.sigmoid(self.res_gate) * r
        else:
            out = warped
        out = torch.clamp(out, -1.0, 1.0)
        if return_shift:
            return out, {"dx": dx, "inc": inc, "sign_logit": sign_logit,
                         "warped": warped}
        return out

    def forward(self, *a, **k):
        return self.predict(*a, **k)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_steer_checkpoint(path: str, model: SteeringModel, config: dict,
                          val_loss: float, step: int, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "kind": "steering",
        "model_config": {
            "K": model.K, "H": model.H, "mode": model.mode,
            "base": config.get("base", 32), "feat_dim": FEAT_DIM,
            "spatial_click": config.get("spatial_click", True),
            "resid_base": config.get("resid_base", 24),
            "resid_down": config.get("resid_down", 4),
        },
        "config": dict(config),
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_steer_checkpoint(path: str, device: str = "cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SteeringModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
