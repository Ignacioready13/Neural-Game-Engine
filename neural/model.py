"""The baseline action-conditioned next-frame model.

A small, deterministic, residual U-Net: K context frames plus their K actions in,
one predicted next frame out. It predicts the *change* from the last context frame
(a residual) rather than the whole frame — a natural choice when consecutive
frames are nearly identical — and adds it back in float32 so mixed precision
never loses the base frame.

The only interface the evaluation harness (and any future model) needs is::

    predict(context, action) -> next_frame
        context : [B, K, 3, H, W] float in [-1, 1]
        action  : [B, K, 2] float  [actionX, actionHasClicked]
        next    : [B, 3, H, W] float in [-1, 1]

Actions are injected the simplest way possible: each context frame's 2-dim action
is broadcast as two extra constant image channels, concatenated to the frame's RGB
channels, so the network sees ``K * (3 + 2)`` input channels. This is "the simplest
option first" from the brief; if the action ablation shows the model ignores the
action, the fallback conditioning scheme goes here.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """conv(3x3) + group-norm + ReLU, with a residual path when channels match."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = _gcd(cin, cout)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups if groups <= 32 else 32, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups if groups <= 32 else 32, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class UNetBaseline(nn.Module):
    """Two-level encoder-decoder with skip connections and a wide receptive field.

    Spatial path: HxW -> H/2xW/2 -> H/4xW/4 (bottleneck) -> back up. Widths 32 /
    64 / 128. Decoder upsamples with nearest-neighbour + conv (no transpose-conv
    checkerboarding) and concatenates the matching encoder skip.
    """

    def __init__(self, in_ch: int, out_ch: int = 3, base: int = 32):
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
        self.dec2 = ConvBlock(base * 4, base * 2)  # up2 (base*2) + skip enc2 (base*2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1, bias=False),
        )
        self.dec1 = ConvBlock(base * 2, base)  # up1 (base) + skip enc1 (base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class FilmUNet(nn.Module):
    """UNet conditioned on the action via FiLM at several scales.

    This is the "conditioning at more than one scale" alternative to the
    broadcast-channel injection, used when that simplest scheme lets the model
    ignore the action. The action [B, K, 2] is flattened and mapped by a small MLP
    to per-level (gamma, beta) channel modulations, applied at the bottleneck and
    both decoder levels. The image path is otherwise the same two-level U-Net.
    """

    def __init__(self, in_ch: int, K: int, out_ch: int = 3, base: int = 32,
                 cond_dim: int = 64):
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
        self.dec2 = ConvBlock(base * 4, base * 2)  # up2 + skip enc2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base * 2, base, 3, padding=1, bias=False),
        )
        self.dec1 = ConvBlock(base * 2, base)  # up1 + skip enc1
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

        self._film_chs = [base * 4, base * 2, base]  # bottleneck, dec2, dec1
        self.film_mlp = nn.Sequential(
            nn.Linear(K * 2, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, 2 * sum(self._film_chs)),
        )

    def _film(self, x, gamma, beta):
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, x, action):
        cond = self.film_mlp(action.reshape(action.shape[0], -1))
        gamma, beta = torch.chunk(cond, 2, dim=1)
        c3, c2, c1 = self._film_chs
        g3, b3 = gamma[:, :c3], beta[:, :c3]
        g2, b2 = gamma[:, c3: c3 + c2], beta[:, c3: c3 + c2]
        g1, b1 = gamma[:, c3 + c2:], beta[:, c3 + c2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e3 = self._film(e3, g3, b3)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d2 = self._film(d2, g2, b2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = self._film(d1, g1, b1)
        return self.out(d1)


# --------------------------------------------------------------------------- #
# Action-image stacking (the simplest action injection)
# --------------------------------------------------------------------------- #

def stack_context_action(context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Concatenate each context frame with its action as extra constant channels.

    ``context`` [B, K, 3, H, W], ``action`` [B, K, 2] -> [B, K * (3 + 2), H, W].
    """
    B, K, _, H, W = context.shape
    ac = action[:, :, :, None, None].expand(B, K, 2, H, W)
    x = torch.cat([context, ac], dim=2)  # [B, K, 5, H, W]
    return x.reshape(B, K * 5, H, W)


# --------------------------------------------------------------------------- #
# Model wrappers (shared interface)
# --------------------------------------------------------------------------- #

class NextFrameModel(nn.Module):
    """Action-conditioned residual next-frame predictor with the common interface."""

    def __init__(self, net: nn.Module, K: int, use_film: bool = False):
        super().__init__()
        self.net = net
        self.K = K
        self.use_film = use_film

    def predict(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.use_film:
            B, K, C, H, W = context.shape
            x = context.reshape(B, K * C, H, W)       # [B, K*3, H, W] RGB only
            resid = self.net(x, action)               # FiLM conditions on the action
        else:
            resid = self.net(stack_context_action(context, action))
        # residual add in float32 so mixed precision never loses the base frame
        out = context[:, -1].float() + resid.float()
        return torch.clamp(out, -1.0, 1.0)

    def forward(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict(context, action)


class PersistenceModel:
    """The reference predictor: repeat the last context frame, ignore the action."""

    def predict(self, context: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        return context[:, -1]


# --------------------------------------------------------------------------- #
# Construction / serialization
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_CONFIG = {
    "K": 4,
    "action_channels": 2,
    "base": 32,
    "conditioning": "channels",  # "channels" (broadcast) or "film" (FiLM at scales)
}


def build_model(config: dict) -> NextFrameModel:
    K = config["K"]
    base = config.get("base", 32)
    if config.get("conditioning", "channels") == "film":
        net = FilmUNet(in_ch=K * 3, K=K, out_ch=3, base=base)
        return NextFrameModel(net, K, use_film=True)
    in_ch = K * (3 + config.get("action_channels", 2))
    net = UNetBaseline(in_ch=in_ch, out_ch=3, base=base)
    return NextFrameModel(net, K)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_checkpoint(path: str, model: NextFrameModel, config: dict, val_loss: float,
                    step: int, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_config": {k: config[k] for k in DEFAULT_MODEL_CONFIG if k in config},
        "config": dict(config),
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cuda") -> tuple[NextFrameModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
