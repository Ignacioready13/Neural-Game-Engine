"""Lightweight image metrics and a phase-correlation shift estimator.

Everything here is dependency-free (numpy/torch only) so it can be reused by the
evaluation harness without pulling in `scikit-image` or similar. All the *image*
metric functions (`l1`, `mse`, `psnr`, `ssim`) take tensors in the loader's
normalized **[-1, 1]** range (the diffusion convention) and convert internally to
[0, 1] for PSNR/SSIM, whose constants assume a [0, 1] dynamic range. `l1`/`mse`
are reported in [0, 1] units too (see `to01`), so a value of 0.02 means "2% of
full scale".

`estimate_shift_x` is the *global horizontal* shift between two frames, estimated
by phase correlation (the x-component of the peak of the cross-power spectrum).
A positive result means the second image's content is shifted **right** relative
to the first (i.e. the camera turned left).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (x + 1.0) * 0.5


def l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error in [0, 1] units (full dynamic range = 1.0)."""
    return float(F.l1_loss(to01(pred), to01(target)).item())


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error in [0, 1]^2 units."""
    return float(F.mse_loss(to01(pred), to01(target)).item())


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB) for a [0, 1] image (MAX = 1.0), capped at 100."""
    m = mse(pred, target)
    if m <= 1e-12:
        return 100.0
    return float(min(100.0, 10.0 * np.log10(1.0 / m)))


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 7) -> float:
    """Structural similarity index (uniform ``window`` x ``window`` filter).

    Standard constants C1 = (0.01*L)^2, C2 = (0.03*L)^2 with L = 1.0 (the [0, 1]
    range). Computed with a box filter via ``avg_pool2d`` — a valid, if plain,
    SSIM window. Returns the mean over batch, channel and space in [-1, 1].
    """
    p = to01(pred)
    t = to01(target)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window // 2
    mu_p = F.avg_pool2d(p, window, 1, pad)
    mu_t = F.avg_pool2d(t, window, 1, pad)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sig_p2 = F.avg_pool2d(p * p, window, 1, pad) - mu_p2
    sig_t2 = F.avg_pool2d(t * t, window, 1, pad) - mu_t2
    sig_pt = F.avg_pool2d(p * t, window, 1, pad) - mu_pt
    num = (2.0 * mu_pt + C1) * (2.0 * sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
    return float((num / den).mean().item())


def _subpixel_peak(corr: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Refine an integer correlation peak to subpixel via 1-D parabolic fit."""
    H, W = corr.shape
    dx = dy = 0.0
    if 0 < x < W - 1:
        a, b, c = corr[y, x - 1], corr[y, x], corr[y, x + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            dx = 0.5 * (a - c) / denom
    if 0 < y < H - 1:
        a, b, c = corr[y - 1, x], corr[y, x], corr[y + 1, x]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            dy = 0.5 * (a - c) / denom
    return float(dx), float(dy)


def estimate_shift(prev: np.ndarray, next_: np.ndarray) -> tuple[float, float]:
    """Global translation between two single-channel images, via phase correlation.

    Returns ``(dx, dy)`` in pixels: the shift of ``next_`` relative to ``prev``
    (positive ``dx`` = content moved right). Hann windowed to suppress edges.
    """
    prev = np.asarray(prev, dtype=np.float64)
    next_ = np.asarray(next_, dtype=np.float64)
    H, W = prev.shape
    # remove the mean so the DC component does not dominate the correlation
    prev = prev - prev.mean()
    next_ = next_ - next_.mean()
    wy = np.hanning(H)
    wx = np.hanning(W)
    win = np.outer(wy, wx)
    F1 = np.fft.rfft2(prev * win)
    F2 = np.fft.rfft2(next_ * win)
    R = F1 * np.conj(F2)
    R /= (np.abs(R) + 1e-12)
    corr = np.fft.irfft2(R, s=(H, W))
    idx = int(np.argmax(np.abs(corr)))
    y, x = np.unravel_index(idx, (H, W))
    dx, dy = _subpixel_peak(np.abs(corr), y, x)
    xf = float(x) + dx
    yf = float(y) + dy
    if xf > W / 2.0:
        xf -= W
    if yf > H / 2.0:
        yf -= H
    # np.fft's cross-power peak reports the *negative* of the content translation
    # (a right-shifted `next_` peaks at -d), so negate to match the documented
    # convention: positive dx = `next_` content moved right (camera turned left).
    return float(-xf), float(-yf)


def estimate_shift_x(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> float:
    """Horizontal shift (pixels) between two RGB frames, on their grayscale mean.

    ``prev_rgb``/``next_rgb`` are ``[H, W, 3]`` (or ``[H, W]``). Returns the
    x-component of the phase-correlation peak.
    """
    prev = np.asarray(prev_rgb)
    next_ = np.asarray(next_rgb)
    if prev.ndim == 3:
        prev = prev.mean(axis=-1)
    if next_.ndim == 3:
        next_ = next_.mean(axis=-1)
    dx, _ = estimate_shift(prev, next_)
    return dx


# --------------------------------------------------------------------------- #
# Stage 5 additive metrics: sharpness, colour, stability, diversity
# --------------------------------------------------------------------------- #

def _as_nchw(x) -> torch.Tensor:
    """Any frame(s) -> [N, C, H, W] float in the loader's [-1, 1] range.

    Accepts the diffusion convention (float tensors/arrays in [-1, 1], the loader's
    ``normalize=True`` output) and raw ``uint8`` frames in [0, 255] (what the packed
    arrays and the rollout driver produce in their display form). Mixing the two
    silently would make every "in [0,1] units" metric meaningless, so the dtype is
    used to decide: integer/uint8 input is rescaled from [0, 255].
    """
    if isinstance(x, torch.Tensor):
        raw_u8 = x.dtype == torch.uint8
        t = x.float()
    else:
        a = np.asarray(x)
        raw_u8 = a.dtype == np.uint8
        # torch.tensor copies, so a read-only memmap slice is safe to consume
        t = torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    if raw_u8:
        t = t / 127.5 - 1.0
    if t.ndim == 3:
        t = t[None]
    if t.shape[1] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(0, 3, 1, 2)
    return t


def sharpness(x) -> float:
    """Mean gradient magnitude in [0, 1] units (higher = sharper).

    A blurry average has a low mean gradient; a crisp frame has a high one. Also
    ``baseline_sharpness`` helper below: same thing, so units are directly
    comparable between real, generated and Stage 4 frames.
    """
    t = to01(_as_nchw(x))
    gx = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs().mean()
    gy = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs().mean()
    return float(0.5 * (gx + gy).item())


def laplacian_var(x) -> float:
    """Variance of the 4-neighbour Laplacian on the luminance, in [0, 1]^2 units."""
    t = to01(_as_nchw(x))
    lum = t.mean(dim=1, keepdim=True)
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=t.device).view(1, 1, 3, 3)
    lap = F.conv2d(lum, k, padding=1)
    return float(lap.var().item())


def color_stats(x) -> dict:
    """Per-channel mean and std in [0, 1] units."""
    t = to01(_as_nchw(x))
    means = t.mean(dim=(0, 2, 3))
    stds = t.std(dim=(0, 2, 3))
    return {
        "mean_rgb": [float(v) for v in means.tolist()],
        "std_rgb": [float(v) for v in stds.tolist()],
        "mean_all": float(t.mean().item()),
    }


def frame_change(a, b) -> float:
    """Mean absolute difference between two frames (or frame stacks) in [0, 1]."""
    ta, tb = to01(_as_nchw(a)), to01(_as_nchw(b))
    return float((ta - tb).abs().mean().item())


def black_frac(x, thr: float = 0.02) -> float:
    """Share of pixels whose luminance is below ``thr`` (collapse-to-black check)."""
    t = to01(_as_nchw(x)).mean(dim=1)
    return float((t < thr).float().mean().item())


def saturated_frac(x, thr: float = 0.98) -> float:
    """Share of pixels whose luminance is above ``thr`` (collapse-to-white check)."""
    t = to01(_as_nchw(x)).mean(dim=1)
    return float((t > thr).float().mean().item())


def seed_diversity(samples) -> float:
    """Mean pairwise L1 between samples of the same context, in [0, 1] units.

    ``samples`` is a list/array of frames (each [3,H,W] or [N,3,H,W] on the same
    grid). Larger = the generator produced visibly different content.
    """
    ts = [to01(_as_nchw(s)) for s in samples]
    if len(ts) < 2:
        return 0.0
    vals = []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            vals.append(float((ts[i] - ts[j]).abs().mean().item()))
    return float(np.mean(vals))


def _crop_border(x, border: int = 4) -> torch.Tensor:
    t = _as_nchw(x)
    return t[:, :, border:-border, border:-border]
