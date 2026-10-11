"""Metric functions and the phase-correlation shift estimator (Stage 4, item 8)."""

import numpy as np
import pytest
import torch

from metrics import estimate_shift_x, l1, mse, psnr, ssim


def _img(n=1, h=32, w=32):
    return torch.rand(n, 3, h, w) * 2 - 1


def test_l1_mse_zero_and_scaling():
    a = _img(4)
    assert l1(a, a) == 0.0
    assert mse(a, a) == 0.0
    # L1 in [0,1] units: a constant offset of 0.2 in [-1,1] = 0.1 in [0,1]
    b = a + 0.2
    assert np.isclose(l1(a, b), 0.1, atol=1e-3)
    assert np.isclose(mse(a, b), 0.01, atol=1e-3)


def test_psnr_and_ssim_identity():
    a = _img(2)
    assert psnr(a, a) == 100.0
    assert np.isclose(ssim(a, a), 1.0, atol=1e-4)


def test_psnr_is_finite_and_monotone():
    a = _img(3)
    b1 = a + 0.05
    b2 = a + 0.2
    p1, p2 = psnr(a, b1), psnr(a, b2)
    assert np.isfinite(p1) and np.isfinite(p2)
    assert p1 > p2  # closer image -> higher PSNR


@pytest.mark.parametrize("k", [-5, -2, 0, 2, 5])
def test_shift_x_recovers_translation(k):
    rng = np.random.default_rng(0)
    img = rng.random((72, 96, 3)).astype(np.float32)
    shifted = np.roll(img, k, axis=1)  # content moved right by k
    dx = estimate_shift_x(img, shifted)
    assert abs(dx - k) < 1.0, f"roll {k} recovered {dx}"


def test_shift_x_works_on_grayscale():
    rng = np.random.default_rng(1)
    img = rng.random((48, 64)).astype(np.float32)
    dx = estimate_shift_x(img, np.roll(img, 3, axis=1))
    assert abs(dx - 3) < 1.0
