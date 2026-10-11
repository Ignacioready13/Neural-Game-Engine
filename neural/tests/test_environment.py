"""Environment verification (item 1): PyTorch runs real GPU work on this machine."""

import torch

from gpu_check import run_gpu_check


def test_gpu_real_compute():
    info = run_gpu_check()
    assert info["cuda_available"] is True
    assert info["arch_in_supported_list"] is True, info.get("arch_list")
    assert info["matmul_finite"] is True
    assert info["conv_backward_grad_ok"] is True
    # Blackwell consumer GPU expected (RTX 5070 = sm_120)
    assert info["gpu_arch"] == "sm_120", info["gpu_arch"]
