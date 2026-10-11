"""Environment check: record versions and prove PyTorch runs real GPU work."""

from __future__ import annotations

import json
import platform
import sys

import torch


def run_gpu_check() -> dict:
    info = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        arch = f"sm_{cap[0] * 10 + cap[1]}"
        arch_list = torch.cuda.get_arch_list()
        info.update({
            "gpu_name": name,
            "gpu_capability": list(cap),
            "gpu_arch": arch,
            "cudnn_version": torch.backends.cudnn.version(),
            "arch_in_supported_list": arch in arch_list,
            "arch_list": arch_list,
        })
        # Real GPU computation (not just is_available): matmul + conv + backward.
        torch.manual_seed(0)
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
        x = torch.randn(4, 3, 64, 64, device="cuda")
        y = conv(x)
        loss = (y.sum() + c.sum())
        loss.backward()
        torch.cuda.synchronize()
        info.update({
            "matmul_finite": bool(torch.isfinite(c).all().item()),
            "conv_backward_grad_ok": conv.weight.grad is not None and bool(
                torch.isfinite(conv.weight.grad).all().item()
            ),
        })
    return info


def main() -> int:
    print(json.dumps(run_gpu_check(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
