"""Step 1 of the action-conditioning investigation: pathway sanity test.

Trains the *same* architecture on a synthetic task where the action obviously
matters: the target is the last context frame shifted horizontally by an amount
proportional to actionX. The context frames are identical, so no frame-based
information can predict the shift — the model must read the action channel.

If it drives the loss near zero, the action-conditioning *code* is fine and the
real issue is the one-step objective; if it cannot, the conditioning has a bug.

Run: .venv/bin/python pathway_sanity.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model import build_model

MAX_SHIFT = 8  # pixels at actionX = +-1


def make_batch(B, K, H, W, device):
    # smooth random texture, per-sample normalized to [-1, 1]
    base = torch.randn(B, 3, H // 4, W // 4, device=device)
    base = F.interpolate(base, size=(H, W), mode="bilinear", align_corners=False)
    lo = base.amin(dim=(1, 2, 3), keepdim=True)
    hi = base.amax(dim=(1, 2, 3), keepdim=True)
    base = (base - lo) / (hi - lo + 1e-6) * 2.0 - 1.0            # [B,3,H,W] in [-1,1]
    ctx = base[:, None].repeat(1, K, 1, 1, 1)                    # K identical frames
    act = torch.zeros(B, K, 2, device=device)
    ax = (torch.rand(B, device=device) * 2 - 1) * 0.8             # actionX in [-0.8, 0.8]
    act[:, -1, 0] = ax
    sh = (ax * MAX_SHIFT).round().long()
    tgt = torch.stack([torch.roll(base[b], s.item(), dims=-1) for b, s in enumerate(sh)])
    return ctx, act, tgt


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    K, H, W = 4, 48, 64
    model = build_model(dict(K=K, base=32, conditioning="channels")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 3001):
        ctx, act, tgt = make_batch(64, K, H, W, device)
        out = model.predict(ctx, act)
        loss = F.l1_loss(out, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 600 == 0:
            print(f"step {step}: l1 {loss.item():.4f}")
    with torch.no_grad():
        ctx, act, tgt = make_batch(256, K, H, W, device)
        final = F.l1_loss(model.predict(ctx, act), tgt).item()
    print(f"final l1 (fresh batch): {final:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
