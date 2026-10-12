"""Stage 5 — the minimal interactive player (Milestone 5).

A thin client, entirely separate from the engine: it opens a checkpoint, seeds the
run with K real frames from an episode, then generates every following frame from
its own outputs plus the click signal. The engine is never involved.

N1/N2 hold by construction: the only things fed to the model are the K most recent
frames (the seed, then its own output) and the click-derived signals. The click
pulse drawn over the displayed image is a *display* overlay and is never fed back.

Click semantics mirror the engine's (``engine`` §2): the held signal is

    actionX = clamp((clientX - left) / width * 2 - 1, -1, 1)      (left = -1)

a click closer than 3 frames (150 ms at 20 Hz) after the last **accepted** click is
ignored, and ``actionHasClicked`` becomes 1 after the first accepted click.

Usage::

    # interactive window
    ./memcap.sh --mem 12G -- .venv/bin/python play.py \
        --checkpoint runs/flow96/best.pt --packed data/bulk_v1_96x72 --split val

    # scripted, headless (writes a GIF and a frame strip)
    .venv/bin/python play.py --checkpoint runs/flow96/best.pt \
        --packed data/bulk_v1_96x72 --headless --frames 200 \
        --clicks "5:0.6,45:-0.8,90:0.2" --out ../neural/review/stage5/play.gif
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from dataset import PackedDataset, frames_to_tensor
from eval_harness import features_from_history, model_predict


class ClickState:
    """The engine's held-click semantics (debounce 3 frames = 150 ms at 20 Hz)."""

    def __init__(self, debounce_frames: int = 3):
        self.debounce = int(debounce_frames)
        self.x = 0.0
        self.has_clicked = 0.0
        self.last_accept = None
        self.accepted = 0

    def click(self, x: float, now_frame: int) -> bool:
        """Offer a click at frame ``now_frame``. Returns True if it was accepted."""
        x = float(np.clip(x, -1.0, 1.0))
        if self.last_accept is None or (now_frame - self.last_accept) >= self.debounce:
            self.last_accept = int(now_frame)
            self.x = x
            self.has_clicked = 1.0
            self.accepted += 1
            return True
        return False

    @property
    def action(self) -> tuple[float, float]:
        return (self.x, self.has_clicked)


class Player:
    """Generates frames in closed loop from a seed context + click signals."""

    def __init__(self, checkpoint: str, packed_path: str, split: str = "val",
                 device: str | None = None, hist_extra: int = 8, seed: int | None = None):
        from gates import load_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, payload = load_model(checkpoint, self.device)
        mc = payload.get("model_config", {})
        self.K = int(mc.get("K", 4))
        self.hist_extra = hist_extra
        self.packed = PackedDataset(packed_path)
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.clicks = ClickState()
        self.frame_index = 0
        self.pulses = []          # (frame_index, x) for the display overlay only

    # -- seeding ----------------------------------------------------------- #

    def _pick_seed(self, episodes=None) -> tuple[int, int]:
        eps = [e for e in self.packed.episodes_by_split[self.split]
               if e["length"] > self.K + 20]
        e = eps[int(self.rng.integers(len(eps)))]
        L = e["length"]
        start = int(self.rng.integers(0, max(1, int(0.1 * L))))
        return e["episode_index"], start

    def reset(self, episode: int | None = None, start: int | None = None):
        """Seed the run with K real frames (random early-episode clip by default)."""
        if episode is None or start is None:
            episode, start = self._pick_seed()
        self.episode, self.start = int(episode), int(start)
        rec = self.packed.episodes[self.episode]
        off = rec["frame_offset"]
        self.frames = frames_to_tensor(
            np.asarray(self.packed.images[off + self.start: off + self.start + self.K])
        ).to(self.device)                                   # [K,3,H,W]
        # click history preceding the seed (real click signals)
        lo = max(0, self.start - self.hist_extra)
        h = self.packed.action_vectors(self.packed.labels[off + lo: off + self.start + self.K])
        self.hist = [row for row in np.nan_to_num(h, nan=0.0).astype(np.float32)]
        self.clicks = ClickState()
        if self.hist and self.hist[-1][1] > 0.5:
            self.clicks.x = float(self.hist[-1][0])
            self.clicks.has_clicked = 1.0
        self.frame_index = 0
        self.pulses = []
        return self.current_frame()

    # -- generation -------------------------------------------------------- #

    def current_frame(self):
        return self.frames[-1]

    @torch.no_grad()
    def step(self, action: tuple[float, float], seed: int | None = None):
        """Advance one frame. ``action`` = the click-derived (actionX, hasClicked)."""
        self.hist.append(np.array([action[0], action[1]], dtype=np.float32))
        feats = features_from_history(np.stack(self.hist), self.K, self.device)
        ctx = self.frames[-self.K:].unsqueeze(0)
        a = torch.tensor([[action[0], action[1]]] * self.K, dtype=torch.float32,
                         device=self.device)[None]
        out = model_predict(self.model, ctx, a, feats, seed, return_shift=True)
        nxt = out[0] if isinstance(out, tuple) else out
        self.frames = torch.cat([self.frames, nxt], dim=0)
        self.frame_index += 1
        return nxt[0]

    def click(self, x: float) -> bool:
        ok = self.clicks.click(x, self.frame_index)
        if ok:
            self.pulses.append((self.frame_index, float(np.clip(x, -1, 1))))
        return ok

    # -- headless ---------------------------------------------------------- #

    def generate(self, n_frames: int, click_script=None, seed: int | None = None):
        """Generate ``n_frames`` with a script (list of ``(frame, x)`` clicks)."""
        frames, pulses = [], []
        for f in range(n_frames):
            for cf, cx in (click_script or []):
                if cf == f:
                    self.click(cx)
            if f == 0:
                frm = self.current_frame()
            else:
                frm = self.step(self.clicks.action, seed=seed)
            frames.append(frm)
            for pf, px in self.pulses:
                if pf <= f:
                    pulses.append((f, px))
        self.pulses = []
        return frames


# --------------------------------------------------------------------------- #
# Rendering / I/O
# --------------------------------------------------------------------------- #

def _to_uint8(frame) -> np.ndarray:
    t = frame.float().detach().cpu()
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    u = ((t * 0.5 + 0.5) * 255).clamp(0, 255).numpy().astype(np.uint8)
    return u


def write_gif(frames, path: str, scale: int = 6, fps: int = 20, max_frames: int = 400):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs = []
    for f in frames[:max_frames]:
        u = _to_uint8(f)
        im = Image.fromarray(u).resize((u.shape[1] * scale, u.shape[0] * scale),
                                       Image.Resampling.NEAREST)
        imgs.append(im)
    if not imgs:
        return path
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=False)
    return path


def write_strip(frames, path: str, every: int = 10, scale: int = 4):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel = frames[::every]
    imgs = [Image.fromarray(_to_uint8(f)).resize(
        (_to_uint8(f).shape[1] * scale, _to_uint8(f).shape[0] * scale),
        Image.Resampling.NEAREST) for f in sel]
    if not imgs:
        return path
    w, h = imgs[0].size
    cols = min(6, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    sheet.save(path)
    return path


def run_interactive(player: Player, scale: int = 8, fps: int = 20):
    """Open a window, show generated frames, read the mouse, show an fps readout."""
    import time
    import pygame
    pygame.init()
    H, W = player.packed.height, player.packed.width
    screen = pygame.display.set_mode((W * scale, H * scale + 24))
    pygame.display.set_caption("NeuralGameEngine — Stage 5 player (click to steer)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    player.reset()
    print("player: click in the window to steer. Esc or window close to quit.",
          flush=True)
    running = True
    fps_est = 0.0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN
                                          and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx = ev.pos[0] / float(W * scale)
                player.click(float(np.clip(mx * 2 - 1, -1, 1)))
        t0 = time.time()
        frm = player.step(player.clicks.action)
        u = _to_uint8(frm)
        surf = pygame.surfarray.make_surface(np.transpose(u, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (W * scale, H * scale))
        screen.fill((0, 0, 0))
        screen.blit(surf, (0, 0))
        # click pulse overlay (display only; never fed back to the model)
        for pf, px in list(player.pulses):
            if player.frame_index - pf < 8:
                cx = int((px * 0.5 + 0.5) * W * scale)
                s = pygame.Surface((3 * scale, H * scale), pygame.SRCALPHA)
                s.fill((255, 255, 255, 60))
                screen.blit(s, (cx - int(1.5 * scale), 0))
            else:
                player.pulses.remove((pf, px))
        dt = time.time() - t0
        fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(1e-6, dt))
        txt = font.render(f"{fps_est:5.1f} fps   click {player.clicks.x:+.2f}  "
                          f"clicked={int(player.clicks.has_clicked)}", True,
                          (200, 255, 200))
        screen.blit(txt, (6, H * scale + 4))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def parse_clicks(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        f, x = part.split(":")
        out.append((int(f), float(x)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5 minimal interactive player")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed-context", dest="seed_context", default=None,
                    help="episode:start override for the seed clip")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--clicks", default="", help='scripted clicks "frame:x,frame:x"')
    ap.add_argument("--out", default=None, help="GIF path (headless)")
    ap.add_argument("--strip", default=None, help="frame-strip PNG path (headless)")
    args = ap.parse_args(argv)

    player = Player(args.checkpoint, args.packed, args.split)
    ep = st = None
    if args.seed_context:
        ep, st = (int(v) for v in args.seed_context.split(":"))
    player.reset(ep, st)
    print(f"seed context: episode {player.episode} start {player.start} "
          f"({player.packed.width}x{player.packed.height})", flush=True)

    if args.headless:
        script = parse_clicks(args.clicks)
        frames = player.generate(args.frames, script)
        out = args.out or "review/stage5/play_headless.gif"
        write_gif(frames, out, scale=max(2, args.scale // 2), fps=args.fps)
        print(f"wrote {out}", flush=True)
        if args.strip:
            write_strip(frames, args.strip)
            print(f"wrote {args.strip}", flush=True)
        return 0

    run_interactive(player, scale=args.scale, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
