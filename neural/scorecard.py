"""Stage 5b scorecard (Sections 4-6 of the overnight brief).

The frozen evaluation set lives in ``eval_sets/scorecard_v1.json``; this module
builds it once (``--part sets``) and scores any candidate against it
(``--part all``), writing ``eval_out/score_<name>.json`` plus a markdown row.

Families
--------
  S  steering  — the Gate G1 numbers, reused verbatim from ``gates.py``.
  R  realness  — radial power spectrum, per-band ratios, Laplacian variance,
                 multi-scale gradients, and a real-vs-generated classifier AUC.
  D  structure — the frozen opening detector: positive share and opening events
                 per 100 frames, real versus generated.
  H  health    — colour, dark-region colour, drift, motion, freeze checks.
  T  speed     — the Gate G4 loop at batch 1.

Every number is produced by code that is frozen before any new training; the
candidate only ever changes the checkpoint.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

from dataset import PackedDataset, frames_to_tensor
from eval_harness import features_from_history, model_predict
from gates import (load_model, near_door_contexts, onset_windows, run_g1, run_g4,
                   sustained_windows)
from judges import (OpeningDetector, health_stats, laplacian_var, motion_px_per_step,
                    multiscale_gradient, opening_events, spectrum_report, to_float01,
                    train_realfake)
from metrics import estimate_shift_x
from analysis import load_analysis
from shift_table import load_target_table
from steer import click_features

HERE = os.path.dirname(os.path.abspath(__file__))
SETS_PATH = os.path.join(HERE, "eval_sets", "scorecard_v1.json")
EVAL_OUT = os.path.join(HERE, "eval_out")
HORIZONS = [20, 60, 150, 300]
SCENARIOS = ["no_click", "real_click", "scripted_alt"]
SEEDS = [0, 1, 2]
LONG_EVERY = 20          # the "long run" clips used for health / motion / opening events
G1_STEPS = 40            # the G1 sustained-steering horizon (a separate thing from the
                         # 300-frame scorecard rollout length)


# --------------------------------------------------------------------------- #
# the frozen sets
# --------------------------------------------------------------------------- #

def build_sets(packed_path="data/bulk_v1_96x72", K=4, H=10, steps=300, n_clips=40,
               n_contexts=300, out=SETS_PATH):
    """Freeze the evaluation sets once; never regenerate without bumping the name."""
    packed = PackedDataset(packed_path)
    val = [e for e in packed.episodes_by_split["val"]]
    val.sort(key=lambda e: e["episode_index"])
    usable = [e for e in val if e["length"] >= K + steps + 4]
    if len(usable) < 2 * n_clips:
        raise SystemExit(f"not enough val episodes: {len(usable)}")

    def clip(e):
        return {"episode": int(e["episode_index"]),
                "start": int((e["length"] - steps - 2) // 2)}

    main = [clip(e) for e in usable[:n_clips]]
    confirm = [clip(e) for e in usable[n_clips: 2 * n_clips]]

    sidecar = load_analysis(packed)
    ctxs, n_ctx_ep = near_door_contexts(packed, sidecar, K, H, n_min=n_contexts,
                                        max_per_ep=2, split="val")
    ctx_list = [[int(c[0]), int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                for c in ctxs][:n_contexts]
    table = load_target_table(packed, packed_path)
    wins = onset_windows(packed, table, K, split="val")[:n_contexts]
    swins = sustained_windows(packed, table, K, split="val", steps=G1_STEPS,
                              n=min(n_contexts, 300))
    long_clips = [main[i] for i in np.linspace(0, len(main) - 1, 10).astype(int)]

    data = {
        "version": "scorecard_v1",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "frozen": True,
        "packed": packed_path,
        "K": K,
        "H": H,
        "rollout_steps": steps,
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "main_clips": main,
        "confirmation_clips": confirm,
        "long_clips": long_clips,
        "steering_contexts": ctx_list,
        "steering_onset_windows": [[int(e), int(s)] for e, s in wins],
        "steering_sustained_windows": [[int(e), int(s)] for e, s in swins],
        "n_context_episodes": int(n_ctx_ep),
        "notes": ("Frozen before any Stage 5b training. main = 40 distinct val episodes, "
                  "mid-episode; confirmation = 40 disjoint val episodes, used once at the "
                  "end; steering_* = the Gate G1 near-door two-door / click-onset / "
                  "sustained-turn selections, frozen so every candidate sees them."),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists — the frozen set is never rewritten")
    with open(out, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {out}: {len(main)} main clips, {len(confirm)} confirmation clips, "
          f"{len(ctx_list)} steering contexts")
    return data


def load_sets(path=SETS_PATH):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #

def script_value(scenario, t):
    """The click x the scripted scenario holds (the pattern uses -0.5 / +0.5)."""
    return -0.5 if (t // 40) % 2 == 0 else 0.5


def _script_for(steps):
    """Scripted alternating left/right pattern: a click every 40 steps, alternating."""
    return [(t, -0.5 if (t // 40) % 2 == 0 else 0.5) for t in range(20, steps, 40)]


def _rollout_actions(packed, model, e, s, K, steps, scenario, seed, device):
    """Closed-loop rollout for one scenario; returns (frames [steps+1,3,H,W], info).

    Frame 0 is the real context's last frame so that per-step statistics are over
    generated steps only.
    """
    if scenario in ("no_click", "real_click"):
        from eval_harness import rollout_v2
        override = None
        if scenario == "no_click":
            override = torch.zeros(1, K, 2)
        gen, info = rollout_v2(packed, model, e, s, K, steps,
                               action_override=override, seed=seed, device=device)
        last = np.asarray(packed.images[packed.episodes[e]["frame_offset"] + s + K - 1])[None]
        return np.concatenate([last, _u8(gen)], axis=0), info

    # scripted_alt: hold each click for 3 steps then release, like a real player
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    lo = max(0, s - 8)
    hist = [r for r in np.nan_to_num(packed.action_vectors(
        packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)]
    frames = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K])).to(device)
    script = dict(_script_for(steps))
    held = (0.0, 0.0)
    release_at = -1
    out = []
    with torch.no_grad():
        for t in range(steps):
            if t in script:
                held = (float(script[t]), 1.0)
                release_at = t + 3
            elif t >= release_at:
                held = (held[0], 0.0)
            hist.append(np.array(held, dtype=np.float32))
            feats = click_features(torch.from_numpy(np.stack(hist))[None].float().to(device))[:, -K:]
            nxt = model_predict(model, frames[-K:].unsqueeze(0), None, feats, seed)
            nxt = nxt[0] if isinstance(nxt, tuple) else nxt
            frames = torch.cat([frames, nxt], dim=0)
            out.append(nxt[0].cpu())
    gen = torch.stack(out)
    last = np.asarray(packed.images[off + s + K - 1])[None]
    return np.concatenate([last, _u8(gen)], axis=0), {"shift_px": None}


def _u8(t):
    t = t.float().detach().cpu()
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)
    else:
        t = t.permute(1, 2, 0)
    return np.clip((t * 0.5 + 0.5) * 255.0, 0, 255).numpy().astype(np.uint8)


@torch.no_grad()
@torch.no_grad()
def commanded_shift(packed, model, jobs, K, steps, seed, device, n_clips=12, n_steps=12):
    """Mean |commanded dx| per step, on real contexts versus on the model's own frames.

    This is the closed-loop steering question in one number. The readout is trained on
    real frames; if it commands a sensible shift on a real context but ~0 once the
    context is made of frames the generator produced, the closed loop freezes: the
    scene stops translating, no doorway can ever come into view, and the player sees
    one long unchanging hallway. The rollouts here are driven by the clip's **real
    clicks** (``rollout_v2``'s ``real_click`` semantics), because with no click at all
    the correct command is ~0 and the comparison would say nothing.
    """
    true_dx, gen_dx = [], []
    for e, s in jobs[:n_clips]:
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        L = rec["length"]
        acts = np.nan_to_num(packed.action_vectors(packed.labels[off: off + L]), nan=0.0).astype(np.float32)
        real = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K]))[None].to(device)
        hist = list(acts[max(0, s - 8): s + K - 1])
        buf = real
        for t in range(n_steps):
            a = acts[s + t: s + t + K]
            hist = hist + [a[-1]]
            feats = click_features(torch.from_numpy(np.stack(hist))[None].float().to(device))[:, -K:]
            a_t = torch.from_numpy(np.ascontiguousarray(a))[None].to(device)
            src = real if t == 0 else buf
            out = model_predict(model, src, a_t, feats, seed, return_shift=True)
            if isinstance(out, tuple):
                nxt, info = out
                dx = info.get("dx") if isinstance(info, dict) else None
                if dx is not None:
                    v = float(dx.abs().mean())
                    (true_dx if t == 0 else gen_dx).append(v)
            else:
                nxt = out
            buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
    return {"mean_abs_dx_real_context": _mean(true_dx),
            "mean_abs_dx_generated_context": _mean(gen_dx),
            "ratio_gen_over_real": (_mean(gen_dx) / _mean(true_dx)) if true_dx and gen_dx
            and _mean(true_dx) else None,
            "n_real": len(true_dx), "n_gen": len(gen_dx)}


@torch.no_grad()
def _rollout_batch(packed, model, jobs, K, steps, scenario, seed, device):
    """Roll a whole batch of independent clips forward together.

    One forward pass per step for the whole batch instead of one per rollout: the
    closed loop is latency-bound at batch 1 (launch overhead dominates), so this is
    the difference between ~12 s and ~3 s per 300-step rollout.  Semantics are the
    same as ``eval_harness.rollout_v2`` for ``no_click``/``real_click`` (verified by
    ``tests/test_stage5b.py``), with the scripted pattern handled inline.
    """
    B = len(jobs)
    L = [packed.episodes[e]["length"] for e, _ in jobs]
    offs = [packed.episodes[e]["frame_offset"] for e, _ in jobs]
    starts = [s for _, s in jobs]
    acts = [np.nan_to_num(packed.action_vectors(packed.labels[o: o + ln]), nan=0.0).astype(np.float32)
            for o, ln in zip(offs, L)]

    # This mirrors eval_harness.rollout_v2 line for line: `a_ctx` is the action
    # tensor handed to the model (its last row is the injected/held action) and
    # `hist` is the click-signal history the player would really have.
    hist, a_ctx, held = [], [], []
    for i, st in enumerate(starts):
        lo = max(0, st - 8)
        real_ctx = acts[i][st: st + K]
        h = list(acts[i][lo: st + K - 1])
        if scenario == "real_click":
            ac = real_ctx.copy()
        else:
            hv = (np.array([script_value("scripted_alt", 0), 1.0], np.float32)
                  if scenario == "scripted_alt" else np.zeros(2, np.float32))
            ac = np.concatenate([real_ctx[: K - 1], hv[None]], axis=0)
        hist.append(list(h) + [ac[K - 1]])
        a_ctx.append(ac)
        held.append(ac[K - 1].copy())

    frames = torch.stack([frames_to_tensor(np.asarray(packed.images[o + st: o + st + K]))
                          for o, st in zip(offs, starts)]).to(device)
    seqs = [[] for _ in range(B)]
    script = dict(_script_for(steps)) if scenario == "scripted_alt" else {}
    release = [-1] * B
    for t in range(steps):
        loop_a = []
        for i in range(B):
            if scenario == "real_click":
                a = acts[i][starts[i] + t: starts[i] + t + K]
            else:
                if scenario == "scripted_alt":
                    if t in script:
                        held[i] = np.array([float(script[t]), 1.0], np.float32)
                        release[i] = t + 3
                    elif t >= release[i]:
                        held[i] = np.array([held[i][0], 0.0], np.float32)
                    a_ctx[i] = a_ctx[i].copy()
                    a_ctx[i][K - 1] = held[i]
                a = a_ctx[i]
            loop_a.append(a)
        feats = torch.cat([click_features(torch.from_numpy(np.stack(hist[i]))[None]
                                          .float().to(device))[:, -K:] for i in range(B)], dim=0)
        a_t = torch.from_numpy(np.stack(loop_a)).to(device)
        nxt = model_predict(model, frames[:, -K:], a_t, feats, seed)
        nxt = nxt[0] if isinstance(nxt, tuple) else nxt
        frames = torch.cat([frames, nxt[:, None]], dim=1)
        for i in range(B):
            seqs[i].append(nxt[i].cpu())
            # the newest action in history for the next step (rollout_v2 rule)
            if scenario == "real_click":
                idx = min(starts[i] + t + K, L[i] - 1)
                hist[i].append(acts[i][idx])
            else:
                hist[i].append(held[i])
    return [torch.stack(s) for s in seqs]


def collect_batched(packed, model, sets, clips, seeds, scenarios, detector=None,
                    device="cuda", progress=None, batch=8):
    """Same as :func:`collect` but with batched rollouts (identical semantics)."""
    K, steps = sets["K"], sets["rollout_steps"]
    bank = {h: [] for h in HORIZONS}
    banks = {sd: {h: [] for h in HORIZONS} for sd in seeds}
    seq_health, seq_motion, seq_events = [], [], []
    long_keys = {(c["episode"], c["start"]) for c in sets["long_clips"]}
    jobs = [(c["episode"], c["start"], sc, sd) for c in clips for sc in scenarios for sd in seeds]
    done = 0
    for seed in seeds:
        for sc in scenarios:
            group = [(e, s) for (e, s, sc2, sd) in jobs if sc2 == sc and sd == seed]
            for i in range(0, len(group), batch):
                chunk = group[i: i + batch]
                gens = _rollout_batch(packed, model, chunk, K, steps, sc, seed, device)
                for (e, s), gen in zip(chunk, gens):
                    seq = np.concatenate([np.asarray(packed.images[
                        packed.episodes[e]["frame_offset"] + s + K - 1])[None], _u8(gen)], axis=0)
                    for h in HORIZONS:
                        bank[h].append(seq[h])
                        banks[seed][h].append(seq[h])
                    if (e, s) in long_keys and seed == seeds[0]:
                        mv = motion_px_per_step(seq[::LONG_EVERY])
                        seq_motion.append({"clip": e, "scenario": sc, "motion_px_per_step": mv})
                        if sc == scenarios[0]:
                            st = health_stats(seq)
                            st.update(clip=e, scenario=sc)
                            seq_health.append(st)
                        if detector is not None:
                            ev = opening_events(detector.positive(seq[1:], device))
                            ev.update(clip=e, scenario=sc)
                            seq_events.append(ev)
                    done += 1
                if progress:
                    progress(f"  rollouts {done}/{len(jobs)}")
    return {"bank": bank, "banks": banks, "health_seq": seq_health,
            "motion_seq": seq_motion, "opening_seq": seq_events,
            "n_rollouts": len(jobs)}


def collect(packed, model, sets, clips, seeds, scenarios, detector=None, device="cuda",
            progress=None):
    """Run every rollout of the frozen set once and return the derived numbers.

    Frames are not kept: horizon snapshots are banked (small) and the per-sequence
    statistics (health, motion, opening events) are computed on the fly for the
    ``long_clips`` subset, so memory stays flat.
    """
    K, steps = sets["K"], sets["rollout_steps"]
    bank = {h: [] for h in HORIZONS}          # generated frames at each horizon
    tags = {h: [] for h in HORIZONS}          # parallel (clip, scenario, seed)
    seq_health, seq_motion, seq_events = [], [], []
    done = 0
    total = len(clips) * len(scenarios) * len(seeds)
    long_keys = {(c["episode"], c["start"]) for c in sets["long_clips"]}
    for ci, clip in enumerate(clips):
        for sc in scenarios:
            for seed in seeds:
                seq, _ = _rollout_actions(packed, model, clip["episode"], clip["start"], K,
                                          steps, sc, seed, device)
                for h in HORIZONS:
                    bank[h].append(seq[h])
                    tags[h].append((clip["episode"], sc, seed))
                if (clip["episode"], clip["start"]) in long_keys and seed == seeds[0]:
                    st = health_stats(seq)
                    st.update(clip=clip["episode"], scenario=sc)
                    seq_health.append(st)
                    mv = motion_px_per_step(seq[::LONG_EVERY])
                    seq_motion.append({"clip": clip["episode"], "scenario": sc,
                                       "motion_px_per_step": mv})
                    if detector is not None:
                        pos = detector.positive(seq[1:], device)
                        ev = opening_events(pos)
                        ev.update(clip=clip["episode"], scenario=sc)
                        seq_events.append(ev)
                done += 1
                if progress and done % 20 == 0:
                    progress(f"  rollouts {done}/{total}")
    return {"bank": bank, "tags": tags, "health_seq": seq_health,
            "motion_seq": seq_motion, "opening_seq": seq_events, "n_rollouts": total}


REAL_OFFSETS = (-2, -1, 0, 1, 2)


def clips_from_split(packed, split, steps, n=40, start_frac=0.5):
    """Deterministic mid-episode clips from a split (used for the test-split run).

    The frozen sets are val-only by design; the test split is scored once at the end
    with the *same* protocol, and this builds its clip list the same way the frozen
    one was built (sorted by episode index, mid-episode start, long enough for the
    full 300-step rollout).
    """
    eps = sorted(packed.episodes_by_split[split], key=lambda e: e["episode_index"])
    usable = [e for e in eps if e["length"] >= steps + 8]
    out = []
    for e in usable[:n]:
        L = e["length"]
        out.append({"episode": int(e["episode_index"]),
                    "start": int((L - steps - 2) * start_frac)})
    return out


def real_reference(packed, sets, long_only=False, clips_override=None, split=None):
    """Real frames at the same horizons from the same clips (the comparison bank).

    Five neighbouring frames per clip and horizon, not one: a single frame per clip
    gives the real-vs-generated classifier 40 real frames against 360 generated
    ones, which is imbalanced enough that the net collapsed to a constant (AUC came
    out at exactly 0.500 at every horizon).  The offsets are inside the same clip
    around the matched horizon, so the comparison stays like-for-like.
    """
    K, steps, H = sets["K"], sets["rollout_steps"], HORIZONS
    clips = clips_override if clips_override is not None else (
        sets["long_clips"] if long_only else sets["main_clips"])
    bank = {h: [] for h in H}
    seq_health, seq_motion, seq_events = [], [], []
    for clip in clips:
        e, s = clip["episode"], clip["start"]
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        seq = np.asarray(packed.images[off + s + K - 1: off + s + K - 1 + steps + 1])
        for h in H:
            for d in REAL_OFFSETS:
                j = h + d
                if 0 <= j < len(seq):
                    bank[h].append(seq[j])
        if clip in (clips_override or sets["long_clips"]):
            st = health_stats(seq)
            st.update(clip=e, scenario="real")
            seq_health.append(st)
            seq_motion.append({"clip": e, "scenario": "real",
                               "motion_px_per_step": motion_px_per_step(seq[::LONG_EVERY])})
    return {"bank": bank, "health_seq": seq_health, "motion_seq": seq_motion}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _leaves(obj, prefix=()):
    """Flatten a nested dict of scalars into {(path...): value}."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, prefix + (str(k),)))
        return out
    return {prefix: obj}


def _rebuild(path, value, root):
    """Put ``value`` back at ``path`` inside a copy of ``root``."""
    cur = root
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return root


def aggregate_seeds(trees, key="sd"):
    """Mean of the per-seed metric trees, plus the seed-to-seed SD of every scalar.

    The brief requires the noise to be visible: a "real" improvement must exceed the
    seed-to-seed spread, so every scalar gets a sibling ``<name>_sd`` (and
    ``<name>_mean``) computed over the sampling seeds.
    """
    flat = [_leaves(t) for t in trees]
    keys = set()
    for f in flat:
        keys |= set(f.keys())
    out = trees[0]
    for path in sorted(keys):
        vals = np.array([f[path] for f in flat
                         if isinstance(f.get(path), (int, float))
                         and not isinstance(f[path], bool)
                         and np.isfinite(float(f[path]))], dtype=np.float64)
        if len(vals) < 2:      # nothing to average: keep the first tree's value
            continue
        out = _rebuild(path, float(vals.mean()), out)
        out = _rebuild(path[:-1] + (path[-1] + f"_{key}",), float(vals.std()), out)
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _sd(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def realness(gen_bank, real_bank, detector_seed=0, device="cuda", log=None):
    """R: spectrum, Laplacian, multi-scale gradients, real-vs-generated AUC."""
    out = {}
    for h in HORIZONS:
        g = np.stack(gen_bank[h])
        r = np.stack(real_bank[h])
        spec = spectrum_report(g, r)
        lv_g, lv_r = laplacian_var(g), laplacian_var(r)
        gs_g, gs_r = multiscale_gradient(g), multiscale_gradient(r)
        row = {
            "spectrum_distance": spec["spectrum_distance"],
            "band_ratio": spec["band_ratio"],
            "laplacian_var": lv_g,
            "laplacian_var_real": lv_r,
            "laplacian_ratio": lv_g / max(1e-9, lv_r),
        }
        for k in gs_g:
            row[f"{k}_ratio"] = gs_g[k] / max(1e-9, gs_r[k])
        # Classifier two-sample test. Train on the first half of the clips, test on
        # the second half, with the two classes balanced in both halves: an
        # imbalanced draw makes the net collapse to the majority class and reports a
        # meaningless 0.500 from tied scores.
        n = len(g)
        half = n // 2
        def _bal(gen_part, real_part, k):
            k = min(k, len(gen_part), len(real_part))
            return gen_part[:k], real_part[:k]
        g_tr, r_tr = _bal(g[:half], r[:half], half)
        g_te, r_te = _bal(g[half:], r[half:], half)
        auc = train_realfake(g_tr, r_tr, g_te, r_te,
                             steps=700, device=device, seed=detector_seed)
        row["realfake_auc"] = auc["auc"]
        out[str(h)] = row
        if log:
            log(f"  R @{h}: spectrum_d={row['spectrum_distance']:.3f} "
                f"lap_ratio={row['laplacian_ratio']:.3f} auc={row['realfake_auc']:.3f}")
    return out


def structure(gen_bank, real_bank, detector, col, device="cuda", log=None, min_prob=0.5):
    """D: opening-detector share and events, real versus generated."""
    out = {}
    for h in HORIZONS:
        g = np.stack(gen_bank[h])
        r = np.stack(real_bank[h])
        pg, pr = detector.positive(g, device), detector.positive(r, device)
        out[str(h)] = {
            "detector_positive_share_gen": float(pg.mean()),
            "detector_positive_share_real": float(pr.mean()),
            "detector_positive_share_ratio": float(pg.mean() / max(1e-9, pr.mean())),
        }
        if log:
            log(f"  D @{h}: positive gen {pg.mean():.3f} vs real {pr.mean():.3f}")
    if col is not None and col["opening_seq"]:
        out["events"] = {
            "gen_per_100": _mean([e["events_per_100"] for e in col["opening_seq"]]),
            "gen_positive_share": _mean([e["positive_share"] for e in col["opening_seq"]]),
        }
    return out


def health(col, real_ref, log=None):
    """H: health aggregates, generated versus real, plus motion."""
    out = {"per_horizon": {}}
    for h in HORIZONS:
        g = np.stack(col["bank"][h])
        r = np.stack(real_ref["bank"][h])
        fg = to_float01(g)
        fr = to_float01(r)
        dark_g = (0.299 * fg[..., 0] + 0.587 * fg[..., 1] + 0.114 * fg[..., 2]).mean(axis=(1, 2)) < 0.06
        out["per_horizon"][str(h)] = {
            "mean_luminance_gen": float(fg.mean()),
            "mean_luminance_real": float(fr.mean()),
            "dark_share_gen": float(dark_g.mean()),
            "dark_r_minus_b_gen": float(fg[dark_g][:, 0].mean() - fg[dark_g][:, 2].mean()) if dark_g.any() else None,
            "near_black_share_gen": float((fg.mean(axis=(1, 2, 3)) < 0.02).mean()),
            "near_white_share_gen": float((fg.mean(axis=(1, 2, 3)) > 0.98).mean()),
        }
    if col["health_seq"]:
        out["drift_per_frame"] = _mean([s["luminance_drift_per_frame"] for s in col["health_seq"]])
        out["drift_per_frame_real"] = _mean([s["luminance_drift_per_frame"] for s in real_ref["health_seq"]])
        out["frame_change"] = _mean([s["frame_change"] for s in col["health_seq"]])
        out["frame_change_real"] = _mean([s["frame_change"] for s in real_ref["health_seq"]])
        out["frozen_any"] = bool(any(s["frozen"] for s in col["health_seq"]))
        out["collapse_any"] = bool(any(s["collapse_black"] or s["collapse_white"] for s in col["health_seq"]))
        out["dark_r_minus_b_seq"] = _mean([s.get("dark_r_minus_b") for s in col["health_seq"]])
        out["dark_r_minus_b_seq_real"] = _mean([s.get("dark_r_minus_b") for s in real_ref["health_seq"]])
        per_scen = {}
        for sc in SCENARIOS:
            rows = [s for s in col["health_seq"] if s["scenario"] == sc]
            if rows:
                per_scen[sc] = {"mean_luminance": _mean([r["mean_luminance"] for r in rows]),
                                "dark_share": _mean([r["dark_share"] for r in rows])}
        out["per_scenario"] = per_scen
    if col["motion_seq"]:
        # Headline motion keeps the longest-standing definition (the no-click scenario)
        # so the rows stay comparable across the night; the per-scenario breakdown is
        # reported alongside it.  In the no-click scenario ~0 horizontal content shift
        # is the *correct* behaviour (walking straight), so the scenario that matters
        # for "does the view turn when the player clicks" is the other two.
        first = SCENARIOS[0]
        head = [m for m in col["motion_seq"] if m["scenario"] == first] or col["motion_seq"]
        out["motion_px_per_step"] = _mean([m["motion_px_per_step"] for m in head])
        out["motion_px_per_step_real"] = _mean([m["motion_px_per_step"] for m in real_ref["motion_seq"]])
        out["motion_by_scenario"] = {
            sc: _mean([m["motion_px_per_step"] for m in col["motion_seq"]
                       if m["scenario"] == sc]) for sc in SCENARIOS
            if any(m["scenario"] == sc for m in col["motion_seq"])}
        out["motion_headline_scenario"] = first
    return out


def score(name, checkpoint, sets_path=SETS_PATH, parts="all", device="cuda",
          clips_key="main_clips", seeds=None, out_dir=EVAL_OUT, notes=None,
          parent_run=None, with_g1=True, split=None):
    import time

    from powerlog import energy_run

    sets = load_sets(sets_path)
    packed = PackedDataset(sets["packed"])
    model, payload = load_model(checkpoint, device)
    clips = sets[clips_key]
    if split and split != "val":
        clips = clips_from_split(packed, split, sets["rollout_steps"])
        with_g1 = False        # the frozen steering contexts are val-only
    seeds = seeds or sets["seeds"]
    det_path = os.path.join(EVAL_OUT, "opening_detector.pt")
    detector = OpeningDetector.load(det_path, device) if os.path.exists(det_path) else None
    if detector is None:
        raise SystemExit("no frozen opening detector — run --part detector first")

    logf = open(os.path.join(out_dir, f"score_{name}.log"), "a")
    logf.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} {name} {checkpoint}\n")
    logf.flush()

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"scoring {name}: {len(clips)} clips x {len(sets['scenarios'])} scenarios x {len(seeds)} seeds")

    rec = {"name": name, "checkpoint": os.path.abspath(checkpoint), "sets": sets_path,
           "clips_key": clips_key, "seeds": seeds, "parts": parts,
           "started": datetime.datetime.now().isoformat(timespec="seconds"),
           "notes": notes, "parent_run": parent_run,
           "model_config": {k: v for k, v in (payload.get("model_config") or {}).items()}}

    with energy_run(f"score_{name}", "eval", parent_run=parent_run,
                    notes=f"scorecard of {name}") as run:
        t0 = time.time()
        col = collect_batched(packed, model, sets, clips, seeds, sets["scenarios"],
                              detector=detector if parts in ("all", "D") else None,
                              device=device, progress=log)
        log(f"  rollouts done in {time.time() - t0:.0f}s")

        ref = real_reference(packed, sets, clips_override=clips)
        np.savez_compressed(os.path.join(out_dir, f"bank_{name}.npz"),
                            **{f"gen_{h}": np.stack(col["bank"][h]) for h in HORIZONS},
                            **{f"real_{h}": np.stack(ref["bank"][h]) for h in HORIZONS})

        def _part(key, fn):
            try:
                rec[key] = fn()
            except Exception as exc:      # one judge failing must not lose the run
                log(f"  !! part {key} failed: {type(exc).__name__}: {exc}")
                rec[key] = {"error": f"{type(exc).__name__}: {exc}"}

        # Per-seed metrics, then mean + seed-to-seed SD for every scalar, so the
        # brief's "a real improvement must exceed the noise" is checkable from the
        # record rather than asserted.
        _part("R", lambda: aggregate_seeds([
            realness(col["banks"][sd], ref["bank"], device=device, log=log if i == 0 else None)
            for i, sd in enumerate(seeds)]) if len(seeds) > 1
            else realness(col["bank"], ref["bank"], device=device, log=log))
        _part("D", lambda: aggregate_seeds([
            structure(col["banks"][sd], ref["bank"], detector, None, device=device,
                      log=log if i == 0 else None)
            for i, sd in enumerate(seeds)]) if len(seeds) > 1
            else structure(col["bank"], ref["bank"], detector, col, device=device, log=log))
        _part("H", lambda: aggregate_seeds([
            health({"bank": col["banks"][sd], "health_seq": col["health_seq"],
                    "motion_seq": col["motion_seq"], "opening_seq": col["opening_seq"]},
                   ref, log=log)
            for sd in seeds]) if len(seeds) > 1 else health(col, ref, log=log))
        if col["opening_seq"]:
            rec["D"]["events"] = {
                "gen_per_100": _mean([e["events_per_100"] for e in col["opening_seq"]]),
                "gen_positive_share": _mean([e["positive_share"] for e in col["opening_seq"]]),
                "per_scenario": {sc: _mean([e["events_per_100"] for e in col["opening_seq"]
                                            if e["scenario"] == sc]) for sc in SCENARIOS},
            }

        if with_g1 and parts in ("all", "S"):
            sidecar = load_analysis(packed)
            table = load_target_table(packed, sets["packed"])
            ctxs = [tuple(c) for c in sets["steering_contexts"]]
            wins = [tuple(c) for c in sets["steering_onset_windows"]]
            swins = [tuple(c) for c in sets["steering_sustained_windows"]]
            K, H = sets["K"], sets["H"]
            def _g1():
                g = run_g1(packed, model, sidecar, table, K, H, device=device, steps=40,
                           n_ctx=len(ctxs), seeds=seeds[:1], ctxs=ctxs, wins=wins,
                           swins=swins)
                c = g["counterfactual"][str(seeds[0])]
                log(f"  S: counterfactual_near={c['both_correct_directions_near']:.3f} "
                    f"shift_ratio={g['shift_ratio']['shift_ratio_h']:.3f} "
                    f"sustained_sign={g['sustained']['sign_agreement']:.3f} "
                    f"sustained_ratio={g['sustained']['median_ratio']:.3f}")
                return g
            _part("S", _g1)
        if parts in ("all", "T"):
            def _t():
                g4 = run_g4(model, packed, sets["K"], device=device, steps=2000, warmup=50)
                log(f"  T: {g4['fps_mean']:.1f} fps mean, peak VRAM {g4['peak_vram_mb']} MB")
                return g4
            _part("T", _t)
        rec["wall_s"] = round(time.time() - t0, 1)
        run.notes = f"scorecard of {name}"

    rec["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"score_{name}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"wrote {path}")
    logf.close()
    return rec


# --------------------------------------------------------------------------- #
# D, second half: one step at real doorway-reveal moments
# --------------------------------------------------------------------------- #

def reveal_windows(packed, detector, n=200, split="val", device="cuda", seed=0,
                   min_gap=4, log=None):
    """Real frames where the frozen detector fires after at least ``min_gap`` quiet ones.

    These are the moments the user is describing: something opening-like has just come
    into view.  The question the scorecard asks at these moments is not "does the 300th
    frame contain a doorway" but the easier and more diagnostic one: **when the world
    reveals an opening, does the model's next frame show it?**
    """
    eps = packed.episodes_by_split[split]
    out = []
    rng = np.random.RandomState(seed)
    order = list(range(len(eps)))
    rng.shuffle(order)
    for ei in order:
        rec = eps[ei]
        off, L = rec["frame_offset"], rec["length"]
        if L < 80:
            continue
        seq = np.asarray(packed.images[off: off + L])
        pos = detector.positive(seq, device)
        gap = min_gap
        for t in range(L):
            if pos[t] and gap >= min_gap and t >= 12 and t + 2 < L:
                out.append((rec["episode_index"], t))
                break              # one reveal per episode keeps the windows distinct
            gap = 0 if pos[t] else gap + 1
        if len(out) >= n:
            break
    if log:
        log(f"  reveal windows: {len(out)} from {len(eps)} val episodes")
    return out


@torch.no_grad()
def reveal_one_step(packed, models, sets, detector, windows, device="cuda", log=None):
    """Detector-positive rate on generated one-step outputs at real reveal moments.

    ``models`` is a list of (name, model).  For each window the *real* next frame and
    each model's one-step prediction from the same real context are scored by the
    frozen detector: the real rate is the ceiling, the Stage 4 baseline is the floor.
    """
    from dataset import frames_to_tensor
    from eval_harness import features_from_history

    K = sets["K"]
    res = {}
    real_frames = {name: [] for name, _ in models}
    real_true = []
    for e, t in windows:
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        ctx = frames_to_tensor(np.asarray(packed.images[off + t - K + 1: off + t + 1]))[None].to(device)
        real_true.append(np.asarray(packed.images[off + t + 1]))
        hist = list(np.nan_to_num(packed.action_vectors(
            packed.labels[off + max(0, t - K + 1 - 8): off + t + 1]), nan=0.0).astype(np.float32))
        feats = features_from_history(np.stack(hist), K, device)
        a = packed.action_vectors(packed.labels[off + t: off + t + K])
        a_t = torch.from_numpy(np.nan_to_num(a, nan=0.0)[None].astype(np.float32)).to(device)
        for name, m in models:
            out = model_predict(m, ctx, a_t, feats, 0)
            out = out[0] if isinstance(out, tuple) else out
            real_frames[name].append(_u8(out[0]))
    real_true = np.stack(real_true)
    rp = detector.positive(real_true, device)
    res["real_next_frame"] = {"positive_share": float(rp.mean()), "n": len(rp)}
    for name, _ in models:
        arr = np.stack(real_frames[name])
        gp = detector.positive(arr, device)
        res[name] = {"positive_share": float(gp.mean()),
                     "ratio_to_real": float(gp.mean() / max(1e-9, rp.mean())),
                     "agreement_with_real": float((gp == rp).mean()), "n": len(gp)}
        if log:
            log(f"  reveals: {name} positive {gp.mean():.3f} vs real {rp.mean():.3f}")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_detector(args):
    from judges import train_opening_detector

    sets = load_sets(args.sets)
    packed = PackedDataset(sets["packed"])
    sidecar = load_analysis(packed)
    os.makedirs(EVAL_OUT, exist_ok=True)
    det, report = train_opening_detector(
        packed, sidecar, steps=args.steps, device=args.device, seed=0,
        log=lambda m: print(m, flush=True),
        out_path=os.path.join(EVAL_OUT, "opening_detector.pt"))
    with open(os.path.join(EVAL_OUT, "opening_detector_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return 0


def _cmd_validate_judges(args):
    """Section 4.3: show the judges separate cases we know are different."""
    import torch.nn.functional as F

    sets = load_sets(args.sets)
    packed = PackedDataset(sets["packed"])
    det = OpeningDetector.load(os.path.join(EVAL_OUT, "opening_detector.pt"), args.device)
    K = sets["K"]
    clips = sets["main_clips"][:24]
    real = np.stack([np.asarray(packed.images[packed.episodes[c["episode"]]["frame_offset"]
                                                 + c["start"] + K - 1]) for c in clips])
    rng = np.random.RandomState(0)
    raw = real.astype(np.float32) / 255.0

    variants = {"real": real}
    variants["real+grain"] = (np.clip(raw + rng.normal(0, 0.08, raw.shape), 0, 1) * 255).astype(np.uint8)
    t = torch.from_numpy(raw).permute(0, 3, 1, 2)
    variants["real+blur"] = (_u8(F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1) * 2 - 1))
    variants["real+heavyblur"] = (_u8(F.avg_pool2d(F.pad(t, (3, 3, 3, 3), mode="replicate"), 7, 1) * 2 - 1))
    # "melted": a strong blur plus a lagged copy, the Stage 4 failure mode
    melted = 0.5 * t + 0.5 * torch.roll(t, 3, dims=0)
    variants["melted"] = _u8(F.avg_pool2d(F.pad(melted, (2, 2, 2, 2), mode="replicate"), 5, 1) * 2 - 1)

    # the real thing: the incumbent's own 300-step closed-loop frames
    model, _ = load_model(args.checkpoint or "runs/flow96/best.pt", args.device)
    gen = []
    for c in clips[:args.validate_clips]:
        seq, _ = _rollout_actions(packed, model, c["episode"], c["start"], K, 300,
                                  "no_click", 0, args.device)
        gen.append(seq[300])
    variants["generator@300"] = np.stack(gen)

    out = {}
    print("judge validation — each judge must separate cases we know differ")
    print(f"{'variant':<16} {'pos_share':>10} {'spec_d':>8} {'lap_ratio':>10} {'auc':>7}")
    for name, frames in variants.items():
        spec = spectrum_report(frames, real)
        lv = laplacian_var(frames) / max(1e-9, laplacian_var(real))
        pos = float(det.positive(frames, args.device).mean())
        auc = train_realfake(frames[: len(frames) // 2], real[: len(real) // 2],
                             frames[len(frames) // 2:], real[len(real) // 2:],
                             steps=600, device=args.device)["auc"]
        out[name] = {"positive_share": pos, "spectrum_distance": spec["spectrum_distance"],
                     "laplacian_ratio": lv, "auc": auc}
        print(f"{name:<16} {pos:>10.3f} {spec['spectrum_distance']:>8.3f} {lv:>10.3f} {auc:>7.3f}")
    with open(os.path.join(EVAL_OUT, "judge_validation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


def _cmd_reveals(args):
    """One-step detector test at real doorway-reveal moments, for several models."""
    from powerlog import energy_run

    sets = load_sets(args.sets)
    packed = PackedDataset(sets["packed"])
    detector = OpeningDetector.load(os.path.join(EVAL_OUT, "opening_detector.pt"),
                                    args.device)
    models = []
    for spec in (args.models or "").split(","):
        if not spec:
            continue
        name, path = spec.split("=", 1)
        models.append((name, load_model(path, args.device)[0]))
    if not models:
        raise SystemExit("pass --models name=path[,name=path]")
    wins = reveal_windows(packed, detector, n=args.n_reveals, device=args.device,
                          log=print)
    with energy_run(f"reveals_{args.name}", "eval",
                    notes=f"one-step doorway-reveal test: {args.models}") as run:
        res = reveal_one_step(packed, models, sets, detector, wins, device=args.device,
                              log=print)
        run.set_frames(len(wins))
    res["n_windows"] = len(wins)
    res["models"] = args.models
    path = os.path.join(EVAL_OUT, f"reveals_{args.name}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    print(f"wrote {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="sets",
                    choices=["sets", "detector", "validate", "score", "reveals"])
    ap.add_argument("--validate-clips", type=int, default=6)
    ap.add_argument("--name", default="v0")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sets", default=SETS_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--clips", default="main_clips", choices=["main_clips", "confirmation_clips"])
    ap.add_argument("--split", default=None, help="score a different split (e.g. test)")
    ap.add_argument("--seeds", default="", help="comma-separated; default = the frozen set")
    ap.add_argument("--no-g1", action="store_true")
    ap.add_argument("--models", default=None,
                    help="for --part reveals: name=checkpoint[,name=checkpoint]")
    ap.add_argument("--n-reveals", dest="n_reveals", type=int, default=200)
    ap.add_argument("--parent-run", default=None)
    ap.add_argument("--notes", default=None)
    args = ap.parse_args(argv)

    if args.part == "sets":
        build_sets(out=args.sets)
        return 0
    if args.part == "detector":
        return _cmd_detector(args)
    if args.part == "validate":
        return _cmd_validate_judges(args)
    if args.part == "reveals":
        return _cmd_reveals(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    score(args.name, args.checkpoint, sets_path=args.sets, clips_key=args.clips,
          seeds=seeds, device=args.device, parent_run=args.parent_run, notes=args.notes,
          with_g1=not args.no_g1, split=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
