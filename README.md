# NeuralGameEngine

A first-person neural game engine styled after the Backrooms, fully implemented by DeepSeek and trained on a single NVIDIA 5070 laptop GPU.

Two halves to this project:

- **`engine/`** — the classical engine (TypeScript + three.js). It is the *teacher*: a
  deterministic, procedurally generated world with click-to-steer navigation, plus a
  headless capture mode that writes `(frames, click signals)` episodes to disk.
- **`neural/`** — the Python/PyTorch half. It packs those episodes into memory-mapped
  arrays, trains on them, evaluates candidates, and plays a trained model in a window.
  It only *reads* the engine's data; it never calls back into the engine.

## Repository layout

```
engine/          the classical engine and the dataset generator
  config/        config.ts — every tunable parameter lives here
  src/           core/ world/ control/ render/ sim/ app/ headless/
  tests/         vitest suite
  docs/          architecture notes, the dataset schema contract, the Stage 2 report
neural/          the Python/PyTorch half
  packer.py      episodes -> packed arrays; dataset.py loads them
  steer.py       the steering core: click signal -> structured view shift
  flow.py        the stochastic generator that renders the next frame
  train_*.py     training;  play.py / play.sh the player;  gates.py / scorecard.py evaluation
  models/        registry.json, the shipped flow96e checkpoint, the seed-context cache
  tests/         pytest suite
  docs/          one report per stage (STAGE3 .. STAGE6)
```

## Architecture, in one paragraph

The engine steps a fixed 20 Hz simulation: the camera always walks forward and a click
only sets a *held* X signal that steers toward a reachable doorway along a smooth path.
That makes the view motion predictable enough to split generation in two. `steer.py`
learns a small **structured** motion output — the per-step horizontal shift of the view —
and warps the last frame with it; `flow.py` learns to generate only the residual the warp
cannot explain, as a stochastic rectified flow conditioned on the click history.
`predict(context, action) -> next_frame` is the one interface every model implements, and
it is all the harness (`eval_harness.py`) and the player (`play.py`) know about.

## Run the model (the shipped `flow96e`)

```bash
cd neural
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
./play.sh --model flow96e
```

`play.sh` runs the model at the training rate (20 Hz), seeded from a real clip; click to
steer and pass `--help` for the flags. `flow96e` is the Stage 5c coherence pick. Its
checkpoint is self-contained — the flow generator embeds the steering core it was trained
on — so it plays with no training run and no dataset. (`--model` may be omitted: the
player then uses the newest revision whose checkpoint is present, which is `flow96e`
here.)

`models/registry.json` also lists the earlier revisions (`flow96`, `flow96c`, …) and their
lineage. That is provenance, not a dependency: only `flow96e`'s checkpoint is in this
repository, so the other names show up as `MISSING` in `./play.sh --list-models` and are
not playable here. See `neural/docs/STAGE5C-REPORT.md`.

## Run the training engine

```bash
cd engine
npm install
npm run dev            # prints a localhost URL — click to steer
```

Requires Node >= 20 and Google Chrome (Puppeteer drives it for headless capture); data
generation additionally needs a GPU with working WebGL.

Generating training data uses the same engine, plan-based and resumable:

```bash
cd engine
npm run build
npm run generate -- --make-plan pilot_v1 --train 120 --val 15 --test 15 --frames 600
npm run generate -- --plan engine/data/pilot_v1.plan.json --concurrency 8
```

## Train your own

```bash
cd neural
.venv/bin/python packer.py --src ../engine/data/pilot_v1 --out data/pilot_v1_64x48 \
    --width 64 --height 48
./memcap.sh --mem 14G -- .venv/bin/python train_steer.py --name steer64 \
    --packed data/pilot_v1_64x48
./memcap.sh --mem 14G -- .venv/bin/python train_flow.py --name flow64 \
    --packed data/pilot_v1_64x48 --core runs/steer64/best.pt
.venv/bin/python register_model.py --run flow64 --core runs/steer64/best.pt
./play.sh --model flow64
```

Everything a run writes lands in `neural/runs/<name>/` (gitignored), which `register_model.py`
freezes into `neural/models/<name>/generator.pt`.

## Tests

```bash
cd engine && npm test                        # vitest, self-contained
cd neural && .venv/bin/python -m pytest -q   # pytest
```

The engine suite is self-contained. The neural suite reads local, gitignored artifacts
(the packed sets under `neural/data/` and the checkpoints under `neural/runs/`), so it is
a developer's check that needs a dataset built first.

## What is not in this repository

Generated datasets, training runs, review and figure output, the reference paper and the
per-stage briefs are all derived or large and are gitignored rather than committed. Every
exclusion is documented in `.gitignore`.
