"""The model registry: named model revisions for the player.

**Naming convention (the user's preference): a model is named after its internal
training run** — ``flow96``, ``flow96c``, ``flow96d``, and whatever the next
``runs/<name>/`` is called. The Stage 5-era letters (``5a`` = the Stage 5 generator
``flow96``, ``5b`` = the Stage 5b winner ``flow96c``) survive only as *aliases*, so older
notes and commands still resolve, but everything the player shows uses the internal name.

Each name maps to a copy of its generator checkpoint (frozen the moment it is
registered, so a later training run that reuses ``runs/<name>/`` cannot change what the
name means) plus the lineage and evaluation records that let a reader trace where the
weights came from. The registry is a small JSON file (``models/registry.json``); the
checkpoint copies live beside it under ``models/``. :mod:`register_model` writes both;
:mod:`play` reads them. Nothing here imports torch, so the registry can be listed cheaply.

Entries also carry a ``selected`` flag. "Selected" means the model passed the project's
selection step and is an official revision you would compare against:

* ``flow96``  (alias ``5a``) — the Stage 5 incumbent, kept.
* ``flow96c`` (alias ``5b``) — the Stage 5b overnight winner, kept (the default).
* ``flow96d`` — **not** selected: the frozen scorecard rejected it
  (``eval_out/decision_flow96c_vs_flow96d.json``, ``keep: false``).

The player's ``M`` key cycles only over selected models, but any registered model — and
any ``runs/<name>/best.pt`` that is not registered at all — can be played by naming it
explicitly, so a rejected candidate can still be inspected by hand.

Layout::

    neural/models/
      registry.json
      _shared/steer96.pt          # the frozen steering core the flow96 family uses
      flow96/generator.pt         # copy of runs/flow96/best.pt
      flow96c/generator.pt        # copy of runs/flow96c/best.pt
      flow96d/generator.pt        # copy of runs/flow96d/best.pt
      seeds.npz                   # cached seed contexts (see build_seeds.py)
      recordings/                 # recorded sessions (.json)

The generator checkpoints are self-contained (the flow model embeds its steering core),
so ``core`` is provenance, not a load dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
DEFAULT_SEEDS = os.path.join(MODELS_DIR, "seeds.npz")
RECORDINGS_DIR = os.path.join(MODELS_DIR, "recordings")
RUNS_DIR = os.path.join(HERE, "runs")

NAME_RE = re.compile(r"^[a-z][a-z0-9]{0,23}$")

# The registry's own schema version; bump only on an incompatible change.
SCHEMA_VERSION = 2


class RegistryError(Exception):
    """Anything the user can fix by hand or by re-running register_model.py."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def resolve(path: str, root: str = HERE) -> str:
    """A registry path (relative to /neural) as an absolute path."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(root, path)


def rel(path: str, root: str = HERE) -> str:
    """Store paths relative to /neural so the registry is portable."""
    ap = os.path.abspath(path)
    try:
        return os.path.relpath(ap, root)
    except ValueError:      # different drive (Windows only)
        return ap


def run_checkpoint(name: str) -> str:
    """The conventional checkpoint for an internal run name, registered or not."""
    return os.path.join(RUNS_DIR, name, "best.pt")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #

def load(path: str | None = None) -> dict:
    """Read the registry. A missing file is an empty registry, not an error."""
    p = path or REGISTRY_PATH
    if not os.path.exists(p):
        return {"version": SCHEMA_VERSION, "models": []}
    with open(p, "r", encoding="utf-8") as f:
        reg = json.load(f)
    if "models" not in reg:
        raise RegistryError(f"{p} has no 'models' list")
    reg.setdefault("version", SCHEMA_VERSION)
    for m in reg["models"]:
        m.setdefault("aliases", [])
        if "selected" not in m:
            # a pre-Stage-6 registry used `hidden`; treat that as "not selected"
            m["selected"] = not m.get("hidden", False)
    return reg


def save(reg: dict, path: str | None = None) -> str:
    p = path or REGISTRY_PATH
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1, sort_keys=False)
        f.write("\n")
    return p


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #

def entries(reg: dict | None = None, selected_only: bool = False) -> list[dict]:
    reg = reg if reg is not None else load()
    out = reg["models"]
    return [m for m in out if m.get("selected")] if selected_only else out


def names(reg: dict | None = None, selected_only: bool = False) -> list[str]:
    """Registered names in registration order (the order ``M`` cycles in)."""
    return [m["name"] for m in entries(reg, selected_only)]


def _matches(entry: dict, name: str) -> bool:
    return name == entry["name"] or name in (entry.get("aliases") or [])


def find(name: str, reg: dict | None = None) -> dict | None:
    """The entry for ``name``, matching the internal name or any alias."""
    reg = reg if reg is not None else load()
    for m in reg["models"]:
        if _matches(m, name):
            return m
    return None


def get(name: str, reg: dict | None = None) -> dict:
    """Like :func:`find`, but raises with the list of known names."""
    entry = find(name, reg)
    if entry is None:
        reg = reg if reg is not None else load()
        known = ", ".join(names(reg)) or "(none registered)"
        raise RegistryError(f"unknown model {name!r}; registered models: {known}")
    return entry


def canonical(name: str, reg: dict | None = None) -> str:
    """Resolve an alias (``5b``) to the internal name (``flow96c``)."""
    entry = find(name, reg)
    return entry["name"] if entry else name


def newest(reg: dict | None = None) -> dict:
    """The most recently registered selected model — the default when no flag is given."""
    visible = entries(reg, selected_only=True)
    if not visible:
        raise RegistryError(
            "no selected models are registered; run "
            "`.venv/bin/python register_model.py --list` for details, or register one "
            "with `--name flow96c --generator runs/flow96c/best.pt ...`")
    return visible[-1]


def checkpoint_for(entry: dict, root: str = HERE) -> tuple[str, str]:
    """``(path, kind)`` for an entry's generator: the frozen copy if present, else the source.

    Returns ``kind`` in {"copy", "source"} so callers can tell whether the name is still
    pinned to a frozen file or is falling back to a live training directory.
    """
    copy = resolve(entry.get("checkpoint", ""), root)
    if copy and os.path.exists(copy):
        return copy, "copy"
    src = resolve(entry.get("checkpoint_source", ""), root)
    if src and os.path.exists(src):
        return src, "source"
    raise RegistryError(
        f"model {entry['name']!r}: no checkpoint found (looked for "
        f"{rel(copy, root) if copy else '<unset>'} and "
        f"{rel(src, root) if src else '<unset>'}); re-copy it with register_model.py")


def resolution(entry: dict) -> tuple[int, int]:
    r = entry.get("resolution") or [96, 72]
    return int(r[0]), int(r[1])


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #

def register(name: str, generator: str, *, core: str | None = None,
             description: str = "", parent: str | None = None,
             width: int = 96, height: int = 72, packed: str | None = None,
             evals: list[str] | None = None, notes: str = "",
             aliases: list[str] | None = None, selected: bool = True,
             copy: bool = True, force: bool = False,
             root: str = HERE, path: str | None = None) -> dict:
    """Register a model under its internal run name (freezing a copy of its checkpoint)."""
    if not NAME_RE.match(name):
        raise RegistryError(
            f"model name {name!r} is not a valid name (expected like 'flow96c', 'flow96e')")
    reg = load(path)
    existing = find(name, reg)
    if existing is not None and not force:
        raise RegistryError(
            f"model {name!r} is already registered (pass --force to replace)")
    src = resolve(generator, root)
    if not os.path.exists(src):
        raise RegistryError(f"generator checkpoint not found: {generator}")
    parent_name = canonical(parent, reg) if parent else None

    alias_list = []
    for a in (aliases or []):
        if a == name:
            continue
        holder = find(a, reg)
        if holder is not None and holder["name"] != name:
            if not force:
                raise RegistryError(
                    f"alias {a!r} already belongs to {holder['name']} "
                    f"(pass --force to take it over)")
            holder["aliases"] = [x for x in holder.get("aliases", []) if x != a]
        if a not in alias_list:
            alias_list.append(a)
    if existing is not None and force:
        # replacing an entry: keep any aliases the caller did not restate
        for a in existing.get("aliases", []):
            if a not in alias_list:
                alias_list.append(a)

    entry = {
        "name": name,
        "aliases": alias_list,
        "description": description or f"generator {rel(src, root)}",
        "registered": date.today().isoformat(),
        "selected": bool(selected),
        "parent": parent_name,
        "checkpoint_source": rel(src, root),
        "checkpoint": None,
        "checkpoint_sha256": sha256(src),
        "core": rel(resolve(core, root), root) if core else None,
        "resolution": [int(width), int(height)],
        "seed_packed": rel(resolve(packed, root), root) if packed else None,
        "evals": list(evals or []),
        "notes": notes,
    }
    if copy:
        dst = os.path.join(MODELS_DIR, name, "generator.pt")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(dst) != os.path.abspath(src):
            shutil.copy2(src, dst)
        entry["checkpoint"] = rel(dst, root)
    if core:
        csrc = resolve(core, root)
        if os.path.exists(csrc):
            base = os.path.basename(csrc)
            if base in ("best.pt", "last.pt"):
                base = os.path.basename(os.path.dirname(csrc)) + ".pt"
            shared = os.path.join(MODELS_DIR, "_shared", base)
            os.makedirs(os.path.dirname(shared), exist_ok=True)
            if os.path.abspath(shared) != os.path.abspath(csrc):
                shutil.copy2(csrc, shared)
            entry["core_copy"] = rel(shared, root)

    reg["models"] = [m for m in reg["models"] if m["name"] != name] + [entry]
    save(reg, path)
    return entry


def format_table(reg: dict | None = None, root: str = HERE) -> str:
    """A human-readable table of registered models (newest last)."""
    reg = reg if reg is not None else load()
    rows = entries(reg, selected_only=False)
    if not rows:
        return "no models registered yet"
    w = max(len(r["name"]) for r in rows)
    a = max([len(", ".join(r.get("aliases") or [])) for r in rows] + [0])
    out = [f"{'name'.ljust(w)}  {'aliases'.ljust(a)}  res    status    description"]
    for r in rows:
        try:
            _, kind = checkpoint_for(r, root)
            status = "ready" if kind == "copy" else "source!"
        except RegistryError:
            status = "MISSING"
        if not r.get("selected") and status != "MISSING":
            status += "+"
        out.append(f"{r['name'].ljust(w)}  "
                   f"{', '.join(r.get('aliases') or []).ljust(a)}  "
                   f"{resolution(r)[0]}x{resolution(r)[1]}  {status:<9} "
                   f"{r.get('description', '')}")
    out.append("")
    out.append("selected (the M key cycles these, and one is the default): "
               + ", ".join(names(reg, selected_only=True)))
    out.append("a + marks a model that was rejected/not promoted; it is still playable "
               "with --model <name>")
    return "\n".join(out)
