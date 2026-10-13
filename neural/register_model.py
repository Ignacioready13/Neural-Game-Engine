#!/usr/bin/env python
"""Register a new model revision for the player (see model_registry.py).

**Models are named after their internal training run** (``flow96``, ``flow96c``,
``flow96e``, …), which is also the name the player shows. Older letters such as ``5a`` /
``5b`` are kept as aliases only.

Examples::

    # show what is registered, which are selected, and where the checkpoints are
    .venv/bin/python register_model.py --list

    # register the next revision straight from its run directory
    .venv/bin/python register_model.py --run flow96e \
        --core runs/steer96/best.pt --description "flow96e: longer self-rollout" \
        --parent flow96c

    # register something that is not an official revision (kept out of the M cycle)
    .venv/bin/python register_model.py --run flow96d --core runs/steer96/best.pt \
        --description "flow96d: rejected edge-weighted candidate" --not-selected

Nothing here trains anything and no training directory is modified; the generator
checkpoint is only read (and copied into ``models/<name>/generator.pt``).
"""

from __future__ import annotations

import argparse
import os
import sys

import model_registry as reg


def _parse_res(spec: str) -> tuple[int, int]:
    try:
        w, h = spec.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"expected WxH (e.g. 96x72), got {spec!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Register / list player model revisions (named after their "
                    "internal training runs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list registered models and exit")
    ap.add_argument("--name", help="the model name, i.e. its internal run name "
                                   "(e.g. flow96e)")
    ap.add_argument("--run", help="shorthand for --name NAME --generator "
                                  "runs/NAME/best.pt")
    ap.add_argument("--generator", help="path to the generator checkpoint to freeze")
    ap.add_argument("--core", default=None,
                    help="optional steering-core checkpoint (provenance)")
    ap.add_argument("--description", default="", help="one-line human description")
    ap.add_argument("--alias", dest="aliases", action="append", default=None,
                    help="an extra name that also resolves (e.g. 5c); repeatable")
    ap.add_argument("--parent", default=None, help="the model this one derives from")
    ap.add_argument("--resolution", type=_parse_res, default=(96, 72),
                    help="training resolution WxH")
    ap.add_argument("--packed", default=None,
                    help="packed dataset used for seed contexts (default: the shared one)")
    ap.add_argument("--eval", dest="evals", action="append", default=None,
                    help="an evaluation record for this model (repeatable)")
    ap.add_argument("--notes", default="", help="free-form notes")
    ap.add_argument("--not-selected", dest="selected", action="store_false",
                    help="a candidate that was rejected/not promoted: excluded from the "
                         "M cycle and from the default, still playable by name")
    ap.add_argument("--no-copy", action="store_true",
                    help="reference the checkpoint in place instead of freezing a copy")
    ap.add_argument("--force", action="store_true", help="replace an existing entry")
    ap.set_defaults(selected=True)
    args = ap.parse_args(argv)

    if args.list:
        print(reg.format_table())
        return 0

    name = args.name or args.run
    if not name:
        ap.print_help()
        return 0
    generator = args.generator or (reg.run_checkpoint(args.run or name)
                                   if args.run else None)
    if not generator:
        ap.error("--generator is required (or use --run NAME to imply runs/NAME/best.pt)")
    try:
        entry = reg.register(
            name, generator, core=args.core, description=args.description,
            parent=args.parent, width=args.resolution[0], height=args.resolution[1],
            packed=args.packed, evals=args.evals, notes=args.notes,
            aliases=args.aliases, selected=args.selected,
            copy=not args.no_copy, force=args.force)
    except reg.RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cp = entry["checkpoint"] or entry["checkpoint_source"]
    print(f"registered {entry['name']}"
          + (f" (alias {', '.join(entry['aliases'])})" if entry.get("aliases") else "")
          + f": {entry['description']}")
    print(f"  checkpoint : {cp}  (sha256 {entry['checkpoint_sha256'][:12]}...)")
    if entry.get("core_copy") or entry.get("core"):
        print(f"  core       : {entry.get('core_copy') or entry['core']}")
    print(f"  resolution : {entry['resolution'][0]}x{entry['resolution'][1]}")
    print(f"  selected   : {entry['selected']}"
          + ("" if entry["selected"] else "  (kept out of the M cycle, playable by name)"))
    print("  play it    : ./play.sh --model " + entry["name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
