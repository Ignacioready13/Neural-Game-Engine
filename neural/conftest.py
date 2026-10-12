import os
import sys

# Make the flat modules (packer.py, dataset.py, ...) importable from /neural.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pathlib

import pytest

from dataset import PackedDataset

ROOT = pathlib.Path(__file__).resolve().parent
# Defaults point at the Stage 2 pilot; env overrides let the same tests run against
# another packed set (Stage 4B uses them for bulk_v1). Defaults are unchanged.
SRC = pathlib.Path(os.environ.get("LNE_SRC", ROOT / ".." / "engine" / "data" / "pilot_v1"))
PACKED_96 = pathlib.Path(os.environ.get("LNE_PACKED_96", ROOT / "data" / "pilot_v1_96x72"))
PACKED_64 = pathlib.Path(os.environ.get("LNE_PACKED_64", ROOT / "data" / "pilot_v1_64x48"))


@pytest.fixture(scope="session")
def src():
    return SRC


@pytest.fixture(scope="session")
def packed96():
    return PackedDataset(str(PACKED_96))


@pytest.fixture(scope="session")
def packed64():
    return PackedDataset(str(PACKED_64))
