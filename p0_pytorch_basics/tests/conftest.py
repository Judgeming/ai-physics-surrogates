"""Tests import your exercise files by default; P0_SOLUTIONS=1 runs them against the reference solutions."""

import importlib
import os

import pytest


@pytest.fixture
def load():
    base = "p0_pytorch_basics.solutions" if os.environ.get("P0_SOLUTIONS") == "1" else "p0_pytorch_basics"
    return lambda name: importlib.import_module(f"{base}.{name}")
