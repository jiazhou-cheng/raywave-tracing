"""Compatibility wrapper for :mod:`src.optics.refractive`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.optics.refractive import *  # noqa: F401,F403
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optics.refractive import *  # noqa: F401,F403

