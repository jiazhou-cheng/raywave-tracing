"""Compatibility wrapper for :mod:`src.loss`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.loss import *  # noqa: F401,F403
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.loss import *  # noqa: F401,F403
