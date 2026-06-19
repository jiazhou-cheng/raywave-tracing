"""Compatibility wrapper for :mod:`src.common`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.common import *  # noqa: F401,F403
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.common import *  # noqa: F401,F403
