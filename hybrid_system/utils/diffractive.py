"""Compatibility wrapper for :mod:`src.optics.diffractive`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src.optics.diffractive import *  # noqa: F401,F403
    from src.optics.diffractive import _phase_to_complex_field  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optics.diffractive import *  # noqa: F401,F403
    from src.optics.diffractive import _phase_to_complex_field  # noqa: F401
