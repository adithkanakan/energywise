"""Pytest configuration: make ``src/`` importable as the ``energywise`` package.

The project keeps its library code under ``src/`` (a src-layout), while the
tests import it as ``energywise``.  Adding ``src`` to ``sys.path`` here makes
that work without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))