"""Run one mutant's targeted tests in a fresh process, with provenance proof.

Usage: run_one.py <mutant_dir> <target> [<target> ...]

Imports lot from the mutant's own src FIRST, prints the resolved lot.__file__,
and refuses to run the tests unless that path lies inside the mutant directory.
Because lot is already in sys.modules by the time pytest loads tests/conftest.py,
the conftest sys.path insertion cannot swap the package underneath the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

mutant = Path(sys.argv[1]).resolve()
targets = sys.argv[2:]

src = mutant / "src"
sys.path.insert(0, str(src))

import lot  # noqa: E402
import lot.geometry, lot.encoders, lot.transport, lot.visibility  # noqa: E402,F401
import lot.correspondence, lot.datasets, lot.evaluate  # noqa: E402,F401

resolved = Path(lot.__file__).resolve()
inside = mutant in resolved.parents
print(f"PROVENANCE lot.__file__ = {resolved} | inside mutant dir = {inside}")
if not inside:
    print("VOID: the loaded lot package is not the mutant's copy", file=sys.stderr)
    raise SystemExit(2)

import pytest  # noqa: E402

raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", *targets]))
