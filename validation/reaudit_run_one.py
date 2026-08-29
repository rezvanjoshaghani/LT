"""Run one mutant's test suite in this fresh process, with import provenance.

Usage: python reaudit_run_one.py <mutant_dir>
Inserts <mutant_dir>/src at sys.path[0], imports lot, refuses to proceed unless
lot.__file__ resolves inside the mutant directory, then runs pytest on the
mutant's tests. Exit code is pytest's.
"""

import sys
from pathlib import Path

mutant = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(mutant / "src"))
import lot  # noqa: E402

resolved = Path(lot.__file__).resolve()
print(f"provenance: lot.__file__ = {resolved}")
if mutant not in resolved.parents:
    print("REFUSING: the imported lot package is not the mutant's copy")
    sys.exit(97)

import pytest  # noqa: E402

sys.exit(pytest.main([str(mutant / "tests"), "-q", "--no-header", "--maxfail=25", "-rf"]))
