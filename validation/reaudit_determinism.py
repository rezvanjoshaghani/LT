"""Re-audit 4.3: evaluate one synthetic scene twice; outputs must be identical.

Drives the real pipeline (probe scene + lot.evaluate) twice in one process and
once more in a fresh scene directory, then compares rows field by field. NaN is
compared as equal to NaN (the permitted centered Mean-Feature representation).
Also re-verifies the overwrite refusal of write_rows.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "validation"))

import probe_scene  # noqa: E402


def rows_for(root: Path) -> list[dict]:
    probe_scene.build_scene(root, feature_mode="surface")
    return probe_scene.evaluate_probe(root)


def canon(rows: list[dict]) -> list[tuple]:
    out = []
    for r in sorted(rows, key=lambda r: (r["context_frame_id"], r["target_frame_id"],
                                         r["encoder"], r["path"], r["variant"])):
        vals = []
        for k in sorted(r):
            v = r[k]
            if isinstance(v, float) and math.isnan(v):
                v = "NaN"
            vals.append((k, v))
        out.append(tuple(vals))
    return out


def main() -> None:
    root_a = Path(tempfile.mkdtemp(prefix="lot_det_a_"))
    rows_a = rows_for(root_a)
    rows_b = probe_scene.evaluate_probe(root_a)      # same inputs, second call
    root_c = Path(tempfile.mkdtemp(prefix="lot_det_c_"))
    rows_c = rows_for(root_c)                         # fresh directory build

    a, b, c = canon(rows_a), canon(rows_b), canon(rows_c)
    print(f"rows: {len(a)} / {len(b)} / {len(c)}")
    print(f"same-input repeat identical:   {a == b}")
    print(f"fresh-rebuild identical:       {a == c}")

    # Overwrite refusal, live.
    from lot.evaluate import write_rows
    out = root_a / "once.parquet"
    write_rows(out, rows_a[:5], {"eval_version": 4, "scene": "probe"})
    try:
        write_rows(out, rows_a[:5], {"eval_version": 4, "scene": "probe"})
        print("overwrite refusal: FAIL (second write succeeded)")
    except FileExistsError:
        print("overwrite refusal: PASS (FileExistsError raised)")

    ok = a == b == c
    print(f"\n4.3 determinism verdict: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
