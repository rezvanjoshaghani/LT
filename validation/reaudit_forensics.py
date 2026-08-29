"""Re-audit Part 4: output forensics on the Stream D evaluation parquet.

Independent of lot: reads parquet via pyarrow/pandas, parses configs/analysis.yaml
directly, and re-implements binning and accounting from the frozen protocol text.
Evidence goes to validation/evidence/reaudit/forensics.json (+ printed report).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "experiment_zero" / "eval"
OUT = ROOT / "validation" / "evidence" / "reaudit"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ("Oracle-Transport", "No-Warp-Copy", "Neighbor-Patch", "Random-Patch", "Mean-Feature")
PATHS = ("per_point", "splat_pool")
SCORE_COLS = ("cosine_mean", "l2_mean", "cosine_centered_mean", "l2_centered_mean")
INTERSECT_COLS = ("cosine_intersect_mean", "cosine_centered_intersect_mean")
KEY = ["scene", "context_frame_id", "target_frame_id", "encoder", "path", "variant"]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "analysis.yaml").read_text())


def load_all() -> tuple[pd.DataFrame, dict[str, dict]]:
    frames, metas = [], {}
    for p in sorted(EVAL_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(p))
        metas[p.stem] = json.loads((EVAL_DIR / f"{p.stem}.meta.json").read_text())
    return pd.concat(frames, ignore_index=True), metas


def popcount_bits(blob: bytes, size: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))[:size]
    return bits.astype(bool)


def bin_label(value: float, edges: list[float], zero_tol: float) -> str:
    """Right-closed binning with a zero bin, from the frozen config text alone."""
    if value < zero_tol:
        return "zero"
    lower = 0.0
    for edge in edges:
        if value <= edge:
            return f"{lower:g}-{edge:g}"
        lower = edge
    return f"{lower:g}+"


def main() -> None:
    cfg = load_config()
    df, metas = load_all()
    report: dict = {"n_rows_total": int(len(df))}
    problems: list[str] = []

    # ---- 4.1 record accounting: expected from run-record counters ----------
    expected_rows = 0
    expected_comparisons_pair_encoder_path = 0
    expected_cross_path_comparisons = 0
    for scene, m in metas.items():
        both, pp, sp = (
            m["pairs_scored_both_paths"],
            m["pairs_scored_per_point_only"],
            m["pairs_scored_splat_only"],
        )
        enc = len(m["encoders"])
        expected_rows += enc * (10 * both + 5 * pp + 5 * sp)
        expected_comparisons_pair_encoder_path += enc * (2 * both + pp + sp)
        expected_cross_path_comparisons += enc * both
    observed_rows = len(df)
    report["accounting"] = {
        "expected_rows_from_run_records": expected_rows,
        "observed_rows": observed_rows,
        "rows_match": expected_rows == observed_rows,
        "expected_comparisons_pair_encoder_path": expected_comparisons_pair_encoder_path,
        "expected_cross_path_pair_encoder": expected_cross_path_comparisons,
        "pairs_considered_total": sum(m["pairs_considered"] for m in metas.values()),
        "pairs_dropped_unscorable_total": sum(m["pairs_dropped_unscorable"] for m in metas.values()),
        "neighbor_omitted_total_meta": sum(
            m["neighbor_patch_omitted_records"] for m in metas.values()
        ),
    }
    if expected_rows != observed_rows:
        problems.append(f"4.1 row count mismatch: expected {expected_rows}, got {observed_rows}")

    # Per-identity reconciliation: rebuild the expected row identity set from the
    # observed (pair, encoder, path) groups and confirm every group carries all 5
    # variants exactly once, then confirm group counts equal the meta counters.
    grp = df.groupby(["scene", "context_frame_id", "target_frame_id", "encoder", "path"])
    sizes = grp.size()
    report["accounting"]["groups_pair_encoder_path"] = int(len(sizes))
    report["accounting"]["groups_with_exactly_5_variants"] = int((sizes == 5).sum())
    if not bool((sizes == 5).all()):
        problems.append("4.1 some (pair,encoder,path) groups do not carry exactly 5 variants")
    want = tuple(sorted(VARIANTS))
    bad_variant_groups = int(
        (~grp["variant"].agg(lambda s: tuple(sorted(s)) == want)).sum()
    )
    if bad_variant_groups:
        problems.append(f"4.1 variant sets differ from the five frozen ones in {bad_variant_groups} groups")

    per_scene_counter_match = {}
    for scene, m in metas.items():
        sub = df[df.scene == scene]
        for enc in m["encoders"]:
            se = sub[sub.encoder == enc]
            paths_by_pair = se.groupby(["context_frame_id", "target_frame_id"])["path"].agg(set)
            found = {
                "both": int(sum(1 for s in paths_by_pair if len(s) == 2)),
                "pp": int(sum(1 for s in paths_by_pair if s == {"per_point"})),
                "sp": int(sum(1 for s in paths_by_pair if s == {"splat_pool"})),
            }
            ok = (
                found["both"] == m["pairs_scored_both_paths"]
                and found["pp"] == m["pairs_scored_per_point_only"]
                and found["sp"] == m["pairs_scored_splat_only"]
            )
            per_scene_counter_match[f"{scene}/{enc}"] = ok
            if not ok:
                problems.append(f"4.1 {scene}/{enc}: rows {found} != run-record counters")
    report["accounting"]["per_scene_counter_match_all"] = all(per_scene_counter_match.values())

    # ---- 4.3 hygiene: grain uniqueness ------------------------------------
    dupes = df.duplicated(subset=KEY).sum()
    report["hygiene"] = {"grain": KEY, "duplicate_rows_at_grain": int(dupes)}
    if dupes:
        problems.append(f"4.3 {dupes} duplicate rows at the frozen grain")

    # ---- 4.3 nonfinite audit ----------------------------------------------
    nonfinite: dict[str, dict[str, int]] = {}
    for col in SCORE_COLS + INTERSECT_COLS + ("coverage_mean",):
        bad = df[~np.isfinite(df[col].to_numpy())]
        nonfinite[col] = {
            f"{v}/{p}": int(((bad.variant == v) & (bad.path == p)).sum())
            for v in VARIANTS
            for p in PATHS
            if ((bad.variant == v) & (bad.path == p)).any()
        }
    report["hygiene"]["nonfinite_by_column"] = nonfinite
    # The single permitted representation: centered columns of Mean-Feature rows.
    for col in ("cosine_centered_mean", "l2_centered_mean"):
        others = df[(df.variant != "Mean-Feature") & ~np.isfinite(df[col].to_numpy())]
        if len(others):
            problems.append(f"4.3 {col}: {len(others)} nonfinite values outside Mean-Feature")
        mf = df[df.variant == "Mean-Feature"]
        if int(np.isfinite(mf[col].to_numpy()).sum()):
            problems.append(f"4.3 {col}: FINITE centered Mean-Feature values present")
    for col in ("cosine_mean", "l2_mean"):
        if int((~np.isfinite(df[col].to_numpy())).sum()):
            problems.append(f"4.3 {col} carries nonfinite values")
    # Representation consistency: Mean-Feature rows PRESENT with NaN, never absent.
    mf_rows = df[df.variant == "Mean-Feature"]
    report["hygiene"]["mean_feature_rows"] = int(len(mf_rows))
    report["hygiene"]["mean_feature_centered_all_nan"] = bool(
        (~np.isfinite(mf_rows["cosine_centered_mean"].to_numpy())).all()
        and (~np.isfinite(mf_rows["l2_centered_mean"].to_numpy())).all()
    )
    # Intersect columns: nonfinite permitted only for Mean-Feature (centered) or
    # empty intersections (n_intersect == 0). Count anything else.
    inter_bad = df[
        (df.n_intersect > 0)
        & (df.variant != "Mean-Feature")
        & ~np.isfinite(df["cosine_intersect_mean"].to_numpy())
    ]
    report["hygiene"]["nonfinite_raw_intersect_with_support"] = int(len(inter_bad))
    if len(inter_bad):
        problems.append(
            f"4.3 {len(inter_bad)} rows have n_intersect>0 but nonfinite raw intersect score"
        )
    report["hygiene"]["rows_with_empty_intersection"] = int((df.n_intersect == 0).sum())
    # coverage_mean: expected NaN on per_point rows (not a score); finite on splat.
    report["hygiene"]["coverage_nan_per_point_rows"] = int(
        (~np.isfinite(df[df.path == "per_point"]["coverage_mean"].to_numpy())).sum()
    )
    report["hygiene"]["coverage_nonfinite_splat_rows"] = int(
        (~np.isfinite(df[df.path == "splat_pool"]["coverage_mean"].to_numpy())).sum()
    )

    # ---- mask-level internal consistency (3.2 pairing identity) -----------
    # For every (pair, encoder): all five variants of a path share one mask;
    # popcount(mask) == n; popcount(pp & sp) == n_intersect on every row;
    # coverage_difference == popcount(own & ~other).
    mask_checks = {"mask_same_within_path": 0, "popcount_eq_n": 0, "n_intersect_ok": 0,
                   "coverage_difference_ok": 0, "violations": []}
    universe = {s: m["universe_size"] for s, m in metas.items()}
    for (scene, cfid, tfid, enc), sub in df.groupby(
        ["scene", "context_frame_id", "target_frame_id", "encoder"]
    ):
        size = universe[scene]
        masks = {}
        for path in PATHS:
            rows = sub[sub.path == path]
            if not len(rows):
                continue
            blobs = set(rows["sample_mask"])
            if len(blobs) != 1:
                mask_checks["violations"].append(f"{scene}/{cfid}->{tfid}/{enc}/{path}: masks differ across variants")
                continue
            mask_checks["mask_same_within_path"] += 1
            mask = popcount_bits(next(iter(blobs)), size)
            masks[path] = mask
            n = rows["n"].iloc[0]
            if int(mask.sum()) == int(n):
                mask_checks["popcount_eq_n"] += 1
            else:
                mask_checks["violations"].append(
                    f"{scene}/{cfid}->{tfid}/{enc}/{path}: popcount {int(mask.sum())} != n {int(n)}"
                )
        if len(masks) == 2:
            inter = int((masks["per_point"] & masks["splat_pool"]).sum())
            if (sub["n_intersect"] == inter).all():
                mask_checks["n_intersect_ok"] += 1
            else:
                mask_checks["violations"].append(f"{scene}/{cfid}->{tfid}/{enc}: n_intersect != popcount(and)")
            cd_pp = int((masks["per_point"] & ~masks["splat_pool"]).sum())
            cd_sp = int((masks["splat_pool"] & ~masks["per_point"]).sum())
            ok = ((sub[sub.path == "per_point"]["coverage_difference"] == cd_pp).all()
                  and (sub[sub.path == "splat_pool"]["coverage_difference"] == cd_sp).all())
            if ok:
                mask_checks["coverage_difference_ok"] += 1
            else:
                mask_checks["violations"].append(f"{scene}/{cfid}->{tfid}/{enc}: coverage_difference mismatch")
    mask_checks["n_violations"] = len(mask_checks["violations"])
    mask_checks["violations"] = mask_checks["violations"][:10]
    report["mask_consistency"] = mask_checks
    if mask_checks["n_violations"]:
        problems.append(f"3.2 mask consistency: {mask_checks['n_violations']} violations")

    # ---- 4.2 population checks --------------------------------------------
    pairs = df[KEY[:3] + ["encoder", "regime", "baseline_m", "rotation_deg", "parallax",
                          "context_median_depth_m", "viewpoint"]].drop_duplicates(
        subset=["scene", "context_frame_id", "target_frame_id"]
    )
    pop: dict = {}
    pop["pairs_distinct"] = int(len(pairs))
    pop["by_regime"] = pairs.groupby("regime").size().to_dict()
    rot = pairs[pairs.regime == "rotation"]
    tra = pairs[pairs.regime == "translation"]
    orb = pairs[pairs.regime == "orbit"]
    pop["rotation_max_baseline_m"] = float(rot.baseline_m.max())
    pop["rotation_max_parallax"] = float(rot.parallax.max())
    pop["rotation_all_parallax_below_zero_tol"] = bool(
        (rot.parallax < cfg["zero_parallax_tol"]).all()
    )
    pop["translation_max_rotation_deg"] = float(tra.rotation_deg.max())
    pop["translation_all_rotation_below_zero_tol"] = bool(
        (tra.rotation_deg < cfg["zero_rotation_tol_deg"]).all()
    )
    pop["translation_rotation_bound_deg_config"] = cfg["translation_rotation_bound_deg"]
    pop["translation_pairs_over_manifest_bound"] = int(
        (tra.rotation_deg > cfg["translation_rotation_bound_deg"]).sum()
    )
    floor = cfg["translation_parallax_design_floor"]
    in_forbidden = tra[(tra.parallax >= cfg["zero_parallax_tol"]) & (tra.parallax < floor)]
    pop["translation_pairs_in_forbidden_interval"] = int(len(in_forbidden))
    if len(in_forbidden):
        problems.append(f"4.2 {len(in_forbidden)} translation pairs inside (0, {floor})")
    pop["orbit_pairs_in_(0,0.025)"] = int(
        len(orb[(orb.parallax >= cfg["zero_parallax_tol"]) & (orb.parallax < floor)])
    )
    if not pop["rotation_all_parallax_below_zero_tol"]:
        problems.append("4.2 rotation pairs above the zero-parallax tolerance")
    if not pop["translation_all_rotation_below_zero_tol"]:
        problems.append("4.2 translation pairs above the zero-rotation tolerance")

    # Stratum-cap check: rebuild the sampling stratum from the whole-frame proxy
    # (baseline / context_median_depth_m) and the frozen stratum edges.
    proxy = pairs.baseline_m / pairs.context_median_depth_m
    strata = pd.DataFrame({
        "scene": pairs.scene,
        "regime": pairs.regime,
        "p_bin": [bin_label(v, list(cfg["stratum_parallax_edges"]), cfg["zero_parallax_tol"]) for v in proxy],
        "r_bin": [bin_label(v, list(cfg["stratum_rotation_edges_deg"]), cfg["zero_rotation_tol_deg"]) for v in pairs.rotation_deg],
    })
    stratum_sizes = strata.groupby(["scene", "regime", "p_bin", "r_bin"]).size()
    pop["strata_count"] = int(len(stratum_sizes))
    pop["max_pairs_in_stratum"] = int(stratum_sizes.max())
    pop["cap"] = cfg["max_pairs_per_stratum"]
    if stratum_sizes.max() > cfg["max_pairs_per_stratum"]:
        problems.append("4.2 a sampling stratum exceeds max_pairs_per_stratum")
    report["population"] = pop

    # ---- binning sanity (4.2): every value lands in a bin, zero bins honest --
    p_edges = list(cfg["parallax_bin_edges"])
    r_edges = list(cfg["rotation_bin_edges_deg"])
    pbins = [bin_label(v, p_edges, cfg["zero_parallax_tol"]) for v in pairs.parallax]
    rbins = [bin_label(v, r_edges, cfg["zero_rotation_tol_deg"]) for v in pairs.rotation_deg]
    report["binning"] = {
        "parallax_bins_seen": sorted(set(pbins)),
        "rotation_bins_seen": sorted(set(rbins)),
        "rotation_regime_parallax_bins": sorted(set(b for b, r in zip(pbins, pairs.regime) if r == "rotation")),
        "translation_regime_rotation_bins": sorted(set(b for b, r in zip(rbins, pairs.regime) if r == "translation")),
    }

    report["problems"] = problems
    (OUT / "forensics.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print(f"\nPROBLEMS: {len(problems)}")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    main()
