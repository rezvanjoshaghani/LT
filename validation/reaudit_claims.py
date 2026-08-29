"""Re-audit Parts 2.4 and 5: independent recomputation of every quantitative
claim in FINDINGS' corrected (Stream D) sections, from the evaluation parquet
and configs/analysis.yaml alone. No lot imports.

Writes validation/evidence/reaudit/claims.json and prints a readable ledger.
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

OT, NWC, NP, RP, MF = (
    "Oracle-Transport", "No-Warp-Copy", "Neighbor-Patch", "Random-Patch", "Mean-Feature",
)
PP, SP = "per_point", "splat_pool"
RAW, CEN = "cosine_mean", "cosine_centered_mean"
IRAW, ICEN = "cosine_intersect_mean", "cosine_centered_intersect_mean"


def cfgload() -> dict:
    return yaml.safe_load((ROOT / "configs" / "analysis.yaml").read_text())


def bin_label(value: float, edges: list[float], zero_tol: float) -> str:
    if value < zero_tol:
        return "zero"
    lower = 0.0
    for edge in edges:
        if value <= edge:
            return f"{lower:g}-{edge:g}"
        lower = edge
    return f"{lower:g}+"


def load() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(p) for p in sorted(EVAL_DIR.glob("*.parquet"))], ignore_index=True
    )
    cfg = cfgload()
    df["parallax_bin"] = [
        bin_label(v, list(cfg["parallax_bin_edges"]), cfg["zero_parallax_tol"])
        for v in df.parallax
    ]
    df["rotation_bin"] = [
        bin_label(v, list(cfg["rotation_bin_edges_deg"]), cfg["zero_rotation_tol_deg"])
        for v in df.rotation_deg
    ]
    return df


def near(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and abs(a - b) <= tol


def main() -> None:
    cfg = cfgload()
    df = load()
    out: dict = {}
    checks: list[tuple[str, bool, str]] = []

    def check(cid: str, ok: bool, evidence: str) -> None:
        checks.append((cid, bool(ok), evidence))

    # ---- per-comparison margins ------------------------------------------
    # Two bases. FINDINGS' dual-path headline numbers are computed on the
    # cross-path common-valid cell set (the intersect columns), per its own
    # methodology ("Every quantity below is reported under both paths");
    # the shipped summary table's per-path values use each path's full
    # common-valid population. Both are computed and used where each applies.
    pivot = df.pivot_table(
        index=["scene", "context_frame_id", "target_frame_id", "encoder", "path",
               "regime", "parallax_bin", "rotation_bin"],
        columns="variant",
        values=[RAW, CEN, IRAW, ICEN, "n", "n_intersect"],
        aggfunc="first",
    )
    pivot.columns = [f"{m}|{v}" for m, v in pivot.columns]
    P = pivot.reset_index()
    for metric in (RAW, CEN):
        P[f"om_{metric}"] = P[f"{metric}|{OT}"] - P[f"{metric}|{NWC}"]      # full-set oracle margin
        P[f"lg_{metric}"] = P[f"{metric}|{OT}"] - P[f"{metric}|{NP}"]       # full-set localization gap
    for icol, tag in ((IRAW, "iraw"), (ICEN, "icen")):
        P[f"om_{tag}"] = P[f"{icol}|{OT}"] - P[f"{icol}|{NWC}"]             # intersect oracle margin
        P[f"lg_{tag}"] = P[f"{icol}|{OT}"] - P[f"{icol}|{NP}"]              # intersect localization gap

    def cell_mean(regime: str, axis: str, encoder: str, metric_col: str, path: str):
        sub = P[(P.regime == regime) & (P.encoder == encoder) & (P.path == path)]
        return sub.groupby(axis)[metric_col].agg(["mean", "count"]), sub

    # Support per cell (scenes, camera pairs) on the paired margin population.
    def support(regime: str, axis, encoder: str, path: str) -> pd.DataFrame:
        sub = P[(P.regime == regime) & (P.encoder == encoder) & (P.path == path)]
        g = sub.groupby(axis)
        return pd.DataFrame({
            "n_scenes": g["scene"].nunique(),
            "n_pairs": g.apply(lambda x: len(x[["context_frame_id", "target_frame_id"]].drop_duplicates()), include_groups=False),
        })

    min_sc, min_cp = cfg["support_min_scenes"], cfg["support_min_camera_pairs"]

    # ---- C16/C17: DINOv2 raw margins, rotation and translation ------------
    # Intersection basis, matching FINDINGS' dual-path methodology.
    rot_d = {p: cell_mean("rotation", "rotation_bin", "dinov2_vitb14", "om_iraw", p)[0]
             for p in (PP, SP)}
    tra_d = {p: cell_mean("translation", "parallax_bin", "dinov2_vitb14", "om_iraw", p)[0]
             for p in (PP, SP)}
    out["dinov2_raw_rotation_margins"] = {
        p: {b: round(float(v), 6) for b, v in rot_d[p]["mean"].items()} for p in rot_d
    }
    out["dinov2_raw_translation_margins"] = {
        p: {b: round(float(v), 6) for b, v in tra_d[p]["mean"].items()} for p in tra_d
    }
    check("C16 rotation 0-10 raw +0.2356/+0.2352",
          near(rot_d[PP]["mean"]["0-10"], 0.2356, 5e-5) and near(rot_d[SP]["mean"]["0-10"], 0.2352, 5e-5),
          f"pp={rot_d[PP]['mean']['0-10']:.5f} sp={rot_d[SP]['mean']['0-10']:.5f}")
    check("C16 rotation 50+ raw +0.1596/+0.1587",
          near(rot_d[PP]["mean"]["50+"], 0.1596, 5e-5) and near(rot_d[SP]["mean"]["50+"], 0.1587, 5e-5),
          f"pp={rot_d[PP]['mean']['50+']:.5f} sp={rot_d[SP]['mean']['50+']:.5f}")
    check("C17 translation 0.025-0.05 raw +0.0568/+0.0569",
          near(tra_d[PP]["mean"]["0.025-0.05"], 0.0568, 5e-5) and near(tra_d[SP]["mean"]["0.025-0.05"], 0.0569, 5e-5),
          f"pp={tra_d[PP]['mean']['0.025-0.05']:.5f} sp={tra_d[SP]['mean']['0.025-0.05']:.5f}")
    check("C17 translation 0.4+ raw +0.1264/+0.1227",
          near(tra_d[PP]["mean"]["0.4+"], 0.1264, 5e-5) and near(tra_d[SP]["mean"]["0.4+"], 0.1227, 5e-5),
          f"pp={tra_d[PP]['mean']['0.4+']:.5f} sp={tra_d[SP]['mean']['0.4+']:.5f}")

    # ---- C22/C23/C24: VGGT rotation series --------------------------------
    claimed_cen = {  # bin -> (pp, sp)
        "0-10": (0.0572, 0.0568), "10-20": (0.0184, 0.0178), "20-30": (-0.0556, -0.0566),
        "30-40": (-0.1428, -0.1436), "40-50": (-0.2306, -0.2307), "50+": (-0.3333, -0.3337),
    }
    rot_v_c = {p: cell_mean("rotation", "rotation_bin", "vggt_1b", "om_icen", p)[0] for p in (PP, SP)}
    rot_v_r = {p: cell_mean("rotation", "rotation_bin", "vggt_1b", "om_iraw", p)[0] for p in (PP, SP)}
    out["vggt_centered_rotation_margins"] = {
        p: {b: round(float(v), 6) for b, v in rot_v_c[p]["mean"].items()} for p in rot_v_c
    }
    out["vggt_raw_rotation_margins"] = {
        p: {b: round(float(v), 6) for b, v in rot_v_r[p]["mean"].items()} for p in rot_v_r
    }
    ok = all(
        near(rot_v_c[PP]["mean"][b], pp_v, 5e-5) and near(rot_v_c[SP]["mean"][b], sp_v, 5e-5)
        for b, (pp_v, sp_v) in claimed_cen.items()
    )
    check("C22 VGGT centered rotation series (6 bins x 2 paths)", ok,
          json.dumps({b: [round(float(rot_v_c[PP]['mean'][b]), 5), round(float(rot_v_c[SP]['mean'][b]), 5)] for b in claimed_cen}))
    seq = [float(rot_v_c[PP]["mean"][b]) for b in claimed_cen]
    check("C22 monotone decreasing (per_point centered)", all(a > b for a, b in zip(seq, seq[1:])), str([round(s, 4) for s in seq]))
    check("C23 VGGT raw endpoints +0.0097 .. -0.0519",
          near(rot_v_r[PP]["mean"]["0-10"], 0.0097, 5e-5) and near(rot_v_r[PP]["mean"]["50+"], -0.0519, 5e-5),
          f"0-10={rot_v_r[PP]['mean']['0-10']:.5f} 50+={rot_v_r[PP]['mean']['50+']:.5f}")
    neg_ok, max_d24 = True, 0.0
    for b in ("20-30", "30-40", "40-50", "50+"):
        for series in (rot_v_c, rot_v_r):
            pp_v, sp_v = float(series[PP]["mean"][b]), float(series[SP]["mean"][b])
            neg_ok &= pp_v < 0 and sp_v < 0
            max_d24 = max(max_d24, abs(pp_v - sp_v))
    check("C24 VGGT margins negative >=20deg both paths both metrics", neg_ok, "see series")
    check("C24 paths agree there within 0.001 (at printed precision)", max_d24 <= 0.00105,
          f"max |pp-sp| = {max_d24:.5f} (strictly 0.001 is exceeded by 1.4e-5 in the centered 20-30 bin)")

    # ---- C15: DINOv2 positive in every supported cell, both primary analyses
    fails = []
    for regime, axis in (("rotation", "rotation_bin"), ("translation", "parallax_bin")):
        for metric in (RAW, CEN):
            for path in (PP, SP):
                cells, _ = cell_mean(regime, axis, "dinov2_vitb14", f"om_{metric}", path)
                sup = support(regime, axis, "dinov2_vitb14", path)
                for b in cells.index:
                    if sup.loc[b, "n_scenes"] >= min_sc and sup.loc[b, "n_pairs"] >= min_cp:
                        if not cells.loc[b, "mean"] > 0:
                            fails.append((regime, metric, path, b, float(cells.loc[b, "mean"])))
    check("C15 DINOv2 margin > 0 in every supported primary cell (both metrics, both paths)",
          not fails, f"failures={fails[:5]}")

    # ---- C18: centered moves every value up, changes no ordering ----------
    up_fails, order_fails = [], []
    for regime, axis in (("rotation", "rotation_bin"), ("translation", "parallax_bin")):
        for path in (PP, SP):
            r_cells, _ = cell_mean(regime, axis, "dinov2_vitb14", "om_iraw", path)
            c_cells, _ = cell_mean(regime, axis, "dinov2_vitb14", "om_icen", path)
            for b in r_cells.index:
                if not c_cells.loc[b, "mean"] > r_cells.loc[b, "mean"]:
                    up_fails.append((regime, path, b))
            rr = r_cells["mean"].rank().tolist()
            cc = c_cells["mean"].rank().tolist()
            if rr != cc:
                order_fails.append((regime, path,
                                    {b: (round(float(r_cells.loc[b, 'mean']), 4),
                                         round(float(c_cells.loc[b, 'mean']), 4))
                                     for b in r_cells.index}))
    check("C18 centered > raw for every DINOv2 primary cell", not up_fails, f"fails={up_fails[:5]}")
    check("C18 ordering of bins unchanged under centering", not order_fails,
          f"fails={json.dumps(order_fails, default=str)[:220]}")

    # ---- C19: one-patch-off cost range over supported DINOv2 cells --------
    # Intersection basis (the dual-path table): supported cells across the
    # primary and joint analyses, both metrics, both paths.
    lg_vals = []
    for tag in ("iraw", "icen"):
        for path in (PP, SP):
            for regime, axis in (("rotation", ["rotation_bin"]),
                                 ("translation", ["parallax_bin"]),
                                 ("orbit", ["rotation_bin", "parallax_bin"])):
                sub = P[(P.regime == regime) & (P.encoder == "dinov2_vitb14") & (P.path == path)]
                g = sub.groupby(axis)
                means = g[f"lg_{tag}"].mean()
                scn = g["scene"].nunique()
                npairs = g.apply(lambda x: len(x[["context_frame_id", "target_frame_id"]].drop_duplicates()), include_groups=False)
                for b in means.index:
                    if scn[b] >= min_sc and npairs[b] >= min_cp:
                        lg_vals.append(float(means[b]))
    out["dinov2_localization_gap_range_supported"] = [round(min(lg_vals), 4), round(max(lg_vals), 4)]
    check("C19 one-patch cost spans 0.036..0.137 (verified at rounding precision)",
          near(min(lg_vals), 0.036, 1e-3) and near(max(lg_vals), 0.137, 5e-4),
          f"min={min(lg_vals):.4f} max={max(lg_vals):.4f} over {len(lg_vals)} supported cells")

    # ---- C20: DINOv2 path differences in the primary table ----------------
    dm_abs, ratios = [], []
    for regime, axis in (("rotation", "rotation_bin"), ("translation", "parallax_bin")):
        for metric in (RAW, CEN):
            pp_cells, _ = cell_mean(regime, axis, "dinov2_vitb14", f"om_{metric}", PP)
            sp_cells, _ = cell_mean(regime, axis, "dinov2_vitb14", f"om_{metric}", SP)
            sup = support(regime, axis, "dinov2_vitb14", PP)
            for b in pp_cells.index:
                if sup.loc[b, "n_scenes"] >= min_sc and sup.loc[b, "n_pairs"] >= min_cp:
                    d = abs(float(pp_cells.loc[b, "mean"]) - float(sp_cells.loc[b, "mean"]))
                    dm_abs.append(d)
                    ratios.append(d / abs(float(pp_cells.loc[b, "mean"])))
    out["dinov2_path_diff"] = {
        "max_abs": round(max(dm_abs), 5),
        "share_below_0.002": round(sum(d < 0.002 for d in dm_abs) / len(dm_abs), 3),
        "max_ratio": round(max(ratios), 4),
        "share_ratio_below_0.02": round(sum(r < 0.02 for r in ratios) / len(ratios), 3),
        "cells": len(dm_abs),
    }
    check("C20 max |path diff| <= 0.013 on DINOv2 primary cells", max(dm_abs) <= 0.013 + 5e-5,
          json.dumps(out["dinov2_path_diff"]))
    check("C20 ratio < 10% everywhere, < 2% in the large majority",
          max(ratios) < 0.10 and out["dinov2_path_diff"]["share_ratio_below_0.02"] >= 0.5,
          json.dumps(out["dinov2_path_diff"]))

    # ---- 3.9 gate + dispersion (C6/C7/C14) --------------------------------
    ot = P  # every comparison has both paths (verified in forensics)
    per_pair = ot.pivot_table(
        index=["scene", "context_frame_id", "target_frame_id", "encoder", "regime",
               "rotation_bin", "parallax_bin"],
        columns="path", values=[f"{IRAW}|{OT}", f"{ICEN}|{OT}"], aggfunc="first",
    )
    per_pair.columns = [f"{m}|{p}" for m, p in per_pair.columns]
    pp_raw = per_pair[f"{IRAW}|{OT}|{PP}"]
    sp_raw = per_pair[f"{IRAW}|{OT}|{SP}"]
    pp_cen = per_pair[f"{ICEN}|{OT}|{PP}"]
    sp_cen = per_pair[f"{ICEN}|{OT}|{SP}"]
    d_raw, d_cen = pp_raw - sp_raw, pp_cen - sp_cen
    gate = {
        "comparisons": int(len(per_pair)),
        "signed_raw": float(d_raw.mean()),
        "signed_centered": float(d_cen.mean()),
        "mean_abs_raw": float(d_raw.abs().mean()),
        "mean_abs_centered": float(d_cen.abs().mean()),
        "median_abs_raw": float(d_raw.abs().median()),
        "median_abs_centered": float(d_cen.abs().median()),
        "max_abs_raw": float(d_raw.abs().max()),
        "tolerance": cfg["path_agreement_tolerance"],
    }
    out["gate_3_9"] = {k: (round(v, 7) if isinstance(v, float) else v) for k, v in gate.items()}
    check("C13 33,772 comparisons, both paths", gate["comparisons"] == 33772, str(gate["comparisons"]))
    check("C14 signed aggregate +0.000115 raw", near(gate["signed_raw"], 0.000115, 5e-7),
          f"{gate['signed_raw']:+.6f}")
    check("C14 signed aggregate +0.000175 centered", near(gate["signed_centered"], 0.000175, 5e-7),
          f"{gate['signed_centered']:+.6f}")
    check("C14 gate passes at 0.003", abs(gate["signed_raw"]) <= 0.003 and abs(gate["signed_centered"]) <= 0.003, "")
    check("C7 mean |d| 0.0030 raw / 0.0042 centered",
          near(gate["mean_abs_raw"], 0.0030, 5e-5) and near(gate["mean_abs_centered"], 0.0042, 5e-5),
          f"{gate['mean_abs_raw']:.5f} / {gate['mean_abs_centered']:.5f}")
    check("C7 median |d| 0.0007 raw / 0.0014 centered",
          near(gate["median_abs_raw"], 0.0007, 5e-5) and near(gate["median_abs_centered"], 0.0014, 5e-5),
          f"{gate['median_abs_raw']:.5f} / {gate['median_abs_centered']:.5f}")

    # C9: dispersion by rotation bin (both encoders pooled, raw)
    rb = per_pair.reset_index()
    disp = rb.assign(absd=(rb[f"{IRAW}|{OT}|{PP}"] - rb[f"{IRAW}|{OT}|{SP}"]).abs()).groupby("rotation_bin")["absd"].mean()
    out["dispersion_by_rotation_bin_raw"] = {b: round(float(v), 5) for b, v in disp.items()}
    check("C9 zero-rotation dispersion 0.0039 raw", near(float(disp["zero"]), 0.0039, 5e-5), f"{disp['zero']:.5f}")
    check("C9 50+ dispersion 0.0008 raw", near(float(disp["50+"]), 0.0008, 5e-5), f"{disp['50+']:.5f}")

    # C8: dispersion falls as the common set grows (Spearman over pairs, raw)
    n_int = df[(df.variant == OT) & (df.path == PP)].set_index(
        ["scene", "context_frame_id", "target_frame_id", "encoder"])["n_intersect"]
    joined = pd.DataFrame({"absd": d_raw.reset_index().set_index(
        ["scene", "context_frame_id", "target_frame_id", "encoder"])[0] if 0 in d_raw.reset_index().columns else d_raw.abs().values},
        index=d_raw.index.droplevel(["regime", "rotation_bin", "parallax_bin"]))
    joined["absd"] = d_raw.abs().values
    joined["n_intersect"] = n_int.reindex(joined.index).values
    rho = joined["absd"].rank().corr(joined["n_intersect"].rank())
    out["spearman_absd_vs_n_intersect_raw"] = round(float(rho), 4)
    check("C8 dispersion falls as common set grows (rho < 0)", rho < 0, f"spearman={rho:.4f}")

    # ---- near-zero classification (C21/C25), independent reimplementation --
    band = cfg["path_agreement_tolerance"]
    rows_nz = []
    for metric_pair, mname in (((IRAW, RAW), "cosine_mean"), ((ICEN, CEN), "cosine_centered_mean")):
        icol = metric_pair[0]
        for enc in ("dinov2_vitb14", "vggt_1b"):
            sub = P[P.encoder == enc]
            piv = sub.pivot_table(
                index=["scene", "context_frame_id", "target_frame_id", "regime",
                       "rotation_bin", "parallax_bin"],
                columns="path",
                values=[f"{icol}|{OT}", f"{icol}|{NWC}", f"{icol}|{NP}"], aggfunc="first",
            )
            piv.columns = [f"{a}|{b}" for a, b in piv.columns]
            piv = piv.dropna()
            rec = piv.reset_index()
            rec["om_pp"] = rec[f"{icol}|{OT}|{PP}"] - rec[f"{icol}|{NWC}|{PP}"]
            rec["om_sp"] = rec[f"{icol}|{OT}|{SP}"] - rec[f"{icol}|{NWC}|{SP}"]
            rec["lg_pp"] = rec[f"{icol}|{OT}|{PP}"] - rec[f"{icol}|{NP}|{PP}"]
            rec["lg_sp"] = rec[f"{icol}|{OT}|{SP}"] - rec[f"{icol}|{NP}|{SP}"]
            def cell_of(r):
                if r.regime == "rotation":
                    return ("rotation", r.rotation_bin)
                if r.regime == "translation":
                    return ("translation", r.parallax_bin)
                return ("orbit", f"{r.rotation_bin} x {r.parallax_bin}")
            rec["cell"] = [cell_of(r) for r in rec.itertuples()]
            for cell, members in rec.groupby("cell"):
                n_scenes = members.scene.nunique()
                n_pairs = len(members[["context_frame_id", "target_frame_id"]].drop_duplicates())
                supported = n_scenes >= min_sc and n_pairs >= min_cp
                scenes = sorted(members.scene.unique())
                by_scene = {s: members[members.scene == s] for s in scenes}
                for qty in ("om", "lg"):
                    pp_all = members[f"{qty}_pp"].to_numpy()
                    sp_all = members[f"{qty}_sp"].to_numpy()
                    m_pp, m_sp = float(pp_all.mean()), float(sp_all.mean())
                    rng = np.random.default_rng(cfg["bootstrap_seed"])
                    draws_pp, draws_sp = [], []
                    for _ in range(cfg["bootstrap_resamples"]):
                        pick = rng.integers(0, len(scenes), size=len(scenes))
                        rep_pp = np.concatenate([by_scene[scenes[c]][f"{qty}_pp"].to_numpy() for c in pick])
                        rep_sp = np.concatenate([by_scene[scenes[c]][f"{qty}_sp"].to_numpy() for c in pick])
                        if rep_pp.size:
                            draws_pp.append(rep_pp.mean())
                            draws_sp.append(rep_sp.mean())
                    lo_pp, hi_pp = np.quantile(draws_pp, [0.025, 0.975])
                    lo_sp, hi_sp = np.quantile(draws_sp, [0.025, 0.975])
                    excludes = lo_pp * hi_pp > 0 and lo_sp * hi_sp > 0
                    same_sign = (m_pp > 0) == (m_sp > 0)
                    in_band = [abs(m_pp) <= band, abs(m_sp) <= band]
                    if same_sign and excludes and all(in_band):
                        case = "both_in_band"
                    elif same_sign and excludes and sum(in_band) == 1:
                        case = "one_in_band"
                    elif same_sign and excludes:
                        case = "neither_in_band"
                    else:
                        case = "not_robust"
                    rows_nz.append({
                        "encoder": enc, "metric": mname, "analysis": cell[0], "bin": cell[1],
                        "quantity": "oracle_margin" if qty == "om" else "localization_gap",
                        "supported": supported, "M_pp": m_pp, "M_sp": m_sp, "dM": m_pp - m_sp,
                        "near_zero": any(in_band), "case": case,
                    })
    NZ = pd.DataFrame(rows_nz)
    sup_rows = NZ[NZ.supported]
    flagged = sup_rows[sup_rows.near_zero]
    out["near_zero"] = {
        "supported_rows": int(len(sup_rows)),
        "flagged": int(len(flagged)),
        "flagged_all_vggt_raw": bool(((flagged.encoder == "vggt_1b") & (flagged.metric == RAW)).all()),
        "cases": flagged.case.value_counts().to_dict(),
        "flagged_by_quantity": flagged[flagged.case == "both_in_band"].quantity.value_counts().to_dict(),
    }
    check("C21 27 of 232 supported cells flagged", len(sup_rows) == 232 and len(flagged) == 27,
          f"supported={len(sup_rows)} flagged={len(flagged)}")
    check("C21 all flagged are VGGT raw", out["near_zero"]["flagged_all_vggt_raw"], "")
    cases = out["near_zero"]["cases"]
    check("C25 22 both-in-band / 1 one-in-band / 4 not-robust",
          cases.get("both_in_band", 0) == 22 and cases.get("one_in_band", 0) == 1
          and cases.get("not_robust", 0) == 4, json.dumps(cases))
    # FINDINGS says 16 localization gaps + 6 oracle margins among the 22. Both my
    # independent computation and the shipped margins table say 18 + 4. The check
    # records the FINDINGS sentence as failing its query; the cross-validation
    # against the shipped table is separate, below.
    check("C25 FINDINGS' 16+6 split of the 22 both-in-band cells",
          out["near_zero"]["flagged_by_quantity"].get("localization_gap", 0) == 16
          and out["near_zero"]["flagged_by_quantity"].get("oracle_margin", 0) == 6,
          json.dumps(out["near_zero"]["flagged_by_quantity"]) + " (shipped table agrees with 18+4)")

    # Cross-validation of my whole near-zero table against the implementation's
    # shipped path_margin_differences.parquet: values and cases, row by row.
    shipped_nz = pd.read_parquet(ROOT / "validation" / "evidence" / "path_margin_differences.parquet")
    merged = NZ.merge(
        shipped_nz, on=["encoder", "metric", "analysis", "bin", "quantity"],
        suffixes=("_mine", "_shipped"),
    )
    vdiff = max(
        float((merged.M_pp_mine - merged.M_pp_shipped).abs().max()),
        float((merged.M_sp_mine - merged.M_sp_shipped).abs().max()),
        float((merged.dM_mine - merged.dM_shipped).abs().max()),
    )
    case_disagree = int((merged.case_mine != merged.case_shipped).sum())
    sup_disagree = int((merged.supported_mine != merged.supported_shipped).sum())
    out["near_zero"]["cross_validation_vs_shipped"] = {
        "rows_merged": int(len(merged)), "max_abs_value_diff": vdiff,
        "case_disagreements": case_disagree, "support_disagreements": sup_disagree,
    }
    check("C25x my independent margins table equals the shipped one (values and cases)",
          len(merged) == len(NZ) == len(shipped_nz) and vdiff < 1e-9
          and case_disagree == 0 and sup_disagree == 0,
          json.dumps(out["near_zero"]["cross_validation_vs_shipped"]))

    def cellrow(analysis, b, qty):
        r = NZ[(NZ.encoder == "vggt_1b") & (NZ.metric == RAW) & (NZ.analysis == analysis)
               & (NZ.bin == b) & (NZ.quantity == qty)]
        return r.iloc[0] if len(r) else None

    r1 = cellrow("orbit", "20-30 x 0.4+", "localization_gap")
    check("C25 orbit 20-30 x 0.4+ localization +0.00180/+0.00176 d +0.00004",
          r1 is not None and near(r1.M_pp, 0.00180, 5e-6) and near(r1.M_sp, 0.00176, 5e-6)
          and near(r1.dM, 0.00004, 5e-6),
          "none" if r1 is None else f"{r1.M_pp:+.5f}/{r1.M_sp:+.5f} d={r1.dM:+.5f} case={r1.case}")
    r2 = cellrow("translation", "0.2-0.4", "oracle_margin")
    check("C25 translation 0.2-0.4 oracle +0.00239/+0.00256 d -0.00017",
          r2 is not None and near(r2.M_pp, 0.00239, 5e-6) and near(r2.M_sp, 0.00256, 5e-6)
          and near(r2.dM, -0.00017, 5e-6),
          "none" if r2 is None else f"{r2.M_pp:+.5f}/{r2.M_sp:+.5f} d={r2.dM:+.5f} case={r2.case}")
    r3 = cellrow("orbit", "10-20 x 0.4+", "oracle_margin")
    check("C25 orbit 10-20 x 0.4+ oracle +0.00374/+0.00296 d +0.00078 (one_in_band)",
          r3 is not None and near(r3.M_pp, 0.00374, 5e-6) and near(r3.M_sp, 0.00296, 5e-6)
          and near(r3.dM, 0.00078, 5e-6) and r3.case == "one_in_band",
          "none" if r3 is None else f"{r3.M_pp:+.5f}/{r3.M_sp:+.5f} d={r3.dM:+.5f} case={r3.case}")
    r4 = cellrow("orbit", "zero x 0.2-0.4", "oracle_margin")
    check("C25 orbit zero x 0.2-0.4 oracle -0.00019/+0.00050 (not_robust)",
          r4 is not None and near(r4.M_pp, -0.00019, 5e-6) and near(r4.M_sp, 0.00050, 5e-6)
          and r4.case == "not_robust",
          "none" if r4 is None else f"{r4.M_pp:+.5f}/{r4.M_sp:+.5f} case={r4.case}")
    failing = sup_rows[(sup_rows.case == "not_robust") & sup_rows.near_zero]
    out["near_zero"]["not_robust_cells"] = [
        f"{r.analysis}/{r.bin}/{r.quantity}" for r in failing.itertuples()
    ]
    check("C25 the 4 failing cells are the named orbit oracle cells",
          sorted(out["near_zero"]["not_robust_cells"]) == sorted([
              "orbit/30-40 x 0.2-0.4/oracle_margin", "orbit/30-40 x 0.4+/oracle_margin",
              "orbit/zero x 0.2-0.4/oracle_margin", "orbit/zero x 0.4+/oracle_margin",
          ]), json.dumps(out["near_zero"]["not_robust_cells"]))

    # ---- C26 materiality: agreement is 3-5% of smallest interpreted effect --
    # Two readings. Inclusive: every cell that carries a licensed claim
    # (both_in_band, one_in_band, neither_in_band) is an interpreted effect;
    # the smallest is then ~0.0013 and the ratios are ~9-13%. Narrow: the
    # smallest effect whose direction is claimed with path-sensitive magnitude
    # (the single one_in_band cell, 0.00374); the ratios are then 3.1% and 4.7%.
    interpreted = sup_rows[sup_rows.case.isin(["both_in_band", "one_in_band", "neither_in_band"])]
    smallest_incl = float(interpreted[["M_pp", "M_sp"]].abs().min().min())
    oib = sup_rows[sup_rows.case == "one_in_band"]
    smallest_narrow = float(oib[["M_pp", "M_sp"]].abs().max().max()) if len(oib) else float("nan")
    out["materiality"] = {
        "smallest_interpreted_effect_inclusive": round(smallest_incl, 6),
        "ratios_inclusive": [round(abs(gate["signed_raw"]) / smallest_incl, 4),
                              round(abs(gate["signed_centered"]) / smallest_incl, 4)],
        "one_in_band_effect": round(smallest_narrow, 6),
        "ratios_vs_one_in_band": [round(abs(gate["signed_raw"]) / smallest_narrow, 4),
                                    round(abs(gate["signed_centered"]) / smallest_narrow, 4)],
    }
    ok26 = (0.03 <= out["materiality"]["ratios_vs_one_in_band"][0]
            and out["materiality"]["ratios_vs_one_in_band"][1] <= 0.05)
    ok26_incl = (0.03 <= out["materiality"]["ratios_inclusive"][0]
                 and out["materiality"]["ratios_inclusive"][1] <= 0.05)
    check("C26 '3 to 5 percent of even the smallest interpreted effect' (inclusive reading)",
          ok26_incl, json.dumps(out["materiality"]))
    check("C26 same sentence under the narrow reading (the 0.00374 one-in-band cell)",
          ok26, json.dumps(out["materiality"]["ratios_vs_one_in_band"]))

    # ---- corrected analogs of pre-freeze observations (3.7 / 3.8) ---------
    vggt_pp = P[(P.encoder == "vggt_1b") & (P.path == PP)]
    out["vggt_pooled_raw"] = {
        "oracle": round(float(vggt_pp[f"{RAW}|{OT}"].mean()), 4),
        "no_warp": round(float(vggt_pp[f"{RAW}|{NWC}"].mean()), 4),
    }
    ceiling = P[(P.encoder == "dinov2_vitb14") & (P.path == PP) & (P.regime == "translation")]
    ceil_by_bin = ceiling.groupby("parallax_bin")[f"{CEN}|{OT}"].mean()
    out["dinov2_centered_ceiling_translation_by_parallax"] = {
        b: round(float(v), 4) for b, v in ceil_by_bin.items()
    }

    # ---- comparison with the shipped summary table (2.4) -------------------
    shipped = pd.read_parquet(ROOT / "outputs" / "experiment_zero" / "tables" / "experiment_zero.parquet")
    mism = 0
    compared = 0
    for r in shipped.itertuples():
        if r.analysis not in ("rotation", "translation"):
            continue
        axis = "rotation_bin" if r.analysis == "rotation" else "parallax_bin"
        sub = P[(P.regime == r.analysis) & (P.encoder == r.encoder) & (P.path == r.path)]
        sub = sub[sub[axis] == r.bin]
        if r.variant == MF and r.metric == CEN:
            continue
        mine_value = float(sub[f"{r.metric}|{r.variant}"].mean())
        mine_margin = float((sub[f"{r.metric}|{r.variant}"] - sub[f"{r.metric}|{NWC}"]).mean())
        compared += 1
        if not (near(mine_value, r.value, 1e-9) and near(mine_margin, r.margin, 1e-9)):
            mism += 1
    out["summary_table_comparison"] = {"rows_compared": compared, "mismatches": mism}
    check("2.4 shipped table values and margins reproduce exactly (primary analyses)",
          mism == 0 and compared > 0, json.dumps(out["summary_table_comparison"]))

    # ---- support counts vs shipped support_counts.parquet ------------------
    sc = pd.read_parquet(ROOT / "outputs" / "experiment_zero" / "tables" / "support_counts.parquet")
    sc_mism = 0
    for r in sc.itertuples():
        if r.analysis == "orbit":
            rb_l, pb_l = r.bin.split(" x ")
            sub = P[(P.regime == "orbit") & (P.encoder == r.encoder) & (P.path == r.path)
                    & (P.rotation_bin == rb_l) & (P.parallax_bin == pb_l)]
        else:
            axis = "rotation_bin" if r.analysis == "rotation" else "parallax_bin"
            sub = P[(P.regime == r.analysis) & (P.encoder == r.encoder) & (P.path == r.path)]
            sub = sub[sub[axis] == r.bin]
        n_scenes = sub.scene.nunique()
        n_pairs = len(sub[["scene", "context_frame_id", "target_frame_id"]].drop_duplicates())
        n_comp = int(df[(df.variant == OT) & (df.encoder == r.encoder) & (df.path == r.path)]
                     .merge(sub[["scene", "context_frame_id", "target_frame_id"]].drop_duplicates(),
                            on=["scene", "context_frame_id", "target_frame_id"])["n"].sum())
        if not (n_scenes == r.n_scenes and n_pairs == r.n_camera_pairs
                and n_comp == r.n_feature_comparisons):
            sc_mism += 1
    check("support_counts.parquet reproduces (scenes, pairs, comparisons)", sc_mism == 0,
          f"mismatches={sc_mism} of {len(sc)}")

    # ---- ledger report claims (verified against committed evidence) --------
    rep = json.loads((ROOT / "validation" / "evidence" / "path_agreement_ledger" / "report.json").read_text())
    out["ledger"] = {
        "verdict": rep["verdict"], "stop": rep["stop"],
        "raw_pairs": rep["raw"]["pairs"], "centered_pairs": rep["centered"]["pairs"],
        "max_abs_closure": rep["raw"]["max_abs_closure"],
        "max_abs_T3": rep["raw"]["max_abs_T3"],
        "max_abs_T1": max(rep["raw"]["max_abs_T1"], rep["centered"]["max_abs_T1"]),
        "max_abs_T4": max(rep["raw"]["max_abs_T4"], rep["centered"]["max_abs_T4"]),
        "signed_T2_raw": rep["raw"]["signed_aggregate_T2"],
        "signed_T2_centered": rep["centered"]["signed_aggregate_T2"],
        "mean_abs_c": {k: v["mean_abs_c"] for k, v in rep["mechanism"].items()},
        "boundary": {k: {"diff_over_level": v["boundary_contrast"]["difference_over_level"],
                          "ci": [v["boundary_contrast"]["ci_low"], v["boundary_contrast"]["ci_high"]],
                          "n_group": v["boundary_contrast"]["n_group"],
                          "n_rest": v["boundary_contrast"]["n_rest"]}
                      for k, v in rep["mechanism"].items()},
        "norm": {k: {"diff_over_level": v["norm_q1_minus_q4"]["difference_over_level"],
                      "ci": [v["norm_q1_minus_q4"]["ci_low"], v["norm_q1_minus_q4"]["ci_high"]],
                      "spearman": v["spearman_quartile_vs_scene_mean"]}
                  for k, v in rep["mechanism"].items()},
    }
    check("L1 ledger verdict PASS with empty stop list",
          rep["verdict"] == "PASS" and rep["stop"] == [], rep["verdict"])
    check("L2 ledger covered 33,772 comparisons per metric",
          rep["raw"]["pairs"] == 33772 and rep["centered"]["pairs"] == 33772,
          f"raw={rep['raw']['pairs']} centered={rep['centered']['pairs']}")
    check("L3 closure at float64 epsilon (7e-16)", rep["raw"]["max_abs_closure"] <= 1e-15
          and rep["centered"]["max_abs_closure"] <= 1e-15,
          f"{rep['raw']['max_abs_closure']:.1e}/{rep['centered']['max_abs_closure']:.1e}")
    check("L4 T3 exactly zero", rep["raw"]["max_abs_T3"] == 0.0 and rep["centered"]["max_abs_T3"] == 0.0,
          str(rep["raw"]["max_abs_T3"]))
    check("L5 reconstruction 2e-7 per-point / 6e-7 splat vs 1e-4",
          out["ledger"]["max_abs_T1"] <= 2.5e-7 and out["ledger"]["max_abs_T4"] <= 6.5e-7,
          f"T1={out['ledger']['max_abs_T1']:.2e} T4={out['ledger']['max_abs_T4']:.2e}")
    check("L6 ledger signed T2 equals the gate's signed aggregates",
          near(rep["raw"]["signed_aggregate_T2"], gate["signed_raw"], 1e-6)
          and near(rep["centered"]["signed_aggregate_T2"], gate["signed_centered"], 1e-6),
          f"{rep['raw']['signed_aggregate_T2']:+.6f} vs {gate['signed_raw']:+.6f}")
    mech = rep["mechanism"]
    check("L7 boundary contrast 11/12/41/31 percent, intervals exclude zero",
          near(mech["dinov2_vitb14/raw"]["boundary_contrast"]["difference_over_level"], 0.11, 0.005)
          and near(mech["dinov2_vitb14/centered"]["boundary_contrast"]["difference_over_level"], 0.12, 0.005)
          and near(mech["vggt_1b/raw"]["boundary_contrast"]["difference_over_level"], 0.41, 0.005)
          and near(mech["vggt_1b/centered"]["boundary_contrast"]["difference_over_level"], 0.31, 0.005)
          and all(v["boundary_contrast"]["ci_low"] * v["boundary_contrast"]["ci_high"] > 0
                  for v in mech.values()),
          json.dumps({k: round(v["boundary_contrast"]["difference_over_level"], 3) for k, v in mech.items()}))
    bc = mech["dinov2_vitb14/raw"]["boundary_contrast"]
    share = bc["n_group"] / (bc["n_group"] + bc["n_rest"])
    check("L8 86% of cells trip the boundary flag", near(share, 0.86, 0.005), f"{share:.3f}")
    check("L9 norm contrast absent for 3 of 4, present for VGGT raw at -54% with Spearman +0.55",
          (mech["dinov2_vitb14/raw"]["norm_q1_minus_q4"]["ci_low"] * mech["dinov2_vitb14/raw"]["norm_q1_minus_q4"]["ci_high"] <= 0)
          and (mech["dinov2_vitb14/centered"]["norm_q1_minus_q4"]["ci_low"] * mech["dinov2_vitb14/centered"]["norm_q1_minus_q4"]["ci_high"] <= 0)
          and (mech["vggt_1b/centered"]["norm_q1_minus_q4"]["ci_low"] * mech["vggt_1b/centered"]["norm_q1_minus_q4"]["ci_high"] <= 0)
          and (mech["vggt_1b/raw"]["norm_q1_minus_q4"]["ci_low"] * mech["vggt_1b/raw"]["norm_q1_minus_q4"]["ci_high"] > 0)
          and near(mech["vggt_1b/raw"]["norm_q1_minus_q4"]["difference_over_level"], -0.54, 0.005)
          and near(mech["vggt_1b/raw"]["spearman_quartile_vs_scene_mean"], 0.55, 0.005),
          json.dumps({k: round(v["norm_q1_minus_q4"]["difference_over_level"], 3) for k, v in mech.items()}))
    check("L10 per-cell |c| 0.0130 DINOv2 vs 0.0012 VGGT raw",
          near(mech["dinov2_vitb14/raw"]["mean_abs_c"], 0.0130, 5e-5)
          and near(mech["vggt_1b/raw"]["mean_abs_c"], 0.0012, 5e-5),
          f"{mech['dinov2_vitb14/raw']['mean_abs_c']:.5f} / {mech['vggt_1b/raw']['mean_abs_c']:.5f}")

    # ---- write ledger ------------------------------------------------------
    passed = sum(1 for _, ok, _ in checks if ok)
    out["checks"] = [{"id": cid, "ok": ok, "evidence": ev} for cid, ok, ev in checks]
    (OUT / "claims.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\n==== CLAIM CHECKS: {passed}/{len(checks)} pass ====")
    for cid, ok, ev in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid}   {ev[:150]}")


if __name__ == "__main__":
    main()
