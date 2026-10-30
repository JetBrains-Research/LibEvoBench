"""Analysis: taxonomy classification, KA decomposition, and report generation.

Reads eval_scored_t{1,2,3} parquets (from score.py). No matrix access needed --
all cross-version lookups were done by score.py.

Usage:
    uv run -m src.eval_benchmark.analysis --lib torch --model openai/gpt-5
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Taxonomy labels
CORRECT = "correct"
ANACHRONISTIC = "anachronistic"
PHANTOM = "phantom"
WRONG_API_VALID = "wrong_api_valid"
NO_API = "no_api"
UNRESOLVABLE = "unresolvable"
UNKNOWN = "unknown"

# Modal-taxonomy tie-breaking priority (most informative first)
TAXONOMY_PRIORITY = [
    CORRECT, WRONG_API_VALID, ANACHRONISTIC, PHANTOM, NO_API, UNRESOLVABLE,
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dc.dataclass
class Args:
    """Analyse scored evaluation results: taxonomy, KA decomposition, report."""

    lib:            str             = "torch"
    model:          str             = "openai/gpt-5"
    eval_dir:       Optional[str]   = None
    """Where to find eval_scored_*.parquet. Default: out/eval_benchmark/{lib}"""
    output_dir:     Optional[str]   = None
    """Where to write report JSON. Default: same as eval_dir"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_version(v: str) -> str:
    parts = str(v).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return str(v)


def _ver_num(v: str) -> int:
    """Convert major.minor version to a comparable integer (major*100 + minor)."""
    parts = str(v).split(".")
    try:
        return int(parts[0]) * 100 + (int(parts[1]) if len(parts) >= 2 else 0)
    except (ValueError, TypeError):
        return 0


def _cramers_v(ct: pd.DataFrame) -> float:
    """Compute Cramer's V from a contingency table (DataFrame)."""
    observed = ct.values
    n = observed.sum()
    r, c = observed.shape
    if n == 0 or min(r, c) <= 1:
        return 0.0
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            expected = row_sums[i] * col_sums[j] / n
            if expected > 0:
                chi2 += (observed[i, j] - expected) ** 2 / expected
    denom = n * (min(r, c) - 1)
    return float((chi2 / denom) ** 0.5) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Task 3 taxonomy
# ---------------------------------------------------------------------------

def classify_t3(df: pd.DataFrame, f1_threshold: float = 0.6) -> pd.DataFrame:
    df = df.copy()

    def _classify(row):
        f1 = row.get("t3_param_f1")
        if pd.isna(f1):
            return UNKNOWN
        if f1 >= f1_threshold:
            return CORRECT
        best_f1 = row.get("t3_best_match_f1")
        best_ver = row.get("t3_best_match_version")
        target_ver = _resolve_version(str(row.get("version", "")))
        if pd.isna(best_f1) or best_f1 < 0.3:
            return PHANTOM
        if best_ver is not None and best_ver != target_ver:
            return ANACHRONISTIC
        return PHANTOM

    df["t3_taxonomy"] = df.apply(_classify, axis=1)

    def _temporal_dir(row):
        if row["t3_taxonomy"] != ANACHRONISTIC:
            return None
        target = row.get("version", "")
        matched = row.get("t3_best_match_version", "")
        if not target or not matched:
            return None
        try:
            return "backward" if matched < target else "forward"
        except TypeError:
            return None

    df["t3_temporal_direction"] = df.apply(_temporal_dir, axis=1)
    return df


# ---------------------------------------------------------------------------
# L0 x L1 outcome matrix
# ---------------------------------------------------------------------------

def l0_l1_matrix(df: pd.DataFrame) -> Dict[str, float]:
    if "t1_L0_api_em" not in df.columns or "t1_L1_api_em" not in df.columns:
        return {}
    l0 = df["t1_L0_api_em"].fillna(False).astype(bool)
    l1 = df["t1_L1_api_em"].fillna(False).astype(bool)
    n = len(df)
    return {
        "both_correct": int((l0 & l1).sum()),
        "l0_only": int((l0 & ~l1).sum()),
        "l1_only": int((~l0 & l1).sum()),
        "both_fail": int((~l0 & ~l1).sum()),
        "total": n,
        "hallucination_rate": float((l0 & ~l1).sum() / n) if n else 0,
        "assistance_rate": float((~l0 & l1).sum() / n) if n else 0,
        "net_version_effect": float(((~l0 & l1).sum() - (l0 & ~l1).sum()) / n) if n else 0,
    }


# ---------------------------------------------------------------------------
# Knowledge-Application decomposition (Channel A: T2 × T1a)
# ---------------------------------------------------------------------------

def ka_decomposition(
    t1_df: pd.DataFrame, knowledge_df: pd.DataFrame,
    knowledge_col: str, app_col: str,
) -> Dict[str, float]:
    if app_col not in t1_df.columns or knowledge_col not in knowledge_df.columns:
        return {}

    t1 = t1_df.copy()
    k = knowledge_df.copy()

    if "lib_version" not in t1.columns or "version" not in k.columns:
        return {}

    t1["_join_ver"] = t1["lib_version"].apply(_resolve_version)
    k["_join_ver"] = k["version"].apply(_resolve_version)

    merged = t1.merge(
        k[["api", "_join_ver", knowledge_col]].drop_duplicates(),
        left_on=["api_norm", "_join_ver"], right_on=["api", "_join_ver"], how="inner",
    )
    valid = merged.dropna(subset=[app_col, knowledge_col])
    if valid.empty:
        return {}

    n = len(valid)
    knows = valid[knowledge_col].astype(bool)
    passes = valid[app_col].astype(bool)
    return {
        "n": n,
        "success": float((knows & passes).sum() / n),
        "application_gap": float((knows & ~passes).sum() / n),
        "pattern_matching": float((~knows & passes).sum() / n),
        "knowledge_gap": float((~knows & ~passes).sum() / n),
    }


# ---------------------------------------------------------------------------
# T1-T3 cross-task: aggregation & three analyses
# ---------------------------------------------------------------------------

def _aggregate_t1_to_api_level(df: pd.DataFrame, lv: str) -> pd.DataFrame:
    """Group T1 scored rows by (api_norm, version) → one row per API.

    Computes api_em_rate and modal_taxonomy (tie-broken by TAXONOMY_PRIORITY).
    """
    em_col = f"t1_{lv}_api_em"
    tax_col = f"t1_{lv}_api_taxonomy"
    res_col = f"t1_{lv}_pred_api_resolved"
    evr_col = f"t1_{lv}_pred_api_exists_versions"
    dir_col = f"t1_{lv}_pred_api_temporal_direction"

    if em_col not in df.columns or "lib_version" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["_ver"] = df["lib_version"].apply(_resolve_version)

    def _agg(group):
        em_rate = float(group[em_col].mean())
        counts = group[tax_col].value_counts()
        if counts.empty:
            modal = UNKNOWN
        else:
            max_count = counts.iloc[0]
            tied = counts[counts == max_count].index.tolist()
            modal = tied[0]
            if len(tied) > 1:
                for prio in TAXONOMY_PRIORITY:
                    if prio in tied:
                        modal = prio
                        break
        resolved = (group[res_col].dropna().iloc[0]
                    if res_col in group.columns and group[res_col].notna().any()
                    else None)
        exists_v = (group[evr_col].dropna().iloc[0]
                    if evr_col in group.columns and group[evr_col].notna().any()
                    else None)
        temp_dir = (group[dir_col].dropna().iloc[0]
                    if dir_col in group.columns and group[dir_col].notna().any()
                    else None)
        return pd.Series({
            "api_em_rate": em_rate,
            "modal_taxonomy": modal,
            "pred_api_resolved": resolved,
            "pred_api_exists_versions": exists_v,
            "pred_api_temporal_direction": temp_dir,
            "n_samples": len(group),
        })

    return (df.groupby(["api_norm", "_ver"])
              .apply(_agg, include_groups=False)
              .reset_index())


def _taxonomy_mirror(t1_agg: pd.DataFrame, t3_df: pd.DataFrame) -> Dict:
    """Contingency table: T1 modal taxonomy × T3 taxonomy."""
    if t1_agg.empty or t3_df.empty:
        return {}
    t3 = t3_df.copy()
    t3["_ver"] = t3["version"].apply(_resolve_version)

    merged = t1_agg.merge(
        t3[["api", "_ver", "t3_taxonomy"]].drop_duplicates(),
        left_on=["api_norm", "_ver"], right_on=["api", "_ver"], how="inner",
    )
    if merged.empty:
        return {}

    ct = pd.crosstab(merged["modal_taxonomy"], merged["t3_taxonomy"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    return {
        "contingency_raw": {
            str(r): {str(c): int(ct.at[r, c]) for c in ct.columns}
            for r in ct.index
        },
        "contingency_pct": {
            str(r): {str(c): round(float(ct_norm.at[r, c]), 3) for c in ct_norm.columns}
            for r in ct_norm.index
        },
        "cramers_v": _cramers_v(ct),
        "n_pairs": len(merged),
    }


def _knowledge_cascade(
    t1_df: pd.DataFrame, t3_df: pd.DataFrame, lv: str,
) -> Dict:
    """Sample-level broadcast: T3 F1 quartiles → T1 api_em & param taxonomy."""
    em_col = f"t1_{lv}_api_em"
    prec_col = f"t1_{lv}_param_precision"
    pcr_col = f"t1_{lv}_param_correct_rate"
    par_col = f"t1_{lv}_param_anachronistic_rate"
    ppr_col = f"t1_{lv}_param_phantom_rate"
    if em_col not in t1_df.columns:
        return {}

    t1 = t1_df.copy()
    t3 = t3_df.copy()
    t1["_ver"] = t1["lib_version"].apply(_resolve_version)
    t3["_ver"] = t3["version"].apply(_resolve_version)

    t3_cols = ["api", "_ver", "t3_param_f1",
               "t3_param_correct_rate", "t3_param_anachronistic_rate",
               "t3_param_phantom_rate"]
    t3_cols = [c for c in t3_cols if c in t3.columns]
    merged = t1.merge(
        t3[t3_cols].drop_duplicates(),
        left_on=["api_norm", "_ver"], right_on=["api", "_ver"], how="inner",
    )
    if merged.empty:
        return {}

    bins = [0, 0.25, 0.5, 0.75, 1.001]
    labels = ["Q1(0-0.25)", "Q2(0.25-0.5)", "Q3(0.5-0.75)", "Q4(0.75-1.0)"]
    merged["_q"] = pd.cut(
        merged["t3_param_f1"].fillna(-1), bins=bins, labels=labels, right=False)

    def _safe_mean(s):
        d = s.dropna()
        return float(d.mean()) if not d.empty else None

    result: Dict = {}
    for q in labels:
        sub = merged[merged["_q"] == q]
        if sub.empty:
            result[q] = {
                "api_em_rate": None, "param_precision_mean": None,
                "param_correct_rate": None,
                "param_anachronistic_rate": None,
                "param_phantom_rate": None,
                "t3_param_correct_rate": None,
                "t3_param_anachronistic_rate": None,
                "t3_param_phantom_rate": None,
                "n_api_pairs": 0, "n_samples": 0}
            continue
        result[q] = {
            "api_em_rate": float(sub[em_col].mean()),
            "param_precision_mean": _safe_mean(sub[prec_col]),
            "param_correct_rate": _safe_mean(sub.get(pcr_col)),
            "param_anachronistic_rate": _safe_mean(sub.get(par_col)),
            "param_phantom_rate": _safe_mean(sub.get(ppr_col)),
            "t3_param_correct_rate": _safe_mean(sub.get("t3_param_correct_rate")),
            "t3_param_anachronistic_rate": _safe_mean(sub.get("t3_param_anachronistic_rate")),
            "t3_param_phantom_rate": _safe_mean(sub.get("t3_param_phantom_rate")),
            "n_api_pairs": int(sub[["api_norm", "_ver"]].drop_duplicates().shape[0]),
            "n_samples": len(sub),
        }

    return {
        "quartiles": result,
        "n_total_pairs": int(
            merged[["api_norm", "_ver"]].drop_duplicates().shape[0]),
        "n_total_samples": len(merged),
    }


def _version_alignment(t1_agg: pd.DataFrame, t3_df: pd.DataFrame) -> Dict:
    """For doubly-anachronistic (api, version) pairs, check version agreement."""
    if t1_agg.empty or t3_df.empty:
        return {}
    t3 = t3_df.copy()
    t3["_ver"] = t3["version"].apply(_resolve_version)

    t1_anach = t1_agg[t1_agg["modal_taxonomy"] == ANACHRONISTIC].copy()
    t3_anach = t3[t3["t3_taxonomy"] == ANACHRONISTIC].copy()
    if t1_anach.empty or t3_anach.empty:
        return {}

    merged = t1_anach.merge(
        t3_anach[["api", "_ver", "t3_best_match_version",
                  "t3_temporal_direction"]].drop_duplicates(),
        left_on=["api_norm", "_ver"], right_on=["api", "_ver"], how="inner",
    )
    if merged.empty:
        return {}

    def _t1_closest(row):
        ev = row.get("pred_api_exists_versions")
        ver = str(row.get("_ver", ""))
        if not ev or pd.isna(ev):
            return None
        versions = str(ev).split(",")
        if not versions:
            return None
        tv = _ver_num(ver)
        return min(versions, key=lambda v: abs(_ver_num(v) - tv))

    merged["t1_closest_version"] = merged.apply(_t1_closest, axis=1)
    valid = merged.dropna(subset=["t1_closest_version", "t3_best_match_version"])
    if valid.empty:
        return {}

    agree = int(
        (valid["t1_closest_version"] == valid["t3_best_match_version"]).sum())
    dir_match = int((
        (valid["pred_api_temporal_direction"] == valid["t3_temporal_direction"])
        & valid["pred_api_temporal_direction"].notna()
    ).sum())

    return {
        "n_pairs": len(valid),
        "version_agreement_rate": float(agree / len(valid)),
        "temporal_direction_agreement": float(dir_match / len(valid)),
    }


# ---------------------------------------------------------------------------
# T2 version-sensitivity analysis
# ---------------------------------------------------------------------------

def t2_version_sensitivity(df: pd.DataFrame) -> Dict:
    """Measure version-blindness for generative T2: does the model predict the
    same API name regardless of the queried version?"""
    if "t2_pred_api" not in df.columns:
        return {}
    valid = df[df["t2_pred_api"].fillna("") != ""].copy()
    if valid.empty:
        return {}

    valid["_ver"] = valid["version"].apply(_resolve_version)

    def _blindness_stats(sub: pd.DataFrame, label: str) -> Dict:
        per_api = sub.groupby("api")["t2_pred_api"].nunique()
        n_apis = len(per_api)
        n_blind = int((per_api == 1).sum())
        return {
            "label": label,
            "n_apis": n_apis,
            "n_version_blind": n_blind,
            "version_blind_rate": float(n_blind / n_apis) if n_apis else 0,
        }

    result: Dict = {}
    result["overall"] = _blindness_stats(valid, "all api_types")

    for at in sorted(valid["api_type"].unique()):
        sub = valid[valid["api_type"] == at]
        result[at] = _blindness_stats(sub, at)

    return result


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------

def _t2_role_stats(sub: pd.DataFrame, label: str) -> Dict:
    ns = len(sub)
    return {
        "label": label,
        "n": ns,
        "em_rate": float(sub["t2_api_em"].mean()) if ns else 0,
        "n_empty": int(sub["t2_empty"].sum()) if "t2_empty" in sub.columns else 0,
    }


def _t3_role_stats(sub: pd.DataFrame, label: str) -> Dict:
    ns = len(sub)
    f1 = sub["t3_param_f1"].dropna()
    return {
        "label": label, "n": ns,
        "f1_mean": float(f1.mean()) if not f1.empty else 0,
        "correct": float((sub["t3_taxonomy"] == CORRECT).sum() / ns) if ns else 0,
        "anachronistic": float((sub["t3_taxonomy"] == ANACHRONISTIC).sum() / ns) if ns else 0,
        "phantom": float((sub["t3_taxonomy"] == PHANTOM).sum() / ns) if ns else 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    t1_df: Optional[pd.DataFrame], t2_df: Optional[pd.DataFrame],
    t3_df: Optional[pd.DataFrame], model: str, lib: str,
    api_comp: Optional[pd.DataFrame] = None,
) -> Dict:
    report: Dict = {"model": model, "library": lib}

    # ---- Task 1 --------------------------------------------------------
    if t1_df is not None:
        levels = [lv for lv in ["L0", "L1", "L2", "L3"]
                  if f"t1_{lv}_api_em" in t1_df.columns]

        # A. Augmentation staircase
        staircase: Dict = {}
        for lv in levels:
            em_col = f"t1_{lv}_api_em"
            sub = t1_df[t1_df[em_col].notna()] if em_col in t1_df.columns else t1_df
            tax_col = f"t1_{lv}_api_taxonomy"
            tax_dist = (sub[tax_col].value_counts(normalize=True).to_dict()
                        if tax_col in sub.columns else {})
            prec = sub[f"t1_{lv}_param_precision"].dropna()
            pcr = sub[f"t1_{lv}_param_correct_rate"].dropna()
            par = sub[f"t1_{lv}_param_anachronistic_rate"].dropna()
            ppr = sub[f"t1_{lv}_param_phantom_rate"].dropna()
            param_tax = {
                "correct_rate_mean": float(pcr.mean()) if not pcr.empty else None,
                "anachronistic_rate_mean": float(par.mean()) if not par.empty else None,
                "phantom_rate_mean": float(ppr.mean()) if not ppr.empty else None,
                "n_with_args": len(pcr),
            }
            staircase[lv] = {
                "api_em_rate": float(sub[em_col].mean()),
                "taxonomy": tax_dist,
                "param_taxonomy": param_tax,
                "param_precision_mean": (float(prec.mean())
                                         if not prec.empty else None),
                "n": len(sub),
                "n_total": len(t1_df),
            }
        report["augmentation_staircase"] = staircase

        # C. Output changed vs L0
        if "t1_L0_completion" in t1_df.columns:
            changed: Dict = {}
            for lv in levels:
                if lv == "L0":
                    continue
                em_col = f"t1_{lv}_api_em"
                col = f"t1_{lv}_completion"
                if col not in t1_df.columns:
                    continue
                sub = t1_df[t1_df[em_col].notna()] if em_col in t1_df.columns else t1_df
                l0_comp = sub["t1_L0_completion"].fillna("").str.strip()
                lv_comp = sub[col].fillna("").str.strip()
                diff_mask = l0_comp != lv_comp
                n_changed = int(diff_mask.sum())
                l0_em = pd.array(sub["t1_L0_api_em"].fillna(False), dtype="boolean")
                lv_em = pd.array(sub[em_col].fillna(False), dtype="boolean")
                n_improved = int((diff_mask & ~l0_em & lv_em).sum())
                n_degraded = int((diff_mask & l0_em & ~lv_em).sum())
                changed[lv] = {
                    "n": len(sub),
                    "n_changed": n_changed,
                    "pct_changed": float(n_changed / len(sub)) if len(sub) else 0,
                    "n_improved": n_improved,
                    "n_degraded": n_degraded,
                    "n_same": n_changed - n_improved - n_degraded,
                }
            report["t1_changed_vs_l0"] = changed

        # B. L0 x L1 matrix
        report["l0_l1_matrix"] = l0_l1_matrix(t1_df)

        # T1 by api_type
        api_type_stats: Dict = {}
        for at in sorted(t1_df["api_type"].unique()):
            sub = t1_df[t1_df["api_type"] == at]
            at_stats: Dict = {}
            for lv in levels:
                em_col = f"t1_{lv}_api_em"
                lv_sub = sub[sub[em_col].notna()] if em_col in sub.columns else sub
                tax_col = f"t1_{lv}_api_taxonomy"
                tax_dist = (lv_sub[tax_col].value_counts(normalize=True).to_dict()
                            if tax_col in lv_sub.columns else {})
                at_stats[lv] = {
                    "api_em_rate": float(lv_sub[em_col].mean()) if len(lv_sub) else 0.0,
                    "taxonomy": tax_dist,
                    "n": len(lv_sub),
                }
            api_type_stats[at] = at_stats
        report["t1_by_api_type"] = api_type_stats

        # D. T1 taxonomy by version (at L0 and L1)
        if "lib_version" in t1_df.columns:
            t1_ver = t1_df.copy()
            t1_ver["_ver"] = t1_ver["lib_version"].apply(_resolve_version)
            for tax_lv in ["L0", "L1"]:
                em_col = f"t1_{tax_lv}_api_em"
                tax_col = f"t1_{tax_lv}_api_taxonomy"
                if tax_col not in t1_ver.columns:
                    continue
                tax_by_ver: Dict = {}
                for v in sorted(t1_ver["_ver"].unique()):
                    sub = t1_ver[t1_ver["_ver"] == v]
                    ns = len(sub)
                    tax_by_ver[v] = {
                        "n": ns,
                        "api_em_rate": float(sub[em_col].mean()),
                        "taxonomy": sub[tax_col].value_counts(
                            normalize=True).to_dict(),
                    }
                report[f"t1_taxonomy_by_version_{tax_lv}"] = tax_by_ver

        # E. T1 lifecycle version analysis (L0)
        if api_comp is not None and "t1_L0_api_em" in t1_df.columns:
            vcols = [c for c in api_comp.columns if c.startswith("v_")]
            api_versions: Dict[str, set] = {}
            for _, row in api_comp.iterrows():
                api = row["api"]
                vers = {c[2:] + ".0" for c in vcols
                        if pd.notna(row[c]) and row[c]}
                api_versions[api] = vers

            em_col = "t1_L0_api_em"
            lc_sub = t1_df[t1_df[em_col].notna()]
            all_versions = sorted(
                {v for vs in api_versions.values() for v in vs})
            lifecycle: Dict = {}
            for v in all_versions:
                valid_apis = {a for a, vs in api_versions.items() if v in vs}
                mask = lc_sub["api_norm"].isin(valid_apis)
                vs = lc_sub[mask]
                if vs.empty:
                    continue
                vl = _resolve_version(v)
                lifecycle[vl] = {
                    "n": len(vs),
                    "api_em_rate": float(vs[em_col].mean()),
                    "n_apis": int(vs["api_norm"].nunique()),
                }
            report["t1_lifecycle_by_version"] = lifecycle

            # Per api_type: constant+sig_changed combined, version_specific separate
            type_groups = {
                "constant+sig_changed": lc_sub["api_type"].isin(
                    ["constant", "sig_changed"]),
                "version_specific": lc_sub["api_type"] == "version_specific",
            }
            lc_by_type: Dict = {}
            for label, grp_mask in type_groups.items():
                at_sub = lc_sub[grp_mask]
                at_lc: Dict = {}
                for v in all_versions:
                    valid_apis = {a for a, vs in api_versions.items()
                                  if v in vs}
                    mask = at_sub["api_norm"].isin(valid_apis)
                    vs = at_sub[mask]
                    if vs.empty:
                        continue
                    vl = _resolve_version(v)
                    at_lc[vl] = {
                        "n": len(vs),
                        "api_em_rate": float(vs[em_col].mean()),
                    }
                lc_by_type[label] = at_lc
            report["t1_lifecycle_by_type"] = lc_by_type

        # T1b staircase (keyword-forced, param F1 as pass criterion)
        t1b_levels = [lv for lv in ["L0", "L1", "L2", "L3"]
                      if f"t1b_{lv}_pass" in t1_df.columns]
        if t1b_levels:
            t1b_staircase: Dict = {}
            for lv in t1b_levels:
                valid = t1_df[f"t1b_{lv}_param_f1"].dropna()
                t1b_staircase[lv] = {
                    "param_f1_mean": float(valid.mean()) if not valid.empty else 0,
                    "pass_rate": float(t1_df[f"t1b_{lv}_pass"].mean()),
                    "n_with_kwargs": int(valid.count()),
                }
            report["t1b_staircase"] = t1b_staircase

            t1b_by_type: Dict = {}
            for at in sorted(t1_df["api_type"].unique()):
                sub = t1_df[t1_df["api_type"] == at]
                at_stats = {}
                for lv in t1b_levels:
                    at_stats[lv] = {
                        "pass_rate": float(sub[f"t1b_{lv}_pass"].mean()),
                        "n": len(sub),
                    }
                t1b_by_type[at] = at_stats
            report["t1b_by_api_type"] = t1b_by_type

    # ---- Task 2 (generative: API identification from redacted doc) ------
    if t2_df is not None and "t2_api_em" in t2_df.columns:
        n_t2 = len(t2_df)
        report["t2_em_rate"] = float(t2_df["t2_api_em"].mean())
        report["t2_empty_rate"] = float(t2_df["t2_empty"].mean())
        report["t2_n"] = n_t2

        if "t2_api_taxonomy" in t2_df.columns:
            report["t2_taxonomy"] = (
                t2_df["t2_api_taxonomy"].value_counts(normalize=True).to_dict())

        t2_by_type: Dict = {}
        for at in sorted(t2_df["api_type"].unique()):
            sub = t2_df[t2_df["api_type"] == at]
            entry: Dict = {
                "n": len(sub),
                "em_rate": float(sub["t2_api_em"].mean()),
            }
            if "t2_api_taxonomy" in sub.columns:
                entry["taxonomy"] = (
                    sub["t2_api_taxonomy"].value_counts(normalize=True).to_dict())
            t2_by_type[at] = entry
        report["t2_by_api_type"] = t2_by_type

        t2_by_ver: Dict = {}
        for v in sorted(t2_df["version"].unique()):
            sub = t2_df[t2_df["version"] == v]
            ns = len(sub)
            entry = {
                "n": ns,
                "em_rate": float(sub["t2_api_em"].mean()),
                "correct": float((sub["t2_api_taxonomy"] == CORRECT).sum() / ns),
                "wrong_api_valid": float(
                    (sub["t2_api_taxonomy"] == WRONG_API_VALID).sum() / ns),
                "anachronistic": float(
                    (sub["t2_api_taxonomy"] == ANACHRONISTIC).sum() / ns),
                "phantom": float(
                    (sub["t2_api_taxonomy"] == PHANTOM).sum() / ns),
            }
            t2_by_ver[v] = entry
        report["t2_by_version"] = t2_by_ver

        t2_int = t2_df[t2_df["api_type"].isin(
            {"sig_changed", "version_specific"})]
        t2_ctrl = t2_df[t2_df["api_type"] == "constant"]
        report["t2_interesting"] = _t2_role_stats(
            t2_int, "sig_changed + version_specific")
        report["t2_control"] = _t2_role_stats(t2_ctrl, "constant")

        report["t2_version_sensitivity"] = t2_version_sensitivity(t2_df)

        if "t2_doc_quality" in t2_df.columns:
            quality_order = ["empty", "stub_property", "too_short",
                             "short_generic", "short", "medium", "good", "rich"]
            dq: Dict = {}
            for q in quality_order:
                sub = t2_df[t2_df["t2_doc_quality"] == q]
                if sub.empty:
                    continue
                dq[q] = {
                    "n": len(sub),
                    "em_rate": float(sub["t2_api_em"].mean()),
                    "pct_of_total": float(len(sub) / n_t2),
                }
            report["t2_by_doc_quality"] = dq

    # ---- Task 3 --------------------------------------------------------
    if t3_df is not None:
        t3_df = classify_t3(t3_df)
        t3_valid = t3_df[t3_df["t3_taxonomy"] != UNKNOWN]
        if not t3_valid.empty:
            report["t3_taxonomy"] = (
                t3_valid["t3_taxonomy"].value_counts(normalize=True).to_dict())
            report["t3_param_f1_mean"] = float(
                t3_valid["t3_param_f1"].dropna().mean())
            prec = t3_valid["t3_param_precision"].dropna()
            if not prec.empty:
                report["t3_param_precision_mean"] = float(prec.mean())
            rec = t3_valid["t3_param_recall"].dropna()
            if not rec.empty:
                report["t3_param_recall_mean"] = float(rec.mean())

            anach = t3_valid[t3_valid["t3_taxonomy"] == ANACHRONISTIC]
            if not anach.empty:
                report["t3_temporal_direction"] = (
                    anach["t3_temporal_direction"].value_counts().to_dict())
                report["t3_anachronistic_version_dist"] = (
                    anach["t3_best_match_version"].value_counts().to_dict())

            by_ver = {}
            for v in sorted(t3_valid["version"].unique()):
                sub = t3_valid[t3_valid["version"] == v]
                ns = len(sub)
                by_ver[v] = {
                    "n": ns,
                    "correct": float((sub["t3_taxonomy"] == CORRECT).sum() / ns),
                    "anachronistic": float(
                        (sub["t3_taxonomy"] == ANACHRONISTIC).sum() / ns),
                    "phantom": float(
                        (sub["t3_taxonomy"] == PHANTOM).sum() / ns),
                }
            report["t3_by_version"] = by_ver

            t3_by_type: Dict = {}
            for at in sorted(t3_valid["api_type"].unique()):
                sub = t3_valid[t3_valid["api_type"] == at]
                ns = len(sub)
                f1_v = sub["t3_param_f1"].dropna()
                entry = {
                    "n": ns,
                    "f1_mean": float(f1_v.mean()) if not f1_v.empty else 0,
                    "correct": float(
                        (sub["t3_taxonomy"] == CORRECT).sum() / ns),
                    "anachronistic": float(
                        (sub["t3_taxonomy"] == ANACHRONISTIC).sum() / ns),
                    "phantom": float(
                        (sub["t3_taxonomy"] == PHANTOM).sum() / ns),
                }
                at_by_ver: Dict = {}
                for v in sorted(sub["version"].unique()):
                    vsub = sub[sub["version"] == v]
                    vn = len(vsub)
                    at_by_ver[v] = {
                        "n": vn,
                        "correct": float(
                            (vsub["t3_taxonomy"] == CORRECT).sum() / vn),
                        "anachronistic": float(
                            (vsub["t3_taxonomy"] == ANACHRONISTIC).sum() / vn),
                        "phantom": float(
                            (vsub["t3_taxonomy"] == PHANTOM).sum() / vn),
                    }
                entry["by_version"] = at_by_ver
                t3_by_type[at] = entry
            report["t3_by_api_type"] = t3_by_type

            t3_int = t3_valid[t3_valid["api_type"].isin(
                {"sig_changed", "version_specific"})]
            t3_ctrl = t3_valid[t3_valid["api_type"] == "constant"]
            report["t3_interesting"] = _t3_role_stats(
                t3_int, "sig_changed + version_specific")
            report["t3_control"] = _t3_role_stats(t3_ctrl, "constant")

    # ---- Channel A: T2 × T1a (API-name KA) ----------------------------
    if t1_df is not None and t2_df is not None:
        for lv in ["L0", "L1", "L2", "L3"]:
            col = f"t1_{lv}_api_em"
            if col in t1_df.columns:
                ka = ka_decomposition(t1_df, t2_df, "t2_api_em", col)
                if ka:
                    report[f"ka_t2_x_t1a_{lv}"] = ka

    # ---- F. T1-T3 cross-connection (three layers) ----------------------
    if t1_df is not None and t3_df is not None:
        for cross_lv in ["L2", "L3"]:
            em_col = f"t1_{cross_lv}_api_em"
            if em_col not in t1_df.columns:
                continue
            suffix = f"_{cross_lv}"
            t1_agg = _aggregate_t1_to_api_level(t1_df, cross_lv)
            if not t1_agg.empty:
                mirror = _taxonomy_mirror(t1_agg, t3_df)
                if mirror:
                    report[f"taxonomy_mirror{suffix}"] = mirror
                align = _version_alignment(t1_agg, t3_df)
                if align:
                    report[f"version_alignment{suffix}"] = align
            cascade = _knowledge_cascade(t1_df, t3_df, cross_lv)
            if cascade:
                report[f"knowledge_cascade{suffix}"] = cascade

    return report


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(report: Dict):
    print(f"\n{'='*70}")
    print(f"ANALYSIS REPORT — {report.get('model', '?')} on "
          f"{report.get('library', '?')}")
    print(f"{'='*70}")

    # ---- T1 staircase --------------------------------------------------
    staircase = report.get("augmentation_staircase", {})
    if staircase:
        print(f"\n--- Augmentation Staircase: API Taxonomy ---")
        print(f"  {'Level':<6} {'N':>6} {'API EM':>8} {'WrValid%':>9} "
              f"{'Anach%':>8} {'Phant%':>8}")
        for lv, stats in staircase.items():
            tax = stats.get("taxonomy", {})
            wrong_valid = tax.get("wrong_api_valid", 0)
            anach = tax.get("anachronistic", 0)
            phantom = tax.get("phantom", 0)
            n = stats.get("n", 0)
            print(f"  {lv:<6} {n:>6} {stats['api_em_rate']:>7.1%} "
                  f"{wrong_valid:>8.1%} "
                  f"{anach:>7.1%} {phantom:>7.1%}")

        print(f"\n--- Augmentation Staircase: Param Taxonomy ---")
        print(f"  {'Level':<6} {'N':>6} {'Prec(kw)':>10} {'Correct%':>9} "
              f"{'Anach%':>8} {'Phant%':>8} {'n(args)':>8}")
        for lv, stats in staircase.items():
            ptax = stats.get("param_taxonomy", {})
            prec = stats.get("param_precision_mean")
            pcr = ptax.get("correct_rate_mean")
            par = ptax.get("anachronistic_rate_mean")
            ppr = ptax.get("phantom_rate_mean")
            n_args = ptax.get("n_with_args", 0)
            n = stats.get("n", 0)
            prec_s = f"{prec:.3f}" if prec is not None else "     n/a"
            pcr_s = f"{pcr:.1%}" if pcr is not None else "    n/a"
            par_s = f"{par:.1%}" if par is not None else "   n/a"
            ppr_s = f"{ppr:.1%}" if ppr is not None else "   n/a"
            print(f"  {lv:<6} {n:>6} {prec_s:>10} {pcr_s:>9} "
                  f"{par_s:>8} {ppr_s:>8} {n_args:>8}")

    # ---- T1 changed vs L0 ---------------------------------------------
    changed = report.get("t1_changed_vs_l0", {})
    if changed:
        print(f"\n--- Output Changed vs L0 ---")
        print(f"  {'Level':<6} {'Changed':>10} {'Improved':>10} "
              f"{'Degraded':>10} {'Same':>10}")
        for lv, stats in changed.items():
            n = stats["n_changed"]
            pct = stats["pct_changed"]
            print(f"  {lv:<6} {n:>6} ({pct:>4.0%}) "
                  f"{stats['n_improved']:>10} {stats['n_degraded']:>10} "
                  f"{stats['n_same']:>10}")

    # ---- L0 x L1 matrix ------------------------------------------------
    l0l1 = report.get("l0_l1_matrix", {})
    if l0l1:
        print(f"\n--- L0 x L1 Outcome Matrix ---")
        print(f"  Version hallucination rate: {l0l1['hallucination_rate']:.1%}")
        print(f"  Version assistance rate:    {l0l1['assistance_rate']:.1%}")
        print(f"  Net version effect:         "
              f"{l0l1['net_version_effect']:+.1%}")

    # ---- T1 by api_type ------------------------------------------------
    by_type = report.get("t1_by_api_type", {})
    if by_type:
        print(f"\n--- T1 by API Type ---")
        for at, stats in by_type.items():
            parts = [f"{lv}={s['api_em_rate']:.0%}"
                     for lv, s in stats.items()]
            first_lv = list(stats.keys())[0]
            print(f"  [{at}] (n={stats[first_lv]['n']}) "
                  f"{' '.join(parts)}")

    # ---- T1 taxonomy by version ----------------------------------------
    for tax_lv in ["L0", "L1"]:
        tax_by_ver = report.get(f"t1_taxonomy_by_version_{tax_lv}", {})
        if tax_by_ver:
            print(f"\n--- T1 Taxonomy by Version ({tax_lv}) ---")
            print(f"  {'Version':>8} {'N':>5} {'API EM':>8} {'WrValid%':>9} "
                  f"{'Anach%':>8} {'Phant%':>8}")
            for v, stats in tax_by_ver.items():
                tax = stats.get("taxonomy", {})
                print(f"  {v:>8} {stats['n']:>5} "
                      f"{stats['api_em_rate']:>7.1%} "
                      f"{tax.get('wrong_api_valid', 0):>8.1%} "
                      f"{tax.get('anachronistic', 0):>7.1%} "
                      f"{tax.get('phantom', 0):>7.1%}")

    # ---- T1 lifecycle by version ----------------------------------------
    lifecycle = report.get("t1_lifecycle_by_version", {})
    if lifecycle:
        print(f"\n--- T1 Lifecycle by Version (L0) ---")
        print(f"  (Each version includes all samples whose API exists at that version)")
        print(f"  {'Version':>8} {'N':>6} {'APIs':>6} {'API EM':>8}")
        for v, stats in lifecycle.items():
            print(f"  {v:>8} {stats['n']:>6} {stats['n_apis']:>6} "
                  f"{stats['api_em_rate']:>7.1%}")

    lc_by_type = report.get("t1_lifecycle_by_type", {})
    if lc_by_type:
        print(f"\n--- T1 Lifecycle by Version × API Type (L0) ---")
        types = list(lc_by_type.keys())
        versions = list(next(iter(lc_by_type.values())).keys()) if types else []
        col_w = 22
        header = f"  {'Version':>8}"
        for at in types:
            header += f"  {at:>{col_w}}"
        print(header)
        for v in versions:
            row_s = f"  {v:>8}"
            for at in types:
                s = lc_by_type[at].get(v, {})
                if s:
                    row_s += f"  {s['api_em_rate']:>{col_w}.1%}"
                else:
                    row_s += f"  {'n/a':>{col_w}}"
            print(row_s)

    # ---- T1b staircase -------------------------------------------------
    t1b_staircase = report.get("t1b_staircase", {})
    if t1b_staircase:
        print(f"\n--- T1b Staircase (keyword-forced, pass = param F1 >= 0.6) ---")
        print(f"  {'Level':<6} {'F1 mean':>8} {'Pass':>8} {'n(kw)':>8}")
        for lv, stats in t1b_staircase.items():
            print(f"  {lv:<6} {stats['param_f1_mean']:>8.3f} "
                  f"{stats['pass_rate']:>7.1%} "
                  f"{stats['n_with_kwargs']:>8}")

    t1b_by_type = report.get("t1b_by_api_type", {})
    if t1b_by_type:
        print(f"\n--- T1b by API Type ---")
        for at, stats in t1b_by_type.items():
            parts = [f"{lv}={s['pass_rate']:.0%}" for lv, s in stats.items()]
            first_lv = list(stats.keys())[0]
            print(f"  [{at}] (n={stats[first_lv]['n']}) "
                  f"{' '.join(parts)}")

    # ---- Task 2 (generative) --------------------------------------------
    t2_em = report.get("t2_em_rate")
    if t2_em is not None:
        n_t2 = report.get("t2_n", "?")
        print(f"\n--- Task 2: API Identification (generative) ---")
        print(f"  EM: {t2_em:.1%}   Empty: {report.get('t2_empty_rate', 0):.1%}   (n={n_t2})")

    t2_tax = report.get("t2_taxonomy", {})
    if t2_tax:
        print(f"\n  Taxonomy distribution:")
        for label, frac in sorted(t2_tax.items(), key=lambda x: -x[1]):
            print(f"    {label:<22} {frac:>6.1%}")

    t2_int = report.get("t2_interesting")
    t2_ctrl = report.get("t2_control")
    if t2_int or t2_ctrl:
        print(f"\n--- Task 2 by Effective Role ---")
        for role, stats in [("INTERESTING", t2_int), ("CONTROL", t2_ctrl)]:
            if stats:
                print(f"  {role} [{stats['label']}] EM={stats['em_rate']:.1%} "
                      f"(n={stats['n']}, empty={stats.get('n_empty', 0)})")

    t2_by_type = report.get("t2_by_api_type", {})
    if t2_by_type:
        print(f"\n--- Task 2 by API Type ---")
        print(f"  {'Type':<22} {'N':>6} {'EM':>8}")
        print(f"  {'-'*38}")
        for at, stats in t2_by_type.items():
            print(f"  {at:<22} {stats['n']:>6} {stats['em_rate']:>7.1%}")

    t2_by_ver = report.get("t2_by_version", {})
    if t2_by_ver:
        print(f"\n--- Task 2 by Version ---")
        print(f"  {'Version':>8} {'N':>5} {'Correct':>9} "
              f"{'WrongVal':>9} {'Anach.':>9} {'Phantom':>9}")
        for v, stats in t2_by_ver.items():
            print(f"  {v:>8} {stats['n']:>5} "
                  f"{stats['correct']:>8.0%} "
                  f"{stats.get('wrong_api_valid', 0):>8.0%} "
                  f"{stats.get('anachronistic', 0):>8.0%} "
                  f"{stats.get('phantom', 0):>8.0%}")

    vs_sens = report.get("t2_version_sensitivity", {})
    if vs_sens:
        print(f"\n--- Task 2: Version Sensitivity ---")
        ov = vs_sens.get("overall", {})
        if ov:
            print(f"  Version-blind APIs (same prediction across all versions): "
                  f"{ov['n_version_blind']}/{ov['n_apis']} "
                  f"({ov['version_blind_rate']:.0%})")
        for at in ["constant", "sig_changed", "version_specific"]:
            s = vs_sens.get(at, {})
            if s:
                print(f"    [{at}] {s['n_version_blind']}/{s['n_apis']} "
                      f"version-blind ({s['version_blind_rate']:.0%})")

    dq = report.get("t2_by_doc_quality", {})
    if dq:
        print(f"\n--- Task 2: EM by Doc Quality ---")
        print(f"  Tiers: medium [50-100), good [100-300), rich [300+ chars]")
        print(f"  {'Quality':<16} {'N':>6} {'EM':>7} {'% total':>8}")
        print(f"  {'-'*40}")
        for q, stats in dq.items():
            print(f"  {q:<16} {stats['n']:>6} {stats['em_rate']:>6.1%} "
                  f"{stats['pct_of_total']:>7.1%}")

    # ---- Task 3 --------------------------------------------------------
    t3_tax = report.get("t3_taxonomy", {})
    if t3_tax:
        print(f"\n--- Task 3: Taxonomy (all APIs) ---")
        print(f"  Param F1 mean: {report.get('t3_param_f1_mean', 0):.3f}")
        prec = report.get("t3_param_precision_mean")
        rec = report.get("t3_param_recall_mean")
        if prec is not None:
            print(f"  Param Precision mean: {prec:.3f}")
        if rec is not None:
            print(f"  Param Recall mean: {rec:.3f}")
        for label, frac in sorted(t3_tax.items(), key=lambda x: -x[1]):
            print(f"  {label:<22} {frac:>6.1%}")

    t3_int = report.get("t3_interesting")
    t3_ctrl = report.get("t3_control")
    if t3_int or t3_ctrl:
        print(f"\n--- Task 3 by Effective Role ---")
        for role, stats in [("INTERESTING", t3_int), ("CONTROL", t3_ctrl)]:
            if stats:
                print(f"  {role} [{stats['label']}] (n={stats['n']}) "
                      f"f1={stats['f1_mean']:.3f} "
                      f"correct={stats['correct']:.0%} "
                      f"anach={stats['anachronistic']:.0%} "
                      f"phantom={stats['phantom']:.0%}")

    t3_by_type = report.get("t3_by_api_type", {})
    if t3_by_type:
        print(f"\n--- Task 3 by API Type (detail) ---")
        for at, stats in t3_by_type.items():
            print(f"  [{at}] (n={stats['n']}) "
                  f"f1={stats['f1_mean']:.3f} "
                  f"correct={stats['correct']:.0%} "
                  f"anach={stats['anachronistic']:.0%} "
                  f"phantom={stats['phantom']:.0%}")

    temp_dir = report.get("t3_temporal_direction", {})
    if temp_dir:
        print(f"\n--- Temporal Direction (T3 anachronistic) ---")
        for d, cnt in temp_dir.items():
            print(f"  {d}: {cnt}")
        ver_dist = report.get("t3_anachronistic_version_dist", {})
        if ver_dist:
            print(f"  Best-matched versions: {ver_dist}")

    by_ver = report.get("t3_by_version", {})
    if by_ver:
        print(f"\n--- Task 3 by Version ---")
        print(f"  {'Version':>8} {'N':>5} {'Correct':>9} "
              f"{'Anach.':>9} {'Phantom':>9}")
        for v, stats in by_ver.items():
            print(f"  {v:>8} {stats['n']:>5} "
                  f"{stats['correct']:>8.0%} "
                  f"{stats['anachronistic']:>8.0%} "
                  f"{stats['phantom']:>8.0%}")
                  
    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    eval_dir = args.eval_dir or f"out/eval_benchmark/{args.lib}"
    output_dir = args.output_dir or eval_dir
    model_tag = args.model.replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)

    t1_df = t2_df = t3_df = None

    t1_path = os.path.join(
        eval_dir, f"eval_scored_t1_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t1_path):
        t1_df = pd.read_parquet(t1_path)
        log.info("Loaded T1: %d rows", len(t1_df))

    t2_path = os.path.join(
        eval_dir, f"eval_scored_t2_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t2_path):
        t2_df = pd.read_parquet(t2_path)
        log.info("Loaded T2: %d rows", len(t2_df))

    t3_path = os.path.join(
        eval_dir, f"eval_scored_t3_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t3_path):
        t3_df = pd.read_parquet(t3_path)
        log.info("Loaded T3: %d rows", len(t3_df))

    data_dir = f"out/benchmark_db/{args.lib}"
    api_comp_path = os.path.join(data_dir, "api_comp_matrix.parquet")
    api_comp = pd.read_parquet(api_comp_path) if os.path.exists(api_comp_path) else None

    report = generate_report(t1_df, t2_df, t3_df, args.model, args.lib,
                             api_comp=api_comp)
    print_report(report)

    report_path = os.path.join(
        output_dir, f"report_{model_tag}_{args.lib}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
