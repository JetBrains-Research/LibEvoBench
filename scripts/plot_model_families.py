#!/usr/bin/env python3
"""One-row multi-family comparison figure for a single plot type.

Each column shows one model family on a fixed library (or aggregated
across libraries). Supported plot types:

    t1-lifecycle-L1          T1 API-C EM by version, level L1
    t1-staircase-aggregated  T1 API-C EM by prompt level, aggregated over libraries
    normalized-versions      T1/T2/T3 mean score over normalized version progression

Usage:
    uv run scripts/plot_model_families.py --plot normalized-versions --libs torch numpy scipy
    uv run scripts/plot_model_families.py --plot t1-lifecycle-L1 --lib torch
    uv run scripts/plot_model_families.py --plot t1-staircase-aggregated --libs torch numpy scipy
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd

from src.eval_benchmark.analysis import classify_t3

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR = Path("out/eval_benchmark")
DB_DIR   = Path("out/benchmark_db")

# ── Model families ─────────────────────────────────────────────────────────────
FAMILIES: dict[str, list[tuple[str, str]]] = {
    "sonnet": [
        ("anthropic_claude-sonnet-4-20250514", "Sonnet 4"),
        ("anthropic_claude-sonnet-4-6",        "Sonnet 4.6"),
    ],
    "opus": [
        ("anthropic_claude-4-opus-20250514",   "Opus 4"),
        ("anthropic_claude-opus-4-6",          "Opus 4.6"),
    ],
    "gpt": [
        ("openai_gpt-4.1",  "GPT-4.1"),
        ("openai_gpt-5",    "GPT-5"),
        ("openai_gpt-5.1",  "GPT-5.1"),
        ("openai_gpt-5.2",  "GPT-5.2"),
        ("openai_gpt-5.4",  "GPT-5.4"),
        ("openai_gpt-5.5",  "GPT-5.5"),
    ],
    "qwen": [
        ("external-collaborators_qwen3_5-35b-a3b",   "Qwen3.5 35B"),
        ("external-collaborators_qwen3_5-122b-a10b",  "Qwen3.5 122B"),
        ("external-collaborators_qwen3_5-397b-a17b",  "Qwen3.5 397B"),
    ],
    "gemini-flash": [
        ("vertex_ai_gemini-2.0-flash",        "gemini-2.0-flash"),
        ("vertex_ai_gemini-2.5-flash",        "gemini-2.5-flash"),
        ("vertex_ai_gemini-3-flash-preview",  "gemini-3-flash"),
    ],
}

FAMILY_GROUPS: dict[str, list[str]] = {
    "sonnet-gemini": ["gemini-flash", "sonnet"],
}

ROW_FAMILIES = ["gpt", "sonnet-gemini", "qwen"]

# ── Colour palettes ────────────────────────────────────────────────────────────
COMBINED_PALETTES: dict[str, list[str]] = {
    "gpt":          ["#B2DFBA", "#6FCF97", "#39B54A", "#2F9D56", "#1B7A3D", "#0D4D2B"],
    "gpt-mini":     ["#FFCC80", "#E65100"],
    "gemini-flash": ["#26A6B8", "#0077FF", "#0033CC"],
    "sonnet-gemini":["#26A6B8", "#0077FF", "#0033CC", "#FFB74D", "#E65100"],
    "qwen":         ["#FF4FC3", "#E0007A", "#9C27B0", "#6A1B9A", "#3B0066"],
    "sonnet":       ["#FFB74D", "#E65100"],
    "opus":         ["#F48FB1", "#880E4F"],
}

# ── Style constants ────────────────────────────────────────────────────────────
CONST_STYLES = [
    {"color": "#BDBDBD", "linewidth": 1.05, "markersize": 2.85, "marker": "s",  "linestyle": "-",               "alpha": 0.45},
    {"color": "#9E9E9E", "linewidth": 1.05, "markersize": 2.85, "marker": "s",  "linestyle": "--",              "alpha": 0.45},
    {"color": "#757575", "linewidth": 1.05, "markersize": 2.85, "marker": "s",  "linestyle": ":",               "alpha": 0.45},
    {"color": "#616161", "linewidth": 1.05, "markersize": 2.85, "marker": "s",  "linestyle": "-.",              "alpha": 0.45},
    {"color": "#424242", "linewidth": 1.05, "markersize": 2.85, "marker": "P",  "linestyle": (0,(3,1,1,1)),     "alpha": 0.45},
]
CONST_MARKERS = ("s", "^", "v", "D", "P")

LINEWIDTH_TARGET  = 2.0
LINEWIDTH_CONST   = 1.35
MARKERSIZE_TARGET = 5.5
MARKERSIZE_CONST  = 4.50
ALPHA_CONST_LINE  = 0.50
ALPHA_CONST_CI    = 0.09
ALPHA_TARGET_CI   = 0.08

FONT_AXIS_LABEL   = 15
FONT_XTICK        = 13
FONT_LEGEND_TITLE = 8
FONT_LEGEND_BODY  = 8
FONT_PANEL_TITLE  = 15

NORMALIZED_VERSION_ANCHORS       = np.linspace(0.0, 1.0, 5)
NORMALIZED_VERSION_TICK_POSITIONS = np.linspace(0.0, 1.0, 5)
NORMALIZED_VERSION_TICK_LABELS   = ["Oldest", "0.25", "Median", "0.75", "Latest"]


# ── Data helpers ───────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 5000, ci: float = 95, seed: int = 19,
) -> Tuple[float, float, float]:
    rng  = np.random.default_rng(seed)
    mean = float(values.mean())
    boots = np.array([
        values[rng.choice(len(values), size=len(values), replace=True)].mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return mean, float(lo), float(hi)


def _ver_cols(api_comp: pd.DataFrame) -> List[str]:
    return sorted(
        (c for c in api_comp.columns if c.startswith("v_")),
        key=lambda c: [int(x) for x in c[2:].split(".")],
    )


def _short_versions(api_comp: pd.DataFrame) -> List[str]:
    return [c[2:] for c in _ver_cols(api_comp)]


def _build_api_versions(api_comp: pd.DataFrame) -> Dict[str, set]:
    vcols = _ver_cols(api_comp)
    out: Dict[str, set] = {}
    for _, row in api_comp.iterrows():
        valid = {c[2:] for c in vcols if pd.notna(row[c]) and bool(row[c])}
        if valid:
            out[row["api"]] = valid
    return out


def _model_colors(n: int, palette: list[str]) -> list:
    if n == 1:
        return [palette[-1]]
    if n <= len(palette):
        return palette[:n]
    from matplotlib.colors import to_rgba
    lo, hi = to_rgba(palette[0]), to_rgba(palette[-1])
    return [
        tuple(lo[j] + (hi[j] - lo[j]) * i / (n - 1) for j in range(4))
        for i in range(n)
    ]


def _try_load(lib: str, task: str, model_tag: str) -> Optional[pd.DataFrame]:
    p = EVAL_DIR / lib / f"eval_scored_{task}_{model_tag}_{lib}.parquet"
    return pd.read_parquet(p) if p.is_file() else None


def _version_sort_key(v: str) -> list[int]:
    return [int(x) for x in str(v).split(".") if x.isdigit()]


def _progression_map(versions: list[str]) -> dict[str, float]:
    if not versions:
        return {}
    n = len(versions)
    out: dict[str, float] = {}
    for i, v in enumerate(versions):
        pos     = 1.0 if n == 1 else i / (n - 1)
        nearest = NORMALIZED_VERSION_ANCHORS[
            int(np.argmin(np.abs(NORMALIZED_VERSION_ANCHORS - pos)))
        ]
        out[str(v)] = float(nearest)
    return out


# ── Family helpers ─────────────────────────────────────────────────────────────

def _family_label(name: str) -> str:
    return {
        "gpt":           "OpenAI GPT",
        "gpt-mini":      "OpenAI GPT mini",
        "gemini-flash":  "Google Gemini Flash",
        "sonnet-gemini": "Sonnet + Gemini",
        "qwen":          "Qwen",
        "sonnet":        "Anthropic Sonnet",
        "opus":          "Anthropic Opus",
    }.get(name, name.replace("-", " ").title())


def _family_models(family_name: str) -> list[tuple[str, str]]:
    if family_name in FAMILY_GROUPS:
        out: list[tuple[str, str]] = []
        for member in FAMILY_GROUPS[family_name]:
            out.extend(FAMILIES[member])
        return out
    return FAMILIES[family_name]


def _load_family(
    family_name: str, lib: str, task: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    out = []
    for model_tag, model_label in _family_models(family_name):
        df = _try_load(lib, task, model_tag)
        if df is not None:
            out.append((model_tag, model_label, df))
    return out


# ── Legend helper ──────────────────────────────────────────────────────────────

def _legend_paired_line_styles(
    ax: plt.Axes,
    *,
    model_order: list[str] | None = None,
    loc: str = "lower left",
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_vs:    dict[str, object] = {}
    by_const: dict[str, object] = {}
    for h, lb in zip(handles, labels):
        if lb.endswith(" · const"):
            by_const[lb.replace(" · const", "").strip()] = h
        else:
            by_vs[lb.strip()] = h

    if model_order is not None:
        order = [n for n in model_order if by_vs.get(n) is not None]
    else:
        order, seen = [], set()
        for lb in labels:
            name = lb.replace(" · const", "").strip() if lb.endswith(" · const") else lb.strip()
            if not lb.endswith(" · const") and name not in seen:
                seen.add(name)
                order.append(name)

    paired_handles, paired_labels = [], []
    for name in order:
        hv = by_vs.get(name)
        if hv is None:
            continue
        hc = by_const.get(name)
        paired_handles.append((hv, hc) if hc is not None else hv)
        paired_labels.append(name)

    if not paired_handles:
        return

    ax.legend(
        paired_handles, paired_labels,
        loc=loc,
        fontsize=FONT_LEGEND_BODY,
        framealpha=0.45,
        title="Stable / Evolving",
        title_fontsize=FONT_LEGEND_TITLE,
        handler_map={tuple: HandlerTuple(ndivide=2)},
    )


# ── T1 Lifecycle ───────────────────────────────────────────────────────────────

def _plot_t1_lifecycle(
    ax: plt.Axes,
    lib: str,
    api_comp: pd.DataFrame,
    models_data: list,
    palette: list[str],
    level: str = "L1",
    *,
    xlabel: str | None = "Library version",
    ylabel: str | None = "API EM (%)",
    auto_ylim: bool = False,
) -> None:
    em_col       = f"t1_{level}_api_em"
    all_vers     = _short_versions(api_comp)
    api_versions = _build_api_versions(api_comp)
    colors       = _model_colors(len(models_data), palette)

    for mi, (model_tag, model_label, t1) in enumerate(models_data):
        if em_col not in t1.columns:
            continue
        sub    = t1[t1[em_col].notna()].copy()
        color  = colors[mi]
        base   = CONST_STYLES[mi % len(CONST_STYLES)]
        const_kw = {
            **base,
            "marker":      CONST_MARKERS[mi % len(CONST_MARKERS)],
            "linewidth":   LINEWIDTH_CONST,
            "markersize":  MARKERSIZE_CONST,
            "alpha":       ALPHA_CONST_LINE,
        }

        for grp_types, kw, lbl in [
            ({"constant", "sig_changed"}, const_kw, f"{model_label} · const"),
            ({"version_specific"},
             {"color": color, "linewidth": LINEWIDTH_TARGET, "markersize": MARKERSIZE_TARGET,
              "marker": CONST_MARKERS[mi % len(CONST_MARKERS)], "linestyle": "-"},
             model_label),
        ]:
            g = sub[sub["api_type"].isin(grp_types)]
            x_lab, y, lo, hi = [], [], [], []
            for v in all_vers:
                valid = {a for a, vs in api_versions.items() if v in vs}
                vs = g[g["api_norm"].isin(valid)]
                if vs.empty:
                    continue
                m, cl, ch = _bootstrap_ci(vs[em_col].values)
                x_lab.append(v); y.append(m * 100)
                lo.append(cl * 100); hi.append(ch * 100)
            x = list(range(len(x_lab)))
            ax.plot(x, y, label=lbl, **kw)
            ax.fill_between(
                x, lo, hi, color=kw["color"],
                alpha=ALPHA_CONST_CI if "constant" in grp_types else ALPHA_TARGET_CI,
            )

    ax.set_xticks(range(len(all_vers)))
    ax.set_xticklabels(all_vers, fontsize=FONT_XTICK, rotation=45, ha="right")
    ax.set_xlabel(xlabel or "", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(ylabel or "", fontsize=FONT_AXIS_LABEL)
    if not auto_ylim:
        ax.set_ylim(45, 97)
    ax.set_title(f"{lib} — API-C Lifecycle ({level})")
    _legend_paired_line_styles(ax)
    ax.grid(True, alpha=0.3)


def make_t1_lifecycle_row(
    lib: str,
    families: list[str],
    out_path: Path,
) -> None:
    n   = len(families)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    api_comp  = pd.read_parquet(DB_DIR / lib / "api_comp_matrix.parquet")

    for col, fam in enumerate(families):
        ax          = axes[0, col]
        palette     = COMBINED_PALETTES.get(fam, ["#90CAF9", "#1565C0"])
        models_data = _load_family(fam, lib, "t1")

        if not models_data:
            ax.set_visible(False)
            continue

        _plot_t1_lifecycle(
            ax, lib, api_comp, models_data, palette, level="L1",
            xlabel="Library version",
            ylabel="API-C EM (%)" if col == 0 else None,
            auto_ylim=(fam == "qwen"),
        )
        ax.set_title(_family_label(fam), fontsize=FONT_PANEL_TITLE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── T1 Staircase (aggregated over libraries) ───────────────────────────────────

def _t1_staircase_levels(df: pd.DataFrame) -> list[str]:
    raw = [
        c.split("_")[1]
        for c in df.columns
        if c.startswith("t1_L") and c.endswith("_api_em")
    ]
    return sorted(set(raw), key=lambda lv: int(str(lv).lstrip("L")))


def _aggregated_staircase_rows(model_tag: str, libs: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for lib in libs:
        t1 = _try_load(lib, "t1", model_tag)
        if t1 is None:
            continue
        levels = _t1_staircase_levels(t1)
        for lv in levels:
            em_col = f"t1_{lv}_api_em"
            if em_col not in t1.columns:
                continue
            valid = t1[t1[em_col].notna()]
            for role, types in [
                ("const",  {"constant", "sig_changed"}),
                ("target", {"version_specific"}),
            ]:
                g = valid[valid["api_type"].isin(types)]
                vals = pd.to_numeric(g[em_col], errors="coerce").dropna().values.astype(float)
                if len(vals):
                    rows.append({"role": role, "lib": lib, "level": lv, "value": float(vals.mean())})
    return pd.DataFrame(rows)


def _aggregate_staircase_levels(
    df: pd.DataFrame, role: str, levels: list[str],
) -> pd.DataFrame:
    rows = []
    sub  = df[df["role"] == role]
    for lv in levels:
        vals = sub[sub["level"] == lv]["value"].values
        if len(vals) == 0:
            rows.append({"level": lv, "mean": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        m, lo, hi = _bootstrap_ci(vals)
        rows.append({"level": lv, "mean": m * 100, "lo": lo * 100, "hi": hi * 100})
    return pd.DataFrame(rows)


def _plot_aggregated_staircase_family(
    ax: plt.Axes,
    family_name: str,
    libs: list[str],
    *,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    models  = _family_models(family_name)
    palette = COMBINED_PALETTES.get(family_name, ["#90CAF9", "#1565C0"])
    colors  = _model_colors(len(models), palette)

    model_frames: list[tuple[int, str, pd.DataFrame]] = []
    level_set:    set[str] = set()
    plotted_labels: list[str] = []

    for mi, (model_tag, model_label) in enumerate(models):
        df = _aggregated_staircase_rows(model_tag, libs)
        if df.empty:
            continue
        plotted_labels.append(model_label)
        model_frames.append((mi, model_label, df))
        level_set.update(str(v) for v in df["level"].unique())

    if not plotted_labels:
        ax.set_visible(False)
        return

    levels = sorted(level_set, key=lambda lv: int(str(lv).lstrip("L")))

    for mi, model_label, df in model_frames:
        color = colors[mi]
        for role, label, linestyle, linewidth, alpha_line, alpha_band in [
            ("const",  f"{model_label} · const", "--", LINEWIDTH_CONST,  ALPHA_CONST_LINE, ALPHA_CONST_CI),
            ("target", model_label,               "-",  LINEWIDTH_TARGET, 0.93,             ALPHA_TARGET_CI),
        ]:
            agg    = _aggregate_staircase_levels(df, role, levels)
            marker = CONST_MARKERS[mi % len(CONST_MARKERS)]
            x      = np.arange(len(agg), dtype=float)
            ax.plot(
                x, agg["mean"],
                color=color, marker=marker, linestyle=linestyle,
                linewidth=linewidth,
                markersize=MARKERSIZE_CONST if role == "const" else MARKERSIZE_TARGET,
                alpha=alpha_line, label=label,
            )
            ax.fill_between(x, agg["lo"].values, agg["hi"].values, color=color, alpha=alpha_band)

    ax.set_xticks(np.arange(len(levels), dtype=float))
    ax.set_xticklabels(levels, fontsize=FONT_XTICK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(35 if family_name == "qwen" else 50, 100)
    ax.set_title(_family_label(family_name), fontsize=FONT_PANEL_TITLE, fontweight="bold")
    _legend_paired_line_styles(ax, model_order=plotted_labels, loc="lower right")
    ax.grid(True, alpha=0.3)


def make_aggregated_staircase(
    libs: list[str],
    families: list[str],
    out_path: Path,
) -> None:
    n   = len(families)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)

    for col, fam in enumerate(families):
        _plot_aggregated_staircase_family(
            axes[0, col], fam, libs,
            xlabel="Prompt augmentation level" if col == n // 2 else None,
            ylabel="API-C EM (%)" if col == 0 else None,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Normalized version progression ─────────────────────────────────────────────

def _normalized_task_version_rows(
    model_tag: str,
    libs: list[str],
    *,
    t1_level: str = "L1",
) -> pd.DataFrame:
    rows: list[dict] = []

    for lib in libs:
        api_comp_path = DB_DIR / lib / "api_comp_matrix.parquet"
        if not api_comp_path.is_file():
            continue
        api_comp     = pd.read_parquet(api_comp_path)
        api_versions = _build_api_versions(api_comp)
        api_vers     = _short_versions(api_comp)
        api_prog     = _progression_map(api_vers)

        t1     = _try_load(lib, "t1", model_tag)
        em_col = f"t1_{t1_level}_api_em"
        if t1 is not None and em_col in t1.columns:
            valid_t1 = t1[t1[em_col].notna()].copy()
            for v in api_vers:
                valid_apis = {api for api, vs in api_versions.items() if v in vs}
                vsub = valid_t1[valid_t1["api_norm"].isin(valid_apis)]
                for role, types in [
                    ("const",  {"constant", "sig_changed"}),
                    ("target", {"version_specific"}),
                ]:
                    g    = vsub[vsub["api_type"].isin(types)]
                    vals = pd.to_numeric(g[em_col], errors="coerce").dropna().values.astype(float)
                    if len(vals):
                        rows.append({
                            "role": role, "task": "T1", "lib": lib,
                            "version": v, "progression": api_prog[str(v)],
                            "value": float(vals.mean()),
                        })

        t2 = _try_load(lib, "t2", model_tag)
        if t2 is not None and "t2_api_em" in t2.columns:
            versions = sorted(
                [str(v) for v in t2["version"].dropna().unique()],
                key=_version_sort_key,
            )
            prog = _progression_map(versions)
            for v in versions:
                vsub = t2[t2["version"].astype(str) == v]
                for role, types in [
                    ("const",  {"constant", "sig_changed"}),
                    ("target", {"version_specific"}),
                ]:
                    g    = vsub[vsub["api_type"].isin(types)]
                    vals = pd.to_numeric(g["t2_api_em"], errors="coerce").dropna().values.astype(float)
                    if len(vals):
                        rows.append({
                            "role": role, "task": "T2", "lib": lib,
                            "version": v, "progression": prog[v],
                            "value": float(vals.mean()),
                        })

        t3 = _try_load(lib, "t3", model_tag)
        if t3 is not None and "t3_param_f1" in t3.columns:
            t3       = classify_t3(t3)
            t3_valid = t3[t3["t3_taxonomy"] != "unknown"].copy()
            versions = sorted(
                [str(v) for v in t3_valid["version"].dropna().unique()],
                key=_version_sort_key,
            )
            prog = _progression_map(versions)
            for v in versions:
                vsub = t3_valid[t3_valid["version"].astype(str) == v]
                for role, types in [
                    ("const",  {"constant"}),
                    ("target", {"sig_changed", "version_specific"}),
                ]:
                    g    = vsub[vsub["api_type"].isin(types)]
                    vals = pd.to_numeric(g["t3_param_f1"], errors="coerce").dropna().values.astype(float)
                    if len(vals):
                        rows.append({
                            "role": role, "task": "T3", "lib": lib,
                            "version": v, "progression": prog[v],
                            "value": float(vals.mean()),
                        })

    return pd.DataFrame(rows)


def _aggregate_normalized_progression(
    df: pd.DataFrame, role: str,
) -> pd.DataFrame:
    rows = []
    sub  = df[df["role"] == role]
    for x in NORMALIZED_VERSION_ANCHORS:
        vals = sub[np.isclose(sub["progression"], x)]["value"].values
        if len(vals) == 0:
            rows.append({"progression": x, "mean": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        m, lo, hi = _bootstrap_ci(vals)
        rows.append({"progression": x, "mean": m * 100, "lo": lo * 100, "hi": hi * 100})
    return pd.DataFrame(rows)


def _plot_normalized_versions_family(
    ax: plt.Axes,
    family_name: str,
    libs: list[str],
    *,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    models  = _family_models(family_name)
    palette = COMBINED_PALETTES.get(family_name, ["#90CAF9", "#1565C0"])
    colors  = _model_colors(len(models), palette)
    plotted_labels: list[str] = []

    for mi, (model_tag, model_label) in enumerate(models):
        df = _normalized_task_version_rows(model_tag, libs)
        if df.empty:
            continue
        color = colors[mi]
        plotted_labels.append(model_label)

        for role, label, linestyle, linewidth, alpha_line, alpha_band in [
            ("const",  f"{model_label} · const", "--", LINEWIDTH_CONST,  ALPHA_CONST_LINE, ALPHA_CONST_CI),
            ("target", model_label,               "-",  LINEWIDTH_TARGET, 0.93,             ALPHA_TARGET_CI),
        ]:
            agg    = _aggregate_normalized_progression(df, role)
            marker = CONST_MARKERS[mi % len(CONST_MARKERS)]
            ax.plot(
                agg["progression"], agg["mean"],
                color=color, marker=marker, linestyle=linestyle,
                linewidth=linewidth,
                markersize=MARKERSIZE_CONST if role == "const" else MARKERSIZE_TARGET,
                alpha=alpha_line, label=label,
            )
            ax.fill_between(
                agg["progression"].values, agg["lo"].values, agg["hi"].values,
                color=color, alpha=alpha_band,
            )

    if not plotted_labels:
        ax.set_visible(False)
        return

    ax.set_xticks(NORMALIZED_VERSION_TICK_POSITIONS)
    ax.set_xticklabels(NORMALIZED_VERSION_TICK_LABELS, fontsize=FONT_XTICK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(35 if family_name == "qwen" else 50, 100)
    ax.set_title(_family_label(family_name), fontsize=FONT_PANEL_TITLE, fontweight="bold")
    _legend_paired_line_styles(ax, model_order=plotted_labels, loc="lower left")
    ax.grid(True, alpha=0.3)


def make_normalized_versions(
    libs: list[str],
    families: list[str],
    out_path: Path,
) -> None:
    n   = len(families)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)

    for col, fam in enumerate(families):
        _plot_normalized_versions_family(
            axes[0, col], fam, libs,
            xlabel="Normalized library version progression" if col == n // 2 else None,
            ylabel="Mean score across all tasks (%)" if col == 0 else None,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ["t1-lifecycle-L1", "t1-staircase-aggregated", "normalized-versions"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot", default="normalized-versions", choices=PLOT_CHOICES,
        help="Plot type to render (default: normalized-versions)",
    )
    parser.add_argument(
        "--lib", default="torch",
        help="Library for t1-lifecycle-L1 (default: torch)",
    )
    parser.add_argument(
        "--libs", nargs="+", default=["torch", "numpy", "scipy"],
        help="Libraries for staircase / normalized plots (default: torch numpy scipy)",
    )
    parser.add_argument(
        "--families", nargs="+", default=ROW_FAMILIES,
        choices=list(FAMILIES.keys()) + list(FAMILY_GROUPS.keys()),
        help="Ordered list of family panels (default: gpt sonnet-gemini qwen)",
    )
    parser.add_argument("--out", default=None, help="Output file path")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    elif args.plot == "t1-lifecycle-L1":
        out_path = EVAL_DIR / f"families_row_t1-lifecycle-L1_{args.lib}.pdf"
    else:
        lib_tag  = "_".join(args.libs)
        out_path = EVAL_DIR / f"families_row_{args.plot}_{lib_tag}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot == "t1-lifecycle-L1":
        make_t1_lifecycle_row(args.lib, args.families, out_path)
    elif args.plot == "t1-staircase-aggregated":
        make_aggregated_staircase(args.libs, args.families, out_path)
    else:
        make_normalized_versions(args.libs, args.families, out_path)


if __name__ == "__main__":
    main()
