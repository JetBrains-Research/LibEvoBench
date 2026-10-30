#!/usr/bin/env python3
"""Plot API and parameter taxonomy stacks in one compact figure.

4×3 grid with two compact legends: top half shows T1-L2 and T2 API taxonomy
(4 categories), bottom half shows T1-L3 and T3 parameter taxonomy (3 categories
+ recall line).

Usage:
    uv run scripts/plot_taxonomy_stacks.py --model openai/gpt-5.1
    uv run scripts/plot_taxonomy_stacks.py --model openai/gpt-5.1 --libs torch numpy scipy
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.eval_benchmark.analysis import classify_t3

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_DIR = Path("out/eval_benchmark")
DB_DIR   = Path("out/benchmark_db")

LIB_LABELS = {"torch": "PyTorch", "numpy": "NumPy", "scipy": "SciPy"}

# ── Font sizes ─────────────────────────────────────────────────────────────────
FONT_AXIS_LABEL  = 15
FONT_XTICK       = 13
FONT_LEGEND_BODY = 8
FONT_PANEL_TITLE = 15

# ── Taxonomy colours ───────────────────────────────────────────────────────────
TAXONOMY_COLORS = {
    "correct":         "#77BFA3",
    "wrong_api_valid": "#F2C14E",
    "anachronistic":   "#F26457",
    "phantom":         "#A6A6A6",
}

# 4-cat list for API taxonomy (T1-L2, T2)
API_TAXONOMY_CATS = [
    ("correct",         "Correct",       TAXONOMY_COLORS["correct"]),
    ("wrong_api_valid", "Wrong",         TAXONOMY_COLORS["wrong_api_valid"]),
    ("anachronistic",   "Anachronistic", TAXONOMY_COLORS["anachronistic"]),
    ("phantom",         "Phantom",       TAXONOMY_COLORS["phantom"]),
]

# 3-cat list for param taxonomy (T1-L3, T3)
PARAM_TAXONOMY_CATS = [
    ("correct",       "Correct",       TAXONOMY_COLORS["correct"]),
    ("anachronistic", "Anachronistic", TAXONOMY_COLORS["anachronistic"]),
    ("phantom",       "Phantom",       TAXONOMY_COLORS["phantom"]),
]

RECALL_COLOR = "#1565C0"

ROW_LABELS = ["API-C@L2", "API-I", "API-C@L3", "SR"]


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


def _ver_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        (c for c in df.columns if c.startswith("v_")),
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


def _resolve_version(v: str) -> str:
    parts = str(v).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(v)


def _version_sort_key(v: str) -> List[int]:
    return [int(x) for x in str(v).split(".") if x.isdigit()]


def _load_data(
    lib: str, model_tag: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
           Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    db_dir   = DB_DIR / lib
    eval_dir = EVAL_DIR / lib
    api_comp  = pd.read_parquet(db_dir / "api_comp_matrix.parquet")
    sign_comp = pd.read_parquet(db_dir / "sign_comp_matrix.parquet")

    def _try(task: str) -> Optional[pd.DataFrame]:
        p = eval_dir / f"eval_scored_{task}_{model_tag}_{lib}.parquet"
        return pd.read_parquet(p) if p.is_file() else None

    return _try("t1"), _try("t2"), _try("t3"), api_comp, sign_comp


# ── Shared plot helpers ────────────────────────────────────────────────────────

def _plot_stacked_rates(
    ax: plt.Axes,
    versions: List[str],
    values_by_cat: Dict[str, np.ndarray],
    cats: list,
) -> None:
    bottoms = np.zeros(len(versions), dtype=float)
    for cat, label, color in cats:
        vals = values_by_cat[cat]
        ax.bar(
            range(len(versions)), vals, bottom=bottoms,
            color=color, label=label, width=0.72, alpha=0.85,
        )
        bottoms += vals


def _style_axis(ax: plt.Axes, versions: List[str], *, upper: bool) -> None:
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=FONT_XTICK - 2, rotation=45, ha="right")
    ax.tick_params(axis="x", labelbottom=not upper, pad=1)
    ax.set_ylim(40, 100)
    ax.tick_params(axis="y", labelsize=FONT_XTICK - 2)
    ax.grid(True, alpha=0.25, axis="y")


def _recall_ci(values: pd.Series) -> Tuple[float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().values.astype(float)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    mean, lo, hi = _bootstrap_ci(vals)
    return mean * 100, lo * 100, hi * 100


def _plot_recall_line(
    ax: plt.Axes, x: List[int],
    means: List[float], lows: List[float], highs: List[float],
) -> None:
    ax.plot(x, means, color=RECALL_COLOR, marker="D", linewidth=2.0,
            markersize=5, label="Recall", zorder=5)
    ax.fill_between(x, lows, highs, color=RECALL_COLOR, alpha=0.12, zorder=4)


# ── API taxonomy panels (T1-L2, T2) ───────────────────────────────────────────

def _plot_t1_l2_api_taxonomy(
    ax: plt.Axes,
    t1: Optional[pd.DataFrame],
    api_comp: pd.DataFrame,
) -> None:
    level   = "L2"
    tax_col = f"t1_{level}_api_taxonomy"
    em_col  = f"t1_{level}_api_em"
    if t1 is None or tax_col not in t1.columns or em_col not in t1.columns:
        ax.set_visible(False)
        return

    versions     = _short_versions(api_comp)
    api_versions = _build_api_versions(api_comp)
    sub = t1[t1[em_col].notna() & (t1["api_type"] == "version_specific")].copy()

    values_by_cat: Dict[str, List[float]] = {cat: [] for cat, _, _ in API_TAXONOMY_CATS}
    for v in versions:
        valid = {api for api, vs in api_versions.items() if v in vs}
        vsub  = sub[sub["api_norm"].isin(valid)]
        for cat, _, _ in API_TAXONOMY_CATS:
            values_by_cat[cat].append(
                (vsub[tax_col] == cat).mean() * 100 if not vsub.empty else 0.0
            )

    _plot_stacked_rates(
        ax, versions,
        {cat: np.array(vals, dtype=float) for cat, vals in values_by_cat.items()},
        API_TAXONOMY_CATS,
    )
    _style_axis(ax, versions, upper=True)


def _plot_t2_api_taxonomy(
    ax: plt.Axes,
    t2: Optional[pd.DataFrame],
) -> None:
    tax_col = "t2_api_taxonomy"
    if t2 is None or tax_col not in t2.columns:
        ax.set_visible(False)
        return

    sub      = t2[t2["api_type"] == "version_specific"].copy()
    versions = sorted(
        [str(v) for v in sub["version"].dropna().unique()],
        key=_version_sort_key,
    )

    values_by_cat: Dict[str, List[float]] = {cat: [] for cat, _, _ in API_TAXONOMY_CATS}
    for v in versions:
        vsub = sub[sub["version"].astype(str) == v]
        for cat, _, _ in API_TAXONOMY_CATS:
            values_by_cat[cat].append(
                (vsub[tax_col] == cat).mean() * 100 if not vsub.empty else 0.0
            )

    _plot_stacked_rates(
        ax, versions,
        {cat: np.array(vals, dtype=float) for cat, vals in values_by_cat.items()},
        API_TAXONOMY_CATS,
    )
    _style_axis(ax, versions, upper=False)


# ── Param taxonomy panels (T1-L3, T3) ─────────────────────────────────────────

def _plot_t1_l3_param_taxonomy(
    ax: plt.Axes,
    t1: Optional[pd.DataFrame],
    sign_comp: pd.DataFrame,
) -> None:
    level      = "L3"
    recall_col = f"t1_{level}_param_recall"
    rate_cols  = {
        "correct":       f"t1_{level}_param_correct_rate",
        "anachronistic": f"t1_{level}_param_anachronistic_rate",
        "phantom":       f"t1_{level}_param_phantom_rate",
    }
    required = {f"t1_{level}_api_em", recall_col, *rate_cols.values()}
    if t1 is None or not required.issubset(t1.columns):
        ax.set_visible(False)
        return

    target_types = {"sig_changed", "version_specific"}
    sub = t1[t1[f"t1_{level}_api_em"].notna() & t1["api_type"].isin(target_types)].copy()
    sub["_short_ver"] = sub["lib_version"].map(_resolve_version)
    versions = _short_versions(sign_comp)

    values_by_cat: Dict[str, List[float]] = {cat: [] for cat, _, _ in PARAM_TAXONOMY_CATS}
    for v in versions:
        vsub = sub[sub["_short_ver"] == v]
        for cat, _, _ in PARAM_TAXONOMY_CATS:
            vals = pd.to_numeric(vsub[rate_cols[cat]], errors="coerce").dropna()
            values_by_cat[cat].append(float(vals.mean()) * 100 if not vals.empty else 0.0)

    _plot_stacked_rates(
        ax, versions,
        {cat: np.array(vals, dtype=float) for cat, vals in values_by_cat.items()},
        PARAM_TAXONOMY_CATS,
    )

    recall_y, recall_lo, recall_hi = [], [], []
    for v in versions:
        mean, lo, hi = _recall_ci(sub[sub["_short_ver"] == v][recall_col])
        recall_y.append(mean); recall_lo.append(lo); recall_hi.append(hi)
    _plot_recall_line(ax, list(range(len(versions))), recall_y, recall_lo, recall_hi)
    _style_axis(ax, versions, upper=True)


def _plot_t3_taxonomy(
    ax: plt.Axes,
    t3: Optional[pd.DataFrame],
) -> None:
    if t3 is None:
        ax.set_visible(False)
        return

    t3 = classify_t3(t3)
    if "t3_taxonomy" not in t3.columns:
        ax.set_visible(False)
        return

    target_types = {"sig_changed", "version_specific"}
    sub = t3[t3["api_type"].isin(target_types) & (t3["t3_taxonomy"] != "unknown")].copy()
    versions = sorted(
        [str(v) for v in sub["version"].dropna().unique()],
        key=_version_sort_key,
    )

    values_by_cat: Dict[str, List[float]] = {cat: [] for cat, _, _ in PARAM_TAXONOMY_CATS}
    for v in versions:
        vsub = sub[sub["version"].astype(str) == v]
        for cat, _, _ in PARAM_TAXONOMY_CATS:
            values_by_cat[cat].append(
                (vsub["t3_taxonomy"] == cat).mean() * 100 if not vsub.empty else 0.0
            )

    _plot_stacked_rates(
        ax, versions,
        {cat: np.array(vals, dtype=float) for cat, vals in values_by_cat.items()},
        PARAM_TAXONOMY_CATS,
    )
    _style_axis(ax, versions, upper=False)


# ── Combined figure ────────────────────────────────────────────────────────────

def plot_taxonomy_stacks_combined(
    model: str,
    libs: List[str],
    out_path: Path,
) -> None:
    model_tag = model.replace("/", "_")
    data = {lib: _load_data(lib, model_tag) for lib in libs}

    n_cols = len(libs)
    fig  = plt.figure(figsize=(5.4 * n_cols, 7.7))
    grid = fig.add_gridspec(5, n_cols, height_ratios=[1.0, 1.0, 0.30, 1.0, 1.0])
    axes = np.array([
        [fig.add_subplot(grid[0, col]) for col in range(n_cols)],
        [fig.add_subplot(grid[1, col]) for col in range(n_cols)],
        [fig.add_subplot(grid[3, col]) for col in range(n_cols)],
        [fig.add_subplot(grid[4, col]) for col in range(n_cols)],
    ])

    for col, lib in enumerate(libs):
        t1, t2, t3, api_comp, sign_comp = data[lib]
        _plot_t1_l2_api_taxonomy(axes[0, col], t1, api_comp)
        _plot_t2_api_taxonomy(axes[1, col], t2)
        _plot_t1_l3_param_taxonomy(axes[2, col], t1, sign_comp)
        _plot_t3_taxonomy(axes[3, col], t3)

        if axes[0, col].get_visible():
            axes[0, col].set_title(
                LIB_LABELS.get(lib, lib),
                fontsize=FONT_PANEL_TITLE, fontweight="bold", pad=10,
            )

    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].text(
            -0.105, 0.5, label, rotation=90,
            fontsize=FONT_PANEL_TITLE, fontweight="bold",
            ha="center", va="center", transform=axes[row, 0].transAxes,
            clip_on=False,
        )

    for row in (1, 3):
        if axes[row, n_cols // 2].get_visible():
            axes[row, n_cols // 2].set_xlabel(
                "Library version", fontsize=FONT_AXIS_LABEL - 1,
            )
    for ax in axes[1]:
        if ax.get_visible():
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("")

    api_legend = fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label, alpha=0.85)
            for _cat, label, color in API_TAXONOMY_CATS
        ],
        loc="upper center", ncol=len(API_TAXONOMY_CATS),
        frameon=True, fontsize=FONT_LEGEND_BODY + 3,
        bbox_to_anchor=(0.5425, 0.985), bbox_transform=fig.transFigure,
    )
    api_legend.get_frame().set_facecolor("white")
    api_legend.get_frame().set_edgecolor("#BDBDBD")
    api_legend.get_frame().set_alpha(0.95)

    param_handles = [
        Patch(facecolor=color, edgecolor="none", label=label, alpha=0.85)
        for _cat, label, color in PARAM_TAXONOMY_CATS
    ]
    param_handles.append(
        Line2D([0], [0], color=RECALL_COLOR, marker="D", linewidth=2.0,
               markersize=5, label="Recall")
    )
    param_legend = fig.legend(
        handles=param_handles,
        loc="center", ncol=len(param_handles),
        frameon=True, fontsize=FONT_LEGEND_BODY + 3,
        bbox_to_anchor=(0.5425, 0.492), bbox_transform=fig.transFigure,
    )
    param_legend.get_frame().set_facecolor("white")
    param_legend.get_frame().set_edgecolor("#BDBDBD")
    param_legend.get_frame().set_alpha(0.95)

    separator_y = 0.492
    for x0, x1 in [(0.09, 0.365), (0.72, 0.995)]:
        fig.add_artist(Line2D(
            [x0, x1], [separator_y, separator_y],
            transform=fig.transFigure, color="#BDBDBD",
            linewidth=0.9, alpha=0.9, zorder=1,
        ))

    fig.subplots_adjust(
        left=0.09, right=0.995, bottom=0.095, top=0.885,
        wspace=0.18, hspace=0.14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="openai/gpt-5.1")
    parser.add_argument("--libs", nargs="+", default=["torch", "numpy", "scipy"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_tag = args.model.replace("/", "_")
    out_path  = (
        Path(args.out) if args.out else
        EVAL_DIR / f"taxonomy_stacks_combined_{model_tag}.pdf"
    )
    plot_taxonomy_stacks_combined(args.model, args.libs, out_path)


if __name__ == "__main__":
    main()
