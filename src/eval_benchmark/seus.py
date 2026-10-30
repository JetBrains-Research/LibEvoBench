"""Software Evolution Understanding Score (SEUS).

Usage:
    uv run -m src.eval_benchmark.seus
    uv run -m src.eval_benchmark.seus --libs torch numpy scipy
    uv run -m src.eval_benchmark.seus --models openai_gpt-5.4 openai_gpt-5.5
"""

from __future__ import annotations

import dataclasses as dc
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tyro

from scripts.plot_model_families import FAMILIES
from src.eval_benchmark.analysis import classify_t3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dc.dataclass
class Args:
    """Compute SEUS from eval_scored_t{1,2,3}_*.parquet files."""

    eval_dir:       str             = "out/eval_benchmark"
    """Directory containing one subdirectory per library with scored parquet files."""
    output_dir:     Optional[str]   = None
    """Where to write SEUS CSVs. Default: same as eval_dir."""
    libs:           Optional[list[str]] = None
    """Libraries to score. Default: discover all library folders with scored files."""
    models:         Optional[list[str]] = None
    """Model tags to score. Default: discover all models with scored files."""
    spread_penalty: float           = 0.5
    """Lambda in mean(B_v) - lambda * std(E_v)."""
    anachronism_penalty: float      = 0.5
    """Gamma multiplier for anachronistic error rate in the headline SEUS score."""


    anachronistic_weight: float     = 0.0
    """Diagnostic taxonomy partial-credit weight for anachronistic predictions."""
    wrong_api_valid_weight: float   = 0.5
    """Diagnostic taxonomy partial-credit weight for valid but wrong APIs."""
    correct_weight: float           = 1.0
    phantom_weight: float           = 0.0
    no_api_weight: float            = 0.0
    unknown_weight: float           = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONST_TYPES = {"constant"}
T1_EVOLVING_TYPES = {"version_specific"}
T2_EVOLVING_TYPES = {"version_specific"}
PARAM_EVOLVING_TYPES = {"sig_changed", "version_specific"}
T3_EVOLVING_TYPES = {"sig_changed", "version_specific"}


def _model_metadata() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for family, models in FAMILIES.items():
        for model_tag, label in models:
            out[model_tag] = (family, label)
    return out


MODEL_METADATA = _model_metadata()


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def _discover_libs(eval_dir: Path) -> list[str]:
    libs: list[str] = []
    for child in sorted(eval_dir.iterdir()):
        if child.is_dir() and any(child.glob("eval_scored_t*_*.parquet")):
            libs.append(child.name)
    return libs


def _parse_scored_name(path: Path, lib: str) -> Optional[tuple[str, str]]:
    prefix = "eval_scored_"
    suffix = f"_{lib}.parquet"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix):-len(suffix)]
    task, _, model_tag = stem.partition("_")
    if task not in {"t1", "t2", "t3"} or not model_tag:
        return None
    return task, model_tag


def _discover_models(eval_dir: Path, libs: list[str]) -> list[str]:
    models: set[str] = set()
    for lib in libs:
        for path in (eval_dir / lib).glob("eval_scored_t*_*.parquet"):
            parsed = _parse_scored_name(path, lib)
            if parsed is not None:
                _task, model_tag = parsed
                models.add(model_tag)
    return sorted(models)


def _try_load(eval_dir: Path, lib: str, task: str, model_tag: str) -> Optional[pd.DataFrame]:
    path = eval_dir / lib / f"eval_scored_{task}_{model_tag}_{lib}.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _family_label(model_tag: str) -> tuple[str, str]:
    if model_tag in MODEL_METADATA:
        return MODEL_METADATA[model_tag]
    return "other", model_tag.replace("_", "/")


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _version_sort_key(version: str) -> list[int]:
    parts = []
    for part in str(version).split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return parts


def _mean_pct(values: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean()) * 100.0


def _role_mean(df: pd.DataFrame, value_col: str, role_types: set[str]) -> Optional[float]:
    if "api_type" not in df.columns:
        return None
    sub = df[df["api_type"].isin(role_types)]
    if sub.empty:
        return None
    return _mean_pct(sub[value_col])


def _harmonic_pair(static_score: Optional[float],
                   evolving_score: Optional[float]) -> Optional[float]:
    if static_score is None or evolving_score is None:
        return None
    if static_score <= 0.0 or evolving_score <= 0.0:
        return 0.0
    return float(2.0 * static_score * evolving_score / (static_score + evolving_score))


def _ratio(evolving_score: Optional[float],
           static_score: Optional[float]) -> Optional[float]:
    if static_score is None or evolving_score is None or static_score <= 0.0:
        return None
    return float(evolving_score / static_score)


def _taxonomy_weights(args: Args) -> dict[str, float]:
    return {
        "correct": args.correct_weight,
        "wrong_api_valid": args.wrong_api_valid_weight,
        "anachronistic": args.anachronistic_weight,
        "phantom": args.phantom_weight,
        "no_api": args.no_api_weight,
        "unknown": args.unknown_weight,
    }


# ---------------------------------------------------------------------------
# Version-level SEUS inputs
# ---------------------------------------------------------------------------

def _append_version_scores(
    rows: list[dict],
    *,
    family: str,
    model_tag: str,
    model: str,
    lib: str,
    task: str,
    df: pd.DataFrame,
    version_col: str,
    value_col: str,
    const_types: set[str],
    evolving_types: set[str],
) -> None:
    if version_col not in df.columns or value_col not in df.columns:
        return

    valid = df[df[value_col].notna()].copy()
    versions = sorted(valid[version_col].dropna().astype(str).unique(), key=_version_sort_key)
    for version in versions:
        vsub = valid[valid[version_col].astype(str) == version]
        static_score = _role_mean(vsub, value_col, const_types)
        evolving_score = _role_mean(vsub, value_col, evolving_types)
        balanced = _harmonic_pair(static_score, evolving_score)
        if balanced is None:
            continue
        rows.append({
            "family": family,
            "model_tag": model_tag,
            "model": model,
            "library": lib,
            "task": task,
            "version": version,
            "static_score": static_score,
            "evolving_score": evolving_score,
            "evolution_retention": _ratio(evolving_score, static_score),
            "balanced_version_score": balanced,
        })


def compute_version_scores(
    eval_dir: Path,
    lib: str,
    model_tag: str,
    family: str,
    model: str,
) -> list[dict]:
    rows: list[dict] = []

    t1 = _try_load(eval_dir, lib, "t1", model_tag)
    if t1 is not None:
        _append_version_scores(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T1-L2", df=t1, version_col="lib_version",
            value_col="t1_L2_api_em", const_types=CONST_TYPES,
            evolving_types=T1_EVOLVING_TYPES,
        )
        _append_version_scores(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T1-L3 Param", df=t1, version_col="lib_version",
            value_col="t1_L3_param_recall", const_types=CONST_TYPES,
            evolving_types=PARAM_EVOLVING_TYPES,
        )

    t2 = _try_load(eval_dir, lib, "t2", model_tag)
    if t2 is not None:
        _append_version_scores(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T2", df=t2, version_col="version",
            value_col="t2_api_em", const_types=CONST_TYPES,
            evolving_types=T2_EVOLVING_TYPES,
        )

    t3 = _try_load(eval_dir, lib, "t3", model_tag)
    if t3 is not None:
        _append_version_scores(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T3", df=t3, version_col="version",
            value_col="t3_param_f1", const_types=CONST_TYPES,
            evolving_types=T3_EVOLVING_TYPES,
        )

    return rows


# ---------------------------------------------------------------------------
# Temporal-confusion penalty
# ---------------------------------------------------------------------------

def _append_anachronism_row(
    rows: list[dict],
    *,
    family: str,
    model_tag: str,
    model: str,
    lib: str,
    task: str,
    rate: Optional[float],
) -> None:
    if rate is None or pd.isna(rate):
        return
    rows.append({
        "family": family,
        "model_tag": model_tag,
        "model": model,
        "library": lib,
        "task": task,
        "anachronism_rate": float(rate),
    })


def _categorical_anachronism_rate(
    df: pd.DataFrame,
    taxonomy_col: str,
    role_types: set[str],
) -> Optional[float]:
    if "api_type" not in df.columns or taxonomy_col not in df.columns:
        return None
    sub = df[df[taxonomy_col].notna() & df["api_type"].isin(role_types)]
    if sub.empty:
        return None
    return float((sub[taxonomy_col].astype(str) == "anachronistic").mean()) * 100.0


def _rate_anachronism_rate(
    df: pd.DataFrame,
    rate_col: str,
    role_types: set[str],
    valid_col: str,
) -> Optional[float]:
    required = {"api_type", rate_col, valid_col}
    if not required.issubset(df.columns):
        return None
    sub = df[df[valid_col].notna() & df["api_type"].isin(role_types)]
    vals = pd.to_numeric(sub[rate_col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean()) * 100.0


def compute_anachronism_rates(
    eval_dir: Path,
    lib: str,
    model_tag: str,
    family: str,
    model: str,
) -> list[dict]:
    rows: list[dict] = []

    t1 = _try_load(eval_dir, lib, "t1", model_tag)
    if t1 is not None:
        _append_anachronism_row(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T1-L2",
            rate=_categorical_anachronism_rate(
                t1, "t1_L2_api_taxonomy", T1_EVOLVING_TYPES,
            ),
        )
        _append_anachronism_row(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T1-L3 Param",
            rate=_rate_anachronism_rate(
                t1, "t1_L3_param_anachronistic_rate",
                PARAM_EVOLVING_TYPES, "t1_L3_api_em",
            ),
        )

    t2 = _try_load(eval_dir, lib, "t2", model_tag)
    if t2 is not None:
        _append_anachronism_row(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T2",
            rate=_categorical_anachronism_rate(t2, "t2_api_taxonomy", T2_EVOLVING_TYPES),
        )

    t3 = _try_load(eval_dir, lib, "t3", model_tag)
    if t3 is not None:
        t3 = classify_t3(t3)
        _append_anachronism_row(
            rows, family=family, model_tag=model_tag, model=model, lib=lib,
            task="T3",
            rate=_categorical_anachronism_rate(t3, "t3_taxonomy", T3_EVOLVING_TYPES),
        )

    return rows


# ---------------------------------------------------------------------------
# Taxonomy diagnostics
# ---------------------------------------------------------------------------

def _taxonomy_score(series: pd.Series, weights: dict[str, float]) -> Optional[float]:
    vals = series.dropna().astype(str)
    if vals.empty:
        return None
    return float(vals.map(lambda v: weights.get(v, 0.0)).mean()) * 100.0


def _taxonomy_rate_score(
    df: pd.DataFrame,
    columns: dict[str, str],
    weights: dict[str, float],
) -> Optional[float]:
    available = {cat: col for cat, col in columns.items() if col in df.columns}
    if not available:
        return None
    scores = []
    for cat, col in available.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        scores.append(vals.fillna(0.0) * weights.get(cat, 0.0))
    if not scores:
        return None
    score = sum(scores)
    if score.dropna().empty:
        return None
    return float(score.mean()) * 100.0


def compute_taxonomy_diagnostics(
    eval_dir: Path,
    lib: str,
    model_tag: str,
    family: str,
    model: str,
    weights: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []

    t1 = _try_load(eval_dir, lib, "t1", model_tag)
    if t1 is not None and "api_type" in t1.columns:
        if "t1_L2_api_taxonomy" in t1.columns:
            sub = t1[
                t1["t1_L2_api_taxonomy"].notna()
                & t1["api_type"].isin(T1_EVOLVING_TYPES)
            ]
            score = _taxonomy_score(sub["t1_L2_api_taxonomy"], weights)
            if score is not None:
                rows.append({
                    "family": family, "model_tag": model_tag, "model": model,
                    "library": lib, "taxonomy_task": "T1-L2 API",
                    "taxonomy_reliability": score,
                })

        rate_cols = {
            "correct": "t1_L3_param_correct_rate",
            "anachronistic": "t1_L3_param_anachronistic_rate",
            "phantom": "t1_L3_param_phantom_rate",
        }
        if "t1_L3_api_em" in t1.columns and any(col in t1.columns for col in rate_cols.values()):
            sub = t1[
                t1["t1_L3_api_em"].notna()
                & t1["api_type"].isin(PARAM_EVOLVING_TYPES)
            ]
            score = _taxonomy_rate_score(sub, rate_cols, weights)
            if score is not None:
                rows.append({
                    "family": family, "model_tag": model_tag, "model": model,
                    "library": lib, "taxonomy_task": "T1-L3 Param",
                    "taxonomy_reliability": score,
                })

    t2 = _try_load(eval_dir, lib, "t2", model_tag)
    if t2 is not None and {"api_type", "t2_api_taxonomy"}.issubset(t2.columns):
        sub = t2[
            t2["t2_api_taxonomy"].notna()
            & t2["api_type"].isin(T2_EVOLVING_TYPES)
        ]
        score = _taxonomy_score(sub["t2_api_taxonomy"], weights)
        if score is not None:
            rows.append({
                "family": family, "model_tag": model_tag, "model": model,
                "library": lib, "taxonomy_task": "T2 API",
                "taxonomy_reliability": score,
            })

    t3 = _try_load(eval_dir, lib, "t3", model_tag)
    if t3 is not None and "api_type" in t3.columns:
        t3 = classify_t3(t3)
        if "t3_taxonomy" in t3.columns:
            sub = t3[t3["api_type"].isin(T3_EVOLVING_TYPES)]
            score = _taxonomy_score(sub["t3_taxonomy"], weights)
            if score is not None:
                rows.append({
                    "family": family, "model_tag": model_tag, "model": model,
                    "library": lib, "taxonomy_task": "T3 API",
                    "taxonomy_reliability": score,
                })

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(version_scores: pd.DataFrame,
              taxonomy_scores: pd.DataFrame,
              anachronism_scores: pd.DataFrame,
              spread_penalty: float,
              anachronism_penalty: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_keys = ["family", "model_tag", "model", "library", "task"]
    library_task = (
        version_scores
        .groupby(task_keys, as_index=False)
        .agg(
            static_score=("static_score", "mean"),
            evolving_score=("evolving_score", "mean"),
            evolving_std=("evolving_score", "std"),
            evolution_retention=("evolution_retention", "mean"),
            balanced_mean=("balanced_version_score", "mean"),
            balanced_min=("balanced_version_score", "min"),
            balanced_max=("balanced_version_score", "max"),
            n_versions=("balanced_version_score", "count"),
        )
    )
    library_task["evolving_std"] = library_task["evolving_std"].fillna(0.0)
    library_task["version_stability_penalty"] = spread_penalty * library_task["evolving_std"]
    if not anachronism_scores.empty:
        library_task = library_task.merge(
            anachronism_scores, on=task_keys, how="left",
        )
    else:
        library_task["anachronism_rate"] = 0.0
    library_task["anachronism_rate"] = library_task["anachronism_rate"].fillna(0.0)
    library_task["temporal_confusion_penalty"] = (
        anachronism_penalty * library_task["anachronism_rate"]
    )
    library_task["seus_library_task"] = (
        library_task["balanced_mean"]
        - library_task["version_stability_penalty"]
        - library_task["temporal_confusion_penalty"]
    ).clip(lower=0.0)
    library_task["balanced_range"] = (
        library_task["balanced_max"] - library_task["balanced_min"]
    )

    model_keys = ["family", "model_tag", "model"]
    model_scores = (
        library_task
        .groupby(model_keys, as_index=False)
        .agg(
            seus=("seus_library_task", "mean"),
            static_score=("static_score", "mean"),
            evolving_score=("evolving_score", "mean"),
            evolution_retention=("evolution_retention", "mean"),
            balanced_mean=("balanced_mean", "mean"),
            evolving_std=("evolving_std", "mean"),
            version_stability_penalty=("version_stability_penalty", "mean"),
            anachronism_rate=("anachronism_rate", "mean"),
            temporal_confusion_penalty=("temporal_confusion_penalty", "mean"),
            balanced_range=("balanced_range", "mean"),
            n_library_tasks=("seus_library_task", "count"),
        )
    )

    if not taxonomy_scores.empty:
        taxonomy_model = (
            taxonomy_scores
            .groupby(model_keys, as_index=False)
            .agg(
                taxonomy_reliability=("taxonomy_reliability", "mean"),
                n_taxonomy_views=("taxonomy_reliability", "count"),
            )
        )
        model_scores = model_scores.merge(taxonomy_model, on=model_keys, how="left")
    else:
        model_scores["taxonomy_reliability"] = np.nan
        model_scores["n_taxonomy_views"] = 0

    model_scores["n_taxonomy_views"] = model_scores["n_taxonomy_views"].fillna(0).astype(int)
    model_scores = model_scores.sort_values("seus", ascending=False)
    library_task = library_task.sort_values("seus_library_task", ascending=False)
    return model_scores, library_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args) -> None:
    if args.spread_penalty < 0.0:
        raise ValueError("spread_penalty must be non-negative")
    if args.anachronism_penalty < 0.0:
        raise ValueError("anachronism_penalty must be non-negative")

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir or args.eval_dir)
    os.makedirs(output_dir, exist_ok=True)

    libs = args.libs or _discover_libs(eval_dir)
    models = args.models or _discover_models(eval_dir, libs)
    weights = _taxonomy_weights(args)

    log.info("Scoring %d models across %d libraries", len(models), len(libs))

    version_rows: list[dict] = []
    taxonomy_rows: list[dict] = []
    anachronism_rows: list[dict] = []
    for model_tag in models:
        family, label = _family_label(model_tag)
        for lib in libs:
            version_rows.extend(compute_version_scores(eval_dir, lib, model_tag, family, label))
            anachronism_rows.extend(
                compute_anachronism_rates(eval_dir, lib, model_tag, family, label)
            )
            taxonomy_rows.extend(
                compute_taxonomy_diagnostics(eval_dir, lib, model_tag, family, label, weights)
            )

    if not version_rows:
        raise RuntimeError("No version score rows found. Check eval_dir/libs/models.")

    version_scores = pd.DataFrame(version_rows)
    taxonomy_scores = pd.DataFrame(taxonomy_rows)
    anachronism_scores = pd.DataFrame(anachronism_rows)
    model_scores, library_task_scores = summarize(
        version_scores, taxonomy_scores, anachronism_scores,
        args.spread_penalty, args.anachronism_penalty,
    )

    model_path = output_dir / "seus_model_scores.csv"
    library_task_path = output_dir / "seus_library_task_scores.csv"
    version_path = output_dir / "seus_version_scores.csv"
    taxonomy_path = output_dir / "seus_taxonomy_scores.csv"
    anachronism_path = output_dir / "seus_anachronism_scores.csv"

    model_scores.to_csv(model_path, index=False)
    library_task_scores.to_csv(library_task_path, index=False)
    version_scores.to_csv(version_path, index=False)
    taxonomy_scores.to_csv(taxonomy_path, index=False)
    anachronism_scores.to_csv(anachronism_path, index=False)

    log.info("Saved: %s", model_path)
    log.info("Saved: %s", library_task_path)
    log.info("Saved: %s", version_path)
    log.info("Saved: %s", taxonomy_path)
    log.info("Saved: %s", anachronism_path)

    cols = [
        "model", "seus", "static_score", "evolving_score", "evolution_retention",
        "version_stability_penalty", "temporal_confusion_penalty",
        "anachronism_rate", "taxonomy_reliability", "n_library_tasks",
    ]
    print(model_scores[cols].head(25).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main(tyro.cli(Args))
