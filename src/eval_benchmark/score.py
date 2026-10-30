"""Scoring: read raw eval results + matrices, compute all scores.

This is the only script that touches matrices for evaluation purposes.
Reads eval_raw_t{1,2,3} parquets and produces eval_scored_t{1,2,3} parquets.

Usage:
    uv run -m src.eval_benchmark.score --lib torch --model openai/gpt-5-mini
"""

from __future__ import annotations

import ast
import dataclasses as dc
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dc.dataclass
class Args:
    """Score raw evaluation results against compatibility matrices."""

    lib:            str             = "torch"
    model:          str             = "openai/gpt-5-mini"
    data_dir:       Optional[str]   = None
    """Where to find api_comp_matrix.parquet, sign_comp_matrix.parquet. Default: out/benchmark_db/{lib}"""
    eval_dir:       Optional[str]   = None
    """Where to find eval_raw_*.parquet. Default: out/eval_benchmark/{lib}"""
    output_dir:     Optional[str]   = None
    """Where to write eval_scored_*.parquet. Default: same as eval_dir"""


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------

def edit_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()


# ---------------------------------------------------------------------------
# API name extraction (legacy, kept for external callers)
# ---------------------------------------------------------------------------

def extract_api_name(completion: str, lib: str) -> Optional[str]:
    completion = completion.strip().strip("`").strip()
    pattern = rf"({re.escape(lib)}\.[a-zA-Z_][\w.]*)"
    m = re.search(pattern, completion)
    if m:
        return m.group(1).rstrip("(),")
    short = completion.split("(")[0].strip()
    if short and "." in short and " " not in short:
        return short
    return None


# ---------------------------------------------------------------------------
# Signature normalisation & param extraction (static, AST-based)
# ---------------------------------------------------------------------------

def _norm_sig(sig: str) -> str:
    sig = sig.strip().strip("`")
    if sig.startswith("def "):
        sig = sig[4:]
    sig = re.sub(r"\s*->\s*.+$", "", sig)
    sig = re.sub(r":\s*[A-Za-z_\[\]|,. ]+(?=[,)])", "", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig


def _param_names_from_sig(sig: str) -> Set[str]:
    m = re.search(r"\((.+)\)", sig)
    if not m:
        return set()
    inner = m.group(1)
    params = set()
    for token in inner.split(","):
        token = token.strip().lstrip("*")
        token = token.split(":")[0].split("=")[0].strip()
        if token and token not in ("self", "cls"):
            params.add(token)
    return params


def extract_keyword_args(code: str) -> Set[str]:
    """Extract keyword argument names from the outermost call only.

    We intentionally skip nested calls (e.g. kwargs inside a lambda body
    or chained method calls) to avoid counting unrelated arguments.
    The regex fallback also restricts to the first balanced parenthesised
    group to avoid picking up kwargs from later statements.
    """
    code = code.strip()
    if not code:
        return set()

    for src in [code, f"_={code}"]:
        try:
            tree = ast.parse(src, mode="eval" if src == code else "exec")
            # Find the first Call node at the top level (or top-level expr)
            call = _find_outermost_call(tree)
            if call is None:
                return set()
            return {kw.arg for kw in call.keywords if kw.arg is not None}
        except SyntaxError:
            continue

    # Regex fallback: restrict to first parenthesised group
    first_call = _first_paren_group(code)
    return set(re.findall(r"(?<![=!<>])(\w+)\s*=(?!=)", first_call))


def _find_outermost_call(tree: ast.AST) -> Optional[ast.Call]:
    """Return the first ast.Call at expression-statement level, without
    recursing into nested calls (lambdas, inner function args, etc.)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expr, ast.Assign, ast.Expression)):
            val = getattr(node, "value", getattr(node, "body", None))
            if isinstance(val, ast.Call):
                return val
            # Handle chained: expr.method(...)
            if isinstance(val, ast.Attribute) or isinstance(val, ast.Subscript):
                inner = val
                while hasattr(inner, "value"):
                    inner = inner.value
                    if isinstance(inner, ast.Call):
                        return inner
    # Fallback: first Call anywhere at depth-1
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Call):
            return node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                return child
    return None


def _first_paren_group(code: str) -> str:
    """Extract text inside the first balanced parenthesised group."""
    start = code.find("(")
    if start < 0:
        return code
    depth, end = 0, start
    for i in range(start, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return code[start + 1:end]


# ---------------------------------------------------------------------------
# Param-name F1
# ---------------------------------------------------------------------------

def param_f1(predicted: Set[str], ground_truth: Set[str]) -> Tuple[float, float, float]:
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _resolve_version(v: str) -> str:
    """Normalise version string to major.minor format."""
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


# ---------------------------------------------------------------------------
# T1 fill extraction & API resolution
# ---------------------------------------------------------------------------

def _extract_fill(text: str, code_str: str) -> str:
    """Strip code_str prefix and leading dot to isolate the model's fill.

    Also strips markdown code fences, which some models (e.g. Haiku) wrap
    the completion in even when the prefix is provided as plain text.
    """
    text, code_str = str(text), str(code_str)
    if code_str and text.startswith(code_str):
        r = text[len(code_str):]
        fill = r[1:] if r.startswith(".") else r
    else:
        fill = text
    return _strip_code_fence(fill)


def _strip_code_fence(text: str) -> str:
    r"""Remove markdown code fences that models sometimes wrap completions in.

    Handles both full fences (```python\\n...\\n```) and partial ones where the
    prefix was already provided by code_str (e.g. \`python\\nfoo()).
    """
    text = text.strip()
    # Full fence wrapping
    m = re.match(r"^```[\w]*\n?(.*?)(?:\n```.*)?$", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # Partial fence: leading backticks + optional lang tag + newline
    text = re.sub(r"^`+[a-zA-Z]*\n?", "", text)
    # Trailing backticks (possibly on their own line)
    text = re.sub(r"\n?`+\s*$", "", text)
    return text.strip()


def _fill_api_path(fill: str) -> str:
    """Extract dotted identifier before '(' from a fill string."""
    before_paren = fill.strip().split("(")[0].strip()
    return "" if " " in before_paren else before_paren


def _resolve_predicted_api(
    pred_api: str, tgt_api: str, api_norm: str,
) -> Optional[str]:
    """Reconstruct full canonical name by combining api_norm prefix with
    the predicted API path.  Returns None when alignment fails."""
    if not pred_api or not tgt_api or not api_norm:
        return None
    # Full path match at dot boundary
    if api_norm.endswith(tgt_api):
        idx = len(api_norm) - len(tgt_api)
        if idx == 0:
            return pred_api
        if api_norm[idx - 1] == ".":
            prefix = api_norm[:idx - 1]
            return f"{prefix}.{pred_api}"
    # Leaf-only fallback (chained method calls)
    tgt_leaf = tgt_api.rsplit(".", 1)[-1]
    if api_norm.endswith("." + tgt_leaf):
        prefix = api_norm[:-(len(tgt_leaf) + 1)]
        pred_leaf = pred_api.rsplit(".", 1)[-1]
        return f"{prefix}.{pred_leaf}" if prefix else pred_leaf
    return None


def _classify_api_taxonomy(
    pred_full: Optional[str],
    api_version_index: Dict[str, Set[str]],
    target_version: str,
) -> Dict[str, Optional[str]]:
    """Classify a (non-correct) predicted API by existence in api_comp."""
    if pred_full is None:
        return {"taxonomy": "unresolvable", "exists_versions": None,
                "temporal_direction": None}
    if pred_full in api_version_index:
        versions = api_version_index[pred_full]
        versions_csv = ",".join(sorted(versions))
        if target_version in versions:
            return {"taxonomy": "wrong_api_valid", "exists_versions": versions_csv,
                    "temporal_direction": None}
        direction = None
        try:
            tv = _ver_num(target_version)
            closest = min(versions, key=lambda v: abs(_ver_num(v) - tv))
            direction = "backward" if _ver_num(closest) < tv else "forward"
        except (ValueError, TypeError):
            pass
        return {"taxonomy": "anachronistic", "exists_versions": versions_csv,
                "temporal_direction": direction}
    return {"taxonomy": "phantom", "exists_versions": None,
            "temporal_direction": None}


# ---------------------------------------------------------------------------
# Cross-version param taxonomy
# ---------------------------------------------------------------------------

def _build_param_version_index(
    sign_comp: pd.DataFrame,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build per-API, per-version param sets + union of all params ever.

    Returns ``{api: {"all": Set[str], "1.10": Set[str], ...}}``.
    """
    vcols = [c for c in sign_comp.columns if c.startswith("v_")]
    index: Dict[str, Dict[str, Set[str]]] = {}
    for _, row in sign_comp.iterrows():
        api = row["api"]
        params = _param_names_from_sig(_norm_sig(str(row["signature"])))
        if api not in index:
            index[api] = {"all": set()}
        index[api]["all"] |= params
        for vcol in vcols:
            if pd.notna(row[vcol]) and row[vcol]:
                ver = vcol[2:]
                index[api].setdefault(ver, set())
                index[api][ver] |= params
    return index


def _classify_params(
    predicted: Set[str],
    gt_params: Set[str],
    all_params_ever: Set[str],
) -> Dict[str, object]:
    """Classify each predicted param as correct / anachronistic / phantom.

    Returns counts and rates (fractions of n_predicted).
    """
    if not predicted:
        return {
            "n_predicted": 0,
            "n_correct": 0, "n_anachronistic": 0, "n_phantom": 0,
            "correct_rate": None, "anachronistic_rate": None, "phantom_rate": None,
        }
    n = len(predicted)
    correct = predicted & gt_params
    wrong = predicted - gt_params
    anachronistic = wrong & all_params_ever
    phantom = wrong - all_params_ever
    return {
        "n_predicted": n,
        "n_correct": len(correct),
        "n_anachronistic": len(anachronistic),
        "n_phantom": len(phantom),
        "correct_rate": len(correct) / n,
        "anachronistic_rate": len(anachronistic) / n,
        "phantom_rate": len(phantom) / n,
    }


# ---------------------------------------------------------------------------
# Cross-version signature matching
# ---------------------------------------------------------------------------

def _build_sign_index(sign_comp: pd.DataFrame) -> Dict[str, List[Tuple[str, str, Set[str]]]]:
    vcols = [c for c in sign_comp.columns if c.startswith("v_")]
    index: Dict[str, List[Tuple[str, str, Set[str]]]] = {}
    for _, row in sign_comp.iterrows():
        api = row["api"]
        sig_norm = _norm_sig(str(row["signature"]))
        params = _param_names_from_sig(sig_norm)
        for vcol in vcols:
            if pd.notna(row[vcol]) and row[vcol]:
                index.setdefault(api, []).append((vcol[2:], sig_norm, params))
    return index


def best_version_match_f1(
    api: str, recalled_sig: str,
    sign_index: Dict[str, List[Tuple[str, str, Set[str]]]],
) -> Tuple[Optional[str], float]:
    entries = sign_index.get(api, [])
    if not entries:
        return None, 0.0
    recalled_params = _param_names_from_sig(_norm_sig(recalled_sig))
    best_ver: Optional[str] = None
    best_f1 = 0.0
    for ver, _sig_norm, gt_params in entries:
        _, _, f1 = param_f1(recalled_params, gt_params)
        if f1 > best_f1:
            best_f1 = f1
            best_ver = ver
    return best_ver, best_f1


# ---------------------------------------------------------------------------
# Task 1 scoring
# ---------------------------------------------------------------------------

def score_t1(df: pd.DataFrame, api_comp: pd.DataFrame,
             sign_comp: pd.DataFrame = None) -> pd.DataFrame:
    log.info("Scoring Task 1 (%d rows)...", len(df))
    df = df.copy()

    # Build API version index for O(1) existence checks
    vcols = [c for c in api_comp.columns if c.startswith("v_")]
    api_version_index: Dict[str, Set[str]] = {}
    for _, row in api_comp.iterrows():
        versions = {c[2:] for c in vcols if pd.notna(row[c]) and row[c]}
        api_version_index[row["api"]] = versions

    # Build param version index for arg taxonomy
    param_ver_index = (_build_param_version_index(sign_comp)
                       if sign_comp is not None else {})

    levels = [lv for lv in ["L0", "L1", "L2", "L3"]
              if f"t1_{lv}_completion" in df.columns]

    for lv in levels:
        comp_col = f"t1_{lv}_completion"
        raw_col = f"t1_{lv}_raw"

        def _score_api(row, _comp=comp_col, _raw=raw_col):
            raw_val = row.get(_raw)
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                return pd.Series({
                    "api_em": None, "api_taxonomy": None,
                    "pred_api_resolved": None,
                    "pred_api_exists_versions": None,
                    "pred_api_temporal_direction": None,
                    "api_leaf_es": None,
                })
            completion = str(row.get(_comp) or "")
            target = str(row.get("target") or "")
            code_str = str(row.get("code_str") or "")
            api_norm = str(row.get("api_norm") or "")
            version = _resolve_version(str(row.get("lib_version") or ""))

            pred_fill = _extract_fill(completion, code_str)
            tgt_fill = _extract_fill(target, code_str)
            pred_api = _fill_api_path(pred_fill)
            tgt_api = _fill_api_path(tgt_fill)

            api_em = bool(pred_api and tgt_api and pred_api == tgt_api)

            if api_em:
                taxonomy, pred_resolved = "correct", None
                exists_vers, temp_dir = None, None
            elif not pred_api:
                taxonomy, pred_resolved = "no_api", None
                exists_vers, temp_dir = None, None
            else:
                pred_resolved = _resolve_predicted_api(pred_api, tgt_api, api_norm)
                info = _classify_api_taxonomy(
                    pred_resolved, api_version_index, version)
                taxonomy = info["taxonomy"]
                exists_vers = info["exists_versions"]
                temp_dir = info["temporal_direction"]

            pred_leaf = pred_api.rsplit(".", 1)[-1] if pred_api else ""
            tgt_leaf = tgt_api.rsplit(".", 1)[-1] if tgt_api else ""
            leaf_es = (edit_similarity(pred_leaf, tgt_leaf)
                       if (pred_leaf and tgt_leaf) else 0.0)

            return pd.Series({
                "api_em": api_em, "api_taxonomy": taxonomy,
                "pred_api_resolved": pred_resolved,
                "pred_api_exists_versions": exists_vers,
                "pred_api_temporal_direction": temp_dir,
                "api_leaf_es": leaf_es,
            })

        api_scores = df.apply(_score_api, axis=1)
        for col in api_scores.columns:
            df[f"t1_{lv}_{col}"] = api_scores[col]

        def _score_params(row, _comp=comp_col, _raw=raw_col):
            raw_val = row.get(_raw)
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                return pd.Series({
                    "param_precision": None, "param_recall": None, "param_f1": None,
                    "param_correct_rate": None,
                    "param_anachronistic_rate": None,
                    "param_phantom_rate": None,
                })
            code_str = str(row.get("code_str") or "")
            completion = _extract_fill(str(row.get(_comp) or ""), code_str)
            predicted = extract_keyword_args(completion)
            gt_completion = _extract_fill(str(row.get("target") or ""), code_str)
            gt_params = extract_keyword_args(gt_completion)
            gt_sig = str(row.get("signature_at_version") or "")
            api_norm = str(row.get("api_norm") or "")
            base = {
                "param_precision": None, "param_recall": None, "param_f1": None,
                "param_correct_rate": None,
                "param_anachronistic_rate": None,
                "param_phantom_rate": None,
            }
            if not predicted or not gt_params:
                return pd.Series(base)
            p, r, f = param_f1(predicted, gt_params)
            base["param_precision"] = p
            base["param_recall"] = r
            base["param_f1"] = f
            sig_params = _param_names_from_sig(_norm_sig(gt_sig)) if gt_sig else set()
            api_entry = param_ver_index.get(api_norm, {})
            all_params = api_entry.get("all", set())
            cls = _classify_params(predicted, sig_params, all_params)
            base["param_correct_rate"] = cls["correct_rate"]
            base["param_anachronistic_rate"] = cls["anachronistic_rate"]
            base["param_phantom_rate"] = cls["phantom_rate"]
            return pd.Series(base)

        param_scores = df.apply(_score_params, axis=1)
        for pcol in ["param_precision", "param_recall", "param_f1",
                      "param_correct_rate", "param_anachronistic_rate",
                      "param_phantom_rate"]:
            df[f"t1_{lv}_{pcol}"] = param_scores[pcol]

        em_rate = df[f"t1_{lv}_api_em"].mean()
        log.info("  T1a %s: api_em=%.1f%%", lv, em_rate * 100)

    # T1b scoring (keyword-forced): param-name F1 is the primary metric
    t1b_levels = [lv for lv in ["L0", "L1", "L2", "L3"]
                  if f"t1b_{lv}_completion" in df.columns]
    for lv in t1b_levels:
        comp_col = f"t1b_{lv}_completion"

        def _score_t1b_params(row, _comp=comp_col):
            completion = _extract_fill(str(row.get(_comp) or ""),
                                       str(row.get("code_str") or ""))
            predicted = extract_keyword_args(completion)
            gt_sig = str(row.get("signature_at_version") or "")
            if not predicted or not gt_sig:
                return pd.Series({
                    "param_precision": None, "param_recall": None,
                    "param_f1": None})
            gt_params = _param_names_from_sig(_norm_sig(gt_sig))
            if not gt_params:
                return pd.Series({
                    "param_precision": None, "param_recall": None,
                    "param_f1": None})
            p, r, f = param_f1(predicted, gt_params)
            return pd.Series({
                "param_precision": p, "param_recall": r, "param_f1": f})

        param_scores = df.apply(_score_t1b_params, axis=1)
        df[f"t1b_{lv}_param_precision"] = param_scores["param_precision"]
        df[f"t1b_{lv}_param_recall"] = param_scores["param_recall"]
        df[f"t1b_{lv}_param_f1"] = param_scores["param_f1"]
        df[f"t1b_{lv}_pass"] = df[f"t1b_{lv}_param_f1"].apply(
            lambda x: x >= 0.6 if pd.notna(x) else False)

    log.info("  Task 1 scoring complete (T1a: %d levels, T1b: %d levels)",
             len(levels), len(t1b_levels))
    return df


# ---------------------------------------------------------------------------
# Task 2 scoring
# ---------------------------------------------------------------------------

def score_t2(df: pd.DataFrame, api_comp: pd.DataFrame) -> pd.DataFrame:
    """Score generative T2: predicted API name vs ground-truth API.

    With the prefix-based prompt, the module path is given; the model only
    predicts the leaf.  The primary metric is exact match (EM).
    """
    log.info("Scoring Task 2 (%d rows)...", len(df))
    df = df.copy()

    vcols = [c for c in api_comp.columns if c.startswith("v_")]
    api_version_index: Dict[str, Set[str]] = {}
    for _, row in api_comp.iterrows():
        versions = {c[2:] for c in vcols if pd.notna(row[c]) and row[c]}
        api_version_index[row["api"]] = versions

    pred = df["t2_pred_api"].fillna("")
    gt = df["api"].fillna("")

    df["t2_api_em"] = pred == gt
    df["t2_empty"] = pred == ""

    def _classify(row):
        if row["t2_api_em"]:
            return "correct"
        p = row["t2_pred_api"]
        if not p or (isinstance(p, float) and pd.isna(p)):
            return "no_api"
        version = _resolve_version(str(row.get("version", "")))
        info = _classify_api_taxonomy(p, api_version_index, version)
        return info["taxonomy"]

    df["t2_api_taxonomy"] = df.apply(_classify, axis=1)

    em = df["t2_api_em"].mean()
    empty = df["t2_empty"].mean()
    log.info("  Task 2: EM=%.1f%%, empty=%.1f%% (n=%d)", em * 100, empty * 100, len(df))
    return df


# ---------------------------------------------------------------------------
# Task 3 scoring
# ---------------------------------------------------------------------------

def score_t3(df: pd.DataFrame, sign_comp: pd.DataFrame) -> pd.DataFrame:
    log.info("Scoring Task 3 (%d rows)...", len(df))
    df = df.copy()
    sign_index = _build_sign_index(sign_comp)
    param_ver_index = _build_param_version_index(sign_comp)

    def _score_row(row):
        resp = _strip_code_fence(str(row.get("t3_response") or "").strip())
        gt = str(row.get("gt_signature") or "").strip()
        api = str(row.get("api", ""))
        base = {
            "t3_resp_norm": "", "t3_gt_norm": "",
            "t3_param_precision": None, "t3_param_recall": None,
            "t3_param_f1": None, "t3_es": None,
            "t3_best_match_version": None, "t3_best_match_f1": None,
            "t3_param_correct_rate": None,
            "t3_param_anachronistic_rate": None,
            "t3_param_phantom_rate": None,
        }
        if not resp or not gt:
            return pd.Series(base)
        resp_norm = _norm_sig(resp)
        gt_norm = _norm_sig(gt)
        resp_params = _param_names_from_sig(resp_norm)
        gt_params = _param_names_from_sig(gt_norm)
        p, r, f1 = param_f1(resp_params, gt_params)
        es = edit_similarity(resp_norm, gt_norm)
        best_ver, best_f1_val = best_version_match_f1(api, resp, sign_index)
        base.update({
            "t3_resp_norm": resp_norm, "t3_gt_norm": gt_norm,
            "t3_param_precision": p, "t3_param_recall": r,
            "t3_param_f1": f1, "t3_es": es,
            "t3_best_match_version": best_ver, "t3_best_match_f1": best_f1_val,
        })
        api_entry = param_ver_index.get(api, {})
        all_params = api_entry.get("all", set())
        cls = _classify_params(resp_params, gt_params, all_params)
        base["t3_param_correct_rate"] = cls["correct_rate"]
        base["t3_param_anachronistic_rate"] = cls["anachronistic_rate"]
        base["t3_param_phantom_rate"] = cls["phantom_rate"]
        return pd.Series(base)

    scores = df.apply(_score_row, axis=1)
    for col in scores.columns:
        df[col] = scores[col]
    valid = df["t3_param_f1"].dropna()
    log.info("  Task 3 mean param F1: %.3f (n=%d)", valid.mean(), len(valid))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    data_dir = args.data_dir or f"out/benchmark_db/{args.lib}"
    eval_dir = args.eval_dir or f"out/eval_benchmark/{args.lib}"
    output_dir = args.output_dir or eval_dir

    model_tag = args.model.replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)

    api_comp = pd.read_parquet(os.path.join(data_dir, "api_comp_matrix.parquet"))
    sign_comp = pd.read_parquet(os.path.join(data_dir, "sign_comp_matrix.parquet"))

    t1_raw = os.path.join(eval_dir, f"eval_raw_t1_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t1_raw):
        t1 = score_t1(pd.read_parquet(t1_raw), api_comp, sign_comp)
        out = os.path.join(output_dir, f"eval_scored_t1_{model_tag}_{args.lib}.parquet")
        t1.to_parquet(out, index=False)
        log.info("Saved: %s", out)

    t2_raw = os.path.join(eval_dir, f"eval_raw_t2_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t2_raw):
        t2 = score_t2(pd.read_parquet(t2_raw), api_comp)
        out = os.path.join(output_dir, f"eval_scored_t2_{model_tag}_{args.lib}.parquet")
        t2.to_parquet(out, index=False)
        log.info("Saved: %s", out)

    t3_raw = os.path.join(eval_dir, f"eval_raw_t3_{model_tag}_{args.lib}.parquet")
    if os.path.exists(t3_raw):
        t3 = score_t3(pd.read_parquet(t3_raw), sign_comp)
        out = os.path.join(output_dir, f"eval_scored_t3_{model_tag}_{args.lib}.parquet")
        t3.to_parquet(out, index=False)
        log.info("Saved: %s", out)


if __name__ == "__main__":
    main(tyro.cli(Args))
