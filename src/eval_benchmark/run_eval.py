"""Evaluation runner: load prompt templates, compose with benchmark data, call model API.

Reads benchmark_t{1,2,3}.parquet and sign_comp_matrix.parquet, applies the
sampling strategy (signature-lifetime for interesting APIs, version-diversified
for constant), then calls the model API and writes raw completions.

Sampling strategy (Option B — full coverage):
  - T1: all interesting APIs (K samples per signature lifetime) +
        N constant APIs (m samples version-diversified)
  - T2: all selected APIs × all versions (full coverage)
  - T3: all selected APIs × all versions with known signatures

Usage:
    uv run -m src.eval_benchmark.run_eval --lib torch --model openai/gpt-5-mini
    uv run -m src.eval_benchmark.run_eval --lib torch --model openai/gpt-5-mini \\
        --n-control-apis 200 --k-per-lifetime 3
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
import pandas as pd
import tyro
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = BASE / "prompts"

T1_LEVELS = ["L0", "L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dc.dataclass
class Args:
    """Run the evaluation benchmark: compose prompts from templates and call model API."""

    lib:            str             = "torch"
    model:          str             = "openai/gpt-5-mini"
    tasks:          Tuple[str, ...] = ("t1", "t2", "t3")
    levels:         Tuple[str, ...] = ("L0", "L1", "L2", "L3")

    n_interesting_apis: Optional[int] = None
    """Number of interesting APIs (sig_changed + version_specific) to include. None = all."""
    n_control_apis:     int           = 1000
    """Number of constant APIs to include as control group."""
    k_per_lifetime:     int           = 10
    """Max versions per signature lifetime."""
    m_per_control:      int           = 5
    """Versions per constant API."""
    n_per_cell:         int           = 3
    """Code samples per (API, version) cell."""

    max_rc_lines:   int             = 4
    """Max lines of right context (suffix) to include in FIM prompt."""
    run_t1b:        bool            = False
    """Run T1b (keyword-arg nudge) alongside T1a."""

    num_workers:    int             = 256
    seed:           int             = 42
    data_dir:       Optional[str]   = None
    """Where to find benchmark_t{1,2,3}.parquet and sign_comp_matrix.parquet. Default: out/benchmark_db/{lib}"""
    output_dir:     Optional[str]   = None
    """Where to write eval_raw_*.parquet. Default: out/eval_benchmark/{lib}"""


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATE_CACHE: dict[str, str] = {}


def _get_template(name: str) -> str:
    if name not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[name] = (PROMPTS_DIR / f"{name}.txt").read_text().strip()
    return _TEMPLATE_CACHE[name]


# ---------------------------------------------------------------------------
# Response post-processing (FIM echo stripping + verbose truncation)
# ---------------------------------------------------------------------------

def strip_echo(raw: str, code_str: str) -> str:
    """Remove redundant code_str prefix the model echoes back.

    With FIM the prompt already contains ``code_str.<FILL_HERE>``, so the model
    should NOT repeat ``code_str.``.  Handles ``code_str.``, ``code_str(``,
    ``code_str[`` prefixes.
    """
    if not raw or not code_str:
        return raw
    text = raw.strip()
    if text.startswith(code_str + "."):
        text = text[len(code_str) + 1:]
    elif text.startswith(code_str) and len(text) > len(code_str):
        if text[len(code_str)] in "([":
            text = text[len(code_str):]
    return text.strip()


def truncate_verbose(raw: str) -> str:
    """Keep only the first complete statement from a multi-line response.

    Tracks bracket depth so multi-line function calls are preserved intact.
    """
    if not raw:
        return raw
    lines = raw.splitlines()
    if len(lines) <= 2:
        return raw
    result_lines = [lines[0]]
    depth = 0
    for ch in lines[0]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    if depth > 0:
        for line in lines[1:]:
            result_lines.append(line)
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
            if depth <= 0:
                break
    return "\n".join(result_lines).strip()


def classify_shape(raw: str, code_str: str) -> str:
    """Classify the shape of a model response.

    Categories: clean, echo, verbose, verbose+echo, wrapped, explanation, empty.
    """
    if not raw or not raw.strip():
        return "empty"
    stripped = raw.strip()
    if "```" in raw:
        return "wrapped"
    lower = stripped.lower()
    if any(lower.startswith(p) for p in
           ("here", "the ", "this ", "sure", "i ", "to ", "note", "answer", "below")):
        return "explanation"
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if code_str and len(code_str) > 1 and stripped.startswith(code_str):
        return "verbose+echo" if len(lines) > 2 else "echo"
    if len(lines) > 2:
        return "verbose"
    return "clean"


_FENCE_RE = re.compile(
    r"```(?:\w*)\s*\n(.*?)```", re.DOTALL
)


_OPEN_FENCE_RE = re.compile(r"```(?:\w*)\s*\n")


def _unwrap_fences(text: str) -> str:
    """Extract code from markdown fences, taking the first block.

    Handles both closed (``` ... ```) and unclosed fences (model hit
    max_tokens before emitting the closing fence).
    """
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _OPEN_FENCE_RE.search(text)
    if m:
        return text[m.end():].strip()
    return text


def _strip_inline_backticks(text: str) -> str:
    """Strip surrounding inline backticks from a response."""
    s = text.strip()
    if s.startswith("`") and s.endswith("`") and "```" not in s:
        return s[1:-1].strip()
    if s.startswith("`") and "`" not in s[1:]:
        return s[1:].strip()
    return text


def _postprocess(raw: str, code_str: str, dot: str) -> tuple:
    """Run the full post-processing pipeline on a raw completion.

    Returns (cleaned_fill, shape_raw, shape_post).
    """
    shape_raw = classify_shape(raw, code_str)
    cleaned = _unwrap_fences(raw) if "```" in raw else raw
    cleaned = _strip_inline_backticks(cleaned)
    cleaned = strip_echo(cleaned, code_str)
    if dot and cleaned.startswith("."):
        cleaned = cleaned[1:]
    if classify_shape(cleaned, "") in ("verbose", "verbose+echo"):
        cleaned = truncate_verbose(cleaned)
    shape_post = classify_shape(cleaned, code_str)
    return cleaned, shape_raw, shape_post


# ---------------------------------------------------------------------------
# T2 generative: doc processing & prompt
# ---------------------------------------------------------------------------

_SECTION_HEADERS = re.compile(
    r"^(?:Parameters|Args|Returns|Return type|Raises|Example|Examples|Note|Warning"
    r"|Keyword [Aa]rg|Shape|Variables|Attributes|Methods|See Also|References"
    r"|Inputs|Outputs|Constraints)\s*:",
    re.MULTILINE,
)


def _strip_signature(doc: str) -> str:
    """Remove leading signature line(s) from a docstring."""
    lines = doc.split("\n")
    cleaned: list[str] = []
    past_sig = False
    for line in lines:
        stripped = line.strip()
        if not past_sig:
            if not stripped:
                continue
            if ("(" in stripped and ")" in stripped) or stripped.startswith("class "):
                continue
            past_sig = True
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_description(doc: str, max_chars: int = 600) -> str:
    """Extract prose description, stripping signature and reference sections."""
    if not doc:
        return ""
    doc = _strip_signature(doc)
    m = _SECTION_HEADERS.search(doc)
    if m:
        doc = doc[:m.start()]
    m2 = re.search(r"^\.\. ", doc, re.MULTILINE)
    if m2:
        doc = doc[:m2.start()]
    doc = doc.strip()
    if len(doc) > max_chars:
        doc = doc[:max_chars].rsplit(" ", 1)[0] + " …"
    return doc


def _redact_api_name(api: str, doc: str) -> str:
    """Replace every component of the API dotted path (case-insensitive)."""
    if not doc:
        return doc
    parts = api.split(".")
    for length in range(len(parts), 0, -1):
        for start in range(len(parts) - length + 1):
            fragment = ".".join(parts[start: start + length])
            doc = doc.replace(fragment, "[REDACTED]")
    for part in parts:
        if len(part) >= 2:
            doc = re.sub(r"\b" + re.escape(part) + r"\b", "[REDACTED]", doc, flags=re.I)
    doc = re.sub(r"(\[REDACTED\][\s.,;:]*){2,}", "[REDACTED] ", doc)
    return doc


_VERSION_PATTERNS = [
    re.compile(r"(?:Deprecated|Changed|Added|New)\s+(?:since|in)\s+version\s+\d+[\d.]*\s*:?", re.I),
    re.compile(r"(?:versionadded|versionchanged|deprecated)::\s*\d+[\d.]*", re.I),
    re.compile(r"(?:since|added in|new in|introduced in|removed in|changed in|deprecated in"
               r"|deprecated since)\s+(?:version\s+)?\d+[\d.]*", re.I),
    re.compile(r"(?:From|In|At)\s+version\s+\d+[\d.]*", re.I),
    re.compile(r"(?:PyTorch|torch)\s+(?:version\s+)?\d+\.\d+[\d.]*", re.I),
    re.compile(r"(?<!Python\s)version\s+\d+\.\d+[\d.]*", re.I),
    re.compile(r"\bv\d+\.\d+\b"),
]


def _scrub_versions(doc: str) -> str:
    """Remove explicit version references to prevent leakage."""
    for p in _VERSION_PATTERNS:
        doc = p.sub("", doc)
    return doc


def _build_doc_lookup(api_dict: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build ``{api: {version: doc_text}}`` preferring online over introspected."""
    online = {
        api: json.loads(s) if s else {}
        for api, s in zip(api_dict["api"], api_dict["online_doc_versions"].fillna(""))
    }
    intro = {
        api: json.loads(s) if s else {}
        for api, s in zip(api_dict["api"], api_dict["doc_versions"].fillna(""))
    }
    merged: Dict[str, Dict[str, str]] = {}
    for api in api_dict["api"]:
        ver_docs: Dict[str, str] = {}
        for ver in set(list(online.get(api, {})) + list(intro.get(api, {}))):
            ver_docs[ver] = online.get(api, {}).get(ver, "") or intro.get(api, {}).get(ver, "")
        if ver_docs:
            merged[api] = ver_docs
    return merged


def _prepare_t2_doc(api: str, version: str, doc_lookup: Dict[str, Dict[str, str]]) -> str:
    """Process raw docstring into a redacted description for T2 generative."""
    raw = doc_lookup.get(api, {}).get(version, "")
    desc = _extract_description(raw)
    desc = _redact_api_name(api, desc)
    desc = _scrub_versions(desc)
    return desc


def _find_collision_apis(
    doc_lookup: Dict[str, Dict[str, str]],
) -> Set[str]:
    """APIs whose (prefix, redacted description) is shared by multiple APIs."""
    key_to_apis: Dict[str, Set[str]] = {}
    for api, ver_docs in doc_lookup.items():
        prefix = ".".join(api.rsplit(".", 1)[:-1])
        for ver in ver_docs:
            desc = _prepare_t2_doc(api, ver, doc_lookup)
            if _classify_doc_quality(desc) not in _DOC_KEEP_QUALITIES:
                continue
            key_to_apis.setdefault(f"{prefix}|||{desc}", set()).add(api)
    return {a for apis in key_to_apis.values() if len(apis) >= 2 for a in apis}


_DOC_QUALITY_GENERIC = [
    re.compile(r"^Returns?\s+(?:the|a)\s", re.I),
    re.compile(r"^Gets?\s+(?:the|a)\s", re.I),
    re.compile(r"^Sets?\s+(?:the|a)\s", re.I),
    re.compile(r"^Creates?\s+(?:a|an)\s", re.I),
    re.compile(r"^Checks?\s", re.I),
    re.compile(r"^Computes?\s", re.I),
    # Forward references ("See ELU for more details.") — no information for the model
    re.compile(r"^See \w", re.I),
    # Meta-commentary, not descriptive ("This function is Array API compatible.")
    re.compile(r"^This (function|method|class)\b", re.I),
]


def _classify_doc_quality(desc: str) -> str:
    """Classify a processed description into a quality tier."""
    if not desc:
        return "empty"
    if desc.startswith("property ") or "= " in desc[:30]:
        return "stub_property"
    words = desc.split()
    if len(words) <= 4:
        return "too_short"
    n = len(desc)
    if n < 50 and any(p.match(desc) for p in _DOC_QUALITY_GENERIC):
        return "short_generic"
    if n < 50:
        return "short"
    if n < 100:
        return "medium"
    if n < 300:
        return "good"
    return "rich"


def _parse_t2_leaf(text: str, prefix: str) -> str:
    """Extract the leaf component from a T2 response and reconstruct the full API path."""
    if not text or text.startswith("ERROR:"):
        return ""
    text = text.strip().strip("`").strip()
    if text.startswith(prefix + "."):
        return text.rstrip("().,")
    leaf = text.splitlines()[0].strip().strip("`").rstrip("().,").strip()
    if "." in leaf:
        leaf = leaf.rsplit(".", 1)[-1]
    if " " in leaf:
        leaf = leaf.split()[0]
    if not leaf or not re.match(r"^[a-zA-Z_]\w*$", leaf):
        return ""
    return f"{prefix}.{leaf}"


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

def _compose_t1(row: dict, level: str, max_rc_lines: int = 4) -> str:
    tpl = _get_template(f"t1_{level}")

    doc_raw = str(row.get("docstring") or "")

    # Redacted docstring: extracted first sentence, API name scrubbed, versions removed
    doc_redacted = _extract_description(doc_raw)
    doc_redacted = _redact_api_name(row.get("api_norm", ""), doc_redacted)
    doc_redacted = _scrub_versions(doc_redacted)
    doc_redacted = doc_redacted if doc_redacted else "(no documentation available)"
    doc_redacted_commented = "\n".join(
        "# " + line for line in doc_redacted.splitlines())

    # Raw (truncated) docstring for legacy levels that still use it
    doc_trimmed = doc_raw[:600] if doc_raw else "(no documentation available)"
    doc_commented = "\n".join("# " + line for line in doc_trimmed.splitlines())

    code_str = row.get("code_str", "")
    dot = "." if code_str else ""

    rc = str(row.get("right_context_few_lines") or "")
    suffix = "\n".join(rc.splitlines()[:max_rc_lines]) if rc else ""

    return tpl.format(
        lib=row.get("library", ""),
        ver=row.get("lib_version", ""),
        api=row.get("api_norm", ""),
        sig=row.get("signature_at_version", ""),
        docstring=doc_commented,
        docstring_redacted=doc_redacted_commented,
        prompt=row.get("prompt", ""),
        code_str=code_str,
        dot=dot,
        suffix=suffix,
    )


def _compose_t2_messages(row: dict, doc_lookup: Dict[str, Dict[str, str]]) -> list:
    """Build multi-turn messages for generative T2 (with few-shot prior turn)."""
    api = row.get("api", "")
    version = row.get("version", "")
    lib = row.get("library", "")
    prefix = ".".join(api.rsplit(".", 1)[:-1])
    desc = _prepare_t2_doc(api, version, doc_lookup)
    prompt = _get_template("t2_user").format(
        lib=lib, version=version, prefix=prefix, doc=desc)
    return [
        {"role": "system", "content": _get_template("t2_system")},
        {"role": "user", "content": _get_template("t2_fewshot_user")},
        {"role": "assistant", "content": _get_template("t2_fewshot_assistant")},
        {"role": "user", "content": prompt},
    ]


def _compose_t3(row: dict) -> str:
    tpl = _get_template("t3")
    return tpl.format(lib=row.get("library", ""), version=row.get("version", ""), api=row.get("api", ""))


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

MAX_RETRIES = 10
REQUEST_TIMEOUT = 60


def _build_kwargs(model: str, max_tokens: int, temperature: float) -> dict:
    is_reasoning = any(tag in model for tag in ("gpt-5", "o1", "o3", "o4"))
    is_claude = any(tag in model for tag in ("claude", "anthropic"))
    effective_max = max_tokens + 512 if is_reasoning else max_tokens
    kwargs: dict = {"model": model, "max_tokens": effective_max, "timeout": REQUEST_TIMEOUT}
    if is_reasoning:
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = temperature
    if is_claude:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


def _call_completion(client: OpenAI, model: str, system: str, prompt: str,
                     max_tokens: int = 64, temperature: float = 0.0,
                     fewshot: list | None = None) -> str:
    kwargs = _build_kwargs(model, max_tokens, temperature)
    msgs = [{"role": "system", "content": system}]
    if fewshot:
        msgs.extend(fewshot)
    msgs.append({"role": "user", "content": prompt})
    kwargs["messages"] = msgs
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < MAX_RETRIES:
                log.debug("Retry %d for completion: %s", attempt + 1, e)
                time.sleep(1)
                continue
            log.warning("API error (completion, %d retries exhausted): %s", MAX_RETRIES, e)
            return ""
    return ""


def _call_chat(client: OpenAI, model: str, prompt: str,
               max_tokens: int = 100, temperature: float = 0.0) -> str:
    kwargs = _build_kwargs(model, max_tokens, temperature)
    kwargs["messages"] = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < MAX_RETRIES:
                log.debug("Retry %d for chat: %s", attempt + 1, e)
                time.sleep(1)
                continue
            log.warning("API error (chat, %d retries exhausted): %s", MAX_RETRIES, e)
            return ""
    return ""


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

_DOC_KEEP_QUALITIES = {"short", "medium", "good", "rich"}


def _run_t1(df: pd.DataFrame, client: OpenAI, model: str,
            levels: List[str], num_workers: int,
            max_rc_lines: int = 4,
            run_t1b: bool = True,
            collision_apis: frozenset = frozenset()) -> pd.DataFrame:
    system_a = _get_template("t1_system")
    system_b = _get_template("t1b_system") if run_t1b else None
    t1_fewshot = [
        {"role": "user", "content": _get_template("t1_fewshot_user")},
        {"role": "assistant", "content": _get_template("t1_fewshot_assistant")},
    ]

    def _process(idx_row):
        idx, row = idx_row
        rd = row.to_dict()
        code_str = str(rd.get("code_str", ""))
        dot = "." if code_str else ""
        doc_quality = _classify_doc_quality(
            _extract_description(str(rd.get("docstring") or "")))
        result = {"_idx": idx, "t1_doc_quality": doc_quality}
        for lv in levels:
            if lv in ("L1", "L2") and (
                doc_quality not in _DOC_KEEP_QUALITIES
                or rd.get("api_norm", "") in collision_apis
            ):
                continue
            prompt = _compose_t1(rd, lv, max_rc_lines)

            raw = _call_completion(client, model, system_a, prompt,
                                   fewshot=t1_fewshot)
            cleaned, shape_raw, shape_post = _postprocess(raw, code_str, dot)
            result[f"t1_{lv}_raw"] = raw
            result[f"t1_{lv}_completion"] = (code_str + dot + cleaned) if cleaned else ""
            result[f"t1_{lv}_shape_raw"] = shape_raw
            result[f"t1_{lv}_shape"] = shape_post

            if system_b is not None:
                raw_b = _call_completion(client, model, system_b, prompt,
                                         fewshot=t1_fewshot)
                cleaned_b, sr_b, sp_b = _postprocess(raw_b, code_str, dot)
                result[f"t1b_{lv}_raw"] = raw_b
                result[f"t1b_{lv}_completion"] = (code_str + dot + cleaned_b) if cleaned_b else ""
                result[f"t1b_{lv}_shape_raw"] = sr_b
                result[f"t1b_{lv}_shape"] = sp_b
        return result

    variant = "T1a+T1b" if run_t1b else "T1a"
    log.info("Running %s (%d samples, levels=%s)...", variant, len(df), levels)
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process, (i, row)): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="T1"):
            results.append(fut.result())

    result_df = df.join(pd.DataFrame(results).set_index("_idx"))
    _log_t1_metrics(result_df, levels, run_t1b)
    doc_levels = [lv for lv in ("L1", "L2") if lv in levels]
    if doc_levels and "t1_doc_quality" in result_df.columns:
        n_total = len(result_df)
        for dlv in doc_levels:
            col = f"t1_{dlv}_raw"
            n_kept = result_df[col].notna().sum() if col in result_df.columns else 0
            log.info("  %s doc quality filter: %d/%d samples kept (%.1f%%)",
                     dlv, n_kept, n_total, 100 * n_kept / n_total if n_total else 0)
    return result_df


def _log_t1_metrics(df: pd.DataFrame, levels: List[str], has_t1b: bool):
    """Log fail rate and echo rate summary for T1 completions."""
    prefixes = [("t1", "T1a")]
    if has_t1b:
        prefixes.append(("t1b", "T1b"))

    for prefix, label in prefixes:
        for lv in levels:
            shape_col = f"{prefix}_{lv}_shape"
            raw_col = f"{prefix}_{lv}_shape_raw"
            if shape_col not in df.columns:
                continue
            n = len(df)
            fail = int((df[shape_col] == "empty").sum())
            echo_raw = int((df[raw_col] == "echo").sum())
            echo_post = int((df[shape_col] == "echo").sum())
            clean = int((df[shape_col] == "clean").sum())
            log.info(
                "  %s %s — fail: %.1f%% (%d/%d), echo: %d raw→%d post, clean: %d/%d",
                label, lv, fail / n * 100, fail, n,
                echo_raw, echo_post, clean, n,
            )


def _run_t2(df: pd.DataFrame, client: OpenAI, model: str,
            doc_lookup: Dict[str, Dict[str, str]], num_workers: int) -> pd.DataFrame:
    kwargs_base = _build_kwargs(model, max_tokens=60, temperature=0.0)

    def _process(idx_row):
        idx, row = idx_row
        rd = row.to_dict()
        api = rd.get("api", "")
        prefix = ".".join(api.rsplit(".", 1)[:-1])
        desc = _prepare_t2_doc(api, rd.get("version", ""), doc_lookup)
        quality = _classify_doc_quality(desc)
        messages = _compose_t2_messages(rd, doc_lookup)
        kw = {**kwargs_base, "messages": messages}
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(**kw)
                raw = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    continue
                log.warning("T2 API error: %s", e)
                raw = ""
        pred = _parse_t2_leaf(raw, prefix)
        return {"_idx": idx, "t2_response": raw, "t2_pred_api": pred,
                "t2_doc_quality": quality}

    log.info("Running Task 2 — generative (%d samples)...", len(df))
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process, (i, row)): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="T2"):
            results.append(fut.result())

    return df.join(pd.DataFrame(results).set_index("_idx"))


def _run_t3(df: pd.DataFrame, client: OpenAI, model: str, num_workers: int) -> pd.DataFrame:
    system = _get_template("t3_system")
    t3_fewshot = [
        {"role": "user", "content": _get_template("t3_fewshot_user")},
        {"role": "assistant", "content": _get_template("t3_fewshot_assistant")},
    ]

    def _process(idx_row):
        idx, row = idx_row
        prompt = _compose_t3(row.to_dict())
        raw = _call_completion(client, model, system, prompt,
                               max_tokens=150, fewshot=t3_fewshot)
        return {"_idx": idx, "t3_response": raw}

    log.info("Running Task 3 (%d samples)...", len(df))
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process, (i, row)): i for i, row in df.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="T3"):
            results.append(fut.result())

    return df.join(pd.DataFrame(results).set_index("_idx"))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _to_short_ver(v: str) -> str:
    parts = str(v).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(v)


def _sample_t1(
    t1: pd.DataFrame,
    sign_comp: pd.DataFrame,
    n_interesting_apis: Optional[int],
    n_control_apis: int,
    k_per_lifetime: int,
    m_per_control: int,
    n_per_cell: int,
    seed: int,
) -> Tuple[pd.DataFrame, Set[str]]:
    """Sample T1 rows using signature-lifetime strategy.

    Returns (sampled_t1_df, selected_api_set).
    """
    rng = np.random.RandomState(seed)
    vcols = [c for c in sign_comp.columns if c.startswith("v_")]
    version_map = {c: c[2:] + ".0" for c in vcols}

    int_mask = t1["api_type"].isin({"sig_changed", "version_specific"})
    int_df = t1[int_mask]
    const_df = t1[~int_mask]
    int_apis = list(int_df["api_norm"].unique())

    # Optionally subsample interesting APIs
    if n_interesting_apis is not None and n_interesting_apis < len(int_apis):
        int_apis = list(rng.choice(int_apis, size=n_interesting_apis, replace=False))

    # -- Interesting: K samples per signature lifetime, version-diversified --
    sc_int = sign_comp[sign_comp["api"].isin(int_apis)]
    int_rows: list[pd.DataFrame] = []
    covered_apis: Set[str] = set()
    for _, row in sc_int.iterrows():
        api = row["api"]
        active_versions = [version_map[c] for c in vcols if pd.notna(row[c]) and bool(row[c])]
        api_rows = int_df[(int_df["api_norm"] == api) & (int_df["lib_version"].isin(active_versions))]
        if api_rows.empty:
            continue
        covered_apis.add(api)
        avail_vers = api_rows["lib_version"].unique()
        chosen_vers = rng.choice(avail_vers, size=min(k_per_lifetime, len(avail_vers)), replace=False)
        for v in chosen_vers:
            cell = api_rows[api_rows["lib_version"] == v]
            int_rows.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=seed))

    # Fallback: interesting APIs whose code samples don't overlap with
    # sign_comp active versions (e.g., version_specific APIs where code
    # comes from a version not tracked in the signature matrix).
    uncovered = set(int_apis) - covered_apis
    uncovered_with_rows = [a for a in uncovered if not int_df[int_df["api_norm"] == a].empty]
    for api in uncovered_with_rows:
        api_df = int_df[int_df["api_norm"] == api]
        avail_vers = api_df["lib_version"].unique()
        chosen_vers = rng.choice(avail_vers, size=min(k_per_lifetime, len(avail_vers)), replace=False)
        for v in chosen_vers:
            cell = api_df[api_df["lib_version"] == v]
            int_rows.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=seed))
    if uncovered_with_rows:
        log.info("T1 fallback: %d interesting APIs sampled outside sign_comp lifetimes",
                 len(uncovered_with_rows))

    # -- Constant: N APIs, m samples version-diversified --
    const_apis = list(const_df["api_norm"].unique())
    if n_control_apis < len(const_apis):
        const_apis = list(rng.choice(const_apis, size=n_control_apis, replace=False))

    const_rows: list[pd.DataFrame] = []
    for api in const_apis:
        api_df = const_df[const_df["api_norm"] == api]
        avail_vers = api_df["lib_version"].unique()
        chosen_vers = rng.choice(avail_vers, size=min(m_per_control, len(avail_vers)), replace=False)
        for v in chosen_vers:
            cell = api_df[api_df["lib_version"] == v]
            const_rows.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=seed))

    parts = int_rows + const_rows
    sampled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    selected_apis = set(int_apis) | set(const_apis)

    n_int = len(int_rows)
    n_const = len(const_rows)
    log.info(
        "T1 sampling: %d interesting APIs (%d rows, %d lifetimes) + "
        "%d constant APIs (%d rows) = %d total rows",
        len(int_apis), n_int, len(sc_int), len(const_apis), n_const, len(sampled),
    )
    return sampled, selected_apis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Args):
    data_dir = args.data_dir or f"out/benchmark_db/{args.lib}"
    output_dir = args.output_dir or f"out/eval_benchmark/{args.lib}"

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        max_retries=0,  # we handle retries ourselves
    )
    model_tag = args.model.replace("/", "_")
    os.makedirs(output_dir, exist_ok=True)

    # Load sign_comp for lifetime-based sampling
    sign_comp = pd.read_parquet(os.path.join(data_dir, "sign_comp_matrix.parquet"))

    # Compute shared API set from T1 sampling
    t1_full = pd.read_parquet(os.path.join(data_dir, "benchmark_t1.parquet"))
    t1_sampled, selected_apis = _sample_t1(
        t1_full, sign_comp,
        args.n_interesting_apis, args.n_control_apis,
        args.k_per_lifetime, args.m_per_control, args.n_per_cell,
        args.seed,
    )

    api_dict = pd.read_parquet(os.path.join(data_dir, "api_dictionary.parquet"))
    doc_lookup = _build_doc_lookup(api_dict)
    collision_apis = _find_collision_apis(doc_lookup)
    log.info("Doc lookup: %d APIs, %d with ambiguous descriptions",
             len(doc_lookup), len(collision_apis))

    if "t1" in args.tasks:
        result = _run_t1(t1_sampled, client, args.model, list(args.levels), args.num_workers,
                         max_rc_lines=args.max_rc_lines, run_t1b=args.run_t1b,
                         collision_apis=collision_apis)
        out = os.path.join(output_dir, f"eval_raw_t1_{model_tag}_{args.lib}.parquet")
        result.to_parquet(out, index=False)
        log.info("Saved T1: %s (%d rows)", out, len(result))

    # For T2/T3: keep ALL interesting APIs + only the T1-selected constant APIs.
    # This gives full coverage of sig_changed & version_specific while keeping
    # constant APIs aligned with T1 for cross-task analysis.
    t1_type_map = dict(zip(t1_full["api_norm"], t1_full["api_type"]))
    const_apis_selected = {a for a in selected_apis
                           if t1_type_map.get(a) == "constant"}
    INTERESTING_TYPES = {"sig_changed", "version_specific"}

    if "t2" in args.tasks:
        df = pd.read_parquet(os.path.join(data_dir, "benchmark_t2.parquet"))
        df = df[
            df["api_type"].isin(INTERESTING_TYPES)
            | df["api"].isin(const_apis_selected)
        ].reset_index(drop=True)
        n_before = len(df)
        df = df[df["gt_exists"]].reset_index(drop=True)
        log.info("T2: filtered gt_exists=True: %d → %d rows", n_before, len(df))

        df["_doc_quality"] = df.apply(
            lambda r: _classify_doc_quality(
                _prepare_t2_doc(r["api"], r["version"], doc_lookup)),
            axis=1,
        )
        n_pre_quality = len(df)
        df = df[df["_doc_quality"].isin(_DOC_KEEP_QUALITIES)].reset_index(drop=True)
        df.drop(columns=["_doc_quality"], inplace=True)
        log.info("T2: filtered doc quality (medium+): %d → %d rows",
                 n_pre_quality, len(df))

        n_pre_coll = len(df)
        df = df[~df["api"].isin(collision_apis)].reset_index(drop=True)
        log.info("T2: filtered ambiguous descriptions: %d → %d rows (%d collision APIs)",
                 n_pre_coll, len(df), len(collision_apis))

        n_int = df[df["api_type"].isin(INTERESTING_TYPES)]["api"].nunique()
        n_const = df[~df["api_type"].isin(INTERESTING_TYPES)]["api"].nunique()
        log.info("T2: %d rows (%d interesting + %d constant APIs)",
                 len(df), n_int, n_const)
        result = _run_t2(df, client, args.model, doc_lookup, args.num_workers)
        out = os.path.join(output_dir, f"eval_raw_t2_{model_tag}_{args.lib}.parquet")
        result.to_parquet(out, index=False)
        log.info("Saved T2: %s (%d rows)", out, len(result))

    if "t3" in args.tasks:
        df = pd.read_parquet(os.path.join(data_dir, "benchmark_t3.parquet"))
        df = df[
            df["api_type"].isin(INTERESTING_TYPES)
            | df["api"].isin(const_apis_selected)
        ].reset_index(drop=True)
        n_int = df[df["api_type"].isin(INTERESTING_TYPES)]["api"].nunique()
        n_const = df[~df["api_type"].isin(INTERESTING_TYPES)]["api"].nunique()
        log.info("T3: %d rows (%d interesting + %d constant APIs)",
                 len(df), n_int, n_const)
        result = _run_t3(df, client, args.model, args.num_workers)
        out = os.path.join(output_dir, f"eval_raw_t3_{model_tag}_{args.lib}.parquet")
        result.to_parquet(out, index=False)
        log.info("Saved T3: %s (%d rows)", out, len(result))


if __name__ == "__main__":
    main(tyro.cli(Args))
