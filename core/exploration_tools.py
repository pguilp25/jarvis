"""Exploration-mode tools: PURPOSE, SEMANTIC, DETAIL.

These work WITHOUT pre-built indices (purpose_map / detailed_map /
semantic embeddings) — they parse source on demand. Designed for
the reading/exploration workflow where the user is orienting in
an unfamiliar codebase.

  PURPOSE  — module + public-symbol docstrings (the "gist" of a file)
  SEMANTIC — keyword-rank docstrings/comments for a NL query
  DETAIL   — deep dive on one symbol: signature, docstring, callers

Cheap. No embeddings, no LSP, no Phase-1 scan. AST + ripgrep only.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path
from typing import Optional


# ───────────────────────── PURPOSE ──────────────────────────────────


def _is_public(name: str) -> bool:
    """Public if it doesn't start with `_` (the Python convention).
    `__init__` is special-cased: it's the only dunder we still treat
    as public surface (a class's constructor is part of its API)."""
    if name == "__init__":
        return True
    return not name.startswith("_")


def _short_signature(node: ast.AST) -> str:
    """Compact signature for a def/class node — args only, no body.

    Preserves: keyword-only `*` marker, default values (shown as `=…`),
    return-annotation (shortened), and class kwargs (e.g. `metaclass=`).
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Build args with kwonly marker preserved.
        a = node.args
        defaults = list(a.defaults)
        n_pos = len(a.args)
        n_def = len(defaults)
        default_start = n_pos - n_def
        parts: list[str] = []
        for i, arg in enumerate(a.args):
            if i >= default_start:
                parts.append(f"{arg.arg}=…")
            else:
                parts.append(arg.arg)
        if a.vararg:
            parts.append(f"*{a.vararg.arg}")
        elif a.kwonlyargs:
            parts.append("*")
        for i, arg in enumerate(a.kwonlyargs):
            if i < len(a.kw_defaults) and a.kw_defaults[i] is not None:
                parts.append(f"{arg.arg}=…")
            else:
                parts.append(arg.arg)
        if a.kwarg:
            parts.append(f"**{a.kwarg.arg}")
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        ret = ""
        if node.returns is not None:
            try:
                ret_str = ast.unparse(node.returns)
                # Trim verbose annotations.
                if len(ret_str) > 40:
                    ret_str = ret_str[:37] + "…"
                ret = f" -> {ret_str}"
            except Exception:
                pass
        return f"{async_prefix}def {node.name}({', '.join(parts)}){ret}"
    if isinstance(node, ast.ClassDef):
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append("…")
        for kw in getattr(node, "keywords", []):
            try:
                bases.append(f"{kw.arg}={ast.unparse(kw.value)}")
            except Exception:
                pass
        suffix = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{suffix}"
    return ""


# Builtins / stdlib names that flood DETAIL's CALLS list without
# carrying useful structural information. Used by `_outgoing_calls`.
_CALL_NOISE = frozenset({
    # builtins / stdlib funcs
    "int", "str", "bool", "float", "list", "dict", "set", "tuple", "bytes",
    "len", "range", "zip", "enumerate", "map", "filter", "sorted", "reversed",
    "min", "max", "sum", "any", "all", "abs", "round", "pow", "divmod",
    "print", "input", "open", "iter", "next", "type", "id", "hash", "repr",
    "isinstance", "issubclass", "callable", "getattr", "setattr", "hasattr",
    "delattr", "vars", "dir", "globals", "locals", "super",
    # common method shortcuts that don't disambiguate symbols
    "append", "extend", "insert", "pop", "remove", "clear",
    "keys", "values", "items", "get", "update", "setdefault", "copy",
    "split", "join", "strip", "lstrip", "rstrip", "replace", "format",
    "startswith", "endswith", "find", "rfind", "index", "rindex",
    "lower", "upper", "title", "capitalize", "encode", "decode",
    "add", "discard", "intersection", "union", "difference",
    # exceptions (usually raised, not "called" in a useful sense)
    "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "AttributeError", "FileNotFoundError", "OSError",
    "ConnectionError", "TimeoutError", "NotImplementedError", "StopIteration",
    "StopAsyncIteration",
    # asyncio / common stdlib shortcuts
    "sleep", "wait_for", "gather", "create_task", "wait", "shield",
    "now", "today", "strftime", "strptime", "fromtimestamp",
    # file/path
    "exists", "isfile", "isdir", "getmtime", "getsize", "abspath", "dirname",
    "basename", "join", "splitext", "expanduser", "expandvars",
})


def _is_method_of_class(class_name: str, target_name: str, tree: ast.AST) -> bool:
    """True if `target_name` is defined as a method directly inside
    `class ClassName` somewhere in `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == target_name:
                        return True
    return False


def _first_line(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip()
    return s.splitlines()[0] if s else ""


_PURPOSE_DOCSTRING_LINE_CAP = 25
_PURPOSE_SYMBOL_CAP = 40
_PURPOSE_CLASS_METHOD_CAP = 12


def extract_purpose(filepath: str, project_root: str) -> str:
    """Return module docstring + each public symbol's signature + first
    docstring line. For classes, also lists their public methods.
    No code bodies; total output is capped so a 200-symbol file stays
    glanceable."""
    full = os.path.join(project_root, filepath) if not os.path.isabs(filepath) else filepath
    if not os.path.isfile(full):
        return f"=== PURPOSE: '{filepath}' — file not found ==="

    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception as e:
        return f"=== PURPOSE: '{filepath}' — read error: {e} ==="

    # Non-Python: try a heuristic header-comment extraction. `.pyi`
    # stubs are full Python syntactically — route them through the
    # AST path so the API surface they're designed to expose is
    # actually returned.
    if not filepath.endswith((".py", ".pyi")):
        return _extract_purpose_non_python(filepath, src)

    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError) as e:
        # ValueError fires on null bytes / non-UTF / surrogate input.
        msg = getattr(e, "msg", str(e))
        lineno = getattr(e, "lineno", None)
        loc = f"line {lineno}" if lineno else "unknown line"
        return f"=== PURPOSE: '{filepath}' — parse error at {loc}: {msg} ==="

    parts: list[str] = []
    parts.append(f"=== PURPOSE: {filepath} ===")
    n_lines = src.count("\n") + 1
    parts.append(f"({n_lines} lines)\n")

    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        parts.append("MODULE:")
        doc_lines = mod_doc.strip().splitlines()
        for ln in doc_lines[:_PURPOSE_DOCSTRING_LINE_CAP]:
            parts.append(ln)
        if len(doc_lines) > _PURPOSE_DOCSTRING_LINE_CAP:
            parts.append(f"… ({len(doc_lines) - _PURPOSE_DOCSTRING_LINE_CAP} more docstring lines)")
        parts.append("")
    else:
        parts.append("MODULE: (no docstring)\n")

    public_defs: list[tuple[ast.AST, str]] = []
    private_defs: list[tuple[ast.AST, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bucket = public_defs if _is_public(node.name) else private_defs
            bucket.append((node, node.name))

    if public_defs:
        parts.append("PUBLIC:")
        shown_defs = public_defs[:_PURPOSE_SYMBOL_CAP]
        for node, _ in shown_defs:
            sig = _short_signature(node)
            doc = _first_line(ast.get_docstring(node))
            line = f"  {sig}  [L{node.lineno}]"
            if doc:
                line += f"\n    — {doc}"
            parts.append(line)
            # For classes, also list public methods so an OO-heavy file
            # actually shows its API surface, not just the class name.
            if isinstance(node, ast.ClassDef):
                public_methods = [
                    m for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _is_public(m.name)
                ]
                if public_methods:
                    for m in public_methods[:_PURPOSE_CLASS_METHOD_CAP]:
                        msig = _short_signature(m)
                        mdoc = _first_line(ast.get_docstring(m))
                        sub = f"      · {msig}  [L{m.lineno}]"
                        if mdoc:
                            sub += f"\n        {mdoc}"
                        parts.append(sub)
                    if len(public_methods) > _PURPOSE_CLASS_METHOD_CAP:
                        parts.append(
                            f"      · … and {len(public_methods) - _PURPOSE_CLASS_METHOD_CAP} more methods"
                        )
        if len(public_defs) > _PURPOSE_SYMBOL_CAP:
            parts.append(f"  … and {len(public_defs) - _PURPOSE_SYMBOL_CAP} more public symbols")
        parts.append("")

    if private_defs:
        parts.append(f"PRIVATE ({len(private_defs)} symbols, names only):")
        names = ", ".join(name for _, name in private_defs[:_PURPOSE_SYMBOL_CAP])
        parts.append(f"  {names}")
        if len(private_defs) > _PURPOSE_SYMBOL_CAP:
            parts.append(f"  … and {len(private_defs) - _PURPOSE_SYMBOL_CAP} more")
        parts.append("")

    # Top-level assignments that look like module-level constants or config.
    consts = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id.isupper() or _is_public(target.id)
                ):
                    consts.append((target.id, node.lineno))
                    break
    if consts:
        parts.append(f"TOP-LEVEL ({len(consts)}):")
        parts.append("  " + ", ".join(f"{name} [L{ln}]" for name, ln in consts[:20]))
        if len(consts) > 20:
            parts.append(f"  … and {len(consts) - 20} more")

    return "\n".join(parts).rstrip() + "\n"


def _extract_purpose_non_python(filepath: str, src: str) -> str:
    """Heuristic: return the file's leading comment block (first 30 lines
    of comments / blank / module-doc style), useful for .md, .sh, .js,
    .ts files where AST parsing isn't worthwhile."""
    lines = src.splitlines()
    n_lines = len(lines)
    header_lines = []
    for i, ln in enumerate(lines[:40]):
        stripped = ln.strip()
        if not stripped:
            if header_lines:
                continue
            else:
                continue
        if stripped.startswith(("#", "//", "/*", "*", "<!--", '"""', "'''")):
            header_lines.append(ln)
        elif header_lines:
            break  # end of comment block
        else:
            break  # no leading comment
    out = [f"=== PURPOSE: {filepath} ===", f"({n_lines} lines)\n"]
    if header_lines:
        out.append("HEADER:")
        out.extend(header_lines)
    else:
        out.append("(no leading header comment)")
    return "\n".join(out) + "\n"


# ───────────────────────── SEMANTIC ─────────────────────────────────


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
    "for", "in", "on", "at", "by", "with", "and", "or", "but", "if", "then",
    "this", "that", "those", "these", "it", "its", "as", "from", "into",
    "do", "does", "did", "you", "your", "we", "our", "they", "their", "i",
    "what", "when", "where", "why", "how", "which", "who", "use", "using",
    "code", "function", "method", "class", "module", "file", "files",
    "would", "should", "could", "can", "will", "may", "might",
}


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, stopwords + 1-char removed.

    Splits on non-letter characters so `WIDGET_MAX_RETRIES` becomes
    {`widget`, `max`, `retries`} — matching the way doc text is
    tokenized in the scorer. Without this split, snake_case identifiers
    in the query would never match anything because the scorer regex
    `[a-z]+` produces single-word tokens.
    """
    raw = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in raw if len(w) > 1 and w not in _STOPWORDS}


def _classify_path(rel_path: str) -> str:
    """SOURCE / TEST / AUDIT bucket for ranking + sectioning.

    SOURCE — core production code (`core/`, `tools/`, `src/`, `lib/`,
             `workflows/`, normal top-level `.py` modules).
    TEST   — test scaffolding (`tests/`, `test_*.py`, `*_test.py`,
             `conftest.py`).
    AUDIT  — behavioral audits, experiments, fixtures, examples,
             benchmarks, documentation-as-code (UPGRADE_*, MIGRATION_*,
             *_GUIDE, *_NOTES). Reads like source but isn't load-bearing
             for the project's runtime behavior.
    """
    # Keep one normalized lower form for path-segment checks, and the
    # ORIGINAL case for the ALL-CAPS docfile heuristic.
    p_lower = rel_path.replace("\\", "/").lower()
    p_orig = rel_path.replace("\\", "/")
    base_lower = os.path.basename(p_lower)
    base_orig = os.path.basename(p_orig)
    if base_lower.startswith("test_") or base_lower.endswith("_test.py") or base_lower == "conftest.py":
        return "TEST"
    if "/tests/" in "/" + p_lower:
        return "TEST"
    if any(seg in "/" + p_lower for seg in (
        "/behavioral_audit/", "/audit/", "/audits/",
        "/fixtures/", "/examples/", "/benchmark/", "/benchmarks/",
        "/eval/", "/evals/", "/scratch/", "/playground/",
    )):
        return "AUDIT"
    # Filename heuristics for "documentation-as-Python" files: a
    # README/UPGRADE/MIGRATION/GUIDE-style top-level script reads like
    # source to grep but isn't a runtime callsite. Demote.
    stem_lower = base_lower[:-3] if base_lower.endswith(".py") else base_lower
    stem_orig = base_orig[:-3] if base_orig.endswith(".py") else base_orig
    if any(stem_lower.startswith(s) for s in (
        "upgrade_", "migration_", "notes_", "readme_", "handoff_"
    )):
        return "AUDIT"
    if stem_lower.endswith(("_guide", "_notes", "_readme", "_handoff")):
        return "AUDIT"
    # ALL-CAPS docs like UPGRADE_V5_GUIDE.py / HANDOFF_V9.py / NOTES.py.
    # Require the stem (without separators/digits) to be entirely
    # uppercase letters — that catches the doc-style filename without
    # demoting legitimate snake_case source files that happen to contain
    # the word "notes" or "guide" somewhere.
    letters_only = re.sub(r"[_\d]+", "", stem_orig)
    if letters_only and letters_only.isupper() and any(
        k in stem_orig.upper() for k in ("GUIDE", "NOTES", "README", "HANDOFF", "UPGRADE")
    ):
        return "AUDIT"
    return "SOURCE"


# Score multipliers applied AFTER raw scoring. SOURCE stays at 1.0
# baseline; TEST/AUDIT are demoted so a real source-of-truth file
# beats a test/audit that happens to mention the same tokens.
_BUCKET_PRIOR = {
    "SOURCE": 1.00,
    "AUDIT":  0.45,
    "TEST":   0.35,
}


# Position weights: where in a file's content a query token hits.
# Module docstring = file's topic statement → strongest signal.
# Public symbol docstring = the file's API surface → medium signal.
# Comments + identifiers = incidental usage → baseline signal.
_POSITION_WEIGHT = {
    "module_doc":  3.0,
    "symbol_doc":  1.5,
    "comment_id":  1.0,
}


def _score_with_positions(
    query_tokens: set[str],
    sections: dict[str, str],
    idf: dict[str, float],
) -> float:
    """Position- and IDF-weighted score for one file.

    `sections` is the file decomposed by signal location:
        {"module_doc": str, "symbol_doc": str, "comment_id": str}
    `idf` is the inverse-document-frequency of each query token,
    so rare tokens (e.g. "LSP") outweigh common ones ("get", "index").
    """
    if not query_tokens:
        return 0.0

    # Tokenize each section ONCE.
    section_tokens: dict[str, list[str]] = {}
    section_counts: dict[str, dict[str, int]] = {}
    total_len = 0
    for name, text in sections.items():
        toks = re.findall(r"[a-z]+", text.lower()) if text else []
        section_tokens[name] = toks
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        section_counts[name] = counts
        total_len += len(toks)

    if total_len == 0:
        return 0.0

    distinct_hits = 0
    weighted_hits = 0.0
    for q in query_tokens:
        q_idf = idf.get(q, 1.0)
        token_hit = False
        for section_name, counts in section_counts.items():
            c = counts.get(q, 0)
            if c > 0:
                token_hit = True
                weighted_hits += c * q_idf * _POSITION_WEIGHT[section_name]
            elif len(q) > 4:
                # Partial-match bonus for longer tokens (plurals / variants).
                partial = sum(
                    cnt for p, cnt in counts.items() if q in p and p != q
                )
                if partial > 0:
                    weighted_hits += partial * 0.5 * q_idf * _POSITION_WEIGHT[section_name]
        if token_hit:
            distinct_hits += 1

    if distinct_hits == 0 and weighted_hits == 0.0:
        return 0.0

    # Coverage: fraction of query tokens that appear (any section).
    coverage = distinct_hits / max(1, len(query_tokens))
    # Density: weighted hits per 1000 tokens.
    density = weighted_hits / max(1, total_len / 1000.0)
    # Coverage is the floor; density adds detail. NO cap so the
    # ranker can break ties on files that talk about the topic more.
    return 10.0 * coverage + density


def _collect_doc_sections(filepath: str) -> dict[str, str]:
    """Decompose a file into ranking-relevant sections.

    Returns a 3-bucket dict:
      module_doc — the file's top-of-file docstring (highest signal).
      symbol_doc — every public def/class's docstring (medium signal).
      comment_id — comments and identifier names (baseline signal).

    Non-Python files: leading comment block → module_doc, the rest of
    the first 200 lines → comment_id. Best-effort, but enough for
    keyword ranking.
    """
    empty = {"module_doc": "", "symbol_doc": "", "comment_id": ""}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception:
        return empty

    if not filepath.endswith((".py", ".pyi")):
        lines = src.splitlines()
        leader: list[str] = []
        for ln in lines[:40]:
            s = ln.strip()
            if s.startswith(("#", "//", "/*", "*", "<!--")):
                leader.append(s.lstrip("#/*<!- "))
            elif leader:
                break
            elif s:
                break
        return {
            "module_doc": "\n".join(leader),
            "symbol_doc": "",
            "comment_id": "\n".join(lines[:200]),
        }

    module_doc = ""
    symbol_doc_parts: list[str] = []
    comment_id_parts: list[str] = []

    try:
        tree = ast.parse(src)
        if (mod := ast.get_docstring(tree)):
            module_doc = mod
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _is_public(node.name) and (d := ast.get_docstring(node)):
                    symbol_doc_parts.append(d)
                # Identifier name goes to baseline bucket regardless.
                comment_id_parts.append(node.name)
        # Module-level constants (top-level NAME = …) — names included
        # in comment_id bucket so SEMANTIC can match config-style symbols.
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        comment_id_parts.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                comment_id_parts.append(node.target.id)
    except (SyntaxError, ValueError):
        # SyntaxError = broken Python; ValueError = null bytes / bad
        # encoding. Either way, fall through to the comment scan.
        pass

    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            comment_id_parts.append(s.lstrip("# "))

    return {
        "module_doc": module_doc,
        "symbol_doc": "\n".join(symbol_doc_parts),
        "comment_id": "\n".join(comment_id_parts),
    }


def _compute_idf(file_sections: list[tuple[str, dict[str, str]]],
                  query_tokens: set[str]) -> dict[str, float]:
    """Inverse document frequency for each query token.

    Rare tokens are weighted higher so a query like
    "LSP get index" doesn't let "get"/"index" dominate just because
    they're common words.
    """
    import math
    n_docs = max(1, len(file_sections))
    df: dict[str, int] = {q: 0 for q in query_tokens}
    for _, sections in file_sections:
        all_text = " ".join(sections.values()).lower()
        present_tokens = set(re.findall(r"[a-z]+", all_text))
        for q in query_tokens:
            if q in present_tokens:
                df[q] += 1
    # idf = log((N + 1) / (df + 1)) + 1.0; never below 1.0.
    return {
        q: math.log((n_docs + 1) / (df[q] + 1)) + 1.0
        for q in query_tokens
    }


_SKIP_DIRS = {
    ".git", "__pycache__", ".jarvis_sandbox", "node_modules", ".venv",
    "venv", "env", ".tox", "dist", "build", ".pytest_cache", ".mypy_cache",
    "logs", ".claude", "site-packages",
}

# Prefixes that match the `startswith` test on a directory name. Matches
# whole-name prefixes only (we still walk dirs whose NAME doesn't match).
_SKIP_DIR_PREFIXES = (
    "backup_", "venv_", "_venv", ".venv",
)

# Path-component substrings that indicate auto-generated / archive content
# we don't want polluting exploration results. Checked against the full
# relative path.
_SKIP_PATH_SUBSTRINGS = (
    "/variants/", "/rendered/", "/rendered_deep/", "/backup_",
)


def _path_is_skipped(rel_path: str) -> bool:
    # Normalize so the leading-slash check uniformly works for paths
    # written as "backup_x/foo.py" (no leading sep) or "a/backup_x/foo.py".
    sentinel = "/" + rel_path.replace("\\", "/")
    for sub in _SKIP_PATH_SUBSTRINGS:
        if sub in sentinel:
            return True
    return False


def _iter_source_files(project_root: str, exts: tuple[str, ...] = (".py",)):
    """Walk project, yielding source-file paths. Skips junk dirs and
    archive directories (backup_*, variants/, rendered/, etc.) so
    exploration results focus on live code."""
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not any(d.startswith(p) for p in _SKIP_DIR_PREFIXES)
        ]
        for fn in filenames:
            if fn.endswith(exts):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, project_root)
                if _path_is_skipped(rel):
                    continue
                yield full


def semantic_search(query: str, project_root: str, top_k: int = 10) -> str:
    """Keyword-rank source files by docstring/comment overlap with the
    natural-language query.

    Ranking combines: bucket prior (source > audit > test), inverse-
    document-frequency weighting on query tokens, and section-position
    weighting (module docstring > symbol docstring > comment).

    Output splits matches into SOURCE and TESTS/AUDIT sections so a
    reader chasing architecture lands on real source first.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        # Distinguish "no tokens after length filter" from "all tokens
        # were stopwords" so the diagnostic actually tells the user
        # what to do next.
        raw_tokens = re.findall(r"[a-zA-Z_]+", query.lower())
        if not raw_tokens:
            reason = "Query contained no letters."
        elif all(len(w) <= 1 for w in raw_tokens):
            reason = "Query tokens are too short (each ≤ 1 character)."
        elif all(w in _STOPWORDS for w in raw_tokens):
            reason = "Query reduced to stopwords (the, and, is, …)."
        else:
            reason = "Query tokens didn't survive the content filter."
        return (
            f"=== SEMANTIC: '{query}' ===\n"
            f"No content words to search. {reason}\n"
            f"Try a more specific phrase like \"where does X decide Y\".\n"
        )

    # Pass 1 — collect sections for every source file. (Cached in the
    # variable so we can compute IDF in pass 2 without re-reading.)
    file_sections: list[tuple[str, dict[str, str]]] = []
    for full_path in _iter_source_files(project_root):
        rel = os.path.relpath(full_path, project_root)
        sections = _collect_doc_sections(full_path)
        file_sections.append((rel, sections))

    if not file_sections:
        return f"=== SEMANTIC: '{query}' ===\nNo source files found under {project_root}."

    # Pass 2 — IDF for the query tokens.
    idf = _compute_idf(file_sections, q_tokens)

    # Pass 3 — score every file, classify by path, apply bucket prior.
    source_hits: list[tuple[float, str, str]] = []
    aux_hits: list[tuple[str, float, str, str]] = []  # (bucket, score, rel, preview)
    for rel, sections in file_sections:
        raw = _score_with_positions(q_tokens, sections, idf)
        if raw < 1.0:
            continue  # require at least one distinct token hit
        bucket = _classify_path(rel)
        score = raw * _BUCKET_PRIOR[bucket]
        preview = _build_preview(sections, q_tokens, rel, project_root)
        if bucket == "SOURCE":
            source_hits.append((score, rel, preview))
        else:
            aux_hits.append((bucket, score, rel, preview))

    source_hits.sort(key=lambda x: -x[0])
    aux_hits.sort(key=lambda x: -x[1])

    out = [f"=== SEMANTIC: '{query}' ==="]
    out.append(f"Query tokens: {', '.join(sorted(q_tokens))}")
    # Show IDF for transparency — a reader can see which tokens were
    # treated as rare/important and re-formulate if needed.
    idf_view = ", ".join(f"{t}={idf[t]:.1f}" for t in sorted(q_tokens))
    out.append(f"Token weights (IDF): {idf_view}")

    if not source_hits and not aux_hits:
        out.append(
            "\nNo files matched. The keyword ranker only looks at docstrings,\n"
            "comments, and identifier names. If your concept isn't named\n"
            "anywhere in those, try [SEARCH: pattern] with a concrete regex,\n"
            "or [REFS: <symbol>] if you know one symbol involved."
        )
        return "\n".join(out) + "\n"

    if source_hits:
        top_source = source_hits[:top_k]
        out.append(f"\nSOURCE ({len(top_source)} of {len(source_hits)}):\n")
        for score, rel, preview in top_source:
            out.append(f"  [score {score:.1f}] {rel}")
            if preview:
                for line in preview.splitlines():
                    out.append(f"    {line}")
            out.append("")

    if aux_hits:
        # Cap auxiliary at half the source budget — keep them visible
        # without burying the source-of-truth lookups.
        top_aux = aux_hits[:max(3, top_k // 2)]
        out.append(f"\nTESTS / AUDIT ({len(top_aux)} of {len(aux_hits)}):\n")
        for bucket, score, rel, preview in top_aux:
            out.append(f"  [{bucket} score {score:.1f}] {rel}")
            if preview:
                for line in preview.splitlines():
                    out.append(f"    {line}")
            out.append("")

    return "\n".join(out).rstrip() + "\n"


def _build_preview(sections: dict[str, str], q_tokens: set[str],
                    rel_path: str, project_root: str) -> str:
    """Pick the best preview line to show under each ranked result.
    Prefers a module-doc line that contains query tokens, then a
    public-symbol-doc line, then a comment from the source."""
    # Try the module docstring first.
    if sections.get("module_doc"):
        preview = _doc_preview_for_query(sections["module_doc"], q_tokens, 1)
        if preview:
            return f"MODULE: {preview}"
    # Try the symbol docstrings.
    if sections.get("symbol_doc"):
        preview = _doc_preview_for_query(sections["symbol_doc"], q_tokens, 1)
        if preview:
            return preview
    # Fall back to inline comments / identifiers.
    full = os.path.join(project_root, rel_path)
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f.read().splitlines(), start=1):
                s = ln.strip()
                if not s.startswith(("#", "//")):
                    continue
                words = set(re.findall(r"[a-z]+", s.lower()))
                if q_tokens & words:
                    return f"L{i}: {s[:120]}"
    except Exception:
        pass
    return ""


def _doc_preview_for_query(doc: str, q_tokens: set[str], max_lines: int) -> str:
    """Return the doc line(s) with the highest token-hit count, capped at
    max_lines. Empty if no line matches at all."""
    best_line = ""
    best_score = 0
    for ln in doc.splitlines():
        words = set(re.findall(r"[a-z]+", ln.lower()))
        hits = len(q_tokens & words)
        if hits > best_score:
            best_score = hits
            best_line = ln.strip()
    if best_score == 0:
        return ""
    return best_line[:160]


# ───────────────────────── DETAIL ───────────────────────────────────


def extract_detail(symbol: str, project_root: str) -> str:
    """Deep dive on one symbol: definition site, signature, docstring,
    body line count, and top callers (with 1 line of context each).

    `symbol` accepts either a bare identifier (`call_with_retry`) or a
    `Parent.member` dotted form. With a dotted form, only methods
    defined directly inside `class Parent` are returned — so a query
    for `DependencyIndex.__init__` doesn't dump every `__init__` in
    the project.
    """
    if not symbol or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", symbol):
        return f"=== DETAIL: '{symbol}' — not a valid identifier ==="

    parts = symbol.split(".")
    if len(parts) == 1:
        base_name = parts[0]
        parent_class = None
    else:
        # Last segment is the symbol; second-to-last is the parent class.
        base_name = parts[-1]
        parent_class = parts[-2]

    matches = _find_definitions(base_name, project_root, parent_class=parent_class)
    if not matches and parent_class is not None:
        # Fall back to bare-name search but flag the parent-class miss
        # so the model knows the qualifier didn't match anything.
        bare_matches = _find_definitions(base_name, project_root)
        if bare_matches:
            return (
                f"=== DETAIL: '{symbol}' ===\n"
                f"No method `{base_name}` found inside `class {parent_class}`.\n"
                f"({len(bare_matches)} unqualified `{base_name}` definitions exist; "
                f"call [DETAIL: {base_name}] to see them all, or check the parent class name.)"
            )

    if not matches:
        return (
            f"=== DETAIL: '{symbol}' ===\n"
            f"No `def {base_name}` or `class {base_name}` found in the project.\n"
            f"Try [REFS: {symbol}] to find usage sites without a definition,\n"
            f"or [SEARCH: pattern] to find string references."
        )

    out = [f"=== DETAIL: {symbol} ==="]
    if parent_class:
        out.append(f"(scoped to methods of class {parent_class})\n")
    elif len(matches) > 1:
        out.append(f"⚠ {len(matches)} definitions found — showing all:\n")
    for m in matches:
        out.append(_format_definition(m))
    # Don't exclude same-file callers wholesale — recursion and inline
    # callers (test harnesses, module-level dispatch) are legitimate
    # sites. Instead exclude only the def-line ranges of the matches
    # so the def itself doesn't get counted as a caller.
    def_ranges = [
        (m["file"], m["line"], getattr(m["node"], "end_lineno", m["line"]))
        for m in matches
    ]
    callers = _find_callers(base_name, project_root, exclude_def_ranges=def_ranges)
    out.append(_format_callers(base_name, callers))
    return "\n".join(out).rstrip() + "\n"


import builtins as _builtins
import keyword as _keyword
# Names that are NEVER a project dependency even if some file happens to define
# a function with the same name — builtins (`list`, `set`, `len`, `print`…),
# keywords, and the implicit method receivers. Excluding these is what keeps
# DEPENDSON precise: without it, a `list()` call resolved to a random project
# function named `list`.
_DEPENDSON_STOPWORDS = (
    set(dir(_builtins)) | set(_keyword.kwlist)
    | {"self", "cls", "args", "kwargs", "_"}
    # Ubiquitous builtin-TYPE methods (str/list/dict/set/file/re). These are
    # attribute calls like `x.get()`/`items.append()` whose receiver is almost
    # never a project type — resolving them by bare name to a same-named project
    # def is a false positive. Precision > recall here (a rare real project
    # method with one of these names is still reachable via REFS/CODE).
    | {"get", "set", "add", "pop", "append", "extend", "insert", "remove",
       "update", "setdefault", "keys", "values", "items", "copy", "clear",
       "sort", "reverse", "join", "split", "rsplit", "strip", "lstrip",
       "rstrip", "replace", "format", "startswith", "endswith", "lower",
       "upper", "title", "count", "find", "index", "encode", "decode",
       "read", "write", "close", "open", "flush", "seek", "readline",
       "readlines", "writelines", "group", "groups", "match", "search",
       "sub", "findall", "finditer", "splitlines", "discard", "popitem"})


def _build_def_name_index(project_root: str) -> dict:
    """name → sorted list of (relpath, lineno) for every def/class/method in the
    project. Built once per [DEPENDSON:] call and used to resolve a symbol's
    outgoing references to their definition sites — project-internal ONLY, so
    builtins/stdlib/third-party names (not defined here) are naturally excluded."""
    index: dict[str, list] = {}
    for full_path in _iter_source_files(project_root):
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
        except Exception:
            continue
        rel = os.path.relpath(full_path, project_root)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index.setdefault(node.name, []).append((rel, node.lineno))
    for name in index:
        index[name] = sorted(set(index[name]))
    return index


def extract_dependencies(symbol: str, project_root: str) -> str:
    """What `symbol` depends ON: the project-defined functions/classes it calls
    or references, each with its definition site(s). This is the REVERSE of the
    caller lookup ([DEPENDENCY:] = what depends on the symbol). AST-walks the
    symbol's body to collect referenced names, then resolves them against a
    project def index (builtins/stdlib drop out — they aren't defined here).

    `symbol` accepts a bare identifier or a `Parent.member` dotted form (the
    dotted form scopes to methods of `class Parent`)."""
    if not symbol or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", symbol):
        return f"=== DEPENDSON: '{symbol}' — not a valid identifier ==="

    parts = symbol.split(".")
    base_name = parts[-1]
    parent_class = parts[-2] if len(parts) > 1 else None

    matches = _find_definitions(base_name, project_root, parent_class=parent_class)
    if not matches:
        return (
            f"=== DEPENDSON: '{symbol}' ===\n"
            f"No `def {base_name}` or `class {base_name}` found in the project.\n"
            f"Try [REFS: {symbol}] to find usage sites, or [SEARCH: pattern]."
        )

    index = _build_def_name_index(project_root)
    out = [f"=== DEPENDSON: {symbol} — what it depends on ==="]
    for m in matches:
        node = m["node"]
        used: set[str] = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Name):
                    used.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    used.add(fn.attr)
            elif isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)

        deps = []
        for name in sorted(used):
            if name == base_name or name in _DEPENDSON_STOPWORDS:
                continue                       # skip self/recursion + builtins
            sites = index.get(name)
            if sites:
                deps.append((name, sites))

        if not deps:
            out.append(
                f"\n{m['file']}:{m['line']} {base_name} — no project-internal "
                f"dependencies (only builtins / stdlib / parameters)."
            )
            continue
        out.append(
            f"\n{m['file']}:{m['line']} {base_name} depends on "
            f"{len(deps)} project symbol(s):"
        )
        for name, sites in deps[:40]:
            loc = ", ".join(f"{f}:{ln}" for f, ln in sites[:3])
            more = f" (+{len(sites) - 3} more)" if len(sites) > 3 else ""
            out.append(f"  • {name}  →  {loc}{more}")
        if len(deps) > 40:
            out.append(f"  … +{len(deps) - 40} more")
    return "\n".join(out).rstrip() + "\n"


def _find_definitions(name: str, project_root: str,
                       parent_class: str | None = None) -> list[dict]:
    """Return list of {file, line, node, source} dicts for every
    `def name` / `class name` / `async def name` matching `name`.

    If `parent_class` is provided, only methods defined directly
    inside `class {parent_class}` are returned — so
    `extract_detail("DependencyIndex.refresh")` no longer dumps every
    `refresh` def in the project.
    """
    found = []
    for full_path in _iter_source_files(project_root):
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue

        if parent_class is not None:
            # Only methods directly inside the named class count.
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == parent_class:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                           and child.name == name:
                            found.append({
                                "file": os.path.relpath(full_path, project_root),
                                "full_path": full_path,
                                "line": child.lineno,
                                "node": child,
                                "source": src,
                                "parent_class": parent_class,
                            })
        else:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == name:
                        found.append({
                            "file": os.path.relpath(full_path, project_root),
                            "full_path": full_path,
                            "line": node.lineno,
                            "node": node,
                            "source": src,
                        })
    return found


def _format_definition(m: dict) -> str:
    node = m["node"]
    sig = _short_signature(node)
    doc = ast.get_docstring(node) or "(no docstring)"
    src_lines = m["source"].splitlines()
    end_line = getattr(node, "end_lineno", node.lineno)
    body_lines = max(0, end_line - node.lineno)

    parts: list[str] = []
    parts.append(f"DEFINED in {m['file']}:{m['line']}")
    parts.append(f"  {sig}")
    parts.append(f"  {body_lines}-line body")
    parts.append("")
    parts.append("  DOCSTRING:")
    for ln in doc.splitlines()[:8]:
        parts.append(f"    {ln}")
    if len(doc.splitlines()) > 8:
        parts.append(f"    … ({len(doc.splitlines()) - 8} more lines)")
    parts.append("")

    # Calls made BY this symbol — outgoing dependencies.
    out_calls = _outgoing_calls(node)
    if out_calls:
        head = sorted(out_calls)[:15]
        parts.append(f"  CALLS ({len(out_calls)}): " + ", ".join(head))
        if len(out_calls) > 15:
            parts.append(f"    … and {len(out_calls) - 15} more")
        parts.append("")
    return "\n".join(parts)


def _outgoing_calls(node: ast.AST) -> set[str]:
    """Set of function-name strings called inside this node's body,
    minus builtins / common stdlib shortcuts that don't carry
    structural signal (`isinstance`, `len`, `get`, `append`, …).
    See `_CALL_NOISE`."""
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                if func.id not in _CALL_NOISE:
                    calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                if func.attr not in _CALL_NOISE:
                    calls.add(func.attr)
    return calls


def _find_callers(name: str, project_root: str,
                   exclude_files: set[str] = None,
                   exclude_def_ranges: list[tuple[str, int, int]] | None = None,
                   max_results: int = 12) -> list[tuple[str, int, str]]:
    """Use ripgrep to find word-bounded matches across the project.
    Returns (relpath, line_no, context_line) tuples. Filters out:
      - paths under archive/snapshot/variant directories (live code only)
      - files listed in `exclude_files` (legacy: whole-file exclusion)
      - the def-line ranges in `exclude_def_ranges` (so recursive
        calls and same-file callers ARE included, but the def itself
        isn't counted as its own caller)
    """
    exclude_files = exclude_files or set()
    exclude_def_ranges = exclude_def_ranges or []
    glob_skips = [
        "!**/__pycache__/**",
        "!**/.jarvis_sandbox/**",
        "!**/logs/**",
        "!**/backup_*/**",
        "!**/venv_*/**",
        "!**/.venv/**",
        "!**/site-packages/**",
        "!**/variants/**",
        "!**/rendered/**",
        "!**/rendered_deep/**",
        "!**/.claude/**",
        "!**/node_modules/**",
    ]
    try:
        rg_args = [
            "rg", "--vimgrep", "--no-heading", "--smart-case",
            "-w", name,
            "--type-add", "py:*.py", "-tpy",
        ]
        for g in glob_skips:
            rg_args += ["--glob", g]
        rg_args.append(project_root)
        proc = subprocess.run(rg_args, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []

    results = []
    for line in proc.stdout.splitlines():
        # vimgrep format: path:line:col:content
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        path, ln, _, content = parts
        rel = os.path.relpath(path, project_root)
        if rel in exclude_files:
            continue
        if _path_is_skipped(rel):
            continue
        # Skip definition lines (handled by DEFINED).
        cs = content.strip()
        if cs.startswith(("def ", "class ", "async def ")) and re.search(
            rf"\b(def|class)\s+{re.escape(name)}\b", cs
        ):
            continue
        try:
            ln_int = int(ln)
        except ValueError:
            continue
        # Skip lines INSIDE a known def-range (the def's own body has
        # been excluded by the caller — but recursive calls or module-
        # level callers in the same file remain).
        inside_def = any(
            rel == def_file and def_start <= ln_int <= def_end
            for def_file, def_start, def_end in exclude_def_ranges
        )
        if inside_def:
            # Allow recursive calls — they appear inside the def range.
            # Detect: the matched word equals `name` and the line itself
            # is not the `def `/`class ` line. We keep these.
            if cs.startswith(("def ", "class ", "async def ")):
                continue
            # Filter false-positive matches that AREN'T real calls:
            # docstrings, comments, string literals. A real recursive
            # call has the name followed by `(` or appears as a member
            # access (e.g. `self.foo()` for a method called `foo`).
            stripped = cs.lstrip()
            if stripped.startswith(("#", '"""', "'''", '"', "'")):
                continue
            # Require the name to appear in a call-shape — either
            # `name(` or `.name(` or `name =` (for assignments to
            # name). If none match, drop the false positive.
            esc = re.escape(name)
            if not re.search(rf"\b{esc}\s*\(|\.{esc}\s*\(", cs):
                continue
            # Real recursive callsite — keep it, label it.
            results.append((rel, ln_int, "[recursive] " + content.strip()[:108]))
            if len(results) >= max_results * 3:
                break
            continue
        results.append((rel, ln_int, content.strip()[:120]))
        if len(results) >= max_results * 3:  # over-fetch, then filter
            break

    # De-dup by (file, line).
    seen = set()
    unique = []
    for r in results:
        key = (r[0], r[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique[:max_results]


def _format_callers(name: str, callers: list[tuple[str, int, str]]) -> str:
    if not callers:
        return f"CALLERS: 0 (use [REFS: {name}] for substring matches)"
    out = [f"CALLERS ({len(callers)} sites):"]
    # Common method names collide with stdlib (`dict.get`, `list.append`,
    # `str.split`). Warn the model that some hits may be unrelated.
    if name in _CALL_NOISE or len(name) <= 4:
        out.append(
            f"  ⚠ '{name}' is a common name — some sites below may be "
            f"unrelated dict/list/str method calls, not calls to this "
            f"specific def. Verify by reading."
        )
    by_file: dict[str, list[tuple[int, str]]] = {}
    for f, ln, ctx in callers:
        by_file.setdefault(f, []).append((ln, ctx))
    for f in sorted(by_file):
        out.append(f"  {f}")
        for ln, ctx in by_file[f][:5]:
            out.append(f"    L{ln}: {ctx}")
        if len(by_file[f]) > 5:
            out.append(f"    … and {len(by_file[f]) - 5} more in this file")
    return "\n".join(out)


# ──────────────────── FILE TREE (planner navigation) ────────────────
# The planner used to receive a `{file_list}` regex-scraped from the
# Phase-1 research AIs' PROSE — so the list reflected what models TYPED,
# not what's on disk. A model that mentioned a wrong path (e.g.
# `galaxy/collection/dataclasses.py` when the code is in
# `galaxy/dependency_resolution/dataclasses.py`) seeded a non-existent
# path the planner then "copied," handing the coder a contradictory
# "modify a file that doesn't exist" step. These helpers ground path
# selection in the REAL filesystem: the planner gets a COLLAPSED tree
# (top-level folders + root files) and EXPANDS folders with [LS:] until
# it sees the exact, copy-paste-ready path. It cannot invent a directory
# it has never seen.

_TREE_SKIP_DIRS = {
    ".git", ".jarvis_sandbox", "__pycache__", "node_modules", ".venv",
    "venv", ".env", ".mypy_cache", ".pytest_cache", ".tox", ".idea",
    ".vscode", ".eggs", "dist", "build", ".cache", ".ruff_cache",
    "site-packages", ".next", ".gradle", "target", ".hg", ".svn",
}
_TREE_SKIP_EXT = {
    ".pyc", ".pyo", ".so", ".o", ".class", ".obj", ".a", ".dll",
    ".dylib", ".lock",
}
_TREE_MAX_ENTRIES = 300       # cap children shown per directory
_TREE_COUNT_CAP = 50_000      # stop counting a subtree past this many files


def _tree_visible(name: str, is_dir: bool) -> bool:
    """Whether a directory entry should be shown (skip VCS/build/cache noise
    and compiled artifacts; keep hidden dirs we care about like .github)."""
    if is_dir:
        if name in _TREE_SKIP_DIRS:
            return False
        if name.startswith(".") and name not in (".github",):
            return False
        return True
    return os.path.splitext(name)[1] not in _TREE_SKIP_EXT


def _count_tree_files(path: str) -> int:
    """Recursive count of visible files under `path` (skips noise dirs).
    Bounded by _TREE_COUNT_CAP so a pathological tree can't stall a prompt."""
    n = 0
    for _root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if _tree_visible(d, True)]
        n += sum(1 for f in files if _tree_visible(f, False))
        if n >= _TREE_COUNT_CAP:
            return _TREE_COUNT_CAP
    return n


def _safe_join(project_root: str, rel: str) -> Optional[str]:
    """Resolve project_root/rel and confirm it stays INSIDE project_root
    (realpath, so `..` and escaping symlinks are caught). None on escape."""
    root = os.path.realpath(project_root)
    target = os.path.realpath(os.path.join(root, rel))
    if target == root or target.startswith(root + os.sep):
        return target
    return None


def _fmt_tree_line(rel_path: str, is_dir: bool, count: Optional[int]) -> str:
    if is_dir:
        label = rel_path.rstrip("/") + "/"
        if count is None:
            return f"  {label}"
        cnt = (f"{count}+ files" if count >= _TREE_COUNT_CAP
               else f"{count} file{'s' if count != 1 else ''}")
        return f"  {label:<52} {cnt}"
    return f"  {rel_path}"


def list_dir_entries(project_root: str, rel: str = "") -> str:
    """List the IMMEDIATE children of project_root/rel — folders first (with
    recursive file counts), then files. Real filesystem, so every path shown
    is exact and copy-paste-ready. Backs the planner's [LS:] expand tool and
    the initial collapsed tree (rel="")."""
    rel = (rel or "").strip().strip("/").replace("\\", "/")
    label = rel or "(project root)"
    base = _safe_join(project_root, rel)
    if base is None:
        return f"=== LS: {rel} ===\n  ✗ path escapes the project root — refused."
    if not os.path.exists(base):
        return (f"=== LS: {label} ===\n  ✗ no such folder. Start from the top: "
                f"[LS: ] shows the top-level folders, then expand one at a time.")
    if not os.path.isdir(base):
        return (f"=== LS: {label} ===\n  ✗ that's a FILE, not a folder — "
                f"read it with [CODE: {rel}].")
    try:
        names = sorted(os.listdir(base))
    except OSError as e:
        return f"=== LS: {label} ===\n  ✗ cannot list: {str(e)[:120]}"
    dirs: list = []
    files: list = []
    for name in names:
        full = os.path.join(base, name)
        is_dir = os.path.isdir(full)
        if not _tree_visible(name, is_dir):
            continue
        child_rel = f"{rel}/{name}" if rel else name
        (dirs if is_dir else files).append((child_rel, full))
    # Root listing IS the PROJECT TREE (matches the prose header in PLAN_PROMPT);
    # deeper listings are [LS: folder] expansions. Same tool, consistent naming.
    header = ("=== PROJECT TREE — top level (expand a folder with [LS: <folder>]) ==="
              if not rel else f"=== LS: {label} ===")
    lines = [header]
    shown = 0
    for child_rel, full in dirs:
        if shown >= _TREE_MAX_ENTRIES:
            break
        lines.append(_fmt_tree_line(child_rel, True, _count_tree_files(full)))
        shown += 1
    for child_rel, _full in files:
        if shown >= _TREE_MAX_ENTRIES:
            break
        line = _fmt_tree_line(child_rel, False, None)
        # Annotate .py files with what they DEFINE, so the planner can see which
        # file holds a symbol instead of guessing (the f327e65d wrong-file bug).
        if child_rel.endswith(".py"):
            syms = _py_top_symbols(_full)
            if syms:
                line = f"{line}    [{syms}]"
        lines.append(line)
        shown += 1
    total = len(dirs) + len(files)
    if total == 0:
        lines.append("  (empty)")
    elif total > shown:
        lines.append(f"  … {total - shown} more entr"
                     f"{'ies' if total - shown != 1 else 'y'} — expand a subfolder to narrow.")
    lines.append("→ expand a folder with [LS: <its path>]; read a file with [CODE: <its path>].")
    return "\n".join(lines)


def build_repo_tree(project_root: str) -> str:
    """The INITIAL collapsed view handed to the planner: top-level folders
    (with recursive file counts) + root files. It drills down via [LS:]."""
    return list_dir_entries(project_root, "")


def _pub_first(names: list) -> list:
    """Public names first (a task usually names the public API), dunders last;
    alphabetical within each group."""
    return sorted(names, key=lambda n: (n.startswith("__"), n.startswith("_"), n.lower()))


def _py_top_symbols(abspath: str, max_classes: int = 10, max_methods: int = 6,
                    max_funcs: int = 12) -> str:
    """One-line summary of what a .py file DEFINES — each class WITH its method
    names, plus top-level functions. [LS:] shows this so the planner can see
    which file holds a symbol the task names (f327e65d misplaced the METHOD
    is_valid_collection_name because nothing showed it lives on AnsibleCollectionRef
    in _collection_finder.py — showing only the class name wasn't enough; the
    method itself must be visible). Public names first, dunders dropped.
    Best-effort: "" on any parse/read failure rather than raising."""
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except Exception:
        return ""
    parts = []
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for cls in classes[:max_classes]:
        methods = _pub_first([
            b.name for b in cls.body
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not (b.name.startswith("__") and b.name.endswith("__"))
        ])
        shown = methods[:max_methods]
        inner = ", ".join(shown)
        if len(methods) > max_methods:
            inner += f", +{len(methods) - max_methods}"
        # Braces, NOT parens: `class X(Base)` is Python base-class syntax, so
        # parens here would read as inheritance. `{...}` clearly means "the
        # methods this class contains".
        parts.append(f"class {cls.name}" + (f" {{{inner}}}" if inner else ""))
    if len(classes) > max_classes:
        parts.append(f"+{len(classes) - max_classes} more classes")
    if funcs:
        funcs = _pub_first(funcs)
        dpart = "def " + ", ".join(funcs[:max_funcs])
        if len(funcs) > max_funcs:
            dpart += f", +{len(funcs) - max_funcs}"
        parts.append(dpart)
    return "; ".join(parts)
