"""
Codebase tools — project scanning, code search, file operations.

Used by the coding agent at every phase:
  - Pre-search: scan project structure + find relevant files
  - On-demand: any AI writes [SEARCH: pattern] → auto-detected, results fed back
"""

import asyncio
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from core.cli import step, status, warn
from clients.gemini import call_flash


# ─── Path Normalization (Windows + Linux) ──────────────────────────────────

def norm_path(p: str) -> str:
    """Normalize path separators to OS native. Accepts both / and \\."""
    return str(Path(p))


def to_forward_slash(p: str) -> str:
    """Convert to forward slashes (for display, ripgrep globs, etc.)."""
    return p.replace("\\", "/")


# ─── Configuration ───────────────────────────────────────────────────────────

LARGE_FILE_THRESHOLD = 50_000  # chars — warn to use KEEP above this
MAX_SEARCH_RESULTS = 200     # v8.10 bumped 100→200. Subagent r5 found
                              # `os.environ.get` returned 13/16 real sites
                              # — last 3 production files missed. 200 with
                              # --max-count=3 surfaces ~66 distinct files,
                              # covers most realistic heavy patterns.
# Source-code file extensions. REFS and SEARCH restrict to these by
# default to avoid matches inside log files, audit outputs, markdown
# docs, etc. dominating the result count.
# v8.4 fix: REFS was returning 88% noise for a real symbol because
# matches in /behavioral_audit/, /logs/, *.md, and previous tool-output
# files were all counted as code references.
SOURCE_EXTENSIONS = {
    # Python
    ".py", ".pyx", ".pyi",
    # JS / TS
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".svelte", ".vue",
    # Web
    ".html", ".css", ".scss", ".sass", ".less",
    # Systems
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
    ".rs", ".go", ".java", ".kt", ".scala",
    # Functional
    ".ml", ".mli", ".hs", ".lhs", ".lean", ".elm", ".clj", ".cljs",
    # Other
    ".rb", ".php", ".swift", ".m", ".mm", ".lua", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".jl", ".pl", ".pm", ".dart", ".cs", ".fs", ".vb",
    # Config-as-code (sometimes searched for symbol references)
    ".toml", ".yaml", ".yml", ".json",
}


def _source_glob_args() -> list[str]:
    """Build ripgrep -g glob args restricting to source extensions."""
    args = []
    for ext in sorted(SOURCE_EXTENSIONS):
        args.extend(["-g", f"*{ext}"])
    return args


IGNORE_DIRS = {
    # Python — caches and virtualenvs only. NOT "lib", "libs", "packages":
    # these are real source-directory names in many Python projects.
    "__pycache__", ".venv", "venv", "env", ".env", "virtualenv",
    "site-packages", ".eggs",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".pytype", ".pyre",
    # Node / JS
    "node_modules", "bower_components", ".npm", ".yarn",
    ".next", ".nuxt", ".svelte-kit", ".astro",
    # Build / dist — keep only the obviously-build ones. NOT "bin", "obj"
    # since some C/C++ projects use them as source directories.
    "build", "dist", "out", "_build", "release", "debug",
    "cmake-build-debug", "cmake-build-release",
    # Version control
    ".git", ".svn", ".hg", ".bzr",
    # IDE / editor
    ".idea", ".vscode", ".vs", ".eclipse", ".settings",
    # Package managers / caches
    ".cache", ".gradle", ".m2", ".cargo",
    # Rust
    ".rustup", "target",  # target is rust's build output
    # Ruby
    ".bundle",
    # Docker
    ".docker",
    # Misc
    ".jarvis", ".jarvis_sandbox", "coverage", "htmlcov",
    ".terraform", ".serverless",
    # v8.7 fix: JARVIS dev/audit/snapshot dirs that contain embedded
    # source code as string literals (for prompt-variant testing).
    # These cause REFS/SEARCH to match symbol names inside the embedded
    # strings, dominating real-source matches. None of these are real
    # source directories.
    "behavioral_audit", "prompt_snapshots", "rendered", "rendered_deep",
    # v8.7 fix #2 (subagent r3): eval/run logs pollute SEARCH/REFS.
    # `logs/run_evaluation/.../test_output.txt` was leaking into
    # `_default_timeout` symbol searches.
    "logs",
    # Timestamped backup copies — `backup_v<N>_<TIMESTAMP>` and similar.
    # Pure historical copies; not callers of anything live. Subagent
    # found them polluting DEPENDENCY output.
    # NOTE: previously these were here but caused silent false-negatives
    # in projects where they're real source dirs:
    #   "pkg"       → Python projects use pkg/ as a regular package name
    #                 (e.g. django's, astropy's. Go conv stays unhandled.)
    #   "lib"/"libs" → common source directory in many projects
    #   "packages"  → common monorepo pattern
    #   "vendor"    → some Python projects vendor deps here as source
    #   "third_party"/"3rdparty"/"external"/"deps" → ditto
    #   "bin"/"obj" → some C projects use these as source
    #   "logs"/"tmp"/"temp" → user-named source dirs sometimes
}

# Directories matching these suffixes are also ignored
IGNORE_DIR_SUFFIXES = {".egg-info", ".dist-info"}

# v8.7: ignore directory PREFIXES (matched at directory-name start)
IGNORE_DIR_PREFIXES = {"backup_v", "backup-v", "snapshot_", "snapshot-"}


def _is_ignored_dir(dirname: str) -> bool:
    """Check if a directory should be ignored (exact match, suffix, or prefix)."""
    if dirname in IGNORE_DIRS:
        return True
    for prefix in IGNORE_DIR_PREFIXES:
        if dirname.startswith(prefix):
            return True
    for suffix in IGNORE_DIR_SUFFIXES:
        if dirname.endswith(suffix):
            return True
    return False

IGNORE_EXTENSIONS = {
    # Compiled
    ".pyc", ".pyo", ".so", ".o", ".a", ".class", ".dll", ".exe", ".dylib",
    ".wasm",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".bmp", ".webp", ".tiff",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    # Data / binary
    ".db", ".sqlite", ".sqlite3", ".bin", ".dat", ".pickle", ".pkl",
    ".h5", ".hdf5", ".parquet", ".arrow", ".feather",
    # Media
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Lock files (large, not useful for planning)
    ".lock",
    # Maps / minified
    ".map", ".min.js", ".min.css",
    # Models
    ".onnx", ".pb", ".pt", ".pth", ".safetensors", ".gguf",
}


# ─── Project Structure ───────────────────────────────────────────────────────

def scan_project(root: str, max_depth: int = 4) -> str:
    """
    Get the full project tree (files + dirs, up to max_depth).
    Returns a formatted string for AI consumption.
    """
    root = Path(root).resolve()
    if not root.exists():
        return f"Directory not found: {root}"

    lines = [f"Project: {root.name}/"]
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter ignored dirs
        dirnames[:] = [d for d in sorted(dirnames) if not _is_ignored_dir(d)]

        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            dirnames.clear()
            continue

        indent = "  " * depth
        if depth > 0:
            lines.append(f"{indent}{rel.name}/")

        for f in sorted(filenames):
            fpath = Path(dirpath) / f
            if fpath.suffix in IGNORE_EXTENSIONS:
                continue
            size = fpath.stat().st_size if fpath.exists() else 0
            size_str = f" ({_human_size(size)})" if size > 10000 else ""
            lines.append(f"{indent}  {f}{size_str}")
            file_count += 1

    lines.append(f"\n({file_count} files)")
    return "\n".join(lines)


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


# ─── File Reading ────────────────────────────────────────────────────────────

def read_file(path: str) -> str | None:
    """Read a file, return raw content or None if binary / unreadable.
    No size limit — callers add KEEP hints based on LARGE_FILE_THRESHOLD."""
    p = Path(norm_path(path))
    if not p.exists():
        return None
    if p.suffix in IGNORE_EXTENSIONS:
        return f"[BINARY FILE: {p.suffix} — skipped]"
    # v8.15 fix: subagent r13 found CODE on a `.pyc` returned 348 lines
    # of mojibake (the suffix gate didn't include .pyc). Add a content-
    # based binary sniff: if the first 4KB contains a NUL byte, treat as
    # binary regardless of extension.
    try:
        with open(p, "rb") as _fb:
            _head = _fb.read(4096)
        if b"\x00" in _head:
            return f"[BINARY FILE: {p.name} contains null bytes — skipped]"
    except Exception:
        pass  # fall through to text read below; mojibake is preferable to crash
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except PermissionError as e:
        # v8.15 fix (B18): subagent r18 found PermissionError was masked
        # as generic READ ERROR, then upstream suggested destructive
        # [REVERT FILE] recovery. Branch the message so the model gets
        # the right diagnosis.
        return (
            f"[READ ERROR: PERMISSION DENIED for {p}: {e}. "
            f"Check file ownership / mode — this is NOT an encoding "
            f"corruption; do not [REVERT FILE] for permission issues.]"
        )
    except UnicodeDecodeError as e:
        return (
            f"[READ ERROR: ENCODING — {p} could not be decoded as UTF-8 "
            f"with errors=replace: {e}. File may be binary or a non-UTF "
            f"text encoding.]"
        )
    except Exception as e:
        return f"[READ ERROR: {e}]"


def read_files(paths: list[str]) -> dict[str, str]:
    """Read multiple files. Returns {path: content}."""
    return {p: read_file(p) or f"[NOT FOUND: {p}]" for p in paths}


# ─── Code Search (ripgrep) ──────────────────────────────────────────────────


def file_uses_tabs(content: str) -> bool:
    """True if any line in the file starts with a TAB character.
    Used by CODE/VIEW/KEEP to attach a "(file uses TAB indentation —
    write REPLACE bodies with tabs, not 4 spaces)" header note when
    the source isn't space-indented. Without this, `expandtabs(4)`
    silently converts tabs to spaces in the rendered view and the
    coder's patch ends up with mixed indentation."""
    for ln in content.splitlines():
        if ln.startswith("\t"):
            return True
    return False


def add_line_numbers(content: str, display_mode: str = "prefix") -> str:
    """Add line numbers to code content.

    display_mode controls how the indent is rendered:

      "prefix"     — v9 LINENO|INDENT|content (default; coder/reviewer).
                     Token-efficient: 12 spaces becomes `12|`. The model
                     reuses the same INDENT|content shape in REPLACE
                     bodies. Required for any role that emits edits.

      "whitespace" — LINENO|<actual spaces><content> (planner/understand/
                     merger / exploration CLI). Reads natural; the model
                     doesn't have to mentally compile `12` into 12 spaces.
                     Read-only roles only — REPLACE-body parsing still
                     expects INDENT|content, so giving this to a coder
                     would invite copy-paste mistakes.

    v9 fix (Q-NUMBERS): line number is now at the FRONT, not the END.
    The previous `i{N}|{code} {LINENO}` format had a silent-corruption
    bug — trailing integers in legitimate code (`MAX = 100`, `return 42`)
    were indistinguishable from leaked line annotations. With the line
    number at the front:
      - Lines ending in integers are ALWAYS pure code.
      - REPLACE bodies that leak the line# prefix show TWO `\\d+\\|`
        patterns (line + indent); legitimate REPLACE has ONE
        (indent only). This makes safe auto-strip possible.

    Examples (prefix mode, v10 — line# uses ':', indent uses '|'):
       def foo():        →  10:0|def foo()
           return 1      →  11:4|return 1
               pass      →  12:8|pass
    A copied view line can be pasted VERBATIM into SEARCH/REPLACE; the
    runtime strips the `N:` line# and the indent-expander handles `N|`.

    Examples (whitespace mode):
       def foo():        →  10|def foo()
           return 1      →  11|    return 1
               pass      →  12|        pass
    """
    TAB_WIDTH = 4
    lines = content.split('\n')
    # v8.7 fix: when content ends with '\n', split('\n') leaves an empty
    # trailing element that gets numbered as a phantom EOF+1 line.
    # `wc -l` and CODE's header (after the earlier v8.7 fix) both report
    # the correct count; the renderer was the missed site. Drop the
    # empty trailing element to match.
    if lines and lines[-1] == '' and content.endswith('\n'):
        lines = lines[:-1]
    out = []
    if display_mode == "whitespace":
        for i, line in enumerate(lines):
            expanded = line.expandtabs(TAB_WIDTH)
            out.append(f"{i+1}:{expanded}")
    elif display_mode == "prefix_ws":
        # NATIVE coder: LINENO:INDENT|<real spaces>content — the indent NUMBER
        # (authoritative; the edit applier re-emits it) AND the real spaces (so the
        # coder SEES the nesting), then code. (root-cause fix for col-0 dedents.)
        for i, line in enumerate(lines):
            expanded = line.expandtabs(TAB_WIDTH)
            stripped = expanded.lstrip(' ')
            n_indent = len(expanded) - len(stripped)
            out.append(f"{i+1}:{n_indent}|{' ' * n_indent}{stripped}")
    else:
        for i, line in enumerate(lines):
            expanded = line.expandtabs(TAB_WIDTH)
            stripped = expanded.lstrip(' ')
            n_indent = len(expanded) - len(stripped)
            # v10: line# uses ':' (distinct from the indent's '|') so a copied
            # line pastes VERBATIM into SEARCH/REPLACE — runtime strips `N:`,
            # indent-expander handles `N|`. No manual line#-stripping needed.
            out.append(f"{i+1}:{n_indent}|{stripped}")
    return '\n'.join(out)


def _make_whitespace_visible(line: str) -> str:
    """Legacy: render leading whitespace visibly. Kept for backwards
    compatibility with any callers that still want the old format."""
    stripped = line.lstrip()
    prefix = line[:len(line) - len(stripped)]
    visible = prefix.replace('\t', '→').replace(' ', '⁃')
    return visible + stripped


def extract_relevant_sections(
    content: str, hints: str, context_lines: int = 100, max_short_file: int = 200,
) -> str:
    """Show only the relevant parts of a file with context_lines padding.

    For short files (<= max_short_file lines), returns the whole file with
    line numbers. For larger files, searches for keywords from `hints`
    (plan details, function names, etc.), finds matching lines, expands
    each match by context_lines above and below, merges overlapping
    ranges, and returns the sections with line numbers + gap markers.

    Always preserves original line numbers so edits can reference them.
    """
    lines = content.split('\n')
    total = len(lines)

    if total <= max_short_file:
        return add_line_numbers(content)

    # Extract searchable keywords from hints (identifiers, function names, etc.)
    # Match words that look like identifiers (2+ chars, not common English)
    keywords = set()
    for word in re.findall(r'[A-Za-z_]\w{2,}', hints):
        low = word.lower()
        # Skip common English words and plan-format words
        if low in {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'has',
            'will', 'should', 'must', 'into', 'when', 'then', 'each', 'make',
            'use', 'new', 'old', 'add', 'not', 'all', 'but', 'can', 'get',
            'set', 'put', 'also', 'any', 'file', 'code', 'line', 'step',
            'current', 'behavior', 'logic', 'details', 'modify', 'create',
            'change', 'update', 'function', 'method', 'class', 'return',
            'import', 'export', 'edit', 'replace', 'none', 'existing',
        }:
            continue
        keywords.add(word)

    if not keywords:
        # No useful keywords — show first and last context_lines
        return add_line_numbers(content)

    # Find all lines that match any keyword
    matched_lines: set[int] = set()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                matched_lines.add(i)
                break

    if not matched_lines:
        # No matches — show the whole file
        return add_line_numbers(content)

    # Build ranges with context_lines padding, merge overlapping
    ranges: list[tuple[int, int]] = []
    for line_idx in sorted(matched_lines):
        start = max(0, line_idx - context_lines)
        end = min(total - 1, line_idx + context_lines)
        if ranges and start <= ranges[-1][1] + 1:
            # Merge with previous range
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    # Build output with line numbers and gap markers
    width = len(str(total))
    output_parts = []
    output_parts.append(f"(file has {total} lines total, showing relevant sections)")

    for i, (start, end) in enumerate(ranges):
        if i == 0 and start > 0:
            output_parts.append(f"{'·' * 40} (lines 1-{start} omitted)")
        elif i > 0:
            prev_end = ranges[i - 1][1]
            output_parts.append(f"{'·' * 40} (lines {prev_end + 2}-{start} omitted)")

        for j in range(start, end + 1):
            output_parts.append(f"{j+1:>{width}}\t{lines[j]}")

        if i == len(ranges) - 1 and end < total - 1:
            output_parts.append(f"{'·' * 40} (lines {end + 2}-{total} omitted)")

    return '\n'.join(output_parts)


_EMPTY_PATTERN_RE = re.compile(r"^\s*(?:\.\*|\$\$\.\*\$\$|\$\.\*\$)?\s*$")

_TEST_PATH_RE = re.compile(r"(^|/)(tests?)/|(^|/)test_[^/]*$|(^|/)[^/]*_test\.[^/]*$")


def _is_test_path(fp: str) -> bool:
    """A test file (in a tests/ or test/ dir, or named test_* / *_test.*)."""
    return bool(_TEST_PATH_RE.search(fp or ""))


def search_code(pattern: str, root: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Search codebase using ripgrep (rg) or grep fallback.
    Returns list of {file, line_num, line, context}.

    Two-pass strategy to ensure test-file hits are never crowded out:
      Pass 1: search ONLY in test directories (paths containing /tests/,
              /test/, or files named test_*.py / *_test.py). Cap at
              max_results/2 so tests always get top billing in the result.
      Pass 2: search the whole project as before. Cap at the remaining
              slots; tests/ hits already returned are deduped.

    Observed failure (astropy-13033): the agent fired a T1 search for
    the failing test's error-message string. The single-pass scan hit
    its 30-result cap on irrelevant files (cfitsio docs, the source
    file itself) BEFORE reaching `astropy/timeseries/tests/test_sampled.py`,
    where the actual assertion lives. Agent saw "no test matches" → fell
    back to paraphrasing the bug description → wrong fix.
    """
    # v8.10 fix: reject empty/wildcard-only patterns. Subagent r6 found
    # `SEARCH ""` and `SEARCH '$$.*$$'` (after shell expansion) matched
    # every line of the first file instead of erroring.
    if _EMPTY_PATTERN_RE.match(pattern or ""):
        return [{
            "file": "__error__",
            "line_num": 0,
            "line": f"SEARCH pattern is empty or matches everything ({pattern!r}). "
                    "Provide a specific substring or regex.",
            "is_match": True,
        }]
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])
    for prefix in IGNORE_DIR_PREFIXES:
        ignore_args.extend(["--glob", f"!{prefix}*/"])
    # Source-only filter — restricts results to code files. Excludes
    # log/audit/markdown noise that dominated counts pre-v8.4.
    source_args = _source_glob_args()

    # v8.11 fix: SEARCH treats patterns as fixed strings unless they contain
    # explicit regex metacharacters. v13: a bare `[` no longer flips to regex —
    # `[STOP]`, `[DONE]`, `[CODE: x]` are LITERAL JARVIS tags the coder greps,
    # not char-classes (regex mode silently matched any of S/T/O/P and flooded
    # results). Only a `[...]` holding a `-` RANGE counts as a real char-class.
    # (Defined here so PASS 1 and PASS 2 share the same -F decision.)
    _looks_regex = (
        "\\" in pattern                                     # any escape: \( \. \d \b \w …
        or any(tok in pattern for tok in ("^", "$", "|", ".*", ".+"))
        or bool(re.search(r"\[[^\]]*-[^\]]*\]", pattern))   # range class, e.g. [a-z]
    )
    rg_mode_flag = [] if _looks_regex else ["-F"]

    test_results: list[dict] = []
    # v8.9: was max_results // 2 (half the budget for tests). Heavy
    # patterns like `DONE_TAG` returned only tests/ matches because tests
    # consumed the cap before core/ files could be reached. Cap test pass
    # at min(20, 1/4 of budget): tests still get guaranteed representation
    # for the astropy-13033 use case, but production code dominates by
    # default. ~80 of 100 slots remain for PASS 2 (whole-project scan).
    test_quota = min(20, max_results // 4) if max_results >= 4 else 0

    # PASS 1 — test directories only
    if test_quota > 0:
        try:
            # v8.7: drop heavy -C 5 to 1 line of context. Same bug as REFS:
            # -C 5 inflates per-file output and fills the result cap before
            # other files are reached, so production-code matches go missing.
            test_cmd = [
                "rg", "--line-number", "--no-heading", "--color=never",
                "--threads=1",  # v8.13: deterministic ordering
                *rg_mode_flag,  # v8.11: same -F decision as PASS 2
                "--max-count=5",
                "-C", "1",
                # Test paths: /tests/ subtrees, test_*.py / *_test.py files
                "-g", "**/tests/**",
                "-g", "**/test/**",
                "-g", "**/test_*",
                "-g", "**/*_test.*",
                *ignore_args, *source_args,
                "-e", pattern, root,
            ]
            r = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if r.returncode <= 1:
                # rg ORs the test-path `-g` globs with the source-extension
                # globs, so PASS 1's output can include non-test files. Keep
                # only real test paths so the test-dir pass actually leads (and
                # test hits can't be crowded out of the cap by source matches).
                _p1 = _parse_rg_output(r.stdout, 10_000)
                test_results = [x for x in _p1 if _is_test_path(x["file"])][:test_quota]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # PASS 2 — whole project (still source-only)
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--threads=1",  # v8.13: deterministic ordering
            *rg_mode_flag,
            # v8.9: max-count=3 per file (was 10). Combined with the
            # MAX_SEARCH_RESULTS=100 cap, this surfaces ~33 files —
            # much better coverage on heavy patterns where
            # alphabetically-first files would otherwise dominate.
            "--max-count=3",
            # No --max-filesize: old 100K cap silently skipped large files
            # (e.g. workflows/code.py at 260KB) — giving false "no results".
            "-C", "1",  # was 5 — same bug as REFS, see PASS 1 comment
            *ignore_args, *source_args,
            "-e", pattern, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # v8.15 fix: subagent r11 found `SEARCH "[]"` leaked ripgrep's
        # regex internal error. If our heuristic chose regex mode but
        # ripgrep rejects the pattern, transparently retry as -F
        # (fixed-string). That's almost always what the user meant.
        _regex_fell_back = False
        if (result.returncode > 1
            and "regex parse error" in (result.stderr or "").lower()
            and "-F" not in cmd):
            try:
                fixed_cmd = ["rg", "-F"] + cmd[1:]
                result = subprocess.run(
                    fixed_cmd, capture_output=True, text=True, timeout=10
                )
                _regex_fell_back = True
            except Exception:
                pass
        # v8.11 fix: surface invalid-regex errors that survived the
        # auto-retry above (e.g. genuinely malformed character classes).
        if result.returncode > 1 and "regex parse error" in (
            result.stderr or ""
        ).lower():
            return [{
                "file": "__error__",
                "line_num": 0,
                "line": (f"SEARCH could not parse pattern {pattern!r}: "
                         f"{(result.stderr or '').strip()[:200]}."),
                "is_match": True,
            }]
        if result.returncode <= 1:  # 0 = found, 1 = not found
            remaining = max(0, max_results - len(test_results))
            general = _parse_rg_output(result.stdout, remaining + len(test_results))
            # De-dup against test_results by (file, line_num)
            seen_keys = {(r["file"], r["line_num"]) for r in test_results}
            merged = list(test_results)
            for item in general:
                if (item["file"], item["line_num"]) in seen_keys:
                    continue
                if len(merged) >= max_results:
                    break
                merged.append(item)
                seen_keys.add((item["file"], item["line_num"]))
            # Test-file hits first (stable): the failing test is the spec, so a
            # weak model should see it before source/docs. PASS 1's test globs
            # get OR'd with the source-extension globs, so they don't reliably
            # lead — this stable re-sort guarantees test-first without disturbing
            # relative order within each group.
            merged.sort(key=lambda r: 0 if _is_test_path(r.get("file", "")) else 1)
            if _regex_fell_back:
                # Tell the model its pattern wasn't a valid regex (so it doesn't
                # read "0 / few matches" as "not in the codebase").
                merged.insert(0, {
                    "file": "__note__", "line_num": 0, "is_match": False,
                    "line": (f"⚠ {pattern!r} is not a valid regex — searched it as a "
                             f"LITERAL string instead. If you meant a regex, fix/escape it."),
                })
            return merged
    except FileNotFoundError:
        pass  # ripgrep not installed, fall through to grep
    except subprocess.TimeoutExpired:
        warn("Code search timed out")
        return test_results  # at least return what we have from tests pass

    # Fallback: grep
    try:
        cmd = [
            "grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
            "--include=*.lean", "--include=*.c", "--include=*.cpp", "--include=*.h",
            "--include=*.rs", "--include=*.java",
            "-C", "25",
            pattern, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return _parse_grep_output(result.stdout, max_results)
    except Exception:
        return []


def _parse_rg_output(output: str, max_results: int) -> list[dict]:
    """Parse ripgrep output into structured results.

    Ripgrep formats:
      match line:    file:LINE:content
      context line:  file-LINE-content

    The previous parser used `(.+?)[:-](\\d+)[:-](.*)$` which allowed
    EITHER separator on EITHER side of the line number. That mis-parses
    file paths containing `-`: for `dir-name/file.py:42:content`, the
    lazy `(.+?)` could split before the `-` between `dir` and `name`,
    sending `name/file.py:42` into the line-number group and failing.
    Backtracking eventually finds a correct split, but the same logic
    accepts inconsistent forms like `file:LINE-content` (mixed match
    and context separators) which can never occur in real rg output.

    The strict form below requires the two separators on the SAME line
    to match: `:LINE:` or `-LINE-`. Eliminates the cross-form ambiguity.
    """
    results = []

    for line in output.split("\n"):
        if not line.strip():
            continue
        # Strict: file<sep>LINE<sep>content where both <sep> are the same char.
        # `:` = match line; `-` = context line. v8.4 fix: track which is which
        # so format_search_results can mark the actual match.
        match = re.match(r'^(.+?)([-:])(\d+)\2(.*)$', line)
        if match and len(results) < max_results:
            filepath, sep, line_num, content = match.groups()
            results.append({
                "file": filepath,
                "line_num": int(line_num),
                "line": content.strip(),
                "is_match": sep == ":",
            })

    return results


def _parse_grep_output(output: str, max_results: int) -> list[dict]:
    """Parse grep output into structured results."""
    results = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        match = re.match(r'^(.+?)[:-](\d+)[:-](.*)$', line)
        if match and len(results) < max_results:
            filepath, line_num, content = match.groups()
            results.append({
                "file": filepath,
                "line_num": int(line_num),
                "line": content.strip(),
            })
    return results


def format_search_results(results: list[dict]) -> str:
    """Format search results for AI consumption.

    v8.4 fix: mark MATCH lines with `→` so the model can distinguish
    them from CONTEXT lines (which the parser emits surrounding each
    match). Previously the format was identical for both, forcing
    the model to mentally re-scan each line for the search pattern.
    """
    # Informational notes (e.g. "your pattern wasn't a valid regex, searched as a
    # literal") are rendered plainly, FIRST — so a broken regex is never a silent
    # "no matches". They carry file == "__note__".
    notes = [r["line"] for r in results if r.get("file") == "__note__"]
    results = [r for r in results if r.get("file") != "__note__"]
    note_str = ("\n".join(notes) + "\n") if notes else ""

    if not results:
        return (note_str + "(no matches found)") if note_str else "(no matches found)"

    # v8.10: handle empty-pattern error sentinel from search_code.
    if len(results) == 1 and results[0].get("file") == "__error__":
        return f"ERROR: {results[0]['line']}"

    lines = []
    current_file = ""
    for r in results:
        if r["file"] != current_file:
            current_file = r["file"]
            lines.append(f"\n── {current_file} ──")
        marker = "→" if r.get("is_match", True) else " "
        lines.append(f"  {marker} {r['line_num']:>4}: {r['line']}")

    return note_str + "\n".join(lines)


# ─── On-Demand Search Tag Detection ─────────────────────────────────────────

SEARCH_TAG_PATTERN = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)


def extract_search_requests(text: str) -> list[str]:
    """Extract [SEARCH: pattern] tags from AI response."""
    return SEARCH_TAG_PATTERN.findall(text)


async def run_on_demand_searches(text: str, project_root: str) -> str:
    """
    Detect [SEARCH: pattern] tags in AI output, run them, return results.
    Returns empty string if no tags found.
    """
    patterns = extract_search_requests(text)
    if not patterns:
        return ""

    all_results = []
    for pattern in patterns[:5]:  # Cap at 5 searches
        status(f"On-demand search: {pattern}")
        results = search_code(pattern, project_root)
        if results:
            all_results.append(f"\n=== Search: {pattern} ===")
            all_results.append(format_search_results(results))

    return "\n".join(all_results)


# ─── Pre-Search (Gemini Flash) ──────────────────────────────────────────────

async def pre_search(task_description: str, project_structure: str, project_root: str) -> str:
    """
    Gemini Flash analyzes the task + project structure and searches
    for likely relevant files/patterns BEFORE the main AIs start.
    """
    step("Pre-search: Gemini Flash scanning project...")

    prompt = f"""You are a code search assistant. Given a coding task and project structure,
identify the most relevant files and code patterns to search for.

TASK: {task_description}

PROJECT STRUCTURE:
{project_structure[:8000]}

Output ONLY a JSON list of search queries (strings to grep for):
["pattern1", "pattern2", "pattern3"]

Think about:
- Function names that would be involved
- Class names / module names
- Variable names
- Import statements
- File paths to read directly

Keep it to 5-8 specific, targeted patterns."""

    try:
        result = await call_flash(prompt, max_tokens=512)
        # Parse patterns
        import json
        # Clean markdown fences
        cleaned = result.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        patterns = json.loads(cleaned)

        if not isinstance(patterns, list):
            return ""

        # Run all searches
        all_results = []
        files_found = set()

        for pattern in patterns[:8]:
            results = search_code(str(pattern), project_root)
            if results:
                all_results.append(f"\n=== Pre-search: {pattern} ===")
                all_results.append(format_search_results(results))
                for r in results:
                    files_found.add(r["file"])

        # Also read the most relevant files in full
        file_contents = []
        for fpath in sorted(files_found)[:10]:  # Cap at 10 files
            content = read_file(fpath)
            if content and not content.startswith("["):
                file_contents.append(f"\n══ {fpath} ══\n{content}")

        status(f"Pre-search: {len(patterns)} patterns, {len(files_found)} files found")

        pre_search_output = "\n".join(all_results)
        if file_contents:
            pre_search_output += "\n\n=== FULL FILE CONTENTS ===\n" + "\n".join(file_contents)

        return pre_search_output

    except Exception as e:
        warn(f"Pre-search failed: {e}")
        return ""


# ─── Reference Search ──────────────────────────────────────────────────────

def _refs_definition_pass(name: str, root: str, ignore_args: list[str]) -> list[dict]:
    """Narrow ripgrep for DEFINITION lines only, with NO global cap.

    Observed failure on xarray__xarray-6938: a normal REFS pass with
    max_results=30 returned 30 usage entries from tests/docs and ZERO
    definition entries — the actual `def to_index_variable(self):` lines
    in xarray/core/variable.py were truncated out by the cap. The
    planner had no signal about WHERE the method is defined.

    This pass runs a precise regex matching common definition syntaxes
    across languages, with `--max-count` not applied so every definition
    surfaces regardless of how many usages exist elsewhere. The hit
    count is small by definition (a symbol usually has 1-5 definition
    sites), so there's no risk of context explosion.
    """
    # Definition patterns per language, all anchored to start-of-line
    # (with optional indentation). Each pattern uses `\b{name}\b` for
    # word-boundary safety and ripgrep's `-P` PCRE mode so we can use
    # `(?:async\s+)?` and `\bNAME\b` together.
    escaped = re.escape(name)
    patterns = [
        # Python: def/async def/class
        rf"^\s*(?:async\s+)?def\s+{escaped}\b",
        rf"^\s*class\s+{escaped}\b",
        # v8.11 fix: Python module-level constant / type-alias assignment.
        # Matches `STOP_TAG = re.compile(...)` and `Result: TypeAlias = ...`.
        # Subagent r7 found STOP_TAG wasn't surfacing in DEFINED because
        # the alphabetically-earlier test files saturated the pass-2 cap
        # before reaching core/tool_call.py:75. The definition pass needs
        # to catch bare-name assignment to keep this bulletproof.
        rf"^{escaped}(?:\s*:\s*\S+)?\s*=(?!=)",
        # JS/TS: function/const/let/var/class/export
        rf"^\s*(?:export\s+(?:default\s+)?)?function\s+{escaped}\b",
        rf"^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\b",
        rf"^\s*(?:export\s+)?class\s+{escaped}\b",
        # Rust: fn/struct/enum/trait
        rf"^\s*(?:pub\s+)?(?:async\s+)?fn\s+{escaped}\b",
        rf"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+{escaped}\b",
        # Go: func / type
        rf"^\s*func\s+(?:\([^)]+\)\s+)?{escaped}\b",
        rf"^\s*type\s+{escaped}\b",
        # C/C++/Java: ret-type Name(  — looser, accept method/function signatures
        rf"^\s*[A-Za-z_][\w:<>,\s\*&]*\s+{escaped}\s*\(",
    ]
    combined = "|".join(f"(?:{p})" for p in patterns)
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--threads=1",  # v8.13: deterministic ordering
            "-P",  # PCRE mode for the alternations
            *ignore_args,
            "-e", combined, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode > 1 or not result.stdout.strip():
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    # No cap: parse all hits.
    return _parse_rg_output(result.stdout, max_results=10_000)


def _refs_no_match(name: str) -> str:
    """'No matches' guidance for REFS. Branches on whether `name` is a dotted
    method: 'qualify with ClassName.' is nonsense when the name ALREADY has a
    dot (it produced `ClassName.Foo.bar`), so for those the real recovery is the
    bare method name or LSP."""
    head = f"Search for '{name}': no matches found."
    if "." in name:
        bare = name.rsplit(".", 1)[-1]
        return "\n".join([
            head,
            f"  • Instance method calls look like `obj.{bare}(...)`, not "
            f"`{name}` — try [REFS: {bare}] for the definition + call sites, "
            f"or [LSP: {name}] for type-resolved results.",
            f"  • If it appears only in docstrings/comments/strings, "
            f"try [SEARCH: {name}].",
        ])
    return "\n".join([
        head,
        f"  • If the name appears only in docstrings/comments/strings, REFS "
        f"won't see it — try [SEARCH: {name}] instead.",
        f"  • If it's a method on a class, qualify it: `ClassName.{name}`.",
        f"  • If imported under an alias elsewhere, query the alias name.",
    ])


def search_refs(name: str, root: str, max_results: int = 150) -> str:
    """
    Find all references to a function/class/variable by name.
    Uses word-boundary matching so searching "render" won't match "prerender".
    Groups results by: definitions, imports, and usages.
    Returns formatted string.

    Two-pass strategy to avoid the xarray-6938 truncation bug:
      1. Narrow definition-only pass (no cap) — guarantees every `def name`
         / `class name` / language equivalent appears in the DEFINED bucket.
      2. Standard usage pass capped at max_results — usages live here.
    Definitions from pass 1 are merged into the final DEFINED list even if
    they didn't survive the cap in pass 2.
    """
    # v8.15 fix: subagent r12 found `REFS ''` returned 54 spurious matches
    # because ripgrep with empty pattern matches every line. Other tools
    # (SEARCH, LSP) already guard this; REFS should too.
    if not name or not name.strip():
        return (
            "=== References: empty/whitespace-only symbol — provide a "
            "non-empty identifier. ==="
        )
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])
    for prefix in IGNORE_DIR_PREFIXES:
        ignore_args.extend(["--glob", f"!{prefix}*/"])
    # v8.4 fix: restrict REFS to source-only files. Previously a search
    # for a nonexistent symbol returned 14 matches from the very
    # capture scripts and prior tool outputs; for real symbols 88% of
    # USED entries were noise from logs/audit/docs.
    source_args = _source_glob_args()

    # PASS 1 — definitions only, uncapped. See helper docstring.
    definition_hits = _refs_definition_pass(name, root, ignore_args + source_args)

    # Use ripgrep with word boundary.
    # No --max-filesize: the old 200K cap silently skipped large files like
    # workflows/code.py (260KB), causing REFS to return empty results even
    # when the symbol was clearly defined there.
    # No -C context: ripgrep context lines use a dash separator which the old
    # split(':', 2) parser dropped entirely. We now use _parse_rg_output which
    # handles both separators, so context works correctly.
    try:
        # v8.7 fix: REFS used to take `-C 3` context (~7 lines per match)
        # then the 30-line `max_results` cap in _parse_rg_output filled up
        # before reaching the alphabetically-later files.
        # v8.13 fix: subagent r9 found REFS was non-deterministic — three
        # runs returned three different file lists because ripgrep's
        # default parallel walk produces results in thread-completion
        # order. With a per-symbol cap, alphabetically-later files were
        # silently truncated on some runs. Add `--threads=1` for
        # deterministic ordering + bump cap (150 default).
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--threads=1",     # v8.13: deterministic output ordering
            "--max-count=30",
            "-w",              # word boundary match
            *ignore_args, *source_args,
            "-e", name, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode > 1:
            return f"Search for '{name}': ripgrep error (rc={result.returncode})"
    except FileNotFoundError:
        # Fallback to grep
        try:
            cmd = [
                "grep", "-rnw", "--include=*.py", "--include=*.js", "--include=*.ts",
                "--include=*.jsx", "--include=*.tsx", "--include=*.c", "--include=*.cpp",
                "--include=*.h", "--include=*.rs", "--include=*.java", "--include=*.lean",
                "-C", "3",
                "-e", name, root,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            return f"Search for '{name}': search failed"
    except subprocess.TimeoutExpired:
        return f"Search for '{name}': timed out"

    if not result.stdout.strip():
        return _refs_no_match(name)

    # Parse with _parse_rg_output which handles both match lines (`:` sep) and
    # context lines (`-` sep). Old split(':', 2) silently dropped context lines.
    raw_results = _parse_rg_output(result.stdout, max_results)

    if not raw_results:
        # Even if pass 2 has nothing, pass 1 may have caught definitions.
        if not definition_hits:
            return _refs_no_match(name)

    # Merge pass-1 definitions in front so they're never lost to the cap.
    # Dedupe by (file, line_num).
    seen_keys: set[tuple[str, int]] = set()
    merged: list[dict] = []
    for item in (definition_hits + raw_results):
        key = (item.get("file", ""), item.get("line_num", 0))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(item)
    raw_results = merged

    # Categorize results
    definitions = []
    imports = []
    usages = []

    for item in raw_results:
        # v8.4 fix: skip CONTEXT lines (ripgrep `-C 3` returns ~3 lines
        # around each match, but those surrounding lines often don't
        # contain the symbol at all). Previously these were classified
        # as "usages" by default, inflating REFS noise to 88% on real
        # symbols. We only report lines where the symbol literally
        # appears (is_match=True from _parse_rg_output).
        if not item.get("is_match", True):
            continue

        filepath = item["file"]
        linenum  = item["line_num"]
        content  = item["line"]

        # v8.7 fix: skip lines that are comments or pure prose mentions.
        # Symbol-name mentions inside `#` comments, docstrings, or quoted
        # strings are not real call-sites — they polluted USED (3 of 8
        # entries for _default_timeout were comments/docstrings).
        stripped_for_classify = content.lstrip()
        if stripped_for_classify.startswith('#'):
            continue
        # Quoted-string-only lines.
        if (stripped_for_classify.startswith(('"', "'"))
            and f"{name}(" not in content):
            continue
        # Catch-all prose detector: if the symbol does NOT appear in
        # any code-like position (call `name(`, attribute `.name`,
        # assign `name =`, decorator `@name`, import statement, def
        # site), it's a mention in narrative text. Common in
        # documentation files like prompts/specs.
        is_call       = f"{name}(" in content
        # v8.10 fix: was only `.{name}` (dot BEFORE name). That caught
        # `obj.NAME` but missed `NAME.method()` — using the symbol AS a
        # receiver. Subagent r6: REFS on `_PERMANENT_STATUS` missed
        # line 33 `_PERMANENT_STATUS.search(...)` because the only check
        # was for `dot-name` not `name-dot`.
        is_attribute  = f".{name}" in content or f"{name}." in content
        is_assign     = bool(re.search(rf"\b{re.escape(name)}\s*=", content))
        is_decorator  = f"@{name}" in content
        is_import     = ("import" in stripped_for_classify.lower()
                          or "require" in stripped_for_classify.lower())
        is_def_site   = bool(re.search(
            rf"^(?:async\s+)?(?:def|class|fn|function|const|let|var)\s+{re.escape(name)}\b",
            stripped_for_classify,
        ))
        is_param_or_arg = bool(re.search(
            rf"(\(|,)\s*{re.escape(name)}\s*(=|[,)])", content
        ))
        # v8.10 fix: also accept use as a function ARG anywhere on a line
        # (e.g. `foo(_PERMANENT_STATUS)` — passes constant as parameter).
        is_arg_value  = bool(re.search(
            rf"[\(\[,]\s*{re.escape(name)}\b\s*[,)\]]", content
        ))
        # v8.15 fix (B15): containment operators `in` / `not in` and
        # comparison operands. `x in SYMBOL`, `SYMBOL in x`,
        # `x is SYMBOL`. Subagent r18 found `REFS _LSP_SKIP_DIRS`
        # missed `if d not in _LSP_SKIP_DIRS:` usage.
        is_in_operator = bool(re.search(
            rf"(?:\bin\s+{re.escape(name)}\b"
            rf"|\b{re.escape(name)}\s+in\b"
            rf"|\bis\s+{re.escape(name)}\b"
            rf"|\b{re.escape(name)}\s+is\b"
            rf"|return\s+{re.escape(name)}\b"
            rf"|yield\s+{re.escape(name)}\b"
            rf"|\b{re.escape(name)}\s*\[)",
            content,
        ))
        # v8.15 fix (B19): f-string interpolation `f"...{name}..."`
        # `f"...{name.attr}..."` `f"...{name[idx]}..."`. Subagent r19
        # found `REFS model_id` missed 18 logging lines like
        # `f"All retries failed for {model_id}: {last_error}"`. Match
        # `{name}` `{name.}` `{name[`. Doubled `{{` is a literal so
        # require single `{` not preceded by another `{`.
        is_fstring_interp = bool(re.search(
            rf"(?:(?<!{{){{{re.escape(name)}\b)",
            content,
        ))
        # v8.15 fix (B21): type-annotated parameter or attribute. The
        # patterns `def f(arg: NAME)`, `var: NAME`, `attr: NAME =`,
        # `-> NAME` are all real references. Subagent r20 found REFS
        # missed `def _default_timeout(model_id: str)` for `model_id`.
        # Recognize:
        #   - `<name>: <TYPE>` (parameter or variable with annotation)
        #   - `: <name>` (the type itself, after a colon)
        #   - `-> <name>` (return type annotation)
        is_type_annotation = bool(re.search(
            rf"(?:\b{re.escape(name)}\s*:\s*[A-Za-z_]"
            rf"|:\s*{re.escape(name)}\b"
            rf"|->\s*{re.escape(name)}\b"
            rf"|->\s*[A-Za-z_\[\],\s\"']*\b{re.escape(name)}\b"
            rf"|\[\s*{re.escape(name)}\s*[\],])",
            content,
        ))
        if not (is_call or is_attribute or is_assign or is_decorator
                or is_import or is_def_site or is_param_or_arg
                or is_arg_value or is_in_operator or is_fstring_interp
                or is_type_annotation):
            continue

        # Make path relative
        try:
            rel = str(Path(filepath).relative_to(root))
        except ValueError:
            rel = filepath

        entry = f"  {rel}:{linenum}  {content}"

        # Categorize based on line content. The prefixes below are followed
        # by an identifier-boundary check (next char must not be a word char
        # or `_`) so `def {name}_other` does NOT get classified as a
        # definition of `name`.
        stripped = content.lstrip()
        def_prefixes = [
            f"def {name}", f"class {name}", f"async def {name}",
            f"function {name}", f"const {name}", f"let {name}", f"var {name}",
            f"export function {name}", f"export const {name}",
            f"export default function {name}",
            f"fn {name}", f"pub fn {name}", f"struct {name}", f"enum {name}",
        ]
        is_definition = False
        for kw in def_prefixes:
            if stripped.startswith(kw):
                next_char = stripped[len(kw):len(kw) + 1]
                # Identifier ends here only if next char is NOT another
                # identifier char. `def foo(` / `class Foo:` / `class Foo `
                # all qualify; `def foo_bar(` does not.
                if not (next_char.isalnum() or next_char == '_'):
                    is_definition = True
                    break
        # v8.10 fix: module-level constant assignment IS a definition.
        # `_PERMANENT_STATUS = re.compile(...)` was previously classified
        # as USED, masking that line 23 is the def. Recognize an
        # assignment at column 0 with the symbol name as the LHS.
        # Heuristic: line starts with `name =` or `name: type =`.
        if not is_definition:
            if re.match(
                rf"^{re.escape(name)}(?:\s*:\s*\S+)?\s*=(?!=)",
                stripped,
            ):
                is_definition = True
        if is_definition:
            definitions.append(entry)
        # v8.15 fix: stricter import classifier. Prose like "find imports
        # of X" was being labeled IMPORTED. Require the line to actually
        # be an import statement: start with `import `, `from `, or be
        # a `require(...)` call.
        elif (stripped_lc := stripped.lower()).startswith(("import ", "from "))\
             or "require(" in stripped_lc or stripped_lc.startswith("require "):
            imports.append(entry)
        else:
            usages.append(entry)

    # SECOND PASS — detect multi-line parenthesized imports the single-line
    # categorizer would miss. Pattern: `from X import (... Name ...)` where
    # `Name` lands on a continuation line. Without this, a re-export like
    #   from .table import (Table, QTable, ...,
    #                       NdarrayMixin, ...)
    # gets classified as USED (the continuation line just lists names)
    # instead of IMPORTED, leaving the consumer invisible to the agent.
    # Observed failure on astropy-13236: agent ran [REFS: NdarrayMixin],
    # didn't see the `__init__.py` re-export was an import, deleted it,
    # broke 644 tests.
    #
    # We run a SEPARATE project-wide ripgrep with multi-line mode (-U) so
    # we catch consumers in files that didn't make the main result's
    # 30-entry cap. The pattern matches parenthesized `from X import (`
    # blocks containing the symbol.
    multiline_hits: list[str] = []
    try:
        ml_pattern = rf"from\s+\S+\s+import\s+\([^)]*\b{re.escape(name)}\b[^)]*\)"
        cp_ml = subprocess.run(
            ["rg", "-U", "--line-number", "--no-heading", "--color=never",
             "-g", "*.py",   # ripgrep uses --glob / -g, not --include
             *ignore_args,
             "-e", ml_pattern, root],
            capture_output=True, text=True, timeout=15,
        )
        if cp_ml.returncode <= 1 and cp_ml.stdout.strip():
            for line in cp_ml.stdout.splitlines()[:20]:
                # ripgrep -U returns `path:line:content` with content being
                # the FIRST line of the multi-line match
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                fp, ln, content = parts
                # Skip the file we're searching against (if name is defined
                # there). This is rare for re-export checks.
                try:
                    rel = str(Path(fp).resolve().relative_to(Path(root).resolve()))
                except ValueError:
                    rel = fp
                opener = content.strip()
                multiline_hits.append(
                    f"  {rel}:{ln}  {opener}  (multi-line import contains {name})"
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # De-dup multiline hits — both against single-line imports and against
    # other multiline-hit lines pointing at the SAME multi-line block.
    # ripgrep -U emits one output line per matched line, so a 3-line
    # import block produces 3 hits all pointing at the same block. Keep
    # only the FIRST (the `from X import (` opener line); skip continuation
    # lines (which start with whitespace + just names).
    def _file_line_key(entry: str) -> str:
        # Format: "  <rel>:<lineno>  <content>  …" — extract "<rel>:<lineno>"
        stripped = entry.lstrip()
        return stripped.split("  ", 1)[0]

    existing_keys = {_file_line_key(imp) for imp in imports}
    deduped_ml: list[str] = []
    seen_ml_files: set[str] = set()  # per-file: keep only the OPENER hit
    for h in multiline_hits:
        key = _file_line_key(h)
        file_part = key.rsplit(":", 1)[0] if ":" in key else key
        if key in existing_keys:
            continue
        # Only keep the FIRST hit per file (the opener line)
        if file_part in seen_ml_files:
            continue
        # Skip continuation-line entries (content starts with whitespace +
        # bare names; doesn't contain `from` or `import`)
        content_part = h.lstrip().split("  ", 1)[1] if "  " in h.lstrip() else ""
        if "from " not in content_part.split("(")[0]:
            continue
        # v8.15 fix: subagent r13 found prose lines like `- [REFS: of]`
        # in docstrings were classified as IMPORTED because they contain
        # the name and a `(` somewhere. Require the line to actually look
        # like a python import statement: stripped content must start
        # with `from ` (not contain it embedded in prose).
        stripped_content = content_part.lstrip()
        if not stripped_content.startswith("from "):
            continue
        seen_ml_files.add(file_part)
        deduped_ml.append(h)
    multiline_hits = deduped_ml

    parts = [f"=== References for '{name}' ==="]
    if definitions:
        parts.append(f"\nDEFINED ({len(definitions)}):")
        parts.extend(definitions)
    if imports or multiline_hits:
        parts.append(f"\nIMPORTED ({len(imports) + len(multiline_hits)}):")
        parts.extend(imports)
        parts.extend(multiline_hits)
    if usages:
        parts.append(f"\nUSED ({len(usages)}):")
        parts.extend(usages)

    # v8.15 fix (B14): subagent r17 found REFS on a name that exists only
    # in docstrings/comments returned just the header with no body and
    # no "0 matches" indication. Make the empty-result case explicit so
    # the model knows REFS truly found nothing (not a tool failure).
    if not definitions and not imports and not multiline_hits and not usages:
        if "." in name:
            _bare = name.rsplit(".", 1)[-1]
            parts.append(
                f"\nNo references found for '{name}'.\n"
                f"  • Instance method calls look like `obj.{_bare}(...)`, not "
                f"`{name}` — try [REFS: {_bare}] for the definition + call sites, "
                f"or [LSP: {name}] for type-resolved results.\n"
                f"  • If it appears only in docstrings/comments/strings, "
                f"try [SEARCH: {name}]."
            )
        else:
            parts.append(
                f"\nNo references found for '{name}'.\n"
                f"  • If the name appears only in docstrings/comments/strings, "
                f"REFS won't see it — try [SEARCH: {name}] instead.\n"
                f"  • If it's a method name on a class, qualify it: "
                f"`ClassName.{name}`.\n"
                f"  • If it's defined in a different module under an alias, "
                f"the alias name is what to query."
            )

    # v8.7: detect import-alias patterns and warn. If any IMPORTED entry
    # uses `as <local_name>`, callers in those files use the local name —
    # which this REFS call won't find (it searches for the original
    # symbol only). Surface that for cross-module accuracy.
    aliased_files = []
    for imp in imports:
        # entries look like: "  path/to/file.py:42  from X import name as Y"
        if re.search(rf"\bimport\s+\S+\s+as\s+\w+", imp) or re.search(
            rf"\b{re.escape(name)}\s+as\s+(\w+)", imp
        ):
            try:
                file_part = imp.lstrip().split(":", 1)[0]
                local_name_match = re.search(
                    rf"\b{re.escape(name)}\s+as\s+(\w+)", imp
                )
                if local_name_match:
                    aliased_files.append(
                        f"  {file_part} aliases as '{local_name_match.group(1)}'"
                    )
            except Exception:
                pass
    if aliased_files:
        parts.append(
            f"\nNOTE: import alias detected in {len(aliased_files)} file(s). "
            f"REFS only finds direct uses of '{name}'; callers in these files "
            f"use the local alias name. For accurate cross-module callers, "
            f"use [DEPENDENCY: #tag] (LSP-resolved, follows aliases):"
        )
        parts.extend(aliased_files)

    # v8.10: subagent r5 observed REFS missed 12 of 29 USED sites that LSP
    # found (REFS = ripgrep word-boundary; LSP = type-resolved). For
    # refactor-safety analysis, the LSP count is authoritative.
    # v8.11: subagent r6 noted LSP is the WRONG tool for module-level
    # constants (LSP only resolves functions/classes). Recommend
    # DEPENDENCY instead for ALL_CAPS names.
    # v8.11-fix: subagent r7 caught that "_underscore-prefix → constant"
    # over-matched on private FUNCTIONS like `_apply_extracted_code`.
    # The new heuristic: ALL_CAPS-only (true constants), OR there's a
    # def/class at the same name in the index (which means LSP works,
    # not DEPENDENCY).
    if len(usages) >= 5 or aliased_files:
        is_likely_constant = name.isupper()
        if is_likely_constant:
            parts.append(
                f"\n(REFS uses ripgrep word-boundary matching. For "
                f"module-level constants like '{name}', use "
                f"[DEPENDENCY: #tag] (AST-resolved, gets the right tag "
                f"from the def-line annotation). LSP doesn't resolve "
                f"constants.)"
            )
        else:
            parts.append(
                f"\n(REFS uses ripgrep word-boundary matching; "
                f"for type-resolved cross-module callers and import-alias "
                f"resolution, use [DEPENDENCY: #tag] — slower but authoritative.)"
            )

    return "\n".join(parts)
