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
MAX_SEARCH_RESULTS = 30      # ripgrep matches
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


def _is_ignored_dir(dirname: str) -> bool:
    """Check if a directory should be ignored (exact match or suffix)."""
    if dirname in IGNORE_DIRS:
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
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[READ ERROR: {e}]"


def read_files(paths: list[str]) -> dict[str, str]:
    """Read multiple files. Returns {path: content}."""
    return {p: read_file(p) or f"[NOT FOUND: {p}]" for p in paths}


# ─── Code Search (ripgrep) ──────────────────────────────────────────────────


def add_line_numbers(content: str) -> str:
    """Add line numbers to code content in the iN| format the model expects.

    Format per line:  iN|{code without leading whitespace} {lineno}
    where N is the number of leading spaces (tabs are expanded to TAB_WIDTH=4)
    and {lineno} is separated from the code by a single space — matching the
    format the prompts document and the format _filter_by_ranges (used by KEEP)
    produces. Reading and writing are symmetric across [CODE:] and [KEEP:].

    Examples:
       def foo():        →  i0|def foo() 10
           return 1      →  i4|return 1 11
               pass      →  i8|pass 12
       (blank)           →  i0| 13
    """
    TAB_WIDTH = 4
    lines = content.split('\n')
    out = []
    for i, line in enumerate(lines):
        expanded = line.expandtabs(TAB_WIDTH)
        stripped = expanded.lstrip(' ')
        n_indent = len(expanded) - len(stripped)
        out.append(f"i{n_indent}|{stripped} {i+1}")
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
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])

    test_results: list[dict] = []
    test_quota = max_results // 2 if max_results >= 4 else 0

    # PASS 1 — test directories only
    if test_quota > 0:
        try:
            test_cmd = [
                "rg", "--line-number", "--no-heading", "--color=never",
                "--max-count=5",
                "-C", "5",
                # Test paths: /tests/ subtrees, test_*.py / *_test.py files
                "-g", "**/tests/**",
                "-g", "**/test/**",
                "-g", "**/test_*",
                "-g", "**/*_test.*",
                *ignore_args,
                pattern, root,
            ]
            r = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            if r.returncode <= 1:
                test_results = _parse_rg_output(r.stdout, test_quota)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # PASS 2 — whole project
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--max-count=5",  # max 5 matches per file
            # No --max-filesize: old 100K cap silently skipped large files
            # (e.g. workflows/code.py at 260KB) — giving false "no results".
            "-C", "5",  # 5 lines context (was 25 — most was wasted)
            *ignore_args,
            pattern, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
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
        match = re.match(r'^(.+?)([-:])(\d+)\2(.*)$', line)
        if match and len(results) < max_results:
            filepath, _sep, line_num, content = match.groups()
            results.append({
                "file": filepath,
                "line_num": int(line_num),
                "line": content.strip(),
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
    """Format search results for AI consumption."""
    if not results:
        return "(no matches found)"

    lines = []
    current_file = ""
    for r in results:
        if r["file"] != current_file:
            current_file = r["file"]
            lines.append(f"\n── {current_file} ──")
        lines.append(f"  {r['line_num']:>4}: {r['line']}")

    return "\n".join(lines)


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
            "-P",  # PCRE mode for the alternations
            *ignore_args,
            combined, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode > 1 or not result.stdout.strip():
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    # No cap: parse all hits.
    return _parse_rg_output(result.stdout, max_results=10_000)


def search_refs(name: str, root: str, max_results: int = 30) -> str:
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
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])

    # PASS 1 — definitions only, uncapped. See helper docstring.
    definition_hits = _refs_definition_pass(name, root, ignore_args)

    # Use ripgrep with word boundary.
    # No --max-filesize: the old 200K cap silently skipped large files like
    # workflows/code.py (260KB), causing REFS to return empty results even
    # when the symbol was clearly defined there.
    # No -C context: ripgrep context lines use a dash separator which the old
    # split(':', 2) parser dropped entirely. We now use _parse_rg_output which
    # handles both separators, so context works correctly.
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--max-count=10",
            "-w",  # word boundary match
            "-C", "3",  # 3 lines context — enough to see usage without bloat
            *ignore_args,
            name, root,
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
                name, root,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            return f"Search for '{name}': search failed"
    except subprocess.TimeoutExpired:
        return f"Search for '{name}': timed out"

    if not result.stdout.strip():
        return f"Search for '{name}': no matches found"

    # Parse with _parse_rg_output which handles both match lines (`:` sep) and
    # context lines (`-` sep). Old split(':', 2) silently dropped context lines.
    raw_results = _parse_rg_output(result.stdout, max_results)

    if not raw_results:
        # Even if pass 2 has nothing, pass 1 may have caught definitions.
        if not definition_hits:
            return f"Search for '{name}': no matches found"

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
        filepath = item["file"]
        linenum  = item["line_num"]
        content  = item["line"]

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
        if is_definition:
            definitions.append(entry)
        elif "import" in stripped.lower() or "require" in stripped.lower():
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
             ml_pattern, root],
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

    return "\n".join(parts)
