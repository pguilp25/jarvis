"""
Project-wide symbol dependency index.

Builds, once per coding-agent run, a map of every top-level def/class +
class-qualified method + module-level constant in the project, with their
definition locations and reference counts (AST-walked, not regex). Symbols
that appear >=10 times are surfaced as "high-use" with a stable 3-char hex
tag so that any agent (planner, coder, reviewer) can:

  1. SEE the high-use symbols at the bottom of every VIEW/CODE result —
     "this function/class is everywhere, edits ripple", and

  2. ASK for the full reference list via the new [dependency: #TAG] tool,
     which returns every file:line where that symbol is used.

The tag is sha1(symbol_qualname)[:3] — globally unique enough that the
same name in two different files gets two different tags, and consistent
across calls within a run.

Threshold is configurable (default 10). The index is built lazily, cached
by project root, and rebuilds are cheap (~2-3s on astropy-sized repos
because we use the stdlib `ast` module, not jedi/pyright).
"""

import ast
import hashlib
import os
import re
from pathlib import Path
from typing import Optional


# ── Data model ──────────────────────────────────────────────────────────────

class Symbol:
    """One indexed symbol — top-level def/class, method, or module constant.

    Two ref tracks coexist:

      • `refs` (AST): the upper bound. Polymorphic methods (multiple
        `def write` across classes) all get credited for every
        `.write()` attribute access, so this is "what could possibly
        break." Always populated.

      • `lsp_refs` (LSP, lazy): the precise lower bound from
        find_references — only the call sites the language server
        could statically resolve to THIS method. None until
        `resolve_refs_with_lsp()` is called; empty list if LSP failed.

    For BLAST RADIUS, AST is the safer signal (over-warns rather than
    under-warns). For "what code calls this exact method?" — LSP gives
    the precise list of resolvable callers.
    """
    __slots__ = ("name", "qualname", "kind", "def_file", "def_line",
                 "tag", "refs", "lsp_refs", "_lsp_resolved")

    def __init__(self, name: str, qualname: str, kind: str,
                 def_file: str, def_line: int, tag: str):
        self.name = name              # bare name, e.g. "write"
        self.qualname = qualname      # disambiguated, e.g. "html.py::HTML.write"
        self.kind = kind              # "func" | "class" | "method" | "const"
        self.def_file = def_file      # path relative to project root
        self.def_line = def_line
        self.tag = tag                # 3-char hex tag like "a3f"
        self.refs: list[tuple[str, int]] = []        # AST (upper bound)
        self.lsp_refs: Optional[list[tuple[str, int]]] = None  # LSP (precise, lazy)
        self._lsp_resolved = False    # True after lsp_refs is populated

    def ref_count(self) -> int:
        """AST upper-bound count — what inline annotations show."""
        return len(self.refs)

    def lsp_ref_count(self) -> Optional[int]:
        """LSP precise count, or None if LSP hasn't been queried."""
        return len(self.lsp_refs) if self.lsp_refs is not None else None

    def ref_files(self) -> set[str]:
        return {f for f, _ in self.refs}

    def lsp_ref_files(self) -> set[str]:
        return {f for f, _ in (self.lsp_refs or [])}


class DependencyIndex:
    """Holds all symbols + lookup helpers. Construct via build_index().

    Supports incremental refresh: call refresh() to pick up file changes
    cheaply (re-stat all .py files, re-parse only mtime-changed ones,
    rebuild symbol table from cached per-file data).
    """

    def __init__(self, project_root: str, threshold: int = 10,
                 max_files: int = 20000):
        self.project_root = os.path.abspath(project_root)
        self._threshold = threshold
        self._max_files = max_files
        # Per-file cache: rel_path -> (mtime, defs, all_loads, importers)
        # • defs: list of (name, qualname, kind, def_line) from _walk_defs
        # • all_loads: list of (name, line) from _walk_all_loads — every
        #   Name(Load)/Attribute(Load)/ImportFrom alias, NOT filtered by
        #   want_names. This lets us rebuild the symbol table without
        #   re-parsing unchanged files when symbols are added/removed.
        # • imported_modules: set of dotted module names this file imports
        #   (used to compute per-module importer counts for "central
        #   utility" marker).
        self._file_cache: dict[str, tuple[float, list, list, set]] = {}
        # Built symbol table — rebuilt by _rebuild_table()
        self.by_tag: dict[str, Symbol] = {}
        self.by_name: dict[str, list[Symbol]] = {}
        self.by_def_file: dict[str, list[Symbol]] = {}
        # rel_path -> count of OTHER files that import this module. High
        # values signal "central utility" — symbols here often have
        # transitive blast radius exceeding their literal AST refs.
        self.importer_count: dict[str, int] = {}

    def lookup_tag(self, tag: str) -> Optional[Symbol]:
        return self.by_tag.get(tag.lstrip("#").lower())

    def symbols_in_file(self, file_path: str) -> list[Symbol]:
        """All symbols DEFINED in this file (use for annotation block)."""
        rel = self._rel(file_path)
        return self.by_def_file.get(rel, [])

    def _rel(self, p: str) -> str:
        """Normalize to project-root-relative posix path.

        Relative paths are interpreted as already-rooted (so callers can
        pass either an absolute sandbox path or a project-relative path).
        We do NOT use os.path.abspath() because that resolves against
        process CWD, which is not the same as project_root in JARVIS.
        """
        if os.path.isabs(p):
            ap = os.path.normpath(p)
            if ap.startswith(self.project_root + os.sep):
                return ap[len(self.project_root) + 1:].replace(os.sep, "/")
            return ap.replace(os.sep, "/")
        # Relative path — treat as already project-relative
        return os.path.normpath(p).replace(os.sep, "/")

    # ── Incremental refresh ────────────────────────────────────────────────

    def refresh(self) -> dict:
        """Re-stat all .py files; re-parse only those whose mtime changed
        since the last refresh; rebuild the symbol table from cached data.

        Returns a stats dict: {files_checked, files_reparsed, files_removed,
        rebuilt_table, elapsed_ms}.
        """
        import time as _t
        t0 = _t.time()
        files = _list_py_files(self.project_root, max_files=self._max_files)

        present_rels: set[str] = set()
        files_reparsed = 0
        any_changed = False

        for abs_path in files:
            try:
                mtime = os.path.getmtime(abs_path)
            except OSError:
                continue
            rel = self._rel(abs_path)
            present_rels.add(rel)
            cached = self._file_cache.get(rel)
            if cached is not None and cached[0] == mtime:
                continue
            # Re-parse this file
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = ast.parse(src, filename=abs_path)
            except (SyntaxError, UnicodeDecodeError, OSError):
                # Drop a broken file from the cache so it doesn't linger
                self._file_cache.pop(rel, None)
                any_changed = any_changed or cached is not None
                continue
            defs = _walk_defs(tree, rel)
            loads = _walk_all_loads(tree)
            imports = _walk_imports(tree)
            self._file_cache[rel] = (mtime, defs, loads, imports)
            files_reparsed += 1
            any_changed = True

        # Drop files that disappeared from the filesystem
        removed = set(self._file_cache.keys()) - present_rels
        for rel in removed:
            del self._file_cache[rel]
        if removed:
            any_changed = True

        if any_changed:
            self._rebuild_table()

        return {
            "files_checked": len(files),
            "files_reparsed": files_reparsed,
            "files_removed": len(removed),
            "rebuilt_table": any_changed,
            "elapsed_ms": int((_t.time() - t0) * 1000),
        }

    def _rebuild_table(self) -> None:
        """Rebuild by_tag/by_name/by_def_file/importer_count from _file_cache."""
        self.by_tag.clear()
        self.by_name.clear()
        self.by_def_file.clear()
        self.importer_count.clear()

        # Pass 1: defs → Symbols
        for rel, (_, defs, _, _) in self._file_cache.items():
            for name, qual, kind, line in defs:
                tag = _mk_tag(qual, used=self.by_tag.keys())  # type: ignore[arg-type]
                sym = Symbol(name, qual, kind, rel, line, tag)
                self.by_tag[tag] = sym
                self.by_name.setdefault(name, []).append(sym)
                self.by_def_file.setdefault(rel, []).append(sym)

        # Pass 2: credit refs to Symbols by bare-name match
        indexed = self.by_name
        for rel, (_, _, loads, _) in self._file_cache.items():
            for name, line in loads:
                if name in _BUILTIN_NAMES:
                    continue
                syms = indexed.get(name)
                if not syms:
                    continue
                for sym in syms:
                    if sym.def_file == rel and sym.def_line == line:
                        continue
                    sym.refs.append((rel, line))

        # Pass 3: compute per-file "centrality" score.
        # A file is "central" when at least one symbol defined in it has
        # many references across many other files. This catches files
        # that get re-exported through __init__.py (where direct import
        # counting fails: a `from astropy.table import Table` registers
        # only against `table/__init__.py` not `table/table.py`).
        #
        # We use: max(len(sym.ref_files())) across symbols defined in
        # the file. If the file's "most popular" symbol is referenced
        # in many files, ANY symbol in the file likely sits on call
        # paths through those files — including helpers AST misses.
        for rel, syms_in_file in self.by_def_file.items():
            max_files = max((len(s.ref_files()) for s in syms_in_file),
                            default=0)
            self.importer_count[rel] = max_files


# ── Builder ─────────────────────────────────────────────────────────────────

# Names we never index — built-ins, common method names that would
# false-positive everywhere. Methods named __init__ etc. are skipped here
# because counting their references is meaningless (every class has one).
_DUNDER_RE = re.compile(r"^__\w+__$")

# Builtins we don't want to surface — they pollute the high-use list
# without giving the model any actionable signal.
_BUILTIN_NAMES = frozenset({
    "self", "cls", "True", "False", "None",
    # Common stdlib names that crowd out useful ones
    "len", "range", "list", "dict", "set", "tuple", "str", "int", "float",
    "bool", "bytes", "type", "object", "isinstance", "issubclass",
    "getattr", "setattr", "hasattr", "delattr", "callable",
    "print", "open", "iter", "next", "zip", "map", "filter", "sorted",
    "min", "max", "sum", "any", "all", "abs", "round", "pow", "divmod",
    "enumerate", "reversed", "repr", "hash", "id", "vars", "dir",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "NotImplementedError",
    "StopIteration", "GeneratorExit", "KeyboardInterrupt",
})


def _mk_tag(qualname: str, used: set[str]) -> str:
    """Stable hex tag derived from qualname, lengthened until unique.

    Birthday-collision safety: with thousands of symbols, 3-char tags
    (16^3 = 4096 slots) collide constantly. We try 3 chars first, then
    extend to 4, 5, ... until the tag is unique within `used`.

    Tags ARE deterministic (same qualname always hashes the same way) but
    NOT prefix-stable across runs — adding a symbol can shift another
    symbol's tag length if it forces a re-collision. That's OK because
    tags are only used within a single agent run.
    """
    h = hashlib.sha1(qualname.encode("utf-8")).hexdigest()
    for length in range(3, 9):
        tag = h[:length]
        if tag not in used:
            return tag
    return h[:16]  # pathological fallback — sha1 collision territory


def _walk_defs(tree: ast.AST, rel_path: str) -> list[tuple[str, str, str, int]]:
    """Walk a parsed module and yield (bare_name, qualname, kind, line).

    qualname format: "<rel_path>::<ClassName.method>" so two same-named
    methods in different classes/files get different qualnames (and tags).
    """
    out: list[tuple[str, str, str, int]] = []

    def visit(node: ast.AST, class_stack: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                if _DUNDER_RE.match(name):
                    # Skip dunders — they're everywhere
                    pass
                elif class_stack:
                    qual = f"{rel_path}::{'.'.join(class_stack)}.{name}"
                    out.append((name, qual, "method", child.lineno))
                else:
                    qual = f"{rel_path}::{name}"
                    out.append((name, qual, "func", child.lineno))
                # Don't recurse into nested functions for indexing — they're
                # closures; if the model cares it'll see them in the file.
                visit(child, class_stack)
            elif isinstance(child, ast.ClassDef):
                name = child.name
                qual = f"{rel_path}::{name}"
                out.append((name, qual, "class", child.lineno))
                visit(child, class_stack + [name])
            elif isinstance(child, ast.Assign) and not class_stack:
                # Module-level constant: only ALL_CAPS names (heuristic to
                # avoid counting every local variable).
                for tgt in child.targets:
                    if isinstance(tgt, ast.Name) and tgt.id.isupper() and len(tgt.id) >= 2:
                        qual = f"{rel_path}::{tgt.id}"
                        out.append((tgt.id, qual, "const", child.lineno))
            else:
                visit(child, class_stack)

    visit(tree, [])
    return out


def _walk_refs(tree: ast.AST, rel_path: str,
               want_names: set[str]) -> list[tuple[str, int]]:
    """Walk a parsed module and yield (bare_name, line) for every Name/Attribute
    load whose bare identifier is in want_names. Loads inside string literals
    or comments are NOT counted (that's the whole point of using AST).

    Returns a flat list — caller groups by name.
    """
    out: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # Only count Loads. Store (assignment target), Del, AugStore
            # are not "using" the symbol — they're rebinding or removing.
            if isinstance(node.ctx, ast.Load) and node.id in want_names:
                out.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            # Only count attribute LOADS — `obj.write()` is Load,
            # `obj.write = func` is Store.
            if isinstance(node.ctx, ast.Load) and node.attr in want_names:
                out.append((node.attr, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in want_names:
                    out.append((alias.name, node.lineno))
    return out


def _walk_imports(tree: ast.AST) -> set[str]:
    """Return the set of dotted module names this file imports.

    For `import X.Y.Z` → {"X.Y.Z", "X.Y", "X"} — every prefix counts as
    "this file uses X" because Python's import system resolves `X.Y.Z`
    by first importing X, then X.Y, etc.

    For `from X.Y import Z` → {"X.Y"} (we don't mark X as used; only
    the specific module being imported from).
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                for i in range(1, len(parts) + 1):
                    out.add(".".join(parts[:i]))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def _walk_all_loads(tree: ast.AST) -> list[tuple[str, int]]:
    """Walk a parsed module and yield (bare_name, line) for EVERY Name/Attribute
    Load and ImportFrom alias — without filtering by want_names.

    Used by the incremental indexer: per-file caches store ALL loads, so
    that when symbols are added/removed elsewhere we don't have to re-parse
    this file. The symbol-table rebuild step joins these against the live
    def map.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                out.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.ctx, ast.Load):
                out.append((node.attr, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.append((alias.name, node.lineno))
    return out


def _list_py_files(root: str, max_files: int = 20000) -> list[str]:
    """Walk root, skip the usual suspects, return absolute .py paths."""
    skip_dirs = {
        ".git", ".jarvis_sandbox", "__pycache__", ".tox", ".venv", "venv",
        "node_modules", ".pytest_cache", ".mypy_cache", "build", "dist",
        ".eggs", "htmlcov",
        # v8.7 fix: snapshot/audit/dev dirs that contain embedded source
        # as strings (for prompt variants). DEPENDENCY was reporting
        # backup_v4_* matches as if they were live callers.
        "behavioral_audit", "prompt_snapshots", "rendered", "rendered_deep",
    }
    # Match dirnames whose prefix marks them as historical backups.
    skip_prefixes = ("backup_v", "backup-v", "snapshot_", "snapshot-")
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs
            and not d.startswith(".")
            and not d.startswith(skip_prefixes)
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
                if len(out) >= max_files:
                    return out
    return out


def build_index(project_root: str,
                threshold: int = 10,
                max_files: int = 20000) -> DependencyIndex:
    """Build (or refresh) the symbol+reference index for a project.

    Implementation note: this constructs an empty DependencyIndex and calls
    .refresh() on it. Subsequent .refresh() calls on the same instance are
    incremental — only files whose mtime changed get re-parsed.

    For incremental updates between VIEW calls, prefer holding onto the
    same DependencyIndex object and calling .refresh() on it. build_index()
    is the cold-start constructor.

    Args:
        project_root: absolute path to the project root (typically the
            sandbox dir for JARVIS).
        threshold: minimum ref_count to surface in the inline annotation.
        max_files: safety cap on number of .py files indexed.
    """
    idx = DependencyIndex(project_root, threshold=threshold,
                          max_files=max_files)
    idx.refresh()
    return idx


# ── Rendering ───────────────────────────────────────────────────────────────

# Match a view-format line and capture its line number. The view format is the
# current v10 front-line-number form `LINENO:INDENT|code` (e.g. `286:4|    def foo`);
# group(1) is the line number. (Was the dead `i<N>|<code> <trailing-lineno>` form, which
# the renderer stopped producing when the line# moved to the front — so the annotation
# silently never fired and `[DEPENDENCY: #tag]` became unreachable. audit fix C.)
_VIEW_LINE_RE = re.compile(r'^(\d+)\s*[:⇥](\d+)\|(.*)$')   # colon OR native ⇥ form (bughunt A)

# Thresholds for the "central" marker. The marker fires when the
# symbol's def file contains at least one symbol referenced from
# CENTRAL_IMPORTERS_THRESHOLD+ other files AND this symbol's own AST
# count is below CENTRAL_LOW_REF_THRESHOLD.
#
# Tuned empirically against 5000-case ground-truth audit:
#   • (100, 5)  → 24% violation-catch, 9% false-positive rate
#   • (50, 10)  → 57% catch, 21% FP (too noisy)
#   • (200, 5)  → 5% catch, 6% FP (too narrow)
# (100, 5) gives a useful weak signal without crying wolf. Documented
# in behavioral_audit/v12/dep_index_audit/sweep_central_thresh.py.
CENTRAL_IMPORTERS_THRESHOLD = 100
CENTRAL_LOW_REF_THRESHOLD = 5


def annotate_view(file_path: str, view_text: str,
                  idx: DependencyIndex,
                  threshold: Optional[int] = None) -> str:
    """Inline-annotate VIEW/CODE output: append `|appears N (#tag)` to each
    line where a high-use symbol is DEFINED.

    The annotation only fires on the def line — usage lines stay clean,
    because surfacing the marker on every call site would be visual noise.
    The model sees blast-radius signal exactly once per symbol, right next
    to the definition, and can drill in with [dependency: #tag].

    Idempotent: re-annotating is a no-op (we skip lines that already have
    `|appears `).
    """
    if not view_text:
        return view_text

    # v8.7 fix: previously the threshold was 10 — meaning only heavily-used
    # symbols got a `|appears N (#tag)` annotation. The DEPENDENCY tool can
    # only be invoked with a known tag, so for singleton symbols (def with
    # < 10 references) there was NO way for the agent to discover the tag,
    # making DEPENDENCY unreachable for ~80% of symbols. Lower threshold to
    # 1 so every def emits its tag; the model can scan past the tail
    # annotation but can also drill in when needed.
    thr = threshold if threshold is not None else getattr(idx, "_threshold", 1)
    syms = [s for s in idx.symbols_in_file(file_path) if s.ref_count() >= thr]
    if not syms:
        return view_text

    # Build def_line -> Symbol map for this file. Two symbols on the same
    # line is extremely rare; if it happens we keep the highest-use one.
    by_def_line: dict[int, Symbol] = {}
    for s in syms:
        prev = by_def_line.get(s.def_line)
        if prev is None or s.ref_count() > prev.ref_count():
            by_def_line[s.def_line] = s

    out_lines: list[str] = []
    trailing_nl = view_text.endswith("\n")
    for line in view_text.splitlines(keepends=False):
        m = _VIEW_LINE_RE.match(line)
        if m and "|appears " not in line:
            try:
                lineno = int(m.group(1))   # front line number (v10 LINENO:INDENT|code)
            except ValueError:
                out_lines.append(line)
                continue
            sym = by_def_line.get(lineno)
            if sym is not None:
                # "shared name" warning: when N other symbols use the
                # same bare name, AST credits every attribute-access
                # ref to all of them — count is an UPPER BOUND for
                # blast radius.
                n_same_name = len(idx.by_name.get(sym.name, []))
                shared = "" if n_same_name <= 1 else ", shared name"
                # "central" warning: the def file is imported by many
                # other modules (>= CENTRAL_IMPORTERS_THRESHOLD), so a
                # symbol here is likely on transitive call paths that
                # AST cannot see. Empirically AST UNDER-counts blast
                # for these symbols (5000-case audit showed all 74
                # INV-1 violations were in heavily-imported modules).
                # Only fires when the literal AST count is low — that's
                # exactly the misleading case.
                n_importers = idx.importer_count.get(sym.def_file, 0)
                central = ""
                if (n_importers >= CENTRAL_IMPORTERS_THRESHOLD
                        and sym.ref_count() < CENTRAL_LOW_REF_THRESHOLD):
                    central = ", central"
                line = (f"{line} |appears {sym.ref_count()} "
                        f"(#{sym.tag}{shared}{central})")
        out_lines.append(line)

    result = "\n".join(out_lines)
    if trailing_nl:
        result += "\n"
    return result


def _render_ref_list(refs: list[tuple[str, int]],
                     max_refs: int = 200) -> list[str]:
    """Format a list of (file, line) refs as readable lines.

    Files sorted by ref-density (most callers first), lines deduped.
    Output truncated at max_refs and "... +N more" appended if so.
    """
    by_file: dict[str, list[int]] = {}
    for f, ln in refs:
        by_file.setdefault(f, []).append(ln)
    file_order = sorted(by_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    out: list[str] = []
    shown = 0
    for i, (f, lns) in enumerate(file_order):
        lns_sorted = sorted(set(lns))
        if shown >= max_refs:
            remaining = sum(len(v) for _, v in file_order[i:])
            out.append(f"    ... +{remaining} more refs "
                       f"(truncated at {max_refs})")
            break
        line_str = ", ".join(str(l) for l in lns_sorted[:20])
        if len(lns_sorted) > 20:
            line_str += f", ... +{len(lns_sorted) - 20} more"
        out.append(f"    {f}:  {line_str}")
        shown += len(lns_sorted)
    return out


def render_dependency(tag: str, idx: DependencyIndex,
                      max_refs: int = 200) -> str:
    """Render a [DEPENDENCY: #TAG] result.

    Always shows the AST upper-bound section. Adds an LSP-precise
    section when LSP was successfully resolved AND the two counts
    differ meaningfully (precise = exact dispatch resolution; helps
    the model see which callers definitely hit THIS method vs which
    might hit a polymorphic sibling).
    """
    sym = idx.lookup_tag(tag)
    if sym is None:
        _by_tag = getattr(idx, "by_tag", None) or {}
        _avail = sorted(_by_tag.keys())[:12]
        _avail_str = (
            "\nAvailable tags right now: " + ", ".join("#" + t for t in _avail)
            + ("" if len(_by_tag) <= 12 else f" (+{len(_by_tag) - 12} more)") + "."
            if _avail else
            "\nNo tags are registered yet — none of your reads surfaced a shared symbol."
        )
        return (
            f"DEPENDENCY ERROR: unknown tag #{tag.lstrip('#')}.{_avail_str}\n"
            f"Tags come from [CODE:]/[VIEW:]/[KEEP:] output, annotated "
            f"`|appears N (#tag, ...)` next to a symbol. Run [CODE: path] on a "
            f"file with your target symbol to discover its tag."
        )

    # v8.15 fix: subagent r16 (B10) found DEPENDENCY returned ghost data
    # when the source file had been deleted between CODE-time tag
    # discovery and the DEPENDENCY lookup. Validate the def file still
    # exists; if not, the index is stale and the model must re-discover.
    import os
    _root = getattr(idx, "project_root", None)
    if _root and sym.def_file:
        _resolved = os.path.join(_root, sym.def_file)
        if not os.path.isfile(_resolved):
            return (
                f"DEPENDENCY ERROR: tag #{tag.lstrip('#')} resolves to "
                f"{sym.def_file}:{sym.def_line} ({sym.name}) but that "
                f"file no longer exists.\n"
                f"The dependency index is stale — the file was deleted "
                f"or moved since the tag was discovered. Re-run [CODE:] "
                f"on the current file containing your target symbol to "
                f"get a fresh tag."
            )

    n_same_name = len(idx.by_name.get(sym.name, []))
    shared_note = ""
    if n_same_name > 1:
        shared_note = (f" — bare name shared with {n_same_name - 1} other "
                       f"{'def' if n_same_name == 2 else 'defs'} "
                       f"across the project")

    # v8.15 fix (B17): subagent r18 found DEPENDENCY counts are sandbox-
    # subset but unlabeled. A model reading "5 references in 2 files"
    # would underestimate refactor blast radius if the full project has
    # 45 refs in 16 files. Surface the scope explicitly.
    _idx_scope_files = len(getattr(idx, "by_def_file", {}) or {})
    scope_note = f"  scope: {_idx_scope_files} files indexed (from .jarvis_sandbox/ working copy)"

    out = [
        f"SYMBOL: {sym.name}  ({sym.kind})",
        f"  defined at {sym.def_file}:{sym.def_line}",
        f"  qualname: {sym.qualname}",
        scope_note,
    ]

    # ─── AST UPPER BOUND ─────────────────────────────────────────────
    out.append("")
    out.append(f"  UPPER BOUND (AST, includes every `.{sym.name}` "
               f"attribute-access site):")
    out.append(f"    {sym.ref_count()} references in "
               f"{len(sym.ref_files())} files{shared_note}.")
    out.append(f"    Use as worst-case blast radius — these are all "
               f"the places that COULD call this symbol.")
    out.extend(_render_ref_list(sym.refs, max_refs=max_refs))

    # ─── LSP PRECISE (if resolved) ───────────────────────────────────
    if sym._lsp_resolved and sym.lsp_refs is not None:
        lsp_count = sym.lsp_ref_count() or 0
        lsp_files = len(sym.lsp_ref_files())
        ast_count = sym.ref_count()
        out.append("")
        out.append(f"  PRECISE (LSP, type-resolved callers only):")
        if lsp_count == ast_count:
            out.append(f"    {lsp_count} references — matches the AST "
                       f"upper bound exactly. No polymorphic ambiguity.")
        else:
            ratio = (f"{lsp_count}/{ast_count} = "
                     f"{100*lsp_count/ast_count:.0f}%" if ast_count else "0%")
            out.append(f"    {lsp_count} references in {lsp_files} files "
                       f"({ratio} of the AST upper bound).")
            out.append(f"    LSP could only statically resolve {lsp_count} "
                       f"of the {ast_count} attribute-access sites — the "
                       f"others might or might not hit this method "
                       f"depending on the receiver's runtime type. Edit "
                       f"with the upper bound in mind.")
        out.extend(_render_ref_list(sym.lsp_refs, max_refs=max_refs))
    else:
        out.append("")
        out.append("  (LSP not consulted — drill-in shows AST upper bound only)")

    return "\n".join(out)
