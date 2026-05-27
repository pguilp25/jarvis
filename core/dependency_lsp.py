"""
LSP-backed precise reference resolution for the dependency index.

The AST-based index gives an UPPER BOUND on references — polymorphic
methods (multiple `def write` across classes) all get credited for any
`.write()` call because we can't resolve the receiver type without
type inference.

LSP gives us PRECISE counts: pylsp/pyright understand that `obj.write()`
resolves to `MyClass.write` (or `BaseClass.write`, depending on the
MRO), and `find_references` returns only the actual call sites.

Design:
  • Lazy — we only query LSP for a symbol when its precise count is
    actually needed (during annotation of a file, or during a
    [DEPENDENCY:] drill-in).
  • Cached per-symbol — the LSP result is stored on the Symbol object
    (sym.refs) and only re-queried after the file's mtime changes.
  • Fallback — if LSP is unavailable, we keep the AST upper-bound and
    add a "(coarse — LSP unavailable)" note. Never blocks annotation.

Cost model:
  • One reference query: ~10-50 ms after the LSP server is warm.
  • One file with N indexed symbols on first annotate: N × 50 ms.
  • Subsequent VIEWs on the same file: 0 ms (cached).
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from core.dependency_index import DependencyIndex, Symbol


async def _find_def_col(sym: Symbol, project_root: str) -> Optional[int]:
    """Find the column where `sym.name` appears on its def line.

    LSP needs (file, line, col) — line we already know, col we have to
    locate inside the line. We use the first occurrence of sym.name as
    a substring of the def line. For `def foo(...):` this lands on `f`
    of `foo`; for `class Bar:` on `B` of `Bar`. Robust enough for
    references resolution.
    """
    abs_path = os.path.join(project_root, sym.def_file)
    if not os.path.isfile(abs_path):
        return None
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if sym.def_line < 1 or sym.def_line > len(lines):
            return None
        line_text = lines[sym.def_line - 1]
        idx = line_text.find(sym.name)
        return idx if idx >= 0 else None
    except OSError:
        return None


async def _query_lsp_refs(sym: Symbol, project_root: str) -> Optional[list[tuple[str, int]]]:
    """Query LSP for the precise reference list of one symbol.

    Returns None if LSP is unavailable (caller falls back to AST count)
    or an empty list if the symbol has no references (legitimate result).
    """
    try:
        from tools.lsp import get_lsp_client
    except ImportError:
        return None

    client = await get_lsp_client(project_root)
    if client is None:
        return None

    col = await _find_def_col(sym, project_root)
    if col is None:
        return None

    try:
        # LSP is 0-indexed for lines; sym.def_line is 1-indexed
        refs = await client.find_references(sym.def_file, sym.def_line - 1, col)
    except Exception:
        return None

    if not refs:
        return []

    out: list[tuple[str, int]] = []
    for r in refs:
        f = r.get("file") or ""
        ln = r.get("line") or 0
        if not f:
            continue
        # Exclude the def itself — Q0/blast-radius cares about USES,
        # not the def. LSP's includeDeclaration=True returns the def
        # too; filter it here.
        if f == sym.def_file and ln == sym.def_line:
            continue
        out.append((f, ln))
    return out


async def resolve_refs_with_lsp(idx: DependencyIndex, sym: Symbol) -> bool:
    """Populate sym.lsp_refs with LSP-precise references. Does NOT touch
    sym.refs (the AST upper bound stays intact — both tracks coexist).

    Returns True if LSP succeeded, False otherwise. After a True
    return, sym.lsp_refs is a list (possibly empty); after False it
    stays None and the caller should rely on sym.refs (AST).
    """
    if sym._lsp_resolved:
        return True

    refs = await _query_lsp_refs(sym, idx.project_root)
    if refs is None:
        sym._lsp_resolved = False
        return False

    sym.lsp_refs = refs
    sym._lsp_resolved = True
    return True


async def resolve_file_refs_with_lsp(idx: DependencyIndex, file_rel: str,
                                     concurrent: int = 4) -> int:
    """Resolve LSP refs for every symbol defined in file_rel.

    Concurrency: LSP servers serialize requests anyway, but we batch
    the asyncio.gather to overlap with our own bookkeeping. Returns the
    number of symbols successfully LSP-resolved.
    """
    syms = idx.symbols_in_file(file_rel)
    if not syms:
        return 0
    # Skip symbols already resolved
    todo = [s for s in syms if not getattr(s, "_lsp_resolved", False)]
    if not todo:
        return 0

    sem = asyncio.Semaphore(concurrent)

    async def _one(s):
        async with sem:
            return await resolve_refs_with_lsp(idx, s)

    results = await asyncio.gather(*[_one(s) for s in todo])
    return sum(1 for ok in results if ok)


def invalidate_lsp_cache(idx: DependencyIndex) -> None:
    """Clear LSP-resolved state on every symbol. Called after a
    refresh() that re-parsed any file: LSP results could be stale
    because the underlying code changed.
    """
    for sym in idx.by_tag.values():
        sym._lsp_resolved = False
        sym.lsp_refs = None
