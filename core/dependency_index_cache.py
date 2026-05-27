"""
Process-wide cache for the dependency index.

JARVIS runs many tool calls inside one agent session against the same
sandbox directory. Re-building the AST index from scratch on every call
would cost ~3s × N calls. Instead we hold one DependencyIndex per
sandbox path and call .refresh() on it before each VIEW/CODE annotation.

The cache is module-global and protected by a lock — JARVIS can run
multiple coding agents in parallel (different sandbox dirs), and within
one agent the lookup loop may also fire concurrently. The lock keeps
get_or_refresh_index() reentrant-safe; the underlying refresh() does
its own per-file mtime check, so it's a no-op when nothing changed.
"""

import threading
from typing import Optional

from core.dependency_index import (
    DependencyIndex,
    build_index,
    annotate_view,
    render_dependency,
)


_CACHE: dict[str, DependencyIndex] = {}
_LOCK = threading.Lock()


def get_or_refresh_index(sandbox_dir: str,
                         threshold: int = 1) -> Optional[DependencyIndex]:
    """Return a DependencyIndex for sandbox_dir, building or refreshing as
    needed.

    On refresh, also invalidate the LSP cache for any symbol that may
    have been affected by a changed file. Coarse — we drop all LSP
    flags — but the cost is bounded: only symbols the model actually
    views next will need re-query.

    Returns None if the sandbox doesn't exist (caller should skip annotation).
    """
    import os
    if not os.path.isdir(sandbox_dir):
        return None
    with _LOCK:
        idx = _CACHE.get(sandbox_dir)
        if idx is None:
            idx = build_index(sandbox_dir, threshold=threshold)
            _CACHE[sandbox_dir] = idx
        else:
            stats = idx.refresh()
            if stats.get("rebuilt_table"):
                try:
                    from core.dependency_lsp import invalidate_lsp_cache
                    invalidate_lsp_cache(idx)
                except ImportError:
                    pass
        return idx


def annotate_code_output(sandbox_dir: str, file_path: str, view_text: str,
                         threshold: int = 1) -> str:
    """SYNC fallback. Uses AST-only counts (upper bound for polymorphic
    methods). Kept for callers that haven't migrated to async LSP path.
    """
    try:
        idx = get_or_refresh_index(sandbox_dir, threshold=threshold)
        if idx is None:
            return view_text
        return annotate_view(file_path, view_text, idx, threshold=threshold)
    except Exception:
        return view_text


async def annotate_code_output_async(sandbox_dir: str, file_path: str,
                                     view_text: str,
                                     threshold: int = 1) -> str:
    """Hybrid inline annotation: uses AST upper-bound counts (safer
    for blast-radius signal) + a `shared name` marker on polymorphic
    methods. LSP is reserved for the [DEPENDENCY:] drill-in path —
    inline stays fast and conservative.

    This is `async` so the call site signature is identical to the
    previous version (which queried LSP eagerly), but we no longer
    hit LSP here. Keeping it async leaves the door open to future
    LSP-pre-warming without another refactor.
    """
    try:
        idx = get_or_refresh_index(sandbox_dir, threshold=threshold)
        if idx is None:
            return view_text
        return annotate_view(file_path, view_text, idx, threshold=threshold)
    except Exception:
        return view_text


def lookup_dependency(sandbox_dir: str, tag: str) -> str:
    """SYNC fallback. AST-only refs. Use lookup_dependency_async for
    LSP-precise lookups.
    """
    try:
        idx = get_or_refresh_index(sandbox_dir)
        if idx is None:
            return (f"DEPENDENCY ERROR: cannot resolve #{tag.lstrip('#')} — "
                    f"no sandbox at {sandbox_dir}")
        return render_dependency(tag, idx)
    except Exception as e:
        return f"DEPENDENCY ERROR: {e}"


async def lookup_dependency_async(sandbox_dir: str, tag: str) -> str:
    """Resolve [DEPENDENCY: #tag] with LSP-precise references. Falls
    back to AST upper bound silently if LSP is down.
    """
    try:
        idx = get_or_refresh_index(sandbox_dir)
        if idx is None:
            return (f"DEPENDENCY ERROR: cannot resolve #{tag.lstrip('#')} — "
                    f"no sandbox at {sandbox_dir}")
        sym = idx.lookup_tag(tag)
        if sym is not None:
            try:
                from core.dependency_lsp import resolve_refs_with_lsp
                await resolve_refs_with_lsp(idx, sym)
            except ImportError:
                pass
        return render_dependency(tag, idx)
    except Exception as e:
        return f"DEPENDENCY ERROR: {e}"


def invalidate(sandbox_dir: str) -> None:
    """Drop the cache entry for a sandbox (used on agent shutdown or
    explicit reset). Next get_or_refresh_index call will rebuild cold."""
    with _LOCK:
        _CACHE.pop(sandbox_dir, None)


def invalidate_all() -> None:
    """Drop all cached indices."""
    with _LOCK:
        _CACHE.clear()
