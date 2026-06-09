"""
NVIDIA NIM embeddings client — uses llama-nemotron-embed-1b-v2.
Generates vector embeddings for semantic search DIRECTLY over the code
(functions/classes, AST-extracted) — no LLM-generated purpose map required.
Semantic search is a first-class, always-available tool: it indexes the source
itself, so it works with zero indexing pre-pass and survives map removal.
"""

import ast
import json
import math
import os
import time
from pathlib import Path

import aiohttp

EMBED_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL   = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_CACHE   = "embeddings.json"   # stored in the maps cache dir


def _get_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


async def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Call NVIDIA NIM embeddings endpoint. Returns one vector per input text.

    input_type:
      "passage"  — for indexing code chunks
      "query"    — for embedding a user query at search time
    """
    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            EMBED_API_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Embed API HTTP {resp.status}: {body[:300]}")
            data = await resp.json()
    # data["data"] is a list of {"embedding": [...], "index": N}. A gateway can
    # return HTTP 200 with an error/moderation envelope (no "data") — raise a
    # clear RuntimeError instead of a bare KeyError so callers surface a useful
    # message and the build path doesn't silently zero-fill (see D#2).
    try:
        ordered = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in ordered]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Embed API returned a 200 with an unexpected shape "
            f"({e}): {str(data)[:300]}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if not (na and nb):
        return 0.0
    sim = dot / (na * nb)
    return sim if math.isfinite(sim) else 0.0


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _embed_cache_path(maps_dir: Path) -> Path:
    return maps_dir / EMBED_CACHE


def load_embed_cache(maps_dir: Path) -> dict | None:
    """Load embedding cache: {"hash": str, "chunks": [{"name": str, "text": str, "vec": [...]}]}"""
    p = _embed_cache_path(maps_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_embed_cache(maps_dir: Path, file_hash: str, chunks: list[dict]):
    """Save embedding cache to disk."""
    p = _embed_cache_path(maps_dir)
    p.write_text(
        json.dumps({"hash": file_hash, "chunks": chunks}),
        encoding="utf-8",
    )


def _vec_is_zero(vec) -> bool:
    """A failed-batch fill is an all-zero vector; treat first-5 zeros as zero."""
    return (not vec) or all(v == 0 for v in vec[:5])


def _build_wholly_failed(chunks: list[dict]) -> bool:
    """True if every chunk got a zero vector — i.e. the embedding endpoint was
    down for the WHOLE build. Persisting that poisons the cache: future calls
    see a hash-matching cache and never rebuild, returning a misleading
    "(no results)" forever. Detect it so callers can skip the save + report
    unavailable instead."""
    if not chunks:
        return False
    return all(_vec_is_zero(c.get("vec")) for c in chunks)


# ─── Purpose-map chunker ─────────────────────────────────────────────────────

def parse_purpose_chunks(purpose_map: str) -> list[dict]:
    """Split purpose map into chunks: [{"name": str, "text": str}]"""
    import re
    chunks = []
    parts = re.split(r'===\s*PURPOSE:\s*', purpose_map)
    for part in parts:
        if not part.strip():
            continue
        title_end = part.find("===")
        if title_end == -1:
            title = part.split("\n")[0].strip()
            body  = part.strip()
        else:
            title = part[:title_end].strip()
            body  = part[title_end + 3:].strip()
        if not title:
            continue
        # Build indexable text: title + description + first few lines of body
        desc = ""
        for line in body.split("\n")[:5]:
            if line.strip():
                desc += line.strip() + " "
        text = f"{title}. {desc}".strip()
        chunks.append({"name": title, "text": text})
    return chunks


# ─── Code chunker (AST — no purpose map needed) ───────────────────────────────

_SKIP_DIR_PARTS = {".git", "node_modules", "__pycache__", ".jarvis", ".venv",
                   "venv", "dist", "build", ".pytest_cache", ".jarvis_sandbox",
                   ".mypy_cache", ".tox", "site-packages"}


# The embedding model's safe input window (chars). A unit that fits is embedded
# WHOLE (max fidelity, no truncation); a unit larger than this is split into
# line-aligned, slightly-overlapping windows so the entire body is still
# represented across chunks. ~6000 chars ≈ 1500 tokens — comfortably under the
# model's limit, so a batch never fails for being too long.
_EMBED_WINDOW_CHARS = 6000
_WINDOW_OVERLAP_LINES = 3


def _line_windows(lines: list[str], max_chars: int = _EMBED_WINDOW_CHARS) -> list[str]:
    """Split a list of source lines into the FEWEST windows that each fit in
    max_chars, breaking only at line boundaries (never mid-line) with a small
    overlap so context isn't lost across a split. A unit that already fits
    returns as a single, complete window — no truncation, no padding."""
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return [joined]
    windows, cur, cur_len = [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) + 1 > max_chars:
            windows.append("\n".join(cur))
            cur = cur[-_WINDOW_OVERLAP_LINES:]            # carry overlap forward
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        windows.append("\n".join(cur))
    return windows


def _unit_start_line(node) -> int:
    """First source line of a def/class INCLUDING decorators. `ast` sets
    `node.lineno` to the `def`/`class` keyword, so a decorated unit would
    otherwise drop its `@decorator` lines — losing fidelity (a `@property` or
    `@app.route(...)` is part of what the symbol IS)."""
    line = node.lineno
    for dec in getattr(node, "decorator_list", []):
        line = min(line, getattr(dec, "lineno", line))
    return line


def parse_code_chunks(project_root: str) -> list[dict]:
    """AST-walk every .py file and return indexable chunks, one per semantic unit
    (function / method / class-header / module-level code):
        {"name": "rel::qualname:line", "text": <full unit source>,
         "file": rel, "line": <1-based start incl. decorators>}.

    Invariants (these are what the test suite pins — break one and retrieval
    quality silently degrades):
      • COMPLETE — every function/class/method in a parseable file is emitted.
      • FAITHFUL — a unit that fits the window is embedded WHOLE (decorators →
        last body line), never truncated; the text begins with `rel::qual` for
        symbol/path grounding.
      • FLEXIBLE — a unit larger than the window is split into line-aligned,
        overlapping windows whose union covers every line (no gap, no mid-line
        cut); small units are a single window (no padding).
      • NON-DUPLICATING — a class emits only its HEADER (class line → first
        method); its methods are separate chunks, never embedded twice.
      • LOSSLESS — module-level code (docstring, imports, constants, top-level
        statements) is captured as a `::<module>` chunk; an unparseable file is
        still indexed via raw line-windows so nothing vanishes from search.
      • DETERMINISTIC — same tree → same chunks, in source order.
    """
    chunks: list[dict] = []
    root = Path(project_root)
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_PARTS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            abs_path = Path(dirpath) / fn
            try:
                src = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = str(abs_path.relative_to(root)).replace(os.sep, "/")
            src_lines = src.splitlines()

            def _emit(qual, start, end, line):
                body = src_lines[start - 1:end]
                if not any(s.strip() for s in body):
                    return
                wins = _line_windows(body)
                for wi, w in enumerate(wins):
                    suffix = f"#{wi + 1}/{len(wins)}" if len(wins) > 1 else ""
                    chunks.append({
                        "name": f"{rel}::{qual}:{line}{suffix}",
                        "text": f"{rel}::{qual}\n{w}",
                        "file": rel, "line": line,
                    })

            try:
                tree = ast.parse(src)
            except SyntaxError:
                # Bulletproof: an unparseable file is still searchable via raw
                # line-windows — code never silently disappears from the index.
                for w in _line_windows(src_lines):
                    if w.strip():
                        chunks.append({"name": f"{rel}::<file>", "text": f"{rel}\n{w}",
                                       "file": rel, "line": 1})
                continue

            def _walk(node, prefix=""):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        s = _unit_start_line(child)
                        _emit(f"{prefix}{child.name}", s,
                              getattr(child, "end_lineno", child.lineno), s)
                    elif isinstance(child, ast.ClassDef):
                        s = _unit_start_line(child)
                        method_starts = [_unit_start_line(c) for c in child.body
                                         if isinstance(c, (ast.FunctionDef,
                                                           ast.AsyncFunctionDef))]
                        header_end = (min(method_starts) - 1 if method_starts
                                      else getattr(child, "end_lineno", child.lineno))
                        _emit(f"{prefix}{child.name}", s, header_end, s)
                        _walk(child, prefix=f"{prefix}{child.name}.")
            _walk(tree)

            # Module-level code: every top-level statement that ISN'T a def/class
            # (module docstring, imports, constants, top-level logic), in order.
            mod_lines: list[str] = []
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    continue
                s = getattr(stmt, "lineno", None)
                e = getattr(stmt, "end_lineno", s)
                if s:
                    mod_lines.extend(src_lines[s - 1:e])
            if any(s.strip() for s in mod_lines):
                wins = _line_windows(mod_lines)
                for wi, w in enumerate(wins):
                    suffix = f"#{wi + 1}/{len(wins)}" if len(wins) > 1 else ""
                    chunks.append({"name": f"{rel}::<module>{suffix}",
                                   "text": f"{rel}\n{w}", "file": rel, "line": 1})
    return chunks


# ─── Main: build / retrieve ───────────────────────────────────────────────────

async def build_embeddings(
    purpose_map: str,
    maps_dir: Path,
    file_hash: str,
    batch_size: int = 32,
) -> list[dict]:
    """Embed all purpose-map chunks and save to cache.
    Returns the list of chunk dicts with "vec" populated.
    """
    from core.cli import status, warn
    chunks = parse_purpose_chunks(purpose_map)
    if not chunks:
        return []

    # Embed in batches (API limit)
    all_vecs: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
        try:
            vecs = await embed_texts(batch_texts, input_type="passage")
            all_vecs.extend(vecs)
            status(f"    Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        except Exception as e:
            warn(f"    Embedding batch {i//batch_size + 1} failed: {e}")
            # Fill with zero vectors so indices stay aligned
            all_vecs.extend([[0.0] * 4096] * len(batch_texts))

    for chunk, vec in zip(chunks, all_vecs):
        chunk["vec"] = vec

    if _build_wholly_failed(chunks):
        warn("    Embedding build wholly failed (endpoint down) — NOT caching poisoned index.")
        return chunks  # caller detects all-zero and reports unavailable
    maps_dir.mkdir(parents=True, exist_ok=True)   # bughunt ckpt-238: mirror build_code_embeddings (else FileNotFoundError on first build)
    save_embed_cache(maps_dir, file_hash, chunks)
    return chunks


async def build_code_embeddings(
    project_root: str,
    maps_dir: Path,
    file_hash: str,
    batch_size: int = 32,
) -> list[dict]:
    """Embed every function/class chunk (AST) and save to cache. No purpose map."""
    from core.cli import status, warn
    chunks = parse_code_chunks(project_root)
    if not chunks:
        return []
    all_vecs: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
        try:
            vecs = await embed_texts(batch_texts, input_type="passage")
            all_vecs.extend(vecs)
            status(f"    Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} code chunks")
        except Exception as e:
            warn(f"    Embedding batch {i // batch_size + 1} failed: {e}")
            all_vecs.extend([[0.0] * 4096] * len(batch_texts))
            # FAST-FAIL on auth/permission errors (ckpt-164): a 401/403/missing-key never
            # recovers on retry, so looping all ~80 batches burns ~90s of a coder round
            # before reporting unavailable. Abort now; zero-fill the rest so the all-zero
            # guard below fires and the caller reports unavailable cleanly.
            if any(s in str(e) for s in ("HTTP 401", "HTTP 403", "Authorization", "Forbidden", "not set")):
                warn("    Embedding auth/permission error — aborting build (won't recover on retry).")
                _rest = len(chunks) - len(all_vecs)
                if _rest > 0:
                    all_vecs.extend([[0.0] * 4096] * _rest)
                break
    for chunk, vec in zip(chunks, all_vecs):
        chunk["vec"] = vec
    if _build_wholly_failed(chunks):
        warn("    Code-embedding build wholly failed (endpoint down) — NOT caching poisoned index.")
        return chunks  # caller detects all-zero and reports unavailable
    maps_dir.mkdir(parents=True, exist_ok=True)
    save_embed_cache(maps_dir, file_hash, chunks)
    return chunks


async def semantic_retrieve(
    query: str,
    project_root: str,
    maps_dir: Path,
    file_hash: str,
    top_n: int = 10,
) -> str:
    """Embed the query, find the top_n most similar CODE chunks (functions/classes,
    AST-extracted from the source), and return each as `file:line` + a snippet.
    No purpose map — semantic search indexes the code itself."""
    from core.cli import status, warn

    cache = load_embed_cache(maps_dir)
    if cache and cache.get("hash") == file_hash and cache.get("chunks"):
        chunks = cache["chunks"]
    else:
        status("    Building semantic code index (first time)...")
        try:
            chunks = await build_code_embeddings(project_root, maps_dir, file_hash)
        except Exception as e:
            warn(f"    Semantic index build failed: {e}")
            return (f"✗ semantic search unavailable: {e}. Do NOT retry it this run — "
                    f"use [SEARCH: text], [REFS: symbol], or [PURPOSE: path] instead.")

    if not chunks:
        return "(no code to search)"

    # Poisoned-index guard (D#2): if EVERY chunk vector is zero, indexing never
    # actually succeeded (endpoint was down during the build / a stale poisoned
    # cache from an older outage). Don't masquerade as "searched, no matches" —
    # tell the model the tool is unavailable and to use lexical tools instead.
    if _build_wholly_failed(chunks):
        warn("    Semantic index is all-zero (build never succeeded) — reporting unavailable.")
        return ("✗ semantic search unavailable: the code index could not be built "
                "(embedding endpoint unreachable). Do NOT retry it this run — use "
                "[SEARCH: text], [REFS: symbol], or [PURPOSE: path] instead.")

    try:
        query_vecs = await embed_texts([query], input_type="query")
        qvec = query_vecs[0]
    except Exception as e:
        warn(f"    Query embedding failed: {e}")
        return (f"✗ semantic search unavailable: {e}. Do NOT retry it this run — "
                    f"use [SEARCH: text], [REFS: symbol], or [PURPOSE: path] instead.")

    scored = []
    for chunk in chunks:
        vec = chunk.get("vec")
        if not vec or all(v == 0 for v in vec[:5]):
            continue
        sim = cosine_similarity(qvec, vec)
        scored.append((sim, chunk))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:top_n]
    if not top:
        return f"(no results for '{query}')"

    parts = [f"=== SEMANTIC: '{query}' — top {len(top)} code matches ===\n"
             "(use [CODE: file] or [VIEW: file start end] to read any match in full)\n"]
    for sim, chunk in top:
        snippet = "\n".join(chunk.get("text", "").splitlines()[:8])
        parts.append(
            f"[{sim:.3f}] {chunk['file']}:{chunk['line']} — {chunk['name'].split('::')[-1]}\n"
            f"{snippet}\n"
        )
    return "\n".join(parts)
