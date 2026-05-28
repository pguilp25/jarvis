"""Native (structured) tool-calling coder path — 2026-05-27.

JARVIS's default coder loop is TEXT-based: the model emits `[tool use]` / `[edit:N]`
as plain text the runtime regex-parses. Models trained primarily for NATIVE
function calling (gpt-oss-120b — the most redundant free model: Groq, OpenRouter,
DeepInfra, NIM) don't speak that idiom; they emit bare tags. This module gives
them a real OpenAI function-calling loop instead.

PARITY: the native coder exposes the SAME capabilities as the text coder — not a
reduced set. Each text tool maps to one native tool, reusing the text coder's
exact executor so behaviour is identical:

    text tag            native tool        executor (core.tool_call)
    ─────────────────   ────────────────   ─────────────────────────────
    [CODE:]/[VIEW:]     read_file          _run_code_reads (skeleton+expand)
    [REFS:]             find_refs          _run_refs_searches
    [DEPENDENCY: #tag]  find_callers       _run_dependency_lookup
    [SEARCH:]           search_text        _run_code_searches
    [PURPOSE:]          file_purpose       _run_purpose_lookups
    [SEMANTIC:]         semantic_search    _run_semantic_lookups
    [DETAIL:]           symbol_detail      _run_detail_lookups
    [REPLACE LINES]     replace_lines      _extract/_apply_extracted_code
    [DONE]              finish             —

(The coder has no [RUN:] — that's planner/reviewer-only, so neither does this.)

INDENT n| FORMAT: read_file renders each line as `LINENO:INDENT|content` (the same
prefix view the text coder sees), so the model reads indentation as a COUNT, not by
eyeballing spaces. replace_lines' new_content uses the same `INDENT|code` convention
(reused indent-expander turns `8|x` → 8 spaces + `x`); raw indentation also passes
through verbatim. This is the weak-model indent-reliability fix, carried into native.

Edit tool = `replace_lines(path, start, end, new_content)` (line-range, NOT the
anchor-based [edit] diff): native models produce clean structured args, and it
REUSES the existing `[REPLACE LINES]` applier + validation gate + reject feedback
(lazy import, to avoid a circular dep with workflows.code).

Public:
  NATIVE_TOOL_MODELS          — model ids that should use this path
  CODER_TOOLS                 — OpenAI tool schemas for the coder
  call_with_native_tools(...) — run the structured tool-use loop
"""
from __future__ import annotations
import asyncio
import json

from core.cli import status, warn

# Models built for native function-calling — use the structured loop, not text.
NATIVE_TOOL_MODELS = {"nvidia/gpt-oss-120b", "groq/gpt-oss-120b"}


def is_native_tool_model(model_id: str) -> bool:
    return model_id in NATIVE_TOOL_MODELS


# ── Tool schemas (OpenAI function-calling format) ────────────────────────────
# One schema per text-coder capability — full parity, not a subset.
CODER_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            "Read a file (optionally a line range) from the project. Each line is "
            "returned as `LINENO:INDENT|content` — LINENO is the 1-based line number, "
            "INDENT is the leading-space COUNT, then the code. A huge file comes back "
            "as a skeleton (top-level defs); pass start_line/end_line to expand a "
            "region. ALWAYS read a file right before you edit it so your replace_lines "
            "line numbers are current."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative file path"},
            "start_line": {"type": "integer", "description": "first line, 1-based (optional)"},
            "end_line": {"type": "integer", "description": "last line, 1-based (optional)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "find_refs",
        "description": (
            "Find usages of a symbol (function/class/variable name) across the project "
            "— cheap word-boundary search. Your FIRST lookup when you need to know who "
            "uses or produces a name before changing it."),
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "exact identifier to search for"},
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "find_callers",
        "description": (
            "Type-resolved callers / blast-radius for a #tag. read_file annotates a "
            "shared symbol as `|appears N (#tag)`; pass that #tag here to see the "
            "resolved callsites and what would break if you change it. Use when N is "
            "large (≥ ~20) instead of reading every callsite."),
        "parameters": {"type": "object", "properties": {
            "tag": {"type": "string", "description": "the #tag from an |appears N (#tag) annotation"},
        }, "required": ["tag"]},
    }},
    {"type": "function", "function": {
        "name": "search_text",
        "description": (
            "Literal/regex text search across the project (ripgrep). Use to locate a "
            "test by name, an error string, or a code pattern when you don't yet know "
            "the file. This is text search, NOT concept search."),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "text or regex to search for"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "file_purpose",
        "description": (
            "File gist: the module docstring + each top-level def/class's one-line "
            "purpose, WITHOUT the bodies. Use to triage which file to read_file when "
            "you're not yet sure where the change goes — far cheaper than reading."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative file path"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "semantic_search",
        "description": (
            "Rank source files by how well their docstrings/comments match a concept "
            "described in plain words. Use when you know WHAT behaviour you want but not "
            "WHERE it lives. Not a substitute for search_text on an exact symbol."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "concept in plain words"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "symbol_detail",
        "description": (
            "Deep dive on ONE named def/class: its full source plus where it's defined "
            "and used. Use to understand a single function/class precisely before "
            "editing it."),
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "a def/class name, e.g. MyClass.method"},
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "replace_lines",
        "description": (
            "Your EDIT. Replace lines start_line..end_line (inclusive, 1-based, from "
            "your most recent read_file) of `path` with new_content. Each line of "
            "new_content should be `INDENT|code` — INDENT is the leading-space count "
            "(copy it from the read_file view; for a new line, count the spaces it "
            "needs). NO LINENO prefix on new content. Raw real indentation also works. "
            "To INSERT after line N without deleting it, set start_line=end_line=N and "
            "make new_content the current line N followed by your new line(s). The "
            "result is parse-checked; a rejection tells you the actual current line so "
            "you can correct and retry."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "new_content": {"type": "string"},
        }, "required": ["path", "start_line", "end_line", "new_content"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": (
            "Call ONLY when the edit is complete and you've verified it does what the "
            "step asked. Ends the task."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "one line: what you changed"},
        }, "required": []},
    }},
]


# ── Tool dispatch ────────────────────────────────────────────────────────────
async def _do_read(args: dict, ctx: dict) -> str:
    # Reuse the text coder's [CODE:]/[VIEW:] executor: skeleton for huge files,
    # range expansion, the `LINENO:INDENT|content` prefix view, AND recording
    # viewed_versions so a following replace_lines anchors on what was just read.
    from core.tool_call import _run_code_reads
    path = args.get("path", "")
    if not path:
        return "✗ read_file needs a path."
    s = args.get("start_line"); e = args.get("end_line")
    arg = f"{path} {int(s)}-{int(e)}" if (s and e) else path
    try:
        out = await _run_code_reads(
            [arg], ctx.get("project_root", ""),
            viewed_versions=ctx.get("viewed_versions"), display_mode="prefix")
    except Exception as ex:
        out = f"✗ read_file failed: {str(ex)[:160]}"
    # Keep file_contents (the replace_lines base) in step with the sandbox.
    sb = ctx.get("sandbox")
    if sb is not None:
        cur = sb.load_file(path)
        if cur is not None:
            ctx["file_contents"][path] = cur
    return out


def _do_replace(args: dict, ctx: dict) -> str:
    # Reuse the proven [REPLACE LINES] machinery (applier + validation gate +
    # actual-line reject feedback) by building the text block the runtime
    # already knows how to apply.
    from workflows.code import _extract_code_blocks, _apply_extracted_code
    path = args.get("path", "")
    s = args.get("start_line"); e = args.get("end_line")
    new = args.get("new_content", "")
    if not path or s is None or e is None:
        return "✗ replace_lines needs path, start_line, end_line, new_content."
    block = (f"=== EDIT: {path} ===\n[REPLACE LINES {int(s)}-{int(e)}]\n"
             f"{new}\n[/REPLACE]\n=== END EDIT ===")
    ext = _extract_code_blocks(block)
    result, matched, attempted, skips = _apply_extracted_code(
        ext, ctx["file_contents"], ctx.get("sandbox"),
        viewed_versions=ctx.get("viewed_versions"))
    # malformed-range messages live on the extracted dict
    skips = list(ext.get("malformed_edits", [])) + list(skips)
    if path in result:
        if ctx.get("sandbox") is not None:
            ctx["sandbox"].write_file(path, result[path])
        ctx["file_contents"][path] = result[path]
        # The edit shifted line numbers; the old read snapshot is now stale.
        if isinstance(ctx.get("viewed_versions"), dict):
            ctx["viewed_versions"][path] = result[path]
        ctx.setdefault("files_changed", set()).add(path)
        n = result[path].count("\n") + 1
        return (f"✓ Applied: {path} lines {s}-{e} replaced. File is now {n} lines. "
                f"Re-read with read_file if you need the new numbering before another edit.")
    reason = " | ".join(str(x).strip().lstrip("-").strip() for x in skips) or \
        "no change produced (range may be invalid)"
    return f"✗ NOT applied to {path}: {reason}"


async def _do_refs(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_refs_searches
    sym = args.get("symbol", "")
    if not sym:
        return "✗ find_refs needs a symbol."
    return await _run_refs_searches([sym], ctx.get("project_root", ""))


async def _do_callers(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_dependency_lookup
    tag = args.get("tag", "")
    if not tag:
        return "✗ find_callers needs a #tag (from an |appears N (#tag) annotation)."
    return await _run_dependency_lookup([tag], ctx.get("project_root", ""))


async def _do_search(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_code_searches
    pat = args.get("pattern", "")
    if not pat:
        return "✗ search_text needs a pattern."
    return await _run_code_searches([pat], ctx.get("project_root", ""))


def _do_purpose(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_purpose_lookups
    path = args.get("path", "")
    if not path:
        return "✗ file_purpose needs a path."
    return _run_purpose_lookups([path], ctx.get("purpose_map") or "",
                                ctx.get("project_root", ""))


def _do_semantic(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_semantic_lookups
    q = args.get("query", "")
    if not q:
        return "✗ semantic_search needs a query."
    return _run_semantic_lookups([q], ctx.get("project_root", ""),
                                 purpose_map=ctx.get("purpose_map"))


def _do_detail(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_detail_lookups
    sym = args.get("symbol", "")
    if not sym:
        return "✗ symbol_detail needs a symbol name."
    return _run_detail_lookups([sym], ctx.get("detailed_map") or "",
                               ctx.get("project_root", ""))


async def _dispatch(name: str, args: dict, ctx: dict):
    if name == "read_file":
        return await _do_read(args, ctx)
    if name == "replace_lines":
        return _do_replace(args, ctx)
    if name == "find_refs":
        return await _do_refs(args, ctx)
    if name == "find_callers":
        return await _do_callers(args, ctx)
    if name == "search_text":
        return await _do_search(args, ctx)
    if name == "file_purpose":
        return _do_purpose(args, ctx)
    if name == "semantic_search":
        return _do_semantic(args, ctx)
    if name == "symbol_detail":
        return _do_detail(args, ctx)
    if name == "finish":
        return ("__FINISH__", args.get("summary", ""))
    return (f"✗ Unknown tool '{name}'. Available: read_file, find_refs, find_callers, "
            f"search_text, file_purpose, semantic_search, symbol_detail, replace_lines, finish.")


# ── The native tool-use loop ─────────────────────────────────────────────────
async def _call_tools_with_retry(model_id, messages, tools, max_tokens):
    """Call the model's tool API with a few retries on transient 429/5xx
    (OR :free rate-limits). Raises if it can't get a response."""
    from clients.nvidia import call_nvidia_tools
    last = None
    for attempt in range(4):
        try:
            return await call_nvidia_tools(model_id, messages, tools, max_tokens=max_tokens)
        except Exception as e:
            last = e
            s = str(e)
            transient = any(c in s for c in ("429", "502", "503", "504", "overloaded", "rate"))
            if not transient or attempt == 3:
                raise
            wait = 3 * (attempt + 1)
            warn(f"  [native:{model_id.split('/')[-1]}] {s[:80]} — retry {attempt+1}/3 in {wait}s")
            await asyncio.sleep(wait)
    raise last


async def call_with_native_tools(model_id: str, system: str, user_content: str,
                                 ctx: dict, max_rounds: int = 16,
                                 max_tokens: int = 8192) -> dict:
    """Run a structured tool-use coding loop. `ctx` carries the mutable state the
    tools act on: {file_contents, sandbox, project_root, viewed_versions,
    purpose_map, detailed_map}. Edits are applied to ctx['file_contents'] + the
    sandbox in place. Returns {answer, done, files_changed, rounds}."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_content}]
    ctx.setdefault("files_changed", set())
    done = False
    final = ""
    rnd = 0
    for rnd in range(1, max_rounds + 1):
        try:
            msg = await _call_tools_with_retry(model_id, messages, CODER_TOOLS, max_tokens)
        except Exception as e:
            warn(f"  [native] giving up after error: {str(e)[:120]}")
            break
        # The assistant message that issued tool_calls MUST precede the tool
        # results in history, with its tool_calls intact.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {}),
        })
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = msg.get("content") or ""
            status(f"  [native:{model_id.split('/')[-1]}] round {rnd}: no tool call — finishing")
            break
        n_edit = sum(1 for tc in tcs if tc.get("function", {}).get("name") == "replace_lines")
        names = ",".join(tc.get("function", {}).get("name", "?") for tc in tcs)
        status(f"  [native:{model_id.split('/')[-1]}] round {rnd}: {len(tcs)} tool call(s) [{names}]"
               + (f", {n_edit} edit(s)" if n_edit else ""))
        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            out = await _dispatch(name, args, ctx)
            if isinstance(out, tuple) and out and out[0] == "__FINISH__":
                done = True
                final = out[1] or "done"
                result_str = "Task marked finished."
            else:
                result_str = str(out)
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": result_str})
        if done:
            break
    return {"answer": final, "done": done,
            "files_changed": sorted(ctx.get("files_changed", set())), "rounds": rnd}
