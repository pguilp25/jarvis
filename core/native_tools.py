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
        "name": "create_file",
        "description": (
            "Create a NEW file at `path` with `content` (the full file text, exact "
            "indentation, NO line-number prefixes). Use this for files that don't "
            "exist yet — a new module, script, or test file (greenfield builds, or "
            "adding a file to an existing project). To change a file that ALREADY "
            "exists, use replace_lines instead — create_file refuses to clobber."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative path of the new file"},
            "content": {"type": "string", "description": "the full contents of the new file"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "replace_lines",
        "description": (
            "Your EDIT. Replace lines start_line..end_line (inclusive, 1-based, from "
            "your most recent read_file) of `path` with new_content. Each line of "
            "new_content should be `INDENT|code` — INDENT is the leading-space count "
            "(copy it from the read_file view; for a new line, count the spaces it "
            "needs). A BLANK line has indent 0 — write it as `0|`, never the "
            "surrounding indent (an indent count on an empty line just adds trailing "
            "whitespace). NO LINENO prefix on new content. Raw real indentation also works. "
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
    if s is not None and e is not None:
        try:
            arg = f"{path} {int(s)}-{int(e)}"
        except (TypeError, ValueError):
            return (f"✗ read_file: start_line/end_line must be integers "
                    f"(got start_line={s!r}, end_line={e!r}).")
    else:
        arg = path
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
    try:
        s_i, e_i = int(s), int(e)
    except (TypeError, ValueError):
        return (f"✗ replace_lines: start_line and end_line must be integers "
                f"(got start_line={s!r}, end_line={e!r}).")
    before = ctx["file_contents"].get(path)
    _before_all = dict(ctx["file_contents"])   # to catch a suffix-resolved key (review #2)
    block = (f"=== EDIT: {path} ===\n[REPLACE LINES {s_i}-{e_i}]\n"
             f"{new}\n[/REPLACE]\n=== END EDIT ===")
    ext = _extract_code_blocks(block)
    result, matched, attempted, skips = _apply_extracted_code(
        ext, ctx["file_contents"], ctx.get("sandbox"),
        viewed_versions=ctx.get("viewed_versions"))
    # malformed-range messages live on the extracted dict
    skips = list(ext.get("malformed_edits", [])) + list(skips)
    if path in result:
        # No-op guard: a byte-identical replace is not a real edit. Reporting it
        # as "✓ Applied" would pollute files_changed and let a coder that changed
        # nothing think it succeeded. (The text coder has this; native must too.)
        if before is not None and result[path] == before:
            return (f"✗ replace_lines was a NO-OP on {path}: new_content is "
                    f"byte-identical to lines {s_i}-{e_i}. Nothing changed — if you "
                    f"intended a change, re-check new_content; if the file is already "
                    f"correct, call finish.")
        sb = ctx.get("sandbox")
        if sb is not None:
            try:
                sb.write_file(path, result[path])
            except Exception as ex:
                return (f"✗ replace_lines: edit computed but FAILED to write the "
                        f"sandbox for {path} ({str(ex)[:120]}). The change did not "
                        f"persist; retry.")
        ctx["file_contents"][path] = result[path]
        # The edit shifted line numbers; the old read snapshot is now stale.
        if isinstance(ctx.get("viewed_versions"), dict):
            ctx["viewed_versions"][path] = result[path]
        ctx.setdefault("files_changed", set()).add(path)
        n = result[path].count("\n") + 1
        return (f"✓ Applied: {path} lines {s_i}-{e_i} replaced. File is now {n} lines. "
                f"Re-read with read_file if you need the new numbering before another edit.")
    # Safety net: the coder may spell `path` differently from a known file, and
    # _match_fp can suffix-resolve it to another key — mutating file_contents
    # WITHOUT the write above (path not in result) → silent sandbox divergence.
    # Sync any key that actually changed so disk and memory never disagree. (review #2)
    _changed = {k: v for k, v in result.items()
                if k != path and v != _before_all.get(k)}
    if _changed:
        for k, v in _changed.items():
            if ctx.get("sandbox") is not None:
                try:
                    ctx["sandbox"].write_file(k, v)
                except Exception:
                    pass
            ctx["file_contents"][k] = v
            if isinstance(ctx.get("viewed_versions"), dict):
                ctx["viewed_versions"][k] = v
            ctx.setdefault("files_changed", set()).add(k)
        _keys = ", ".join(_changed)
        return (f"✓ Applied (your path '{path}' resolved to {_keys}). Use that exact "
                f"path for further edits so line numbers anchor cleanly.")
    reason = " | ".join(str(x).strip().lstrip("-").strip() for x in skips) or \
        "no change produced (range may be invalid)"
    return f"✗ NOT applied to {path}: {reason}"


def _do_create(args: dict, ctx: dict) -> str:
    # Reuse the `=== FILE: path ===` new-file machinery (same applier the text
    # coder uses for new files) so greenfield / new-module work is possible —
    # replace_lines can only edit existing lines.
    from workflows.code import _extract_code_blocks, _apply_extracted_code
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "✗ create_file needs a path."
    existing = ctx["file_contents"].get(path)
    if existing is None and ctx.get("sandbox") is not None:
        existing = ctx["sandbox"].load_file(path)
    if existing:
        n = existing.count("\n") + 1
        return (f"✗ create_file: {path} already exists ({n} lines). Use replace_lines "
                f"to modify an existing file, or read_file to see it first.")
    block = f"=== FILE: {path} ===\n{content}\n=== END FILE ==="
    ext = _extract_code_blocks(block)
    result, matched, attempted, skips = _apply_extracted_code(
        ext, ctx["file_contents"], ctx.get("sandbox"),
        viewed_versions=ctx.get("viewed_versions"))
    produced = result.get(path) if path in result else ext.get("new_files", {}).get(path)
    if produced is not None:
        if ctx.get("sandbox") is not None:
            try:
                ctx["sandbox"].write_file(path, produced)
            except Exception as ex:
                return (f"✗ create_file: computed {path} but FAILED to write the "
                        f"sandbox ({str(ex)[:120]}); retry.")
        ctx["file_contents"][path] = produced
        if isinstance(ctx.get("viewed_versions"), dict):
            ctx["viewed_versions"][path] = produced
        ctx.setdefault("files_changed", set()).add(path)
        n = produced.count("\n") + 1
        return f"✓ Created: {path} ({n} lines)."
    reason = " | ".join(str(x).strip().lstrip("-").strip() for x in skips) or \
        "no file produced (content may be empty or malformed)"
    return f"✗ create_file NOT applied for {path}: {reason}"


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
    if name == "create_file":
        return _do_create(args, ctx)
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
            f"search_text, file_purpose, semantic_search, symbol_detail, create_file, "
            f"replace_lines, finish.")


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
            s = str(e).lower()
            tname = type(e).__name__.lower()
            # Treat gateway 5xx / 429 / overload AND network blips (timeouts,
            # connection resets) as transient — a dropped socket should retry,
            # not end the coder's step. (Audit #5/#6.)
            transient = (
                isinstance(e, asyncio.TimeoutError)
                or any(c in s for c in ("429", "502", "503", "504", "overloaded",
                                        "rate limit", "rate-limit", "timed out",
                                        "timeout", "connection", "temporarily"))
                or any(c in tname for c in ("timeout", "clienterror", "connector",
                                            "serverdisconnected", "connectionreset"))
            )
            if not transient or attempt == 3:
                raise
            wait = 3 * (attempt + 1)
            warn(f"  [native:{model_id.split('/')[-1]}] {str(e)[:80]} — retry {attempt+1}/3 in {wait}s")
            await asyncio.sleep(wait)
    raise last


def _est_chars(messages) -> int:
    return sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
               for m in messages)


def _trim_history(messages: list, max_chars: int, model_id: str) -> list:
    """Bound message-history growth so a long loop never drifts into a silent
    HTTP-400 context overflow (audit #47/#7). Keeps system + user + the newest
    assistant/tool groups, evicting the oldest groups first. Pairing is preserved
    (an assistant-with-tool_calls and its tool results are dropped together) so
    the request stays API-valid."""
    if _est_chars(messages) <= max_chars or len(messages) <= 4:
        return messages
    head, rest = messages[:2], messages[2:]
    groups, i = [], 0
    while i < len(rest):
        grp = [rest[i]]
        j = i + 1
        while j < len(rest) and rest[j].get("role") == "tool":
            grp.append(rest[j]); j += 1
        groups.append(grp); i = j
    dropped = 0
    while len(groups) > 2 and _est_chars(head + [m for g in groups for m in g]) > max_chars:
        groups.pop(0); dropped += 1
    if dropped:
        warn(f"  [native:{model_id.split('/')[-1]}] context near cap — dropped "
             f"{dropped} old tool round(s) to avoid overflow")
    return head + [m for g in groups for m in g]


# Forced self-check before the native coder is allowed to finish. The native
# coder repeatedly got the APPROACH and FILE right but botched a DETAIL and then
# exited without ever re-checking (django-14053: correct dict-collect idea, wrong
# yield ORDER → failed; matplotlib: used self._mapping before __init__ set it).
# The text coder has a SCENARIO TRACE self-check; the native path skipped it.
# This injects one trace pass before finish. Fired AT MOST once per step.
_VERIFY_NUDGE = (
    "⚠ SELF-CHECK before you finish — you edited: {files}.\n"
    "FIRST, plan-adherence: re-read the STEP and confirm your edit does EXACTLY "
    "what it says. If the step treats two groups DIFFERENTLY (e.g. 'yield group A "
    "immediately, COLLECT group B and yield it at the end'), verify your code "
    "actually BRANCHES that way — don't collapse them into one path (collecting "
    "everything, or yielding everything) just because it's simpler. A simpler "
    "shape that drops the step's distinction is WRONG.\n"
    "THEN trace the step's requirement (the failing scenario / expected behaviour) "
    "through your code line by line, checking the easy-to-miss details:\n"
    "  • execution ORDER — does each statement run when it should? (yield/return "
    "placement, a pass that overwrites an earlier one)\n"
    "  • a name used BEFORE it's assigned (e.g. an attribute __init__ would set "
    "but isn't set on this path)\n"
    "  • off-by-one / wrong boundary / wrong comparison\n"
    "  • the exact TYPE or shape returned, and every case the requirement names\n"
    "If you find a CONCRETE problem, fix it with replace_lines now. If the code is "
    "correct as written, call finish — do NOT change it just to change something."
)


async def call_with_native_tools(model_id: str, system: str, user_content: str,
                                 ctx: dict, max_rounds: int = 16,
                                 max_tokens: int = 8192,
                                 max_history_chars: int = 400_000) -> dict:
    """Run a structured tool-use coding loop. `ctx` carries the mutable state the
    tools act on: {file_contents, sandbox, project_root, viewed_versions,
    purpose_map, detailed_map}. Edits are applied to ctx['file_contents'] + the
    sandbox in place. Returns {answer, done, files_changed, rounds, reason} where
    reason ∈ {finished, no-tool-call, empty-turn, budget-exhausted, api-error}."""
    short = model_id.split('/')[-1]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_content}]
    ctx.setdefault("files_changed", set())
    done = False
    final = ""
    reason = "budget-exhausted"
    rnd = 0
    _fail_counts: dict = {}   # (tool, raw_args) → consecutive-reject count (audit #46)
    _verify_nudged = False    # one forced self-check before finishing (per step)
    _stuck = False            # hard-stop flag: same edit rejected ≥3× → fall over

    def _verify_nudge_msg():
        return {"role": "user",
                "content": _VERIFY_NUDGE.format(
                    files=", ".join(sorted(ctx.get("files_changed", set()))) or "(none)")}
    for rnd in range(1, max_rounds + 1):
        messages = _trim_history(messages, max_history_chars, model_id)
        try:
            msg = await _call_tools_with_retry(model_id, messages, CODER_TOOLS, max_tokens)
        except Exception as e:
            warn(f"  [native:{short}] giving up after error: {str(e)[:120]}")
            reason = "api-error"
            break
        if not isinstance(msg, dict):
            warn(f"  [native:{short}] non-dict model message — stopping")
            reason = "api-error"
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
            # Stopped WITH edits but never verified → force one self-check pass
            # before accepting the stop (catches the detail-level bugs).
            if ctx.get("files_changed") and not _verify_nudged:
                _verify_nudged = True
                messages.append(_verify_nudge_msg())
                status(f"  [native:{short}] round {rnd}: stopped with edits — "
                       f"one self-check pass before finishing")
                continue
            # An empty assistant turn (no tool calls, no content) is a STALL, not
            # a finish — distinguish so the workflow can tell "model did nothing"
            # from a deliberate stop. (Audit #11/#45.)
            reason = "no-tool-call" if final.strip() else "empty-turn"
            status(f"  [native:{short}] round {rnd}: no tool call ({reason})")
            break
        n_edit = sum(1 for tc in tcs if tc.get("function", {}).get("name") == "replace_lines")
        names = ",".join(tc.get("function", {}).get("name", "?") for tc in tcs)
        status(f"  [native:{short}] round {rnd}: {len(tcs)} tool call(s) [{names}]"
               + (f", {n_edit} edit(s)" if n_edit else ""))
        for tc in tcs:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except Exception:
                args = None
            if not isinstance(args, dict):
                # Malformed / non-object arguments — tell the coder exactly that
                # instead of the misleading "needs a path". (Audit #15/#16.)
                result_str = (f"✗ {name or 'tool'}: arguments were not a valid JSON "
                              f"object — re-emit the call with a proper object "
                              f"(got: {str(raw_args)[:80]}).")
            else:
                # NEVER let a tool executor exception kill the run. Any raise
                # becomes a role-coherent ✗ the coder can react to. (Audit #1/#22.)
                try:
                    out = await _dispatch(name, args, ctx)
                except Exception as e:
                    out = (f"✗ {name or 'tool'} failed internally: {str(e)[:160]} — "
                           f"try a different tool or a narrower input.")
                if isinstance(out, tuple) and out and out[0] == "__FINISH__":
                    done = True
                    final = out[1] or "done"
                    reason = "finished"
                    result_str = "Task marked finished."
                else:
                    result_str = str(out)
            # Loop-breaker: if the SAME tool call keeps getting rejected, escalate
            # so the coder changes approach instead of spinning the same failing
            # edit until the budget runs out. (Audit #46.)
            if isinstance(result_str, str) and result_str.startswith("✗"):
                _sig = (name, raw_args)
                _fail_counts[_sig] = _fail_counts.get(_sig, 0) + 1
                if _fail_counts[_sig] == 2:
                    result_str += (f"\n⚠ You have now sent this EXACT {name or 'tool'} "
                                   f"call {_fail_counts[_sig]}× and it was rejected each "
                                   f"time. STOP repeating it — change the arguments, "
                                   f"read_file to get the CURRENT line numbers, try a "
                                   f"different approach, or call finish if the file is "
                                   f"already correct.")
                # HARD STOP: a model that re-sends the SAME rejected call ≥3× is
                # stuck (pylint-4551: 13 identical rejected edits burned the whole
                # budget). Break out so the workflow falls over to another coder
                # instead of spinning. (Strengthens the audit-#46 nag.)
                if _fail_counts[_sig] >= 3:
                    warn(f"  [native:{short}] same {name or 'tool'} call rejected "
                         f"{_fail_counts[_sig]}× — stuck; stopping for fallover")
                    _stuck = True
            else:
                _fail_counts.pop((name, raw_args), None)   # success clears the streak
            messages.append({"role": "tool", "tool_call_id": tc.get("id", "") if isinstance(tc, dict) else "",
                             "content": result_str})
        if _stuck:
            reason = "stuck-repeating"
            break
        if done:
            # Force ONE self-check pass before accepting finish, if the coder
            # made edits and hasn't verified yet. It may fix a detail bug, or
            # re-finish unchanged. (The native analog of the text SCENARIO TRACE.)
            if ctx.get("files_changed") and not _verify_nudged:
                _verify_nudged = True
                done = False
                messages.append(_verify_nudge_msg())
                status(f"  [native:{short}] round {rnd}: finish requested — "
                       f"one self-check pass first")
                continue
            break
    files = sorted(ctx.get("files_changed", set()))
    # Make a step that produced NOTHING visible (audit #44/#48): a clean finish
    # with no edits, a stall, or a budget blow-out should never look like success.
    if reason == "budget-exhausted":
        warn(f"  [native:{short}] hit the {max_rounds}-round budget without finishing "
             f"— step may be incomplete ({len(files)} file(s) edited).")
    elif reason in ("empty-turn", "no-tool-call") and not files:
        warn(f"  [native:{short}] stopped ({reason}) with ZERO edits — step produced nothing.")
    elif done and not files:
        warn(f"  [native:{short}] called finish but made ZERO edits — step produced nothing.")
    return {"answer": final, "done": done, "files_changed": files,
            "rounds": rnd, "reason": reason}
