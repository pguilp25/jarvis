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
    [PURPOSE:]          file_purpose       _run_purpose_lookups (file gist)
    [SEMANTIC:]         semantic_search    embeddings.semantic_retrieve (over code)
    [DEPENDSON:]        depends_on         exploration_tools.extract_dependencies
    [REPLACE LINES]     replace_lines      _extract/_apply_extracted_code
    [DONE]              finish             —

(The coder has no [RUN:] — that's planner/reviewer-only, so neither does this.)

INDENT FORMAT (ckpt 88): read_file renders each line as `LINENO: <code with REAL
leading spaces>` and edit_file's old/new are real code lines — ONE format across read
and write, so getting indentation right is a COPY job, not a count→spaces conversion.
(Earlier the read used a `LINENO:INDENT|count` view aligned with replace_lines' `8|x`
write form; edit_file copies verbatim, so the whole native path moved to real
whitespace. The applier still accepts `N|` counts too, for back-compat.)

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
import os
import re

from core.cli import status, warn

# Models built for native function-calling — use the structured loop, not text.
NATIVE_TOOL_MODELS = {"nvidia/gpt-oss-120b", "nvidia/gpt-oss-nim", "groq/gpt-oss-120b"}

# WHITESPACE read view (default ON for the native coder; JARVIS_NATIVE_WS=0 to
# A/B-disable). The READ format must MATCH the WRITE format: edit_file is the
# primary edit tool and it COPIES real code lines (old/new are verbatim lines
# with real leading spaces), so read_file shows REAL indentation — the model
# copies the exact spaces it sees instead of converting a `LINENO:INDENT|count`
# into spaces in its head (that count→spaces conversion was a top source of
# IndentationErrors: f327 had 6/16 edit rejects from bad indent). The earlier
# count format was aligned with replace_lines (which WRITES `8|code`); under
# edit_file that alignment flipped. replace_lines still works (its applier
# accepts raw whitespace too), it's just no longer the primary path.
_WS_MODE = os.environ.get("JARVIS_NATIVE_WS", "1") != "0"


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
            "Embedding search over the CODE itself (functions/classes) — returns the "
            "top matching file:line units. Use when you know WHAT behaviour you want "
            "but not WHERE it lives. Not a substitute for search_text on an exact symbol."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "concept in plain words"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "depends_on",
        "description": (
            "What a symbol depends ON: the project functions/classes it calls or uses, "
            "with their definition sites (builtins/stdlib excluded). The reverse of "
            "dependents — use to learn what a function relies on before changing it."),
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
            "exists, use edit_file instead — create_file refuses to clobber."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative path of the new file"},
            "content": {"type": "string", "description": "the full contents of the new file"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Your PRIMARY edit tool. Change an existing file by giving `hunks` — a list "
            "of {\"start_line\": N, \"old\": [...], \"new\": [...]} objects. `old` is the "
            "EXACT existing line(s), copied VERBATIM from read_file (real indentation, no "
            "LINENO/INDENT prefixes). `start_line` is the read_file line number where "
            "`old` begins. `new` is what those lines become. The match is by CONTENT "
            "(verified against the file) with `start_line` to pin WHICH occurrence — so "
            "a stale number can't misfire AND a line that repeats isn't ambiguous. "
            "Copying the real lines keeps you focused on exactly what you change.\n"
            "  • CHANGE lines: old=[current lines], new=[replacements].\n"
            "  • INSERT new code: give start_line = the line to add AFTER, new=[your new "
            "line(s)], and leave old empty.\n"
            "  • DELETE: old=[the lines to remove], new=[] (empty).\n"
            "Multiple hunks edit several spots in one call. Parse-checked; a rejection "
            "says exactly what to fix."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative path to edit"},
            "hunks": {
                "type": "array",
                "description": "one or more content-anchored edits",
                "items": {"type": "object", "properties": {
                    "start_line": {"type": "integer",
                                   "description": "read_file line number where `old` begins (pins which occurrence)"},
                    "old": {"type": "array", "items": {"type": "string"},
                            "description": "exact existing line(s), copied verbatim from read_file"},
                    "new": {"type": "array", "items": {"type": "string"},
                            "description": "replacement line(s); empty array to delete"},
                }, "required": ["start_line", "old", "new"]},
            },
        }, "required": ["path", "hunks"]},
    }},
    {"type": "function", "function": {
        "name": "replace_lines",
        "description": (
            "SECONDARY edit tool — prefer edit_file. Use ONLY for a clean whole-range "
            "swap where copying the old lines is pointless. Replace lines "
            "start_line..end_line (inclusive, 1-based, from your most recent read_file) "
            "of `path` with new_content. Write new_content with REAL leading spaces, "
            "exactly like the read_file view shows them (NO `LINENO:` prefix, NO `N|` "
            "count). To INSERT after line N, set start_line=end_line=N and make "
            "new_content the current line N followed by your new line(s). Line numbers "
            "go stale after any edit — that's why edit_file (content-matched) is safer."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "new_content": {"type": "string"},
        }, "required": ["path", "start_line", "end_line", "new_content"]},
    }},
    {"type": "function", "function": {
        "name": "run_code",
        "description": (
            "RUN a shell command against your EDITED code and see what it actually does "
            "— the only way to catch a bug that READS fine but BEHAVES wrong (valid code, "
            "wrong logic). Runs in your sandbox (your edits are live) with the repo's deps "
            "+ pytest available; read-only + no network. Use it to VERIFY before finish: "
            "write a tiny check that exercises your change and asserts the expected "
            "behaviour — e.g. python -c \"from pkg.mod import Thing; t=Thing(); "
            "assert t.method(...) == expected\" — or run the module's existing tests: "
            "python -m pytest path/to/test_file.py -q. If exit≠0, the OUTPUT is your "
            "edit's real behaviour; fix it and re-run. Don't finish on an unverified edit."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "shell command, e.g. python -c \"...\" or python -m pytest <path> -q"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": (
            "Call ONLY when the edit is complete and you've verified it does what the "
            "step asked — ideally by run_code, not just by reading. Ends the task."),
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
    if (s is None) != (e is None):
        # Exactly one bound given → the model asked for a region but the bound it
        # dropped would be silently ignored (whole file returned). Tell it.
        return ("✗ read_file: give BOTH start_line and end_line for a range, or "
                "NEITHER to read the whole file (you provided only "
                f"{'start_line' if s is not None else 'end_line'}).")
    if s is not None and e is not None:
        try:
            s_i, e_i = int(s), int(e)
        except (TypeError, ValueError):
            return (f"✗ read_file: start_line/end_line must be integers "
                    f"(got start_line={s!r}, end_line={e!r}).")
        # Validate the range BEFORE delegating. _run_code_reads renders an
        # inverted / out-of-bounds / negative range as a header with an EMPTY
        # body (or, for negatives, a misleading "FILE NOT FOUND") — both leave
        # the coder with nothing and a hallucination risk. Tell it precisely
        # what's wrong and how to fix it, the same way the text loop would.
        if s_i < 1 or e_i < 1:
            return (f"✗ read_file: line numbers must be ≥ 1 (got start_line={s_i}, "
                    f"end_line={e_i}). Re-issue with a positive 1-based range, or "
                    f"omit start_line/end_line to read the whole file.")
        if e_i < s_i:
            return (f"✗ read_file: invalid range — start_line ({s_i}) must be ≤ "
                    f"end_line ({e_i}). Put the smaller line number first.")
        # Out-of-bounds start (beyond EOF) — name the file's real length.
        _base = ctx.get("file_contents", {}).get(path)
        if _base is None:
            sb0 = ctx.get("sandbox")
            if sb0 is not None:
                _base = sb0.load_file(path)
        if _base is not None:
            _total = _base.count("\n") + 1
            if s_i > _total:
                return (f"✗ read_file: start_line {s_i} is out of range — {path} has "
                        f"only {_total} line(s). Read within 1-{_total}, or omit the "
                        f"range to read the whole file.")
        arg = f"{path} {s_i}-{e_i}"
    else:
        arg = path
    try:
        out = await _run_code_reads(
            [arg], ctx.get("project_root", ""),
            viewed_versions=ctx.get("viewed_versions"),
            display_mode="whitespace" if _WS_MODE else "prefix")
    except Exception as ex:
        out = f"✗ read_file failed: {str(ex)[:160]}"
    # Keep file_contents (the replace_lines base) in step with the sandbox.
    sb = ctx.get("sandbox")
    if sb is not None:
        cur = sb.load_file(path)
        if cur is not None:
            ctx["file_contents"][path] = cur
    return out


def _post_edit_syntax_gate(path: str, new_content: str, before, *,
                           tool: str, resend: str) -> "str | None":
    """Shared parse / dead-code / dup gate for native edits. Returns a REJECTION
    string (the edit must NOT be written) or None (safe to write). Only flags a
    problem THIS edit introduced — a pre-existing breakage passes through so the
    coder isn't sent chasing an unrelated error. `resend` is the tool-specific
    "try again" hint appended to each rejection."""
    if not path.endswith(".py"):
        return None
    from workflows.code import (_check_syntax, _unreachable_after_jump,
                                _duplicate_adjacent_stmts)
    ok_after, _serr = _check_syntax(path, new_content)
    if not ok_after and (before is None or _check_syntax(path, before)[0]):
        return (f"✗ {tool} NOT applied to {path}: your change makes the file fail to "
                f"parse, so it was NOT written (the file is unchanged). Fix the error "
                f"below and {resend}.\n{_serr}")
    if ok_after:
        new_dead = _unreachable_after_jump(new_content)
        old_dead = _unreachable_after_jump(before) if before else {}
        if len(new_dead) > len(old_dead):
            where = "; ".join(f"line {ln}: `{txt}`"
                              for ln, txt in sorted(new_dead.items())[:3])
            return (f"✗ {tool} NOT applied to {path}: your edit leaves UNREACHABLE "
                    f"code — {where} comes right after a return/raise at the same "
                    f"indent, so it never runs (the file is unchanged). If that logic "
                    f"should run on the success path, DEDENT it OUT of the guard "
                    f"block, then {resend}.")
        new_dup = _duplicate_adjacent_stmts(new_content)
        old_dup = _duplicate_adjacent_stmts(before) if before else {}
        if len(new_dup) > len(old_dup):
            where = "; ".join(f"line {ln}: `{txt}`"
                              for ln, txt in sorted(new_dup.items())[:3])
            return (f"✗ {tool} NOT applied to {path}: your edit creates DUPLICATE "
                    f"adjacent code — {where} repeats the statement right before it "
                    f"(the file is unchanged). You likely re-emitted an anchor block "
                    f"AND your new copy. Keep ONE; {resend}.")
    return None


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
    # Validate the range up front. The text [REPLACE LINES] regex only matches
    # `\d+`, so a NEGATIVE start (e.g. -1) never reaches the applier's range
    # check — it falls through to the vague "no change produced (range may be
    # invalid)". Catch it here with a precise, actionable message. (An inverted
    # range IS caught downstream, but the message comes back duplicated; we
    # de-dup it below.)
    if s_i < 1 or e_i < 1:
        return (f"✗ replace_lines: invalid range — line numbers must be positive "
                f"(1 ≤ start ≤ end); got start_line={s_i}, end_line={e_i}. Use the "
                f"1-based line numbers from your most recent read_file.")
    if e_i < s_i:
        return (f"✗ replace_lines: invalid range — start_line ({s_i}) must be ≤ "
                f"end_line ({e_i}). Put the smaller line number first (use the "
                f"numbers from your most recent read_file).")
    before = ctx["file_contents"].get(path)
    _before_all = dict(ctx["file_contents"])   # to catch a suffix-resolved key (review #2)
    block = (f"=== EDIT: {path} ===\n[REPLACE LINES {s_i}-{e_i}]\n"
             f"{new}\n[/REPLACE]\n=== END EDIT ===")
    ext = _extract_code_blocks(block)
    result, matched, attempted, skips = _apply_extracted_code(
        ext, ctx["file_contents"], ctx.get("sandbox"),
        viewed_versions=ctx.get("viewed_versions"))
    # malformed-range messages live on the extracted dict. The same message can
    # appear in BOTH malformed_edits and skips (e.g. an inverted range) — merge
    # while preserving order and dropping exact duplicates so the coder sees the
    # reason ONCE, not "...invalid range... | ...invalid range...".
    _seen: set = set()
    skips = [x for x in (list(ext.get("malformed_edits", [])) + list(skips))
             if not (str(x).strip() in _seen or _seen.add(str(x).strip()))]
    if path in result:
        # No-op guard: a byte-identical replace is not a real edit. Reporting it
        # as "✓ Applied" would pollute files_changed and let a coder that changed
        # nothing think it succeeded. (The text coder has this; native must too.)
        if before is not None and result[path] == before:
            return (f"✗ replace_lines was a NO-OP on {path}: new_content is "
                    f"byte-identical to lines {s_i}-{e_i}. Nothing changed — if you "
                    f"intended a change, re-check new_content; if the file is already "
                    f"correct, call finish.")
        # SYNTAX GATE — parity with the text coder's parse gate (code.py:11374).
        # A native edit that makes a previously-parseable .py file un-importable
        # must NOT ship silently (that's how an IndentationError reached a final
        # patch). Reject WITHOUT writing or mutating state so the loop re-targets
        # the same (now-unchanged) lines and retries. Only block errors THIS edit
        # introduced — if the file was already broken, let it through rather than
        # send the coder chasing a pre-existing error in unrelated code.
        _gate = _post_edit_syntax_gate(
            path, result[path], before, tool="replace_lines",
            resend=f"re-send the SAME line range {s_i}-{e_i}")
        if _gate:
            return _gate
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


def _locate_block(cur_lines: list, old_list: list) -> "tuple[int | None, int]":
    """Find where the consecutive `old_list` lines appear in cur_lines, matching
    by STRIPPED content (whitespace/prefix-insensitive). Returns (1-based start
    line of the FIRST match, total match count)."""
    stripped = [str(o).strip() for o in old_list]
    n = len(stripped)
    if n == 0:
        return None, 0
    hits = []
    for idx in range(len(cur_lines) - n + 1):
        if [cur_lines[idx + k].strip() for k in range(n)] == stripped:
            hits.append(idx + 1)
    return (hits[0] if hits else None), len(hits)


def _old_not_found_msg(i: int, path: str, ctx: dict) -> str:
    """`old` matched nowhere in the file. Two very different causes — name the
    likely one. If we've already edited this file, the model is almost certainly
    copying `old` from a STALE earlier read (the file moved under it); telling it
    'wrong file' would wrongly send it away from the right file. If we HAVEN'T
    touched it, the symbol probably lives elsewhere (the f327e65d wrong-file bug)."""
    if path in ctx.get("files_changed", set()):
        return (f"✗ edit_file hunk #{i}: the `old` line(s) aren't in {path} — but you've "
                f"already EDITED {path}, so your `old` is STALE (copied from a read taken "
                f"BEFORE that edit). read_file {path} AGAIN and copy `old` from the CURRENT "
                f"view — do not reuse line text from an earlier read in this conversation.")
    return (f"✗ edit_file hunk #{i}: the `old` line(s) are NOT in {path}. Likely the WRONG "
            f"FILE — is the code you're changing actually defined here? [SEARCH] the symbol "
            f"to find the file it's defined in, then edit THAT. Otherwise re-read {path} and "
            f"copy `old` verbatim.")


def _do_edit(args: dict, ctx: dict) -> str:
    """CONTENT-ANCHORED edit (the primary edit tool), expressed as JSON `hunks` —
    each {start_line, old:[lines], new:[lines]}. `old` = the EXACT existing lines
    copied verbatim from read_file (the context that anchors + focuses); `new` =
    what they become. `start_line` = the read-view line number where `old` begins
    (disambiguates when `old` repeats; the content is still verified, so an
    approximate number self-corrects). We translate to the text coder's NUMBERED
    [edit] diff (number-first, content-verified) and apply via the same
    _extract/_apply path replace_lines uses — so a stale number can't misfire AND
    a non-unique match isn't ambiguous. DELETE: new=[]. INSERT: old=[line you add
    after], new=[that line, then the additions]."""
    path = args.get("path", "")
    hunks = args.get("hunks")
    if not path:
        return "✗ edit_file needs a path."
    if not hunks or not isinstance(hunks, list):
        return ("✗ edit_file needs a non-empty `hunks` array. Each hunk is "
                "{\"start_line\": N, \"old\": [exact existing line(s) from read_file], "
                "\"new\": [replacement line(s)]}.")

    cur = ctx["file_contents"].get(path)
    if cur is None:
        sb0 = ctx.get("sandbox")
        if sb0 is not None:
            cur = sb0.load_file(path)
    if cur is None:
        return (f"✗ edit_file: {path} is not in context — read it with read_file "
                f"first, then copy the exact `old` lines (and their start_line) from "
                f"that view.")
    cur_lines = cur.split("\n")

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return str(v).split("\n")

    resolved = []   # (start_line, old_list, new_list) per hunk, validated
    for i, h in enumerate(hunks, 1):
        if not isinstance(h, dict):
            return (f"✗ edit_file hunk #{i} is not an object — each hunk must be "
                    f"{{\"start_line\": N, \"old\": [...], \"new\": [...]}}.")
        old_list = _as_list(h.get("old"))
        new_list = _as_list(h.get("new"))
        if not any(o.strip() for o in old_list):
            # PURE INSERT ergonomics: the model naturally leaves `old` empty when
            # ADDING new code (a method/function) and gives start_line + new. Don't
            # reject — treat it as "insert `new` AFTER start_line" by anchoring on
            # that existing line (keep it, then add). (f327: 5× 'old is empty'.)
            sl_raw = h.get("start_line")
            try:
                sl_i = int(sl_raw)
            except (TypeError, ValueError):
                sl_i = 0
            if new_list and 1 <= sl_i <= len(cur_lines):
                anchor = cur_lines[sl_i - 1]
                old_list = [anchor]
                new_list = [anchor] + new_list   # keep the anchor, add new below it
            else:
                return (f"✗ edit_file hunk #{i}: `old` is empty. To CHANGE code, put the "
                        f"exact existing line(s) in `old`. To INSERT new code, give "
                        f"`start_line` = the line you want to add AFTER and put the new "
                        f"line(s) in `new` (leave `old` empty).")
        sl = h.get("start_line")
        if sl is None:
            # No number given — resolve from content; reject if it's not unique.
            sl, n_hits = _locate_block(cur_lines, old_list)
            if n_hits == 0:
                return _old_not_found_msg(i, path, ctx)
            if n_hits > 1:
                return (f"✗ edit_file hunk #{i}: `old` appears {n_hits} times in {path} "
                        f"— add `start_line` (the read_file line number where this "
                        f"occurrence begins) so I edit the right one, or include more "
                        f"surrounding lines in `old`.")
        else:
            try:
                sl = int(sl)
            except (TypeError, ValueError):
                return (f"✗ edit_file hunk #{i}: start_line must be an integer line "
                        f"number from read_file (got {sl!r}).")
            if sl < 1:
                return (f"✗ edit_file hunk #{i}: start_line must be ≥ 1 (got {sl}).")
            # Verify `old` actually exists in THIS file. If not, the model is most
            # likely editing the WRONG FILE (e.g. trying to change a class that
            # lives in another module) — say so plainly instead of the line-range
            # applier's "stale view? line N" wording, which sends it chasing
            # numbers. This is the f327e65d failure: editing AnsibleCollectionRef
            # in dataclasses.py when the class is in _collection_finder.py.
            _, _n = _locate_block(cur_lines, old_list)
            if _n == 0:
                return _old_not_found_msg(i, path, ctx)
        resolved.append((sl, old_list, new_list))

    # The numbered [edit] applier requires lines top-to-bottom in FILE order.
    # The model may send hunks in any order, so sort by start_line ourselves
    # (stable, so same-line hunks keep their given order) instead of rejecting
    # with "edit lines out of order" and making it retry — that retry-loop is a
    # top cause of round pile-up.
    resolved.sort(key=lambda t: t[0])
    edit_lines = []
    for sl, old_list, new_list in resolved:
        for j, o in enumerate(old_list):
            edit_lines.append(f"{sl + j}:-{o}")
        for nw in new_list:
            edit_lines.append(f"+{nw}")

    block = (f"=== EDIT: {path} ===\n[edit]\n" + "\n".join(edit_lines)
             + "\n[/edit]\n=== END EDIT ===")
    from workflows.code import _extract_code_blocks, _apply_extracted_code
    before = ctx["file_contents"].get(path)
    _before_all = dict(ctx["file_contents"])
    ext = _extract_code_blocks(block)
    result, matched, attempted, skips = _apply_extracted_code(
        ext, ctx["file_contents"], ctx.get("sandbox"),
        viewed_versions=ctx.get("viewed_versions"))
    _seen: set = set()
    skips = [x for x in (list(ext.get("malformed_edits", [])) + list(skips))
             if not (str(x).strip() in _seen or _seen.add(str(x).strip()))]

    if path in result:
        if before is not None and result[path] == before:
            return (f"✗ edit_file was a NO-OP on {path}: the `new` lines are identical "
                    f"to `old`. Nothing changed — re-check your hunk, or call finish if "
                    f"the file is already correct.")
        _gate = _post_edit_syntax_gate(
            path, result[path], before, tool="edit_file",
            resend="re-send the corrected hunk")
        if _gate:
            return _gate
        sb = ctx.get("sandbox")
        if sb is not None:
            try:
                sb.write_file(path, result[path])
            except Exception as ex:
                return (f"✗ edit_file: change computed but FAILED to write the sandbox "
                        f"for {path} ({str(ex)[:120]}). It did not persist; retry.")
        ctx["file_contents"][path] = result[path]
        if isinstance(ctx.get("viewed_versions"), dict):
            ctx["viewed_versions"][path] = result[path]
        ctx.setdefault("files_changed", set()).add(path)
        n = result[path].count("\n") + 1
        return (f"✓ Applied {len(hunks)} hunk(s) to {path}. File is now {n} lines. "
                f"Re-read with read_file before another edit if you need fresh context.")

    # Suffix-resolved key safety net (mirror _do_replace).
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
        return (f"✓ Applied (your path '{path}' resolved to {', '.join(_changed)}). "
                f"Use that exact path for further edits.")
    reason = " | ".join(str(x).strip().lstrip("-").strip() for x in skips) or \
        ("the `old` text wasn't found in the file — copy it VERBATIM from read_file "
         "(exact spaces/punctuation), and include enough lines to be unique")
    return f"✗ edit_file NOT applied to {path}: {reason}"


def _do_create(args: dict, ctx: dict) -> str:
    # Reuse the `=== FILE: path ===` new-file machinery (same applier the text
    # coder uses for new files) so greenfield / new-module work is possible —
    # replace_lines can only edit existing lines.
    from workflows.code import _extract_code_blocks, _apply_extracted_code
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "✗ create_file needs a path."
    if not str(content).strip():
        # Empty/whitespace content would write a 0-byte file and report "✓ Created
        # (1 lines)" — a silent no-op the model can't tell from real success.
        return (f"✗ create_file: no content for {path}. Pass the full file body in "
                f"`content` (the complete code for the new file).")
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
        # SYNTAX GATE — a new .py module that doesn't parse would ImportError the
        # moment anything (incl. the test) imports it. Reject before writing so the
        # coder re-sends a corrected body, rather than ship a dead file.
        if path.endswith(".py"):
            from workflows.code import (_check_syntax, _unreachable_after_jump,
                                        _duplicate_adjacent_stmts)
            ok_new, _serr = _check_syntax(path, produced)
            if not ok_new:
                return (f"✗ create_file NOT written: {path} fails to parse, so it was "
                        f"not created. Fix the error below and re-send the full file "
                        f"body in `content`.\n{_serr}")
            dead = _unreachable_after_jump(produced)
            if dead:
                where = "; ".join(f"line {ln}: `{txt}`"
                                  for ln, txt in sorted(dead.items())[:3])
                return (f"✗ create_file NOT written: {path} has UNREACHABLE code — "
                        f"{where} comes right after a return/raise at the same indent, "
                        f"so it never runs. Fix the indentation and re-send the full body.")
            dup = _duplicate_adjacent_stmts(produced)
            if dup:
                where = "; ".join(f"line {ln}: `{txt}`"
                                  for ln, txt in sorted(dup.items())[:3])
                return (f"✗ create_file NOT written: {path} has DUPLICATE adjacent code — "
                        f"{where} repeats the statement right before it. Keep ONE copy "
                        f"and re-send the full body.")
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


async def _do_semantic(args: dict, ctx: dict) -> str:
    # Embedding search over the CODE (same path as the text loop) — no purpose map.
    from tools.code_index import _maps_dir, _load_all_code
    from tools.embeddings import semantic_retrieve
    q = args.get("query", "")
    if not q:
        return "✗ semantic_search needs a query."
    project_root = ctx.get("project_root", "")
    if not project_root:
        return "✗ semantic_search needs a project_root."
    maps_dir = _maps_dir(project_root)
    try:
        _, file_hash = _load_all_code(project_root)
        out = await semantic_retrieve(q, project_root, maps_dir, file_hash, top_n=10)
    except Exception as ex:
        return (f"✗ semantic_search failed ({str(ex)[:120]}). Embeddings may be "
                f"unavailable — use search_text for an exact symbol/string, or "
                f"find_refs for a known name.")
    # semantic_retrieve signals trouble with a parenthetical, NOT a ✗ — so a
    # weak native coder can't tell it failed and isn't told the alternative.
    # Normalise: when embeddings are unavailable or there's nothing to search,
    # return a ✗ that names the fallback tools (parity with how the text loop
    # would flag a no-result lookup). A real hit list passes through unchanged.
    low = (out or "").lower()
    if not (out or "").strip():
        return ("✗ semantic_search returned nothing for that query. Try search_text "
                "for an exact symbol/string, or rephrase the concept.")
    if low.startswith("(semantic search unavailable") or low.startswith("(no code to search"):
        _detail = out.strip().strip("()")
        if _detail.lower().startswith("semantic search unavailable:"):
            _detail = _detail.split(":", 1)[1].strip()
        return (f"✗ semantic_search unavailable: {_detail}. "
                f"Fall back to search_text (exact text/regex) or find_refs (a known "
                f"symbol name) instead.")
    if "no " in low and ("match" in low or "result" in low) and len(out.strip()) < 80:
        return (f"✗ semantic_search: {out.strip()} — 0 matches. Try search_text with "
                f"an exact term, or rephrase the concept.")
    return out


def _do_dependson(args: dict, ctx: dict) -> str:
    from core.exploration_tools import extract_dependencies
    sym = args.get("symbol", "")
    if not sym:
        return "✗ depends_on needs a symbol name."
    return extract_dependencies(sym, ctx.get("project_root", ""))


def _do_run(args: dict, ctx: dict) -> str:
    """Run a shell command against the coder's EDITED sandbox and return the
    output — so the coder can OBSERVE its change's runtime behaviour instead of
    SIMULATING it in its head (the static gates prove a patch parses + names
    resolve; only running proves it DOES the right thing). cwd = the sandbox dir
    where edits land; the bwrap sandbox is read-only/no-net but binds the venv
    (the repo's deps + pytest), so `python -c …` / `python -m pytest …` work."""
    from core.safe_exec import run_sandboxed
    cmd = (args.get("command") or "").strip()
    if not cmd:
        return ("✗ run_code needs a `command` — e.g. python -c \"<a check that "
                "constructs the object, calls your changed code, and asserts the "
                "expected result>\", or python -m pytest <path::test> -q.")
    sb = ctx.get("sandbox")
    cwd = str(getattr(sb, "sandbox_dir", "") or "") if sb is not None else ""
    if not cwd:
        cwd = ctx.get("project_root", "")
    if not cwd:
        return "✗ run_code: no sandbox to run in."
    try:
        res = run_sandboxed(cmd, cwd=cwd, timeout=90, project_root=cwd)
    except Exception as e:
        return f"✗ run_code failed to launch: {str(e)[:160]}"
    if res.get("blocked"):
        return (f"✗ run_code blocked: {str(res.get('reason', ''))[:200]} "
                f"(read-only/no-net sandbox; write-ops and network are off).")
    code = res.get("exit_code", -1)
    out = (res.get("output") or "").strip()
    timed = " (TIMED OUT at 90s)" if res.get("timed_out") else ""
    head = (f"✓ ran in your edited sandbox — exit 0{timed}" if code == 0
            else f"✗ ran in your edited sandbox — exit {code}{timed} "
                 f"(your edit's behaviour, NOT a tool error — read the output)")
    return f"{head}\n{out[:4000] or '(no output)'}"


def _debug_edit_trace(tool: str, args: dict, result: str) -> None:
    """Env-gated (JARVIS_DEBUG_EDITS=<path>) trace of every edit call + its result —
    so we can see EXACTLY what new_content the coder emitted and why it was rejected.
    Inert unless the env var is set. (Diagnostic for the 'why the coder failed' audit.)"""
    import os
    path = os.environ.get("JARVIS_DEBUG_EDITS")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\nTOOL {tool}  args: start={args.get('start_line')} "
                    f"end={args.get('end_line')} path={args.get('path')}\n"
                    f"--- new_content ---\n{args.get('new_content', args.get('content',''))}\n"
                    f"--- RESULT ---\n{result}\n")
    except Exception:
        pass


async def _dispatch(name: str, args: dict, ctx: dict):
    if name == "read_file":
        return await _do_read(args, ctx)
    if name == "edit_file":
        res = _do_edit(args, ctx)
        _debug_edit_trace("edit_file", args, res)
        return res
    if name == "replace_lines":
        res = _do_replace(args, ctx)
        _debug_edit_trace("replace_lines", args, res)
        return res
    if name == "create_file":
        res = _do_create(args, ctx)
        _debug_edit_trace("create_file", args, res)
        return res
    if name == "find_refs":
        return await _do_refs(args, ctx)
    if name == "find_callers":
        return await _do_callers(args, ctx)
    if name == "search_text":
        return await _do_search(args, ctx)
    if name == "file_purpose":
        return _do_purpose(args, ctx)
    if name == "semantic_search":
        return await _do_semantic(args, ctx)
    if name == "depends_on":
        return _do_dependson(args, ctx)
    if name == "run_code":
        return await asyncio.get_event_loop().run_in_executor(None, _do_run, args, ctx)
    if name == "finish":
        return ("__FINISH__", args.get("summary", ""))
    # A weak/native model often reaches for a name from another idiom (the text
    # tags, or generic verbs). Name the LIKELY intended native tool first — same
    # courtesy the text loop gives ([READ]→[CODE], [GREP]→[SEARCH]) — then list
    # the full set, so the coder corrects in one step instead of guessing.
    _ALIAS = {
        "read": "read_file", "open": "read_file", "cat": "read_file",
        "get": "read_file", "view": "read_file", "code": "read_file",
        "keep": "read_file", "show": "read_file",
        "grep": "search_text", "search": "search_text", "find": "search_text",
        "rg": "search_text", "ls": "search_text", "list": "search_text",
        "glob": "search_text", "ripgrep": "search_text",
        "refs": "find_refs", "references": "find_refs", "usages": "find_refs",
        "callers": "find_callers", "dependency": "find_callers",
        "dependson": "depends_on", "depends": "depends_on",
        "purpose": "file_purpose", "summary": "file_purpose", "gist": "file_purpose",
        "semantic": "semantic_search",
        "write": "create_file", "new_file": "create_file", "touch": "create_file",
        "edit": "edit_file", "replace": "edit_file", "edit_lines": "edit_file",
        "modify": "edit_file", "patch": "edit_file", "apply": "edit_file",
        "search_replace": "edit_file", "str_replace": "edit_file",
        "run": "run_code", "run_test": "run_code", "run_tests": "run_code",
        "pytest": "run_code", "test": "run_code", "bash": "run_code",
        "shell": "run_code", "exec": "run_code", "python": "run_code",
        "done": "finish", "stop": "finish", "complete": "finish", "end": "finish",
    }
    suggestion = _ALIAS.get((name or "").strip().lower().lstrip("[").rstrip("]:"))
    hint = (f" Did you mean '{suggestion}'?" if suggestion else "")
    return (f"✗ Unknown tool '{name}'.{hint} Available: read_file, find_refs, "
            f"find_callers, search_text, file_purpose, semantic_search, depends_on, "
            f"edit_file, create_file, replace_lines, run_code, finish.")


# ── The native tool-use loop ─────────────────────────────────────────────────
# Coder chain (user 2026-05-29) places gpt-oss on each infra at a DISTINCT slot:
# nvidia/gpt-oss-120b = OpenRouter :free (slot 1, primary); nvidia/gpt-oss-nim =
# NVIDIA NIM (slot 4, after qwen+mistral). So each native gpt model pins ONE
# endpoint here — the chain ORDER is orchestrated in workflows/code.py, not by
# cycling endpoints inside one call. (Groq excluded: 8K free-tier throttle.)
_GPT_OSS_ENDPOINT = {"gpt-oss-120b": "openrouter", "gpt-oss-nim": "nvidia"}
_PERM = re.compile(r'HTTP\s*(?:400|401|403|404|410)\b', re.IGNORECASE)


def _is_transient(e) -> bool:
    """A transient error is worth retrying the SAME endpoint (rate-limit, gateway
    5xx, network blip). A permanent one (4xx auth/not-found/bad-request) is not —
    move to the next endpoint immediately. (user: 'retry the same ai when the
    error is not permanent.')"""
    s = str(e).lower(); tname = type(e).__name__.lower()
    if _PERM.search(str(e)):
        return False
    return (
        isinstance(e, asyncio.TimeoutError)
        or any(c in s for c in ("429", "500", "502", "503", "504", "overloaded",
                                "rate limit", "rate-limit", "timed out", "timeout",
                                "connection", "temporarily", "capacity", "provider returned"))
        or any(c in tname for c in ("timeout", "clienterror", "connector",
                                    "serverdisconnected", "connectionreset"))
    )


async def _call_tools_with_retry(model_id, messages, tools, max_tokens,
                                 per_provider_retries: int = 4):
    """Call the model's native tool API, cycling its gpt-oss endpoints. For each
    provider: retry the SAME endpoint on transient errors, skip to the next
    provider on a permanent error. Only raises once EVERY provider is exhausted —
    so the workflow switches to a different MODEL only after gpt-oss has had every
    endpoint. Non-gpt-oss models keep the single-endpoint behavior."""
    from clients.nvidia import call_nvidia_tools
    short = model_id.split('/')[-1]
    providers = [_GPT_OSS_ENDPOINT[short]] if short in _GPT_OSS_ENDPOINT else [""]
    last = None
    for pi, provider in enumerate(providers):
        for attempt in range(per_provider_retries):
            try:
                return await call_nvidia_tools(model_id, messages, tools,
                                               max_tokens=max_tokens,
                                               force_provider=provider)
            except Exception as e:
                last = e
                where = f"{short}@{provider or 'auto'}"
                if not _is_transient(e):
                    warn(f"  [native:{where}] permanent ({str(e)[:70]}) — "
                         + ("next endpoint" if pi < len(providers) - 1 else "out of endpoints"))
                    break  # permanent on this endpoint → try the next provider
                if attempt == per_provider_retries - 1:
                    warn(f"  [native:{where}] transient, retries spent — "
                         + ("next endpoint" if pi < len(providers) - 1 else "out of endpoints"))
                    break
                wait = 3 * (attempt + 1)
                warn(f"  [native:{where}] {str(e)[:70]} — retry {attempt+1}/{per_provider_retries} in {wait}s")
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
                                 ctx: dict, max_rounds: int = 40,
                                 max_tokens: int = 32000,
                                 max_history_chars: int = 400_000) -> dict:
    """Run a structured tool-use coding loop. `ctx` carries the mutable state the
    tools act on: {file_contents, sandbox, project_root, viewed_versions,
    purpose_map, detailed_map}. Edits are applied to ctx['file_contents'] + the
    sandbox in place. Returns {answer, done, files_changed, rounds, reason} where
    reason ∈ {finished, no-tool-call, empty-turn, budget-exhausted, api-error}."""
    short = model_id.split('/')[-1]
    if _WS_MODE:
        # Authoritative override of any `INDENT|` count instructions in the base
        # prompt. Appended LAST so it wins. read_file shows REAL indentation, which
        # matches how edit_file works (you COPY lines) — so indentation is a
        # copy job, not a count-to-spaces conversion.
        system = system + (
            "\n\n## INDENTATION — read_file shows REAL spaces (overrides any `N|` count rule above)\n"
            "Every read_file line is `LINENO: <the code with its ACTUAL leading spaces>` — "
            "the indentation you see IS the indentation. So in edit_file:\n"
            "  • `old` lines: copy them CHARACTER-FOR-CHARACTER from the read view, leading "
            "spaces included (drop only the `LINENO: ` prefix). Never re-type or re-indent "
            "them — paste exactly, so the content-match can't fail on whitespace.\n"
            "  • `new` lines: write the real leading spaces too. For a CHANGED line, keep the "
            "same indentation as the line it replaces. For a NEW line, indent it to line up "
            "with the surrounding code you can SEE (a block body sits one level — usually 4 "
            "spaces — deeper than its `def`/`if`/`for`/`class` header, never at the header's "
            "level). A blank line is empty.\n"
            "Do NOT write `N|` counts anywhere — that's the old format; use real spaces.")
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
        n_edit = sum(1 for tc in tcs if tc.get("function", {}).get("name")
                     in ("edit_file", "replace_lines"))
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
            # edit until the budget runs out. (Audit #46.) EXCLUDE run_code: a
            # non-zero exit is legitimate behavioural FEEDBACK to iterate on, not a
            # malformed call — counting it would cut off the fix loop we just added.
            if (isinstance(result_str, str) and result_str.startswith("✗")
                    and name != "run_code"):
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
            # STALE-READ guard: once an edit LANDS, any earlier read_file view of
            # that file still sitting in the history is outdated — and the coder
            # was copying `old` from those stale views (f327: 12 'old not found'
            # rejects after 3 edits applied). Mark them so it re-reads instead.
            if (name in ("edit_file", "replace_lines")
                    and isinstance(result_str, str) and result_str.startswith("✓")):
                _ep = (args.get("path") or "") if isinstance(args, dict) else ""
                if _ep:
                    for _m in messages[:-1]:
                        _c = _m.get("content")
                        if (_m.get("role") == "tool" and isinstance(_c, str)
                                and "=== Code:" in _c and _ep in _c
                                and "⟪STALE" not in _c):
                            _m["content"] = (
                                f"⟪STALE — {_ep} was EDITED after this read; its line "
                                f"numbers and content below are OUTDATED. Do NOT copy "
                                f"`old` from here — read_file {_ep} again for the current "
                                f"view.⟫\n" + _c)
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
