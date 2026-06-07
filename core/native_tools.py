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

INDENT FORMAT (ckpt 125/129; gutter naturalized ckpt 143): read_file + the pre-loaded file
block render each line as `LINENO ⇥INDENT|<real spaces>code` (prefix_ws) — line# bare on the
left, `⇥INDENT` marks the indent number, then real spaces + code. The coder SEES the
indentation AND its number. Edits declare indent by NUMBER: old/new lines are `INDENT|code`
and the applier (`_expand_indent_lines`) re-emits the spaces — so the coder states a number
and never types (or drops) leading spaces (the col-0 dedent root cause). STRICT (ckpt 129):
the applier transforms ONLY the two UNAMBIGUOUS forms — `INDENT|code` and a full view line
`LINENO ⇥INDENT|code` (also tolerating the old `LINENO:INDENT|`, plus stripping the harness's
own `|appears N (#hex)` tail). A diff
row is NOT editable input (its gutter is shape-ambiguous with YAML/config, so it's taken
literally and the reject tells the coder to re-send as `INDENT|code`). Anything else is
literal — no guessed transform can corrupt real content.

Primary edit = `edit_file(path, hunks)` (content-anchored, can't go stale); `replace_lines`
is the secondary line-range tool. Both REUSE the existing applier + validation gate +
reject feedback (lazy import, to avoid a circular dep with workflows.code). The coder also
has `run_code` (a sandboxed runner — read-only, no network).

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
from core import thought_logger

# Models built for native function-calling — use the structured loop, not text.
NATIVE_TOOL_MODELS = {"nvidia/gpt-oss-120b", "nvidia/gpt-oss-nim", "groq/gpt-oss-120b",
                      # mistral/medium speaks the standard tools= function-calling API
                      # (api.mistral.ai). As a TEXT coder it kept omitting JARVIS's
                      # [tool use] wrapper (0 extractable edits); as a NATIVE-tool coder
                      # it emits structured tool_calls instead. (user 2026-06-02)
                      "mistral/medium"}

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

# TRACE-to-test for the CODER (flag-gated, A/B; default OFF). The semantic fails
# are CODER errors (right files, wrong logic), and the coder is who run_codes — so
# the trace→edge→discriminating-test→run_code→fix loop belongs HERE, in one agent.
# The trace's "ONE edge → MINIMAL change" is also a scope-corset against the
# over-editing regression. Enabled with JARVIS_TRACE.
_TRACE_MODE = bool(os.environ.get("JARVIS_TRACE"))

# EDIT-COT (flag-gated, A/B; default OFF). The captured coder reasoning (ckpt-118)
# showed the coder GUESSES the contract ("likely/probably/simpler/basic") instead of
# doing the 4-move CoT — and an OPTIONAL CoT is ignored. So bake the grounding INTO
# the writing tools: every edit must carry goal/traced/check, and the harness REJECTS
# the edit if they're missing OR ungrounded (the `traced` field must quote a REAL line
# from the file, so a guess can't satisfy it). Forcing the grounding as a tool arg is
# the structural enforcement the diagnosis calls for. Enabled with JARVIS_EDIT_COT.
_EDIT_COT = bool(os.environ.get("JARVIS_EDIT_COT"))
_HEDGE = ("likely", "probably", "maybe", "i think", "i guess", "not sure",
          "should be", "presumably", "perhaps", "might be")


def _check_edit_cot(args: dict, cur: str, is_insert: bool) -> "str | None":
    """VERIFICATION REMOVED (ckpt-133). The grounding SLOTS (goal/traced/check) and the
    prompt that invites them are KEPT — the coder may still reason in them — but the harness
    NO LONGER REJECTS an edit for missing/ungrounded fields. The verbatim-`traced`-quote
    teeth were tripping weak models into 8×-reject loops on hard steps (3/8 instances on the
    ckpt-132 run; cost 395e5e20 a timeout) — the model gamed the FORMAT instead of thinking,
    and a forced rigid reasoning template tends to hurt, not help. The `old` field already
    carries the real line (content-verified at apply time), so dropping the second copy loses
    no actual grounding. Always returns None (never rejects); kept as a no-op so call sites
    and the A/B flag stay in place and re-enabling is a one-function change."""
    return None


def is_native_tool_model(model_id: str) -> bool:
    return model_id in NATIVE_TOOL_MODELS


# ── Tool schemas (OpenAI function-calling format) ────────────────────────────
# One schema per text-coder capability — full parity, not a subset.
CODER_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            "Read a file (optionally a line range) from the project. Each line is "
            "returned as `LINENO ⇥INDENT|<real spaces>content` — LINENO (bare, on the left) "
            "is the 1-based line number; `⇥INDENT` is the indentation — the ⇥ marks it, "
            "INDENT is the leading-space COUNT (the number you reuse in edits); then `|`, the "
            "real leading spaces, and the code. (So `286 ⇥4|    def foo` = line 286, indent 4.) "
            "A huge file comes back "
            "as a skeleton (top-level defs); pass start_line/end_line to expand a "
            "region. You usually don't NEED to re-read a file you've already been shown "
            "(the step's injected file(s), one you read, or one you just edited — its diff "
            "IS the live state), because edit_file anchors on BOTH the line number you "
            "copied AND the content, so a shifted view self-corrects. If you're genuinely "
            "unsure of the CURRENT state after several edits, re-reading a SPECIFIC unseen "
            "range is fine — but re-dumping a whole file you've already seen wastes your turn "
            "budget; the diff after each edit already gives you accurate, live line numbers. "
            "(A huge file comes back as a skeleton, not a full re-dump.)"),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative file path"},
            "start_line": {"type": "integer", "description": "first line, 1-based (optional)"},
            "end_line": {"type": "integer", "description": "last line, 1-based (optional)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": (
            "See the project's filesystem TREE — the planner's view, EXPANDABLE one level at "
            "a time. With no path (or path=''), it lists the top-level folders (each with a "
            "recursive file count) and the root files (each with its LINE count). Pass a "
            "folder path to expand THAT folder. Use it to find WHERE a file lives before "
            "read_file — never guess a path; every path it shows is exact and copy-paste-ready. "
            "It shows ONE level per call (it's not the whole tree at once) — expand the folder "
            "you care about."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "folder to expand (repo-relative); omit or '' for the top level"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "keep",
        "description": (
            "Free up context by KEEPING only the line ranges of a file you still need and "
            "dropping the rest of that file's view from your context. Use it when you've read "
            "several files/ranges and only a few spots actually matter: read a few → keep the "
            "relevant ranges → read more → keep — so a many-file step never overflows. You can "
            "ONLY keep ranges of a file you have already VIEWED (read_file first); keeping a "
            "file or range you haven't viewed is an error. The kept ranges stay; everything else "
            "of that file is replaced by a short pointer (read_file a range to bring it back)."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative file you've already read"},
            "ranges": {"type": "array", "description": "the line ranges to KEEP, each as [start, end] (1-based, inclusive)",
                       "items": {"type": "array", "items": {"type": "integer"}}},
        }, "required": ["path", "ranges"]},
    }},
    {"type": "function", "function": {
        "name": "batch",
        "description": (
            "Run SEVERAL read-only LOOKUPS in ONE round. (Each normal tool call costs a whole "
            "model round-trip; your opening 'look' at a step usually needs several files — do "
            "them together here instead of one-per-round.) `calls` = a list of {tool, args}; you "
            "get every result back at once. Batchable: read_file, search_text, find_refs, "
            "file_purpose, semantic_search, depends_on, list_dir. NOT batchable: edit_file / "
            "create_file (edits have order dependencies — one per round), keep, run_code, finish. "
            "Name what each lookup answers; don't ask the same thing two ways."),
        "parameters": {"type": "object", "properties": {
            "calls": {"type": "array", "description": "the lookups to run together, each {\"tool\": <name>, \"args\": {...}}",
                      "items": {"type": "object", "properties": {
                          "tool": {"type": "string", "description": "read_file / search_text / find_refs / file_purpose / semantic_search / depends_on / list_dir"},
                          "args": {"type": "object", "description": "that tool's arguments"}}}},
        }, "required": ["calls"]},
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
            "Create a NEW file at `path` with `content` — the full file text, written as "
            "normal code with real indentation (no line numbers, no gutter — you're authoring "
            "a whole new file, not editing a view). Use this for files that don't "
            "exist yet — a new module, script, or test file (greenfield builds, or "
            "adding a file to an existing project). To change a file that ALREADY "
            "exists, use edit_file instead — create_file refuses to clobber."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative path of the new file"},
            "content": {"type": "string", "description": "the full contents of the new file"},
            "goal": {"type": "string", "description": "GROUNDING (optional): the spec behaviour this new file provides — 1 concrete sentence"},
            "traced": {"type": "string", "description": "GROUNDING: the spec/interface line this file implements (quote it) — not a guess about what's wanted"},
            "check": {"type": "string", "description": "GROUNDING: one concrete input→expected-output case the new code satisfies"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Edit a file with content-matched search→replace. Make ALL your changes to this "
            "file in ONE call: `edits` = a LIST of SMALL {old, new} hunks — ONE hunk per "
            "change. MANY small hunks beats one big hunk. Changes far apart (a def up top AND "
            "its caller 200 lines below AND an import) are SEPARATE hunks in the SAME `edits` "
            "list — NEVER one giant `old` swallowing the gap between them. All hunks anchor to "
            "the file's CURRENT view and apply TOGETHER, so their ORDER doesn't matter, line "
            "numbers never shift between them, and you get back ONE consolidated diff = the "
            "file's new state. (For a single change you may pass `old`/`new` directly.) Per "
            "hunk: `old` = the exact CONTIGUOUS lines THAT ONE hunk changes — copy the view "
            "line(s) VERBATIM, keeping the whole `LINENO ⇥INDENT|` prefix (e.g. `286 ⇥4|    def "
            "setvalue`); the harness anchors on BOTH the line number AND the content (a stale "
            "number self-corrects). Bracket each hunk with ~1-2 UNCHANGED lines above and below "
            "— copy them verbatim into BOTH `old` and `new`: it makes the match UNIQUE (no "
            "'appears N times' / 'not found' rejects) and the real surrounding lines SHOW you "
            "the exact indent to reuse (read the `⇥INDENT` of the line your code belongs under). "
            "Nearby hunks may share those context lines — that's fine, they're merged; the only "
            "thing to avoid is two hunks CHANGING the same line. "
            "`new` = the replacement as `INDENT|code` — the indent NUMBER (the `⇥` value), a "
            "pipe, then code with NO leading spaces (e.g. `4|def f():`, `8|return x`; the harness "
            "re-emits the spaces FROM THE NUMBER, so put indent in the NUMBER, never as spaces in "
            "the code — `0|    x` is WRONG, it means indent 0). To INSERT, add your new lines "
            "between the unchanged bracket lines. To DELETE, new=[the bracket lines only]. "
            "Removing a name (a def/import) AND the code that still USES it must go in the SAME "
            "`edits` call as separate hunks, or it's rejected. A rejection leaves the file "
            "UNCHANGED — your view is still current, so just fix the call and resend; don't re-read."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "repo-relative path to edit"},
            "edits": {"type": "array", "description": "ALL your changes to this file as a LIST of "
                      "SMALL {old, new} hunks — one per change; changes far apart go in SEPARATE hunks "
                      "here (not one big `old`). They apply together in ANY order against the current "
                      "view and return ONE consolidated diff. Always prefer this over separate edit_file calls.",
                      "items": {"type": "object", "properties": {
                          "old": {"type": "array", "items": {"type": "string"}},
                          "new": {"type": "array", "items": {"type": "string"}}}}},
            "old": {"type": "array", "items": {"type": "string"},
                    "description": "single-edit shorthand: the EXACT existing lines, copied VERBATIM from your read (keep the `LINENO ⇥INDENT|` prefix). Include ~2 UNCHANGED lines above and below the change so the match is unique and the surrounding indent is visible. Use `edits` to batch several changes."},
            "new": {"type": "array", "items": {"type": "string"},
                    "description": "single-edit shorthand: the replacement as `INDENT|code` (indent in the NUMBER, code with no leading spaces; no line number) — re-include the same ~2 bracketing context lines unchanged; [] to delete"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_code",
        "description": (
            "OPTIONAL: run a shell command in your sandbox (your edits are live; read-only + "
            "no network; a MINIMAL python — the standard library only, NOT every framework dep) "
            "to check ONE concrete fact — e.g. python -c \"from pkg.mod import Thing; "
            "print(Thing().method(...))\" to see a real value. Prefer a tiny targeted python -c "
            "over a broad `pytest` run: a full test run usually hits ModuleNotFoundError on an "
            "un-installed dep (PyQt5, jinja2, web, test plugins) — that is an ENVIRONMENT limit, "
            "NOT your bug, and reacting to it (editing imports, creating stub modules) corrupts "
            "your patch. exit 0 = ok (a passing assert prints nothing); exit≠0 = the output is "
            "the real behaviour. Not required — TRACE-and-reason is the main path (see HOW TO THINK)."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "shell command — prefer python -c \"...\" for one targeted fact; a single python -m pytest <path::test> only if it doesn't import a framework dep the minimal sandbox lacks"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": (
            "Call ONLY when the edit is complete and you've verified it does what the "
            "step asked — by a targeted run_code check OR a careful re-trace of the spec's "
            "example. Ends the task."),
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "one line: what you changed"},
        }, "required": []},
    }},
]

# TRACE-to-test tool — flag-gated (JARVIS_TRACE) so the DEFAULT tool surface is
# unchanged. A "format-enforcer": it does no computation, it returns a strict
# template that makes the coder TRACE the real flow to the behavioural EDGE
# (citing real lines) and design a test that CATCHES the bug — which it then
# run_codes. Turns "understand the nuance" into a procedure + bounds scope to the
# edge (anti-over-edit).
_TRACE_TOOL = {"type": "function", "function": {
    "name": "trace_to_test",
    "description": (
        "AFTER you've edited a behaviour you're unsure of, call this to PROVE the edit "
        "instead of hoping. It returns a strict template to fill: GOAL → a line-grounded "
        "FLOW trace (each step cites the EXACT code line @ file:line — read_file first; "
        "imagined lines are rejected) → the EDGE where correct vs the naive impl diverges "
        "→ a TEST that CATCHES the bug (adversarial setup + the assertion that fails for a "
        "naive impl). Then run_code that test against your edit: green proves the fix, red "
        "shows exactly what the edit still gets wrong. Use it as the CLOSING step for any "
        "subtle/conditional requirement — it turns 'I think this is right' into 'I ran the "
        "discriminating test and it passed.'"),
    "parameters": {"type": "object", "properties": {
        "target": {"type": "string", "description": "the behaviour/symbol to trace to a test"},
    }, "required": ["target"]},
}}
if _TRACE_MODE:
    CODER_TOOLS.append(_TRACE_TOOL)

# JARVIS_BATCH_ONLY experiment (ckpt-182): expose `batch` as the SOLE tool, so the model — which
# emits exactly one tool_call per round — packs MULTIPLE ops into that one call (collapsing the
# look phase to a ~4-round step). `batch` carries every op as {tool, args}. Flag-gated: the
# default tool surface (CODER_TOOLS) is unchanged.
_BATCH_ONLY_TOOLS = [{"type": "function", "function": {
    "name": "batch",
    "description": (
        "Your ONLY tool. Each round, emit ONE batch carrying the operations you want run this "
        "round, as `calls` = a list of {\"tool\": <name>, \"args\": {...}}. The harness runs them "
        "IN ORDER and returns all results at once. Available ops and their args:\n"
        "  read_file{path[,start_line,end_line]} · list_dir{[path]} · search_text{pattern} · "
        "find_refs{symbol} · find_callers{tag} · file_purpose{path} · semantic_search{query} · "
        "depends_on{symbol}\n"
        "  edit_file{path, edits:[{old:[lines],new:[lines]}]} · create_file{path, content} · "
        "keep{path, ranges:[[start,end]]} · run_code{command} · finish{summary}\n"
        "HOW TO USE IT — the ideal step is ~4 rounds:\n"
        "  1) LOOK: batch all the read_file/search lookups you need (gather many files in one "
        "round). You get every result back together.\n"
        "  2) After you SEE them, batch your edit_file op(s). DO NOT read_file and edit_file the "
        "SAME file in one batch — the edit would run before you've seen the file. Put all hunks "
        "for one file in ONE edit_file (its `edits` list); separate files = separate ops in the "
        "batch.\n"
        "  3) VERIFY: batch a run_code check.\n"
        "  4) finish{summary} — put it as the LAST op (alone, or after a final mechanical edit).\n"
        "Name what each lookup answers; don't ask the same thing twice. A file >1000 lines comes "
        "back as a def-index → batch the ranges you need next round."),
    "parameters": {"type": "object", "properties": {
        "calls": {"type": "array", "description": "this round's ops, each {\"tool\": <name>, \"args\": {...}}",
                  "items": {"type": "object", "properties": {
                      "tool": {"type": "string"},
                      "args": {"type": "object"}}}},
    }, "required": ["calls"]},
}}]


# ── Tool dispatch ────────────────────────────────────────────────────────────
def _view_stamp(ctx: dict) -> str:
    """A human 'when' for a file view/edit — 'step S, round R' (round alone if no
    step is set). Stamped onto every diff and view so the coder can ORDER multiple
    views of the same file and know which one is current (its latest)."""
    r = ctx.get("round")
    s = ctx.get("step_num")
    if s is not None and r is not None:
        return f"step {s}, round {r}"
    if r is not None:
        return f"round {r}"
    return "this step"


def _unassigned_enum_members(src: str) -> list:
    """Enum members DEFINED but never ASSIGNED/compared anywhere in `src` — a likely
    UNHANDLED case (the coder defined `VersionChange.unknown` but no branch ever sets
    a field to it). AST-based, best-effort; returns ['EnumName.member', ...]. Used as a
    pre-finish ADVISORY (harness computes the global 'which members are dead', the coder
    decides locally). NOTE: catches a FORGOTTEN case, not a WRONG-value case (setting a
    branch to the wrong member is semantic — only run_code / the spec catches that)."""
    import ast
    try:
        tree = ast.parse(src)
    except Exception:
        return []
    enums = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
                (isinstance(b, ast.Name) and b.id.endswith("Enum")) or
                (isinstance(b, ast.Attribute) and b.attr.endswith("Enum"))
                for b in node.bases):
            members = set()
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    members |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    members.add(stmt.target.id)
            members = {m for m in members if not m.startswith("_")}
            if members:
                enums[node.name] = members
    if not enums:
        return []
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used.add((node.value.id, node.attr))
    out = []
    for ename, members in enums.items():
        for m in sorted(members):
            if (ename, m) not in used:
                out.append(f"{ename}.{m}")
    return out


def _note_view(ctx: dict, path: str) -> None:
    """Record that the coder now has {path}'s CURRENT full state in its context —
    set by a full read_file and by every applied edit (the edit's diff + the
    unchanged remainder = a current view). Used to short-circuit redundant
    full re-reads (the #1 context-blowup + thrash cause: f631 re-read a 900-line
    file 5× and blew the 131072-token window)."""
    ctx.setdefault("view_at", {})[path] = _view_stamp(ctx)


def _str_or_err(args: dict, key: str, tool: str):
    """Coerce a tool's primary string arg, or return (None, ✗msg) for a missing /
    empty / NON-STRING value. A weak model that passes 5 / [] / null / true for a
    string field must get a clean ✗ to react to — never crash the tool with a
    TypeError/AttributeError downstream (.strip(), regex, a dict key). (ckpt-149
    stability fuzz: 6 tools raised on wrong-typed args before this.)"""
    v = args.get(key)
    if isinstance(v, str) and "\x00" in v:
        # a null byte crashes ripgrep / subprocess / compile() downstream with a
        # ValueError; reject it up front as the invalid input it is.
        return None, f"✗ {tool}: `{key}` contains a null byte (invalid). Re-issue with clean text."
    if isinstance(v, str) and v.strip():
        return v, None
    return None, (f"✗ {tool}: `{key}` must be a non-empty string "
                  f"(got {type(v).__name__}). Re-issue with a string {key}.")


_FULL_VIEW_CAP = 1000    # a file up to this many lines is shown IN FULL on read; larger → the
                         # ACCUMULATING VIEW (ckpt-196): ONE growing per-file view = the def/class
                         # index with the ranges the coder has READ filled in as real code, and the
                         # un-read GAPS shown as labelled holes ("⋯ lines X-Y not read ⋯" + the def
                         # names inside them). Each read EXPANDS that one view and REPLACES the prior
                         # one in history (the loop collapses older copies to a pointer), so there's
                         # exactly ONE, ever-more-complete view per file — no duplicate dumps, no
                         # "you already read a-b" refusals, and `keep` trims it to the ranges that
                         # matter. This replaced the old slice-and-short-circuit model, which made a
                         # >cap file feel like 10 disconnected searches and looped gpt-oss to timeout
                         # (a26: 69 reads / 9 edits / 1800s). Kept at 1000 so the growing view — not a
                         # full dump ×several files — is what fills context (the ckpt-179 overflow).
_FULL_REREAD_CAP = _FULL_VIEW_CAP   # back-compat alias (a few call sites / tests still name it)


def _repo_base(ctx: dict) -> str:
    """Absolute dir holding the live repo for THIS run — the sandbox (edits are live
    there) if present, else the project root. '' if neither is set."""
    sb = ctx.get("sandbox")
    base = (getattr(sb, "sandbox_dir", "") or "") if sb is not None else ""
    return base or (ctx.get("project_root") or "")


def _file_line_count(ctx: dict, rel: str, full_abs: str = "") -> int:
    """Best-effort line count of a repo file. Prefer the LIVE in-memory content
    (current — includes this step's edits), then the sandbox, then disk. 0 if the
    file can't be read (shown as '?'). Never raises."""
    c = ctx.get("file_contents", {}).get(rel)
    if not isinstance(c, str):
        sb = ctx.get("sandbox")
        if sb is not None:
            try:
                c = sb.load_file(rel)
            except Exception:
                c = None
    if not isinstance(c, str) and full_abs:
        try:
            with open(full_abs, "r", errors="replace") as _f:
                c = _f.read()
        except Exception:
            c = None
    if not isinstance(c, str) or not c:
        return 0
    return c.count("\n") + (0 if c.endswith("\n") else 1)


def _incrusted_tree(ctx: dict, rel_path: str) -> str:
    """Render WHERE `rel_path` sits in the filesystem — its folder breadcrumb plus the
    sibling entries in its immediate folder (sub-dirs with recursive file-counts, files
    with LINE counts), marking the viewed file with ◀ VIEWING. So a file view is never a
    bare floating block: the coder always sees the file's place in the tree and what's
    beside it (the planner's tree, incrusted into the read). Bounded (≤ ~40 entries) and
    best-effort — returns '' on any failure so a read can never break. Expand elsewhere
    with list_dir."""
    try:
        import os as _os
        from core.exploration_tools import _tree_visible, _count_tree_files
        base = _repo_base(ctx)
        if not base or not _os.path.isdir(base):
            return ""
        rp = rel_path.replace("\\", "/").strip("/")
        parent_rel = _os.path.dirname(rp)
        fname = _os.path.basename(rp)
        parent_abs = _os.path.join(base, parent_rel) if parent_rel else base
        if not _os.path.isdir(parent_abs):
            return ""
        # containment (defense-in-depth: case-b reaches here gated only on view_at): a `..`
        # path must not list a folder outside the repo. Separator boundary, not bare prefix.
        _rb, _rp = _os.path.realpath(base), _os.path.realpath(parent_abs)
        if _rp != _rb and not _rp.startswith(_rb + _os.sep):
            return ""
        crumb_parts = [p for p in parent_rel.split("/") if p]
        crumb = "(project root)" if not crumb_parts else " ▸ ".join(crumb_parts) + "/"
        try:
            names = sorted(_os.listdir(parent_abs))
        except OSError:
            return ""
        dir_ents, file_ents, view_ent = [], [], None
        for nm in names:
            full = _os.path.join(parent_abs, nm)
            isd = _os.path.isdir(full)
            if not _tree_visible(nm, isd):
                continue
            if isd:
                dir_ents.append(f"{nm}/ ({_count_tree_files(full)} files)")
            else:
                rel_n = f"{parent_rel}/{nm}" if parent_rel else nm
                ln = _file_line_count(ctx, rel_n, full)
                if nm == fname:
                    view_ent = f"[▸ {nm} ({ln} lines)  ◀ VIEWING]"
                else:
                    file_ents.append(f"{nm} ({ln} lines)")
        # The incrusted header is for ORIENTATION (where am I + a few neighbours), not the full
        # listing — that's what list_dir is for. Keep it tight so a read in a big folder doesn't
        # dump 40 siblings every time. Always keep the viewed file visible.
        ents = dir_ents + ([view_ent] if view_ent else []) + file_ents
        _MAX = 12
        shown = ents[:_MAX]
        if view_ent and view_ent not in shown:
            shown = shown[:_MAX - 1] + [view_ent]
        more = len(ents) - len(shown)
        tail = (f"   … +{more} more here (list_dir '{parent_rel or '.'}' for the full folder)"
                if more > 0 else "")
        return f"  {crumb}  — folder ({len(ents)} entries):\n      {'   '.join(shown)}{tail}"
    except Exception:
        return ""


def _def_index(content: str, path: str) -> str:
    """A compact navigation INDEX for a too-large (>cap) file: every def/class with its line
    number, NO bodies — just enough to know WHERE to look, then read a range. (ckpt-179: the
    big-file view is this index, not a full dump or a verbose skeleton.) Best-effort; falls back
    to a def/class line-grep for non-Python / unparseable files. Never raises."""
    try:
        import ast as _ast
        _t = _ast.parse(content)
        rows = []
        for _n in _ast.walk(_t):
            if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                _kind = "class" if isinstance(_n, _ast.ClassDef) else "def"
                _ind = "  " * (getattr(_n, "col_offset", 0) // 4)
                rows.append((_n.lineno, f"  {_n.lineno}: {_ind}{_kind} {_n.name}"))
        rows.sort()
        if rows:
            body = "\n".join(r[1] for r in rows[:500])
            if len(rows) > 500:    # don't silently hide defs past the cap (adversarial review)
                body += (f"\n  … (+{len(rows) - 500} more defs not shown — search_text/find_refs "
                         f"to locate a specific symbol's line)")
            return body
    except Exception:
        pass
    # non-Python / unparseable → grep def/class/function-ish lines
    import re as _re2
    out = [f"  {i+1}: {ln.strip()[:90]}" for i, ln in enumerate(content.split("\n"))
           if _re2.match(r'\s*(def |class |async def |function |[A-Za-z_]+\s*=\s*function)', ln)]
    if not out:
        return "  (no def/class lines detected — read a range to see content)"
    body = "\n".join(out[:500])
    if len(out) > 500:
        body += f"\n  … (+{len(out) - 500} more — search_text/find_refs to locate a symbol)"
    return body


def _too_large_view(ctx: dict, path: str, nlines: int, content: str) -> str:
    """The >cap big-file view (ckpt-179): incrusted header + the def/class INDEX + a pointer to
    read a range. NO bodies, NO full dump — keeps the footprint tiny so many files fit in context."""
    return (_view_header(ctx, path, nlines, structure_only=True)
            + "\nThis file's defs/classes — just to see WHERE to look; read_file a range for the "
              "actual code (or search_text/find_refs to locate a symbol's line):\n"
            + _def_index(content, path)
            + f"\n→ read_file(path='{path}', start_line=N, end_line=M) around the line you need.")


def _merge_ranges(ranges):
    """Merge (start,end) 1-based inclusive ranges into a minimal sorted, non-overlapping set —
    touching/adjacent ranges combine (so the accumulating view shows one block, not two). Skips
    malformed pairs. Never raises."""
    norm = []
    for r in ranges or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        norm.append((min(a, b), max(a, b)))
    norm.sort()
    out = []
    for s, e in norm:
        if out and s <= out[-1][1] + 1:        # overlapping OR directly adjacent → merge
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _def_lines(content: str):
    """[(lineno, 'kind name'), …] for every def/class, sorted by line — powers the hole
    summaries in the accumulating view (which defs sit in an un-read gap). Never raises."""
    rows = []
    try:
        import ast as _ast
        for _n in _ast.walk(_ast.parse(content)):
            if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                _kind = "class" if isinstance(_n, _ast.ClassDef) else "def"
                rows.append((_n.lineno, f"{_kind} {_n.name}"))
    except Exception:
        import re as _re2
        for _i, _ln in enumerate(content.split("\n"), 1):
            _m = _re2.match(r'\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)', _ln)
            if _m:
                rows.append((_i, f"{_m.group(1)} {_m.group(2)}"))
    rows.sort()
    return rows


def _accumulated_view(ctx: dict, path: str, content: str) -> str:
    """ONE growing view of a >cap file (ckpt-196). The def/class index with the line-ranges the
    coder has READ (ctx['_served_ranges'][path]) filled in as real `LINENO ⇥INDENT|code`, and the
    un-read GAPS shown as labelled holes that list the defs inside them. Empty revealed set → the
    plain def-index (the first read). Each read EXPANDS this; the loop collapses older copies of
    this file's view to a pointer, so there's exactly ONE, ever-more-complete view per file — no
    duplicate dumps, no 'you already read a-b' refusals. `keep` trims it to the kept ranges."""
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    total = max(len(lines), 1)
    revealed = _merge_ranges([(s, e) for (s, e) in ctx.get("_served_ranges", {}).get(path, [])])
    revealed = [(max(1, s), min(total, e)) for (s, e) in revealed if s <= total]
    if not revealed:
        return _too_large_view(ctx, path, total, content)        # nothing read yet → def-index
    defs = _def_lines(content)
    def _hole(a, b):
        names = [f"{ln}:{nm}" for (ln, nm) in defs if a <= ln <= b]
        tail = ("  — contains " + "; ".join(names[:10]) + (" …" if len(names) > 10 else "")) if names else ""
        return (f"  ⋯ lines {a}-{b} not read — read_file(start_line={a}, end_line={b}) "
                f"to reveal ⋯{tail}")
    revealed_n = sum(e - s + 1 for s, e in revealed)
    head = (f"=== VIEW: {path} — {total} lines — GROWING VIEW ({revealed_n} revealed; the rest "
            f"are labelled gaps — read a range to fill any in) ===")
    tree = _incrusted_tree(ctx, path)
    sep = "  " + "─" * 46
    out = [f"{head}\n{tree}\n{sep}" if tree else f"{head}\n{sep}"]
    cursor = 1
    for (s, e) in revealed:
        if cursor < s:
            out.append(_hole(cursor, s - 1))
        for i in range(s, e + 1):
            ln = lines[i - 1]
            ind = len(ln) - len(ln.lstrip(' '))
            out.append(f"{i} ⇥{ind}|{ln}")
        cursor = e + 1
    if cursor <= total:
        out.append(_hole(cursor, total))
    return "\n".join(out)


def _msg_is_view_of(content, path: str) -> bool:
    """True iff this tool-message content is a VIEW/Code block (or a prior KEPT block) of EXACTLY
    `path`. HEADER-PRECISE (`=== VIEW: {path} —`, `=== Code: {path} `, `⟪KEPT … of {path}`) so a
    path merely MENTIONED inside another file's view (an import line) does NOT false-match — the
    loose `path in content` test collapsed unrelated files' views (bughunt #15/#16). Single source
    of truth for the supersede / keep-eviction / read-replace loops."""
    if not isinstance(content, str):
        return False
    return (f"=== VIEW: {path} —" in content
            or f"=== Code: {path} " in content
            or ("⟪KEPT only lines" in content and f"of {path} " in content))


def _supersede_prior_file_views(messages: list, path: str) -> int:
    """Collapse every EARLIER tool-message view of `path` (a plain view OR a prior KEPT block) to a
    one-line pointer, leaving the newest, most-complete view (messages[-1]) as the only live copy —
    so there's exactly ONE view per file in context. Called after a read of a >cap file returns the
    growing view. Pure content-swap on existing tool messages (no add/remove → API pairing intact).
    Other files' views are untouched. Returns the number collapsed. (ckpt-196)"""
    n = 0
    for _m in messages[:-1]:
        _c = _m.get("content")
        if not (_m.get("role") == "tool" and isinstance(_c, str)):
            continue
        if "⟪earlier view" in _c:
            continue
        if _msg_is_view_of(_c, path):
            _m["content"] = (f"⟪earlier view of {path} — superseded by your newer, more "
                             f"complete view of it below; scroll down.⟫")
            n += 1
    return n


def _view_header(ctx: dict, rel_path: str, total_lines: int,
                 *, range_desc: str = "", structure_only: bool = False) -> str:
    """The incrusted VIEW header: a clear, unambiguous 'THIS file / FULL vs RANGE vs
    STRUCTURE' banner, then the file's position in the tree (breadcrumb + siblings)."""
    if structure_only:
        head = (f"=== VIEW: {rel_path} — {total_lines} lines — TOO LARGE to show in full "
                f"(>{_FULL_VIEW_CAP}); use read_file with a line range ===")
    elif range_desc:
        head = (f"=== VIEW: {rel_path} — {range_desc} of {total_lines}  "
                f"(RANGE — NOT the full file) ===")
    else:
        head = f"=== VIEW: {rel_path} — {total_lines} lines (FULL file) ==="
    tree = _incrusted_tree(ctx, rel_path)
    sep = "  " + "─" * 46
    return (f"{head}\n{tree}\n{sep}" if tree else f"{head}\n{sep}")


def _reframe_read_body(body: str, header: str) -> str:
    """Swap _run_code_reads' own `=== Code: … ===` line (and its text-coder-only
    `[KEEP: …]` large-file hint, which the native coder cannot emit) for the incrusted
    VIEW header. Everything else — line-numbered content, tab warnings, #tag
    annotations — is kept verbatim."""
    kept = []
    for ln in body.split("\n"):
        st = ln.strip()
        if st.startswith("=== Code:"):
            continue
        if "[KEEP:" in ln and "Large file" in ln:
            continue
        kept.append(ln)
    return header + "\n" + "\n".join(kept).lstrip("\n")


async def _do_read(args: dict, ctx: dict) -> str:
    # Reuse the text coder's [CODE:]/[VIEW:] executor: skeleton for huge files,
    # range expansion, the `LINENO ⇥INDENT|content` prefix view, AND recording
    # viewed_versions so a following edit anchors on what was just read.
    from core.tool_call import _run_code_reads
    path, _e = _str_or_err(args, "path", "read_file")
    if _e:
        return _e
    # Baseline snapshot for the re-read diff (set ONCE, first time we touch this file —
    # whichever of read/edit comes first captures the pre-edit content). (ckpt-150.)
    _fc0 = ctx.get("file_contents", {}).get(path)
    if _fc0 is not None:
        ctx.setdefault("_first_seen", {}).setdefault(path, _fc0)
    s = args.get("start_line"); e = args.get("end_line")
    _whole_note = ""   # set when a ≤3000 range read is upgraded to a whole-file serve (ckpt-178)
    if (s is None) != (e is None):
        # Exactly one bound given → the model asked for a region but the bound it
        # dropped would be silently ignored (whole file returned). Tell it.
        return ("✗ read_file: give BOTH start_line and end_line for a range, or "
                "NEITHER to read the whole file (you provided only "
                f"{'start_line' if s is not None else 'end_line'}).")
    # ── BIG FILE → the ACCUMULATING VIEW (ckpt-196) ─────────────────────────────────────────
    # A file over _FULL_VIEW_CAP that the coder does NOT already hold in full (NOT in view_at —
    # i.e. not injected at step start, not a prior small-file full read) is navigated by ONE
    # growing view: a whole-file read shows the current view (just the def-index until ranges are
    # read); a range read REVEALS that range (merged into _served_ranges[path]) and returns the
    # expanded view; re-reading a revealed range changes nothing (no refusal, no dup). The loop
    # collapses any earlier copy of this file's view to a pointer. (Replaces the slice +
    # short-circuit model that made a big file feel like 10 disconnected searches → gpt-oss looped
    # to a 1800s timeout: a26 read urls.py 69× / edited 9×.) Files the coder already holds in full
    # (view_at set) keep the existing short-circuit path below — re-serving them as a def-index
    # would DOWNGRADE a view it already has.
    if path not in ctx.get("view_at", {}):
        _bcur = ctx.get("file_contents", {}).get(path)
        if not (_bcur and _bcur.strip()):
            _bsb = ctx.get("sandbox")
            _bre = _bsb.load_file(path) if _bsb is not None else None
            if _bre and _bre.strip():
                _bcur = _bre
                ctx.setdefault("file_contents", {})[path] = _bcur
        if isinstance(_bcur, str) and _bcur.strip():
            _bnl = _bcur.count("\n") + (0 if _bcur.endswith("\n") else 1)
            if _bnl > _FULL_VIEW_CAP:
                if s is not None and e is not None:
                    try:
                        s_i, e_i = int(s), int(e)
                    except (TypeError, ValueError):
                        return (f"✗ read_file: start_line/end_line must be integers "
                                f"(got start_line={s!r}, end_line={e!r}).")
                    if s_i < 1 or e_i < 1:
                        return (f"✗ read_file: line numbers must be ≥ 1 (got start_line={s_i}, "
                                f"end_line={e_i}). Re-issue with a positive 1-based range.")
                    if e_i < s_i:
                        return (f"✗ read_file: invalid range — start_line ({s_i}) must be ≤ "
                                f"end_line ({e_i}). Put the smaller line number first.")
                    if s_i > _bnl:
                        return (f"✗ read_file: start_line {s_i} is out of range — {path} has only "
                                f"{_bnl} line(s). Read within 1-{_bnl}.")
                    e_i = min(e_i, _bnl)
                    _rev = ctx.setdefault("_served_ranges", {}).setdefault(path, [])
                    _rev.append((s_i, e_i))
                    ctx["_served_ranges"][path] = _merge_ranges(_rev)
                # else: whole-file read → reveal nothing new; show the current accumulated view.
                # Record viewed_versions (so a following edit anchors on CURRENT content); do NOT
                # set view_at — a big file is never "fully held" via reads, so the next whole
                # re-read keeps returning the growing view rather than a no-body short-circuit.
                if isinstance(ctx.get("viewed_versions"), dict):
                    ctx["viewed_versions"][path] = _bcur
                return _accumulated_view(ctx, path, _bcur)
    # RE-READ of an already-seen file (no range). The coder ALREADY holds this file's
    # current view — loaded at step start, read earlier this step, or handed back inside an
    # edit's diff — and (point 1) its full reasoning + those views are uncapped in history.
    # So re-dumping the file wastes a ~3-minute round AND clutters context with a duplicate
    # block the model can't tell apart from the live one (the #1 confusion + f631 blow-out).
    # If its view is current → ONE line saying so, no body. If the file actually CHANGED
    # since it last saw it (a non-edit_file mutation that didn't sync the view) → serve the
    # current content + the cumulative diff vs the START of this step.
    if s is None and e is None and path in ctx.get("view_at", {}):
        cur = ctx.get("file_contents", {}).get(path)
        # An EMPTY file_contents entry is, in this codebase, a failed-load sentinel
        # (`sandbox.load_file(fp) or read_file(...) or ""` collapses a non-resolving
        # path to ""). Treat empty/whitespace-only as NO content: try a real sandbox
        # reload, else fall through to _run_code_reads (actionable FILE NOT FOUND).
        if not (cur and cur.strip()):
            _sb0 = ctx.get("sandbox")
            _re = _sb0.load_file(path) if _sb0 is not None else None
            cur = _re if (_re and _re.strip()) else None
        if cur is not None:
            # The freshest version the coder has actually seen — its last view if recorded,
            # else what it was first shown this step (injected target / first read).
            _last = ctx.get("viewed_versions", {}).get(path)
            if _last is None:
                _last = ctx.get("_first_seen", {}).get(path)
            _base = ctx.get("_first_seen", {}).get(path)   # step-start baseline (see point 2)
            _rc = ctx.setdefault("_reread_count", {})
            _rc[path] = _rc.get(path, 0) + 1
            # (a) the coder's view is CURRENT (identical to what it last saw) → say so in one
            # line, no body, nothing capped. This is the common case (it just re-asked).
            if _last is not None and _last == cur:
                # BIG FILE (> cap): "you already have THIS file's current view" is a LIE — the
                # coder NEVER held real bodies for a >cap file, only a def-index skeleton + the
                # ranges/diffs it explicitly saw. After an edit (which sets view_at + makes
                # _last==cur), a whole-file re-read landed here and returned a NO-BODY message;
                # the coder read that as "the system returns no content", then GUESSED line
                # content → reject loop → 1800s timeout (a26). Serve the CURRENT def-index so it
                # can pick a fresh range (range re-reads now serve fresh content — see Fix A).
                _nl_cur = cur.count("\n") + (0 if cur.endswith("\n") else 1)
                if _nl_cur > _FULL_VIEW_CAP:
                    _note_view(ctx, path)
                    return _too_large_view(ctx, path, _nl_cur, cur)
                _where = ctx.get("view_at", {}).get(path, "earlier this step")
                _editnote = ""
                if _base is not None and _base != cur:
                    _editnote = (" Your edit(s) to it this step ARE applied — they're in the "
                                 "diff you already received; the view you hold is current.")
                _again = (f" (asked {_rc[path]}× now — edit straight from the view you have, or "
                          f"read a SPECIFIC start_line/end_line range for a region you haven't "
                          f"seen.)" if _rc[path] >= 2 else "")
                return (f"ℹ {path} — you ALREADY have THIS file's current view ({_where}) and it "
                        f"is UP TO DATE (unchanged since you last saw it).{_editnote} No need to "
                        f"re-read — edit straight from the view you hold.{_again}")
            # (b) it CHANGED since the coder last saw it → re-serve the CURRENT content (incrusted
            # in the tree), leading with the cumulative diff vs step-start. Honours the >3000 redirect.
            ctx.setdefault("viewed_versions", {})[path] = cur
            _note_view(ctx, path)
            _nlines = cur.count("\n") + (0 if cur.endswith("\n") else 1)
            if _nlines > _FULL_VIEW_CAP:
                _body = _too_large_view(ctx, path, _nlines, cur)
            else:
                from tools.codebase import add_line_numbers
                _body = (_view_header(ctx, path, _nlines) + "\n"
                         + add_line_numbers(cur, display_mode="prefix_ws"))
            _diff_base = _base if _base is not None else _last
            if _diff_base is not None and _diff_base != cur:
                from core.edit_diff import render_diff
                _diff = render_diff(_diff_base, cur, path)
                _dl = _diff.split("\n")
                if len(_dl) > 400:
                    _diff = "\n".join(_dl[:400]) + (f"\n… (+{len(_dl) - 400} more changed lines "
                            f"— see the current content below / read a range for detail)")
                return (f"✓ {path} — UPDATED since you last saw it; the line numbers below are the "
                        f"CURRENT, live ones. The diff is the cumulative change since the START of "
                        f"this step.\n━━ what changed this step ━━\n{_diff}\n{_body}")
            return _body
        # cur unavailable — fall through to the normal read path.
    # WHOLE-FILE read (no range, and either never seen or its view fell through): enforce the
    # >3000-line redirect ourselves so the native coder gets STRUCTURE + range guidance, never
    # a multi-thousand-line dump (the text loop's own cap is 8000 — too high for this coder).
    if s is None and e is None:
        _cur = ctx.get("file_contents", {}).get(path)
        if not (_cur and _cur.strip()):
            _sbx = ctx.get("sandbox")
            _cur = _sbx.load_file(path) if _sbx is not None else _cur
        if not (_cur and _cur.strip()):
            # disk fallback so the >3000 redirect fires before _run_code_reads' 8000 cap can
            # full-dump a huge file. Read from the SAME place _run_code_reads does (the sandbox
            # dir via _repo_base) first, then the project root.
            import os as _os
            for _root in (_repo_base(ctx), ctx.get("project_root") or ""):
                if not _root:
                    continue
                _fp = _os.path.join(_root, path)
                try:
                    if _os.path.isfile(_fp):
                        with open(_fp, "r", errors="replace") as _f:
                            _cur = _f.read()
                        if _cur and _cur.strip():
                            break
                except Exception:
                    pass
        if isinstance(_cur, str) and _cur.strip():
            _nl = _cur.count("\n") + (0 if _cur.endswith("\n") else 1)
            if _nl > _FULL_VIEW_CAP:
                ctx["file_contents"][path] = _cur          # keep the base in step with disk
                # the def-index is NOT a full view → do NOT _note_view (force explicit range reads).
                return _too_large_view(ctx, path, _nl, _cur)
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
        # RANGE-NIBBLING FIX (ckpt-178). The coder reads a big-but-≤3000 file in MANY narrow,
        # often-overlapping ranges (f631: 119 reads, a26c325b: ~30 ranges) — each a full ~180s
        # round, because range reads never enter view_at and so never short-circuit. Two cures:
        #  (a) ≤3000 lines → serve the WHOLE file ONCE (it's small enough to hold) and mark it
        #      viewed; every later read (range or full) then short-circuits. One bounded read
        #      beats dozens of round-trips. (set arg=path + clear the range so the tail FULL-frames.)
        #  (b) >3000 lines → can't dump; CACHE served ranges and short-circuit a re-read whose
        #      span is already fully covered by one we served (kills the overlapping nibble).
        _rc_cur = ctx.get("file_contents", {}).get(path)
        if not (_rc_cur and _rc_cur.strip()):
            _sbx = ctx.get("sandbox")
            _rc_cur = _sbx.load_file(path) if _sbx is not None else _rc_cur
        if isinstance(_rc_cur, str) and _rc_cur.strip():
            _rc_nl = _rc_cur.count("\n") + (0 if _rc_cur.endswith("\n") else 1)
            if _rc_nl <= _FULL_VIEW_CAP:
                _seen = ctx.get("viewed_versions", {}).get(path)
                if path in ctx.get("view_at", {}) and _seen == _rc_cur:
                    return (f"ℹ {path} — you already hold this WHOLE file ({_rc_nl} lines) in your "
                            f"view; lines {s_i}-{e_i} are in it. No need to re-read — edit straight "
                            f"from the view you have.")
                _whole_note = (f"(You asked for lines {s_i}-{e_i}; this file is only {_rc_nl} lines "
                               f"(≤{_FULL_VIEW_CAP}), so here is ALL of it — you now hold the whole "
                               f"file, no need to read it again.)\n")
                s, e = None, None          # serve WHOLE: tail FULL-frames + _note_views it
                arg = path
            else:
                _sr = ctx.setdefault("_served_ranges", {}).setdefault(path, [])
                for (a, b) in _sr:
                    if a <= s_i and e_i <= b:
                        return (f"ℹ {path} — you already read lines {a}-{b}, which COVER the "
                                f"lines {s_i}-{e_i} you asked for (they're in your context above). "
                                f"Read a region you have NOT seen, or edit from what you have.")
                _sr.append((s_i, e_i))
                arg = f"{path} {s_i}-{e_i}"
        else:
            arg = f"{path} {s_i}-{e_i}"
    else:
        arg = path
    try:
        out = await _run_code_reads(
            [arg], ctx.get("project_root", ""),
            viewed_versions=ctx.get("viewed_versions"),
            # Unified native view: LINENO:INDENT|<real spaces>code — number (authoritative
            # for edits) + visible indent, matching the pre-loaded file block. 2026-06-02.
            display_mode="prefix_ws")
    except Exception as ex:
        out = f"✗ read_file failed: {str(ex)[:160]}"
    # Keep file_contents (the replace_lines base) in step with the sandbox.
    sb = ctx.get("sandbox")
    if sb is not None:
        cur = sb.load_file(path)
        if cur is not None:
            ctx["file_contents"][path] = cur
    # INCRUST the view in the tree: swap _run_code_reads' bare `=== Code: ===` header for the
    # tree-positioned VIEW banner (THIS file, FULL vs RANGE, breadcrumb + siblings + line counts).
    # CRITICAL: _run_code_reads does NOT signal errors with a ✗ — a FILE NOT FOUND / IS A
    # DIRECTORY / EMPTY / SKELETON ONLY / PERMISSION-DENIED result carries that word IN the
    # `=== Code: … ===` title. So reframe ONLY a GENUINE full/range view (title is
    # `(N lines …)` or `(lines A-B of …)` AND not a skeleton) — otherwise we'd relabel an
    # error/skeleton as a "FULL file" view AND _note_view it, making a later re-read falsely
    # short-circuit to "you already have it, up to date" on a file the coder never actually saw.
    _hdr_line = ""
    if isinstance(out, str) and "=== Code:" in out:
        _hdr_line = next((ln for ln in out.split("\n") if ln.strip().startswith("=== Code:")), "")
    _is_real_view = bool(re.search(r"\((?:\d+ lines|lines \d)", _hdr_line)) and "SKELETON" not in _hdr_line
    if _is_real_view:
        _full = ctx.get("file_contents", {}).get(path) or ""
        _tot = _full.count("\n") + (0 if (_full and _full.endswith("\n")) else (1 if _full else 0))
        if s is not None and e is not None:
            _hdr = _view_header(ctx, path, _tot or e_i, range_desc=f"lines {s_i}-{e_i}")
        else:
            _hdr = _view_header(ctx, path, _tot)
        out = _reframe_read_body(out, _hdr)
        if _whole_note:                       # ≤3000 range read upgraded to a whole-file serve
            out = out.split("\n", 1)
            out = out[0] + "\n" + _whole_note + (out[1] if len(out) > 1 else "")
    # A successful FULL read means the coder now holds the whole current file — record it so a
    # later full re-read is short-circuited. (A RANGE read shows only a slice → NOT fully in
    # context; an error/skeleton is NOT a view → never marked, so the coder keeps trying.)
    if s is None and e is None and _is_real_view:
        _note_view(ctx, path)
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
        # Duplicate TOP-LEVEL def/class (ckpt-178): the adjacent check above misses two defs of
        # the SAME name FAR APART — c580 added `def widened_hostnames` at line 77 AND line 546,
        # the 2nd silently shadowing the 1st so half the logic was dead. Flag a module-level
        # name this edit made appear 2+ times when it didn't before (pre-existing dups pass).
        import ast as _ast
        def _toplevel_dupes(src):
            try:
                _t = _ast.parse(src)
            except Exception:
                return {}
            _seen = {}
            for _node in _t.body:
                if isinstance(_node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                    # Skip decorator-driven redefinitions (@typing.overload stub chains,
                    # functools.singledispatch `@fn.register / def _`) and the throwaway name `_`
                    # — these LEGITIMATELY repeat a top-level name and are NOT dead-code shadows
                    # (bughunt #7). Only undecorated, real-name redefinitions count as a dupe.
                    if getattr(_node, "decorator_list", None) or _node.name == "_":
                        continue
                    _seen[_node.name] = _seen.get(_node.name, 0) + 1
            return {k: c for k, c in _seen.items() if c > 1}
        _new_dd = _toplevel_dupes(new_content)
        _old_dd = _toplevel_dupes(before) if before else {}
        _newly = [k for k, c in _new_dd.items() if c > _old_dd.get(k, 0)]
        if _newly:
            _nm = _newly[0]
            return (f"✗ {tool} NOT applied to {path}: your edit defines `{_nm}` {_new_dd[_nm]}× "
                    f"at the top level of {path} — a SECOND `def {_nm}`/`class {_nm}` silently "
                    f"shadows the first, so half your logic is dead code. Keep ONE definition "
                    f"(edit the existing one in place, don't add a new copy); {resend}.")
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
    # INDENT|code: declare indent as a number, harness re-emits the spaces (idempotent;
    # same fix as edit_file). new_content lines may be `N|code` or literal real-space code.
    if new:
        new = "\n".join(_expand_indent_lines(new.split("\n")))
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
    _cot_reject = _check_edit_cot(args, before or "", is_insert=False)
    if _cot_reject:
        return _cot_reject
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
        if isinstance(ctx.get("viewed_versions"), dict):
            ctx["viewed_versions"][path] = result[path]
        ctx.setdefault("files_changed", set()).add(path)
        # A big-file edit shifts the line numbers at/below it, so EVERY range we previously
        # served for this file may now be stale. Drop the served-range cache so a range
        # re-read serves FRESH current content instead of a stale "you already read a-b"
        # short-circuit (the a26 reject-loop→timeout root cause — the coder copied pre-edit
        # lines that no longer matched). keep-validation still works via view_at (set just below).
        ctx.get("_served_ranges", {}).pop(path, None)
        _note_view(ctx, path)   # the diff + unchanged remainder = a current view
        n = result[path].count("\n") + 1
        from core.edit_diff import render_diff
        _dbase = ctx.get("_first_seen", {}).get(path, before)   # step-start baseline (point 2)
        _diff = render_diff(_dbase or "", result[path], path)
        _when = _view_stamp(ctx)
        return (f"✓ Applied: {path} lines {s_i}-{e_i} replaced — change made at {_when}. "
                f"The diff below is the CUMULATIVE change to {path} since the START of this step "
                f"(the file you were given / first read). That start-state + this diff = {path}'s "
                f"CURRENT, live state — TRUST it, your view is NOT stale. Do NOT read_file {path} "
                f"again; for your next edit write `old` as `INDENT|code` (or copy a line from your "
                f"read view) — do NOT paste a diff row. Only for a part of {path} you have NOT seen, "
                f"read it with a start_line/end_line range.\n"
                + (_diff or "(no visible line change)"))
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
            _note_view(ctx, k)   # keep view_at in step with the suffix-resolved key
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


def _actual_region_hint(cur_lines, start_line, old_list) -> str:
    """RECOVERY for the #1 reject-loop cause (audit 2026-06-03): a big file is shown
    as a SKELETON (signatures, no bodies), the coder is told to trust its view, so it
    builds `old` for a function body it never actually saw → `_locate_block` finds
    nothing → reject → it re-sends the same imagined `old` → fallover with only a
    trivial top-level line landed (e.g. just an `import`). Instead of dead-ending,
    SHOW the coder the real current lines at the intended site so it can copy a valid
    `old`. Renders as `LINENO ⇥INDENT|code` (the read-view form). '' if we can't localize."""
    if not cur_lines:
        return ""
    if not isinstance(old_list, list):          # defensive (ckpt-149): tolerate non-list `old`
        old_list = [] if old_list is None else [old_list]
    n = len([o for o in old_list if str(o).strip()]) or 1
    anchor = None
    try:
        sl = int(start_line)
        if 1 <= sl <= len(cur_lines):
            anchor = sl - 1
    except (TypeError, ValueError):
        anchor = None
    if anchor is None and old_list:        # no usable start_line → fuzzy-locate the first old line
        import difflib
        first = next((str(o).strip() for o in old_list if str(o).strip()), "")
        if first:
            best, bi = 0.0, None
            for idx, ln in enumerate(cur_lines):
                r = difflib.SequenceMatcher(None, ln.strip(), first).ratio()
                if r > best:
                    best, bi = r, idx
            if best >= 0.6:
                anchor = bi
    if anchor is None:
        return ""
    lo = max(0, anchor - 2); hi = min(len(cur_lines), anchor + n + 2)
    rows = []
    for idx in range(lo, hi):
        ln = cur_lines[idx]; ind = len(ln) - len(ln.lstrip(' '))
        rows.append(f"{idx+1} ⇥{ind}|{' ' * ind}{ln.strip()}")
    return ("\n   ↪ The ACTUAL current lines at that spot are below — copy your `old` "
            "VERBATIM from these (as INDENT|code), don't reconstruct it from memory:\n"
            + "\n".join(rows))


def _old_not_found_msg(i: int, path: str, ctx: dict, old_raw=None,
                       cur_lines=None, start_line=None) -> str:
    """`old` matched nowhere in the file. Two very different causes — name the
    likely one. If we've already edited this file, the model is almost certainly
    copying `old` from a STALE earlier read (the file moved under it); telling it
    'wrong file' would wrongly send it away from the right file. If we HAVEN'T
    touched it, the symbol probably lives elsewhere (the f327e65d wrong-file bug).
    When we can localize the intended site, we APPEND the real current lines so a
    skeleton-only view isn't a dead end (see _actual_region_hint)."""
    # Defensive (ckpt-149): old_raw is the RAW `old` straight from the call — may be
    # a non-list (int/str/None) for a malformed edit; coerce so we never raise here.
    if isinstance(old_raw, str):
        old_raw = [old_raw]
    elif not isinstance(old_raw, list):
        old_raw = [] if old_raw is None else [str(old_raw)]
    # Strict-input cue: if `old` looks like it was pasted from a DIFF's +/- row
    # (`N:+ ` / `N:- `), that gutter is NOT editable input and isn't silently stripped —
    # tell the coder the canonical form instead of leaving it to guess. (audit pass-4.)
    if old_raw and any(_LOOKS_COPIED_GUTTER_RE.match(str(o)) for o in old_raw):
        return (f"✗ edit_file hunk #{i}: your `old` looks like a line copied from a DIFF "
                f"(it starts with `LINENO:+ ` / `LINENO:- `). A diff row is not editable "
                f"input. For `old`, copy the line from the read VIEW of {path} VERBATIM — it "
                f"shows `LINENO ⇥INDENT|code` (e.g. `286 ⇥4|    def foo`); keep that whole "
                f"prefix. Write `new` lines as `INDENT|code` (the indent NUMBER, a pipe, then "
                f"the code — e.g. `8|return x`); the harness applies the indent from the number.")
    if path in ctx.get("files_changed", set()):
        _when = ctx.get("view_at", {}).get(path, "your last edit")
        return (f"✗ edit_file hunk #{i}: those `old` line(s) aren't in {path} as it is NOW. "
                f"You already EDITED {path} ({_when}), so this `old` was copied from a view "
                f"taken BEFORE that edit. Fix it WITHOUT re-reading the whole file: copy "
                f"`old` from the line as it reads NOW — the LATEST diff above shows the "
                f"current text (use it, but copy the code, don't paste the raw `:+/-` diff "
                f"row); if the line is in a part you have NOT seen since the edit, read_file "
                f"{path} with that exact start_line/end_line range. Don't reuse stale line text."
                + _actual_region_hint(cur_lines, start_line, old_raw))
    return (f"✗ edit_file hunk #{i}: the `old` line(s) are NOT in {path}. The file is "
            f"UNCHANGED and your view is STILL CURRENT — do NOT re-read. Two causes: (1) Your "
            f"`old` doesn't match {path} character-for-character — copy it again EXACTLY from "
            f"the view you ALREADY have (keep the full `LINENO ⇥INDENT|` prefix) and resend. "
            f"(2) WRONG FILE — the symbol may live elsewhere; use search_text (or find_refs) to "
            f"locate it and edit THAT file. (read_file a start_line/end_line range ONLY for a region of {path} you've "
            f"genuinely never seen — never re-dump the whole file.)"
            + _actual_region_hint(cur_lines, start_line, old_raw))


# Indentation is declared as a NUMBER; the harness re-emits the spaces. The model can
# COPY-PASTE any of the forms it actually sees and they all resolve correctly:
#   INDENT|code              the documented edit form          e.g.  4|def foo
#   LINENO ⇥INDENT|code      a line copied verbatim from the read view  e.g.  286 ⇥4|    def foo
#   LINENO:[+|-]code         a line copied from a post-edit diff (real spaces)  e.g.  12:+    return 2
# The INDENT NUMBER (when present) is authoritative — any visible leading spaces in the
# copied code are stripped and re-applied from the number, so the coder cannot mis-indent.
# A plain real-space line (no prefix) is taken literally (back-compat). The `LINENO:` and
# diff-marker strips are anchored so they CANNOT corrupt real code: they only fire when a
# `\d+\|` (indent) or `\d+:[+-]` (diff) shape follows — a normal `key: value` / `5: x` line
# never matches.
# CANONICAL edit forms ONLY — both are UNAMBIGUOUS (the `\d+\|` / `\d+:\d+\|` shape does not
# collide with real code/YAML/config):
_INDENT_LINE_RE = re.compile(r'^(\d+)\|(.*)$')          # INDENT|code            (the write form)
# A copied view line. ckpt-143 naturalized the gutter to `LINENO ⇥INDENT|code`
# (the ⇥ tab-glyph marks the indent); we still accept the old `LINENO:INDENT|`
# colon form so a stale paste never silently fails to match.
_VIEW_LINE_RE   = re.compile(r'^\d+\s*[:⇥](\d+)\|(.*)$')  # LINENO ⇥INDENT|code  (copied view line)
# ckpt-144: a copied view line carries its LINENO up front. We pull it out to ANCHOR
# the edit by BOTH line number AND content — the number locates (and disambiguates
# when the `old` text repeats), the content-verified applier still self-corrects if
# the number is stale. A bare `INDENT|code` write-form (one number) has no lineno → None.
_VIEW_LINENO_RE = re.compile(r'^\s*(\d+)\s*[:⇥]\d+\|')


def _view_lineno(raw_lines: list):
    """The 1-based LINENO from the first copied view line (`LINENO ⇥INDENT|…`), or
    None if the model wrote the bare `INDENT|code` form (no line number to anchor on)."""
    for ln in raw_lines:
        m = _VIEW_LINENO_RE.match(str(ln))
        if m:
            return int(m.group(1))
    return None
# blast-radius annotation a def line may carry: ` |appears N (#hex...)`. Require the
# `(#hex` shape the annotation ALWAYS emits, so a real code line like
# `raise ValueError("x |appears 3 times")` is NOT truncated.
_APPEARS_TAIL_RE = re.compile(r'\s*\|appears \d+ \(#[0-9a-fA-F][^)]*\)\s*$')
# A line that looks like it was copied from a DIFF's +/- row (`N:+ ` / `N:- `). We
# deliberately do NOT silently transform these — but if one is used as `old` and the match
# fails, the reject TELLS the coder to re-send as INDENT|code. Detection drives that message
# ONLY; it never rewrites the line. We match ONLY `+`/`-` (NOT the context `N:  ` form, which
# is shape-ambiguous with YAML `443:  desc` and would mis-fire). (audit pass-4: strict.)
_LOOKS_COPIED_GUTTER_RE = re.compile(r'^\s*\d+:[+\-] ')


def _expand_indent_lines(lines: list, trust_spaces: bool = False) -> list:
    """Resolve every old/new/content line to its real source form. The model declares indent
    by NUMBER (`INDENT|code`) — or copies a view line verbatim (`LINENO:INDENT|code`) — and the
    harness applies the spaces, so the coder never types (and never drops) leading spaces (the
    col-0 dedent root cause). `trust_spaces` (ckpt-155) is a PARSE-FAIL RETRY mode: when the
    code part ALSO carries leading spaces that disagree with the NUMBER, use the typed spaces
    instead of the number. It is used ONLY as a fallback after the number-based expansion failed
    to parse — so it fixes the `0|    x` dual-channel slip (number 0 but 4 spaces typed → indent
    4) WITHOUT touching the default (number-authoritative) path that an intentional dedent
    `4|        x` relies on (that one parses, so the retry never fires). We transform ONLY these
    two UNAMBIGUOUS forms; everything else is
    taken LITERALLY (so a real YAML/code line is never corrupted by a guessed transform). A
    leftover diff/whitespace gutter is NOT stripped — it simply won't match, and the reject
    explains how to re-send (see _do_edit)."""
    # Defensive (ckpt-149): a caller may hand a non-list (int/str/None) or a list with
    # non-string items; coerce so we never raise on `for ln in lines` / regex on a non-str.
    if isinstance(lines, str):
        lines = lines.split("\n")
    elif not isinstance(lines, list):
        lines = [] if lines is None else [str(lines)]
    out = []
    for ln in lines:
        if not isinstance(ln, str):
            ln = str(ln)
        # A def line in the view may carry a ` |appears N (#tag)` blast-radius annotation
        # (the harness's own marker, `(#hex)`-guarded); strip it so copying that line matches.
        ln = _APPEARS_TAIL_RE.sub('', ln)
        m = _INDENT_LINE_RE.match(ln) or _VIEW_LINE_RE.match(ln)
        if m:                                    # INDENT|code or LINENO:INDENT|code
            # STRIP a stray `⇥`/U+21E5 from the CODE part (ckpt-178): a weak coder copying the
            # view's `LINENO ⇥INDENT|` gutter sloppily can leave a ⇥ inside the code (e.g.
            # `0|⇥def f` → `\d+\|` matches `0|`, the ⇥ rides along in `code`). ⇥ is the harness's
            # display glyph, NEVER valid in Python source, so a leftover is always a leaked marker
            # — it caused an endless `SyntaxError: invalid character '⇥'` reject-loop (c580 via
            # mistral). Strip it BEFORE lstrip so the spaces it shielded aren't kept as indent.
            code = m.group(2).replace("⇥", "")
            ind = int(m.group(1))
            if trust_spaces:
                typed = len(code) - len(code.lstrip(' '))
                if typed > 0:                    # typed spaces disagree → trust them (retry mode)
                    ind = typed
            out.append(' ' * ind + code.lstrip(' '))
        else:
            out.append(ln.replace("⇥", ""))      # literal — drop a leaked marker, else verbatim
    return out


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
    path, _pe = _str_or_err(args, "path", "edit_file")
    if _pe:
        return _pe
    hunks = args.get("hunks")
    # ckpt-151: the model-facing BATCH form is `edits` = [{old, new}, ...] — the coder's
    # whole "edit section" for this file in ONE call. All edits anchor to the SAME current
    # view and apply together (so line numbers can't shift between them → no mid-edit
    # re-read churn), and the call returns ONE consolidated diff. `hunks` is the internal
    # alias; a single {old, new} is the shorthand. (all funnel to the same applier below.)
    if not hunks:
        _edits = args.get("edits")
        if isinstance(_edits, list) and _edits:
            hunks = _edits
    if not hunks:
        _o = args.get("old"); _n = args.get("new")
        if _o is not None or _n is not None:
            _as = lambda v: v if isinstance(v, list) else ([] if v in (None, "") else str(v).split("\n"))
            # carry top-level start_line into the synthesized hunk (ckpt-199): without it, the
            # natural single insert `edit_file(path, old=[], new=[...], start_line=N)` lost its
            # anchor → "old is empty" reject, while the identical nested `edits=[{...,start_line}]`
            # inserted cleanly. (Recurring empty-old insert failure, e.g. f327.)
            hunks = [{"old": _as(_o), "new": _as(_n), "start_line": args.get("start_line")}]
    if not path:
        return "✗ edit_file needs a path."
    if not hunks or not isinstance(hunks, list):
        return ("✗ edit_file needs `old` (the exact existing block as INDENT|code lines, "
                "copied verbatim from your read) and `new` (the replacement; [] to delete).")

    cur = ctx["file_contents"].get(path)
    if cur is None:
        sb0 = ctx.get("sandbox")
        if sb0 is not None:
            cur = sb0.load_file(path)
            if cur is not None:
                # persist the sandbox-loaded content (ckpt-201, bughunt #17): otherwise `before`
                # / `_before_all` stay empty for this path → _build_and_apply resets file_contents
                # without it → a valid edit on a never-read-but-existing file falsely rejects.
                ctx.setdefault("file_contents", {})[path] = cur
    if cur is None:
        return (f"✗ edit_file: {path} is not in context — read it with read_file "
                f"first, then copy the exact `old` lines (and their start_line) from "
                f"that view.")
    # Baseline for the re-read diff: capture the PRE-edit content the first time we touch
    # this file (so a later read_file can show what changed since the coder first saw it). (ckpt-150.)
    ctx.setdefault("_first_seen", {}).setdefault(path, cur)
    cur_lines = cur.split("\n")

    # EDIT-COT grounding gate (flag-gated): reject unless the edit carries grounded
    # goal/traced/check — a guess can't quote a real line. Pure-insert = all hunks
    # have empty `old` (no existing line to quote → relax the quote-check).
    _hk = args.get("hunks") if isinstance(args.get("hunks"), list) else []
    _is_insert = bool(_hk) and all(
        not any(str(o).strip()
                for o in (h.get("old") if isinstance(h.get("old"), list) else []))
        for h in _hk if isinstance(h, dict))
    _cot_reject = _check_edit_cot(args, cur, _is_insert)
    if _cot_reject:
        return _cot_reject

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
        # INDENT|code: the model declares indentation as a NUMBER (the view shows it),
        # the harness re-emits that many spaces — so the coder never types (and never
        # drops) leading spaces. Idempotent: copying the view's `4|    def foo` gives
        # the same result as `4|def foo`. (root-cause fix, 2026-06-02.)
        _raw_old = _as_list(h.get("old"))
        old_list = _expand_indent_lines(_raw_old)
        new_raw = _as_list(h.get("new"))          # kept un-expanded; expanded at block-build (parse-fail retry)
        new_list = _expand_indent_lines(new_raw)
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
                new_raw = [anchor] + new_raw     # keep raw in step for parse-fail re-expansion
            else:
                return (f"✗ edit_file hunk #{i}: `old` is empty. `old` must hold the EXACT "
                        f"existing line(s) you're changing — copy them from your view (keep "
                        f"the `LINENO ⇥INDENT|` prefix). To INSERT new code, put a real "
                        f"adjacent line in BOTH `old` and `new`, with your new line(s) next "
                        f"to it in `new` (that anchors the insert).")
        sl = h.get("start_line")
        if sl is None:
            # ckpt-144: anchor on BOTH lineno AND content. If the model copied the view
            # line(s) verbatim, `old` carries the LINENO (`286 ⇥4|…`) — use it as the
            # anchor so a repeated `old` lands on the RIGHT occurrence. The applier below
            # is content-verified, so a stale number self-corrects; the number only
            # disambiguates. (bare `INDENT|code` writes have no lineno → content-only.)
            sl = _view_lineno(_raw_old)
        if sl is None:
            # No number anywhere — resolve from content; reject if it's not unique.
            sl, n_hits = _locate_block(cur_lines, old_list)
            if n_hits == 0:
                return _old_not_found_msg(i, path, ctx, h.get("old"),
                                          cur_lines=cur_lines, start_line=None)
            if n_hits > 1:
                return (f"✗ edit_file hunk #{i}: `old` appears {n_hits} times in {path} "
                        f"— copy the view line(s) VERBATIM, keeping the `LINENO ⇥INDENT|` "
                        f"prefix so the line number picks the RIGHT occurrence (or include "
                        f"more surrounding lines in `old` to make it unique).")
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
                return _old_not_found_msg(i, path, ctx, h.get("old"),
                                          cur_lines=cur_lines, start_line=sl)
        resolved.append((sl, old_list, new_raw))

    # The numbered [edit] applier requires lines top-to-bottom in FILE order.
    # The model may send hunks in any order, so sort by start_line ourselves
    # (stable, so same-line hunks keep their given order) instead of rejecting
    # with "edit lines out of order" and making it retry — that retry-loop is a
    # top cause of round pile-up.
    resolved.sort(key=lambda t: t[0])
    # OVERLAP handling (ckpt-156 → ckpt-158): all hunks anchor on the file as it is BEFORE
    # any of them apply, so two hunks SHARING only UNCHANGED context lines are harmless —
    # the ~1-2-context-line bracketing makes nearby hunks overlap like this all the time.
    # MERGE such hunks (dedupe the shared context) so they apply as one contiguous block
    # (the applier's flat op-list otherwise chokes on the duplicated line). REJECT only the
    # genuine conflict: two hunks that both CHANGE the same line (nonsensical) — or one fully
    # nested in another. (User: "it just needs to not overlap a CHANGED line.")
    _merged = []
    for _h in resolved:
        if not _merged:
            _merged.append(list(_h)); continue
        _sA, _oA, _nA_raw = _merged[-1]
        _sB, _oB, _nB_raw = _h
        _endA = _sA + len(_oA) - 1
        if _sB > _endA:
            _merged.append(list(_h)); continue                 # disjoint → separate hunk
        _endB = _sB + len(_oB) - 1
        _ov = _endA - _sB + 1                                   # shared old-line count
        _nA = _expand_indent_lines(_nA_raw); _nB = _expand_indent_lines(_nB_raw)
        _shared = _oA[-_ov:]
        _conflict = (
            _endB <= _endA                                      # B fully nested in A
            or _oB[:_ov] != _shared                             # shared old text disagrees
            or len(_nA) < _ov or _nA[-_ov:] != _shared          # A CHANGED the shared region
            or len(_nB) < _ov or _nB[:_ov] != _shared)          # B CHANGED the shared region
        if _conflict:
            return (f"✗ edit_file: two of your hunks both change the SAME line(s) around line "
                    f"{_sB} — that's a real conflict (overlapping CHANGES can't both apply). "
                    f"Make ONE hunk for that region: `old` = the contiguous block from line "
                    f"{_sA} to line {max(_endA, _endB)} (copy those real lines verbatim), with "
                    f"all the changes in its `new`. (Sharing only UNCHANGED context lines is "
                    f"fine — this fires only because a CHANGED line is shared.)")
        # context-only overlap → dedupe the shared lines and fuse into one hunk
        _merged[-1] = [_sA, _oA + _oB[_ov:], _nA_raw + _nB_raw[_ov:]]
    resolved = _merged
    from workflows.code import _extract_code_blocks, _apply_extracted_code
    before = ctx["file_contents"].get(path)
    _before_all = dict(ctx["file_contents"])

    def _build_and_apply(trust_spaces: bool):
        edit_lines = []
        for sl, old_list, new_raw in resolved:
            new_list = _expand_indent_lines(new_raw, trust_spaces=trust_spaces)
            for j, o in enumerate(old_list):
                edit_lines.append(f"{sl + j}:-{o}")
            for nw in new_list:
                edit_lines.append(f"+{nw}")
        block = (f"=== EDIT: {path} ===\n[edit]\n" + "\n".join(edit_lines)
                 + "\n[/edit]\n=== END EDIT ===")
        # Start every attempt from the SAME pre-edit state (a rejected attempt
        # reverts itself, but reset explicitly so the retry is clean).
        ctx["file_contents"].clear(); ctx["file_contents"].update(_before_all)
        _ext = _extract_code_blocks(block)
        _res, _m, _a, _sk = _apply_extracted_code(
            _ext, ctx["file_contents"], ctx.get("sandbox"),
            viewed_versions=ctx.get("viewed_versions"))
        return _res, list(_ext.get("malformed_edits", [])) + list(_sk)

    result, skips = _build_and_apply(False)

    # PARSE-FAIL INDENT RETRY (ckpt-155): if the number-based expansion produced a
    # file that does NOT parse AND some `new` line carried typed leading spaces that
    # DISAGREE with its `INDENT|` number (the `0|    x` dual-channel slip — number 0
    # but 4 spaces typed), retry once trusting the typed spaces. Fires ONLY on a parse
    # reject (so it can never alter a passing edit) and ONLY when the disagreement
    # exists (so an unrelated SyntaxError isn't masked). An intentional dedent
    # `4|        x` parses, so it never reaches this path — number stays authoritative.
    _indent_autofixed = False
    from workflows.code import _check_syntax as _csyn
    def _result_is_good(_res):
        # "good" = applied AND parses. Uses _check_syntax (tokenize+compile) — NOT just
        # ast.parse — because ast.parse is lenient about a col-0 `return`/`yield`/`break`
        # that compile() rejects: the applier's own ast-gate lets such a file INTO `_res`,
        # so a plain "path not in _res" check would MISS the very `0|    return x` slip the
        # retry exists to fix (audit ckpt-157). A pre-existing breakage (before also fails)
        # is not this edit's fault → treated as good so we don't loop on it.
        if path not in _res:
            return False
        if before is not None and _res[path] == before:
            return True
        if not (path.endswith(".py") or path.endswith(".pyi")):
            return True            # non-Python: no parse gate applies → applied == good
        ok = _csyn(path, _res[path])[0]
        if not ok and before is not None and not _csyn(path, before)[0]:
            return True
        return ok
    def _has_indent_disagreement():
        for _sl, _ol, new_raw in resolved:
            for ln in new_raw:
                m = _INDENT_LINE_RE.match(str(ln)) or _VIEW_LINE_RE.match(str(ln))
                if m:
                    code = m.group(2); typed = len(code) - len(code.lstrip(' '))
                    if typed > 0 and typed != int(m.group(1)):
                        return True
        return False
    if not _result_is_good(result) and _has_indent_disagreement():
        _r2, _s2 = _build_and_apply(True)
        if _result_is_good(_r2):             # retry now parses (tokenize+compile)
            result, skips, _indent_autofixed = _r2, _s2, True

    # Reconcile ctx["file_contents"] with the CHOSEN `result`. A failed retry leaves the
    # dict holding the retry's content while `result` may be the first attempt — and an
    # ast-accepted-but-compile-bad result would otherwise leave BROKEN content in the dict,
    # making the reject's "file UNCHANGED — your view is current" a LIE (audit ckpt-157,
    # pre-existing leak amplified by ckpt-154's don't-re-read wording). Reset to the pre-edit
    # snapshot, then re-apply ONLY a good result. A non-good `result` still flows to the
    # gate/skip logic below to produce the precise reject, but the file is genuinely unchanged.
    ctx["file_contents"].clear(); ctx["file_contents"].update(_before_all)
    if _result_is_good(result) and path in result:
        ctx["file_contents"][path] = result[path]
        _sb_rec = ctx.get("sandbox")
        if _sb_rec is not None:
            try: _sb_rec.write_file(path, result[path])
            except Exception: pass
    elif ctx.get("sandbox") is not None and before is not None:
        try: ctx["sandbox"].write_file(path, before)   # ensure sandbox not left broken
        except Exception: pass

    _seen: set = set()
    skips = [x for x in skips
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
            # REVERT (ckpt-178): a PARSEABLE-but-bad result (unreachable / duplicate / dup-def)
            # was tentatively written to file_contents + sandbox just above, so a bare reject
            # would leave the file CHANGED while telling the coder "the file is unchanged" — a
            # lie that makes it edit blind. Reset to the pre-edit snapshot so "unchanged" is TRUE.
            ctx["file_contents"].clear(); ctx["file_contents"].update(_before_all)
            _sbr = ctx.get("sandbox")
            if _sbr is not None and before is not None:
                try: _sbr.write_file(path, before)
                except Exception: pass
            # WHOLE-BLOCK route (ckpt-137; replace_lines retired from CODER_TOOLS — the
            # message now routes to edit_file itself). The gate (unreachable / duplicate /
            # parse) fires almost only on a WHOLE-BLOCK rewrite where the coder's hunk
            # stranded the old `return` or re-emitted the anchor — so hand the coder the
            # precise "put the whole contiguous block in ONE hunk" edit_file move below
            # instead of letting it re-loop fragments.
            _old_total = sum(len(o) for _s, o, _n in resolved)
            _is_block = _old_total >= 4 or any(
                re.match(r'\s*(def|class|async def)\b', str(o))
                for _s, ol, _n in resolved for o in ol)
            if _is_block and resolved:
                _start = resolved[0][0]
                _end = max(s + len(o) - 1 for s, o, _n in resolved)
                _gate += (f"\n↪ Your edit left part of THIS block behind (that's the reject "
                          f"above). For THIS one block, put its whole CONTIGUOUS span in `old` "
                          f"— every line from {_start} to {_end} — and the corrected block in "
                          f"`new`. (Distant changes elsewhere stay as their OWN separate hunks "
                          f"in the `edits` list — don't merge them into this one.)")
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
        # A big-file edit shifts the line numbers at/below it, so EVERY range we previously
        # served for this file may now be stale. Drop the served-range cache so a range
        # re-read serves FRESH current content instead of a stale "you already read a-b"
        # short-circuit (the a26 reject-loop→timeout root cause — the coder copied pre-edit
        # lines that no longer matched). keep-validation still works via view_at (set just below).
        ctx.get("_served_ranges", {}).pop(path, None)
        _note_view(ctx, path)   # the diff + unchanged remainder = a current view
        n = result[path].count("\n") + 1
        # Hand back the before/after diff with the file's CURRENT line numbers, so
        # the coder sees exactly what changed AND has fresh, correct numbers to
        # anchor its next edit — instead of re-reading (the re-read/stale-`old`
        # churn was the #1 cause of round pile-up: f327 burned ~16-33 edits nibbling
        # blind). The coder no longer needs read_file between consecutive edits.
        from core.edit_diff import render_diff
        _dbase = ctx.get("_first_seen", {}).get(path, before)   # step-start baseline (point 2)
        _diff = render_diff(_dbase or "", result[path], path)
        _when = _view_stamp(ctx)
        _fixnote = (" ⚠ Your `INDENT|` number disagreed with the spaces you typed and "
                    "wouldn't parse, so I used your typed spaces instead — verify the new "
                    "line(s) are at the right scope in the diff." if _indent_autofixed else "")
        return (f"✓ Applied {len(hunks)} edit(s) to {path} — {_when}.{_fixnote} The diff below is "
                f"the CUMULATIVE change to {path} since the START of this step (the file you were "
                f"given / first read) — EVERY edit you've made to it this step, against ONE stable "
                f"baseline. That start-state + this diff = {path}'s CURRENT, live state; everything "
                f"NOT in the diff is UNCHANGED. TRUST that — your view is NOT stale (your `old` was "
                f"anchored on its line number AND content, so a shifted view self-corrects). For "
                f"your next change here, COPY the relevant line from your view/this diff VERBATIM as "
                f"`old` (keep its `LINENO ⇥INDENT|` so it anchors); write `new` as `INDENT|code`. Do "
                f"NOT paste a raw `LINENO:+/- ` diff row. You do NOT need to read_file again — your "
                f"start-state + this diff IS {path}'s current state (read a range only for a region "
                f"you've truly never seen).\n"
                + (_diff or "(no visible line change)"))

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
            ctx.get("_served_ranges", {}).pop(k, None)  # ckpt-200 #5: edit shifted lines → drop
            # the resolved key's stale revealed-ranges too (the main-path ckpt-194 fix missed this
            # suffix-resolved net), else a range re-read short-circuits on pre-edit content.
            _note_view(ctx, k)   # keep view_at in step with the suffix-resolved key
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
    path, _pe = _str_or_err(args, "path", "create_file")
    if _pe:
        return _pe
    # PATH CONTAINMENT + IMPORT-SHIM GUARD (ckpt-166/167). A new file MUST live inside
    # the project AND must not be an import-shim. The weak coder, after a run_code
    # ModuleNotFoundError on a missing dep/framework module (web, infogami, yaml, PyQt5…),
    # tries to "satisfy" the import by manufacturing a stub package (web/__init__.py,
    # infogami/…) or a root module (yaml.py). That junk pollutes the patch and breaks
    # `git apply` in the real env — it regressed a PASSING instance (4a5d2a7d ✓→broken,
    # ckpt-166: shipped web/__init__.py + an infogami/ create → apply_failed). Block it.
    from os.path import isabs, normpath, exists, join
    import sys as _sys
    _pn = path.replace("\\", "/").strip()
    # (1) escape — absolute or ../ outside the project
    if isabs(_pn) or _pn.startswith("../") or "/../" in _pn or normpath(_pn).startswith(".."):
        return (f"✗ create_file refused: {path} is outside the project. Use a project-relative "
                f"path (no leading '/' or '../').")
    _segs = [s for s in _pn.split("/") if s and s != "."]
    _topseg = _segs[0] if _segs else _pn
    _topname = _topseg[:-3] if _topseg.endswith(".py") else _topseg
    # project/sandbox base — to tell a NEW top-level package from an existing one
    _sb0 = ctx.get("sandbox")
    _base = (getattr(_sb0, "sandbox_dir", "") or "") if _sb0 is not None else ""
    _base = _base or (ctx.get("project_root") or "")
    _base_ok = bool(_base) and exists(_base)
    _top_exists = _base_ok and exists(join(_base, _topseg))
    # (2) DYNAMIC: the coder just failed to import this top module in run_code → it is
    # shimming it. Highest-confidence signal, near-zero false positive.
    if _topname in (ctx.get("_failed_imports") or set()):
        return (f"✗ create_file refused: `{_pn}` would shim `{_topname}`, which run_code just "
                f"failed to import. That ModuleNotFoundError is an ENVIRONMENT limit of the smoke "
                f"sandbox, NOT a missing file — the REAL test env has `{_topname}`. Do NOT create "
                f"it; edit your assigned target file instead.")
    # (3) STATIC: a module (or its top package) shadowing stdlib/common deps that isn't
    # already in the repo → a shim even if run_code was never called.
    _shadow = set(getattr(_sys, "stdlib_module_names", set())) | {
        "yaml", "jinja2", "jinja", "PyQt5", "PyQt6", "web", "django", "numpy", "scipy",
        "pandas", "requests", "six", "pytest", "setuptools", "lxml", "sqlalchemy", "resolvelib"}
    if _topname in _shadow and not _top_exists:
        return (f"✗ create_file refused: `{_pn}` would create `{_topseg}`, shadowing the "
                f"`{_topname}` module (not present in the repo). This is the missing-dependency "
                f"shim rabbit-hole, never a real fix. Edit your assigned target file instead.")
    # (4) NEW TOP-LEVEL PACKAGE: a NESTED new file whose top-level dir is NOT in the repo
    # (e.g. web/__init__.py with no web/ package). A fix adds files to EXISTING packages;
    # a brand-new top-level package is the shim/scratch pattern. Only enforced when the
    # project base is visible (else we can't distinguish new from existing — e.g. tests).
    if _base_ok and len(_segs) > 1 and not _top_exists:
        return (f"✗ create_file refused: `{_pn}` would create a NEW top-level package "
                f"`{_topseg}/` that isn't in the project. A fix adds files to EXISTING packages; "
                f"a brand-new top-level package is the import-shim/scratch pattern. Put a genuinely "
                f"new module inside an existing package, or edit your assigned target file.")
    # (5) SCRATCH TEST (ckpt-172): a NEW root-level test_*.py / *_test.py is the coder
    # writing its OWN verification test. It is never graded (the real tests are separate
    # and hidden) and it pollutes the shipped patch (b748edea leaked a root test_multipart.py).
    # Steer to run_code instead of committing a throwaway test.
    _bn = _segs[-1] if _segs else _pn
    if len(_segs) == 1 and _bn.endswith(".py") and (_bn.startswith("test_") or _bn[:-3].endswith("_test")):
        return (f"✗ create_file refused: `{_pn}` is a scratch verification test at the repo root. "
                f"Your own test is NOT graded (the real tests are separate) and it pollutes the "
                f"patch. To CHECK your edit, use run_code with an inline `python -c \"...\"` "
                f"assertion instead of committing a test file.")
    content = args.get("content", "")
    if not isinstance(content, str):     # tolerate a non-string content (list/int/None)
        content = "" if content is None else str(content)
    if not str(content).strip():
        # Empty/whitespace content would write a 0-byte file and report "✓ Created
        # (1 lines)" — a silent no-op the model can't tell from real success.
        return (f"✗ create_file: no content for {path}. Pass the full file body in "
                f"`content` (the complete code for the new file).")
    content = str(content)
    # NOTE (pass-4 F6, deferred): new-file content is still N|-expanded by the shared
    # `=== FILE:` applier downstream, so a literal data line shaped `123|x` is rewritten to
    # 123 spaces. Rare (needs a new file with digit-pipe data lines); a proper fix means
    # making the shared new-file applier verbatim — tracked, not done here.
    # grounding gate (new file → no line to quote; ground in the spec/interface)
    _cot_reject = _check_edit_cot(args, "", is_insert=True)
    if _cot_reject:
        return _cot_reject
    existing = ctx["file_contents"].get(path)
    if existing is None and ctx.get("sandbox") is not None:
        existing = ctx["sandbox"].load_file(path)
    if existing:
        n = existing.count("\n") + 1
        return (f"✗ create_file: {path} already exists ({n} lines). Use edit_file "
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
        _note_view(ctx, path)   # you just wrote it — it's in context, no read needed
        n = produced.count("\n") + 1
        return f"✓ Created: {path} ({n} lines)."
    reason = " | ".join(str(x).strip().lstrip("-").strip() for x in skips) or \
        "no file produced (content may be empty or malformed)"
    return f"✗ create_file NOT applied for {path}: {reason}"


def _do_list_dir(args: dict, ctx: dict) -> str:
    """Expandable filesystem tree (the planner's view): the IMMEDIATE children of a folder —
    sub-dirs (with recursive file counts) then files (with LINE counts). path='' / omitted →
    the project top level; a folder path expands it one level. Real filesystem (exact,
    copy-paste-ready paths). Bounded; never raises."""
    import os as _os
    rel = args.get("path")
    if rel is None:
        rel = args.get("dir") or args.get("folder") or ""
    if not isinstance(rel, str):
        rel = ""
    rel = rel.replace("\\", "/").strip().strip("/")
    base = _repo_base(ctx)
    if not base or not _os.path.isdir(base):
        return "✗ list_dir: no project root is available for this run."
    target = _os.path.normpath(_os.path.join(base, rel)) if rel else _os.path.normpath(base)
    _rb, _rt = _os.path.realpath(base), _os.path.realpath(target)
    if _rt != _rb and not _rt.startswith(_rb + _os.sep):   # separator boundary: 'bar2' must not match 'bar'
        return f"✗ list_dir: '{rel}' escapes the project root — refused. Use a repo-relative folder."
    if not _os.path.exists(target):
        return (f"✗ list_dir: no such folder '{rel}'. Start at the top (list_dir with no path) and "
                f"expand one folder at a time.")
    if not _os.path.isdir(target):
        return f"✗ list_dir: '{rel}' is a FILE, not a folder — read it with read_file."
    try:
        from core.exploration_tools import _tree_visible, _count_tree_files
        names = sorted(_os.listdir(target))
    except OSError as ex:
        return f"✗ list_dir: cannot list '{rel}': {str(ex)[:120]}"
    dirs, files = [], []
    for nm in names:
        full = _os.path.join(target, nm)
        isd = _os.path.isdir(full)
        if not _tree_visible(nm, isd):
            continue
        child = f"{rel}/{nm}" if rel else nm
        if isd:
            dirs.append(f"  {child}/   ({_count_tree_files(full)} files)")
        else:
            ln = _file_line_count(ctx, child, full)
            files.append(f"  {child}   ({ln} lines)")
    label = rel or "(project root)"
    _MAX = 300
    body = dirs + files
    out = [f"=== TREE: {label} — folders (expand with list_dir) then files (read with read_file) ==="]
    if not body:
        out.append("  (empty)")
    else:
        out += body[:_MAX]
        if len(body) > _MAX:
            out.append(f"  … {len(body) - _MAX} more — expand a subfolder to narrow.")
    out.append("→ expand a folder: list_dir(path='<folder>');  read a file: read_file(path='<file>').")
    return "\n".join(out)


def _render_ranges(content: str, ranges: list) -> str:
    """Render specific 1-based line ranges of `content` in the `LINENO ⇥INDENT|code` view form
    (real line numbers). Used to replace a dropped full view with just the kept ranges."""
    lines = content.split("\n")
    out = []
    for (s, e) in ranges:
        out.append(f"  ── lines {s}-{e} ──")
        for i in range(max(1, s), min(len(lines), e) + 1):
            ln = lines[i - 1]
            ind = len(ln) - len(ln.lstrip(' '))
            out.append(f"{i} ⇥{ind}|{ln}")
    return "\n".join(out)


def _do_keep(args: dict, ctx: dict) -> str:
    """KEEP only the given line ranges of an already-VIEWED file; the loop then drops the rest of
    that file's view from context (ckpt-179). Errors if the file or a range wasn't viewed."""
    path, _e = _str_or_err(args, "path", "keep")
    if _e:
        return _e
    raw = args.get("ranges")
    norm = []
    if isinstance(raw, list):
        for r in raw:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                try:
                    norm.append((int(r[0]), int(r[1])))
                except (TypeError, ValueError):
                    pass
            elif isinstance(r, str):
                m = re.match(r'\s*(\d+)\s*-\s*(\d+)', r)
                if m:
                    norm.append((int(m.group(1)), int(m.group(2))))
    if not norm:
        return ("✗ keep: `ranges` must be a list of [start, end] line pairs "
                "(e.g. [[40, 60], [120, 135]]).")
    cur = ctx.get("file_contents", {}).get(path)
    if cur is None:
        _sb = ctx.get("sandbox")
        cur = _sb.load_file(path) if _sb is not None else None
    viewed = path in ctx.get("view_at", {}) or path in ctx.get("_served_ranges", {})
    if cur is None or not viewed:
        return (f"✗ keep: you haven't viewed {path} — read_file it first, then keep the "
                f"ranges you need.")
    filelen = cur.count("\n") + 1 if cur else 0
    fully = path in ctx.get("view_at", {})            # a whole-file view → every line is viewed
    served = ctx.get("_served_ranges", {}).get(path, [])
    for (s, e) in norm:
        if s < 1 or e < s or e > filelen:
            return (f"✗ keep: range {s}-{e} is outside {path} (it has {filelen} lines). "
                    f"Keep only ranges you actually read.")
        if not fully and not any(a <= s and e <= b for (a, b) in served):
            return (f"✗ keep: you haven't viewed lines {s}-{e} of {path} — read_file that "
                    f"range first, then keep it.")
    ctx.setdefault("_kept", {})[path] = norm
    # Accumulating-view sync (ckpt-196): keep is the trim for the growing view — set the file's
    # REVEALED ranges to exactly what's kept, so a later read of this file shows only the kept
    # ranges (+ whatever it newly reads) + labelled holes. Other files' views are untouched.
    ctx.setdefault("_served_ranges", {})[path] = _merge_ranges(norm)
    # ckpt-200 (bughunt #1/#4): keep evicts the file body from context (the loop renders only the
    # kept ranges). For a SMALL fully-held file, view_at + viewed_versions still claimed "you hold
    # the WHOLE file", so every later read of a DROPPED region was refused ("you already hold it")
    # → the coder could never anchor an edit on a dropped line → stale-anchor loop → budget
    # exhaust (the exact dead-end keep exists to avoid). Drop that "fully held" claim so a later
    # read RE-SERVES the region. No-op for a big file (never in view_at — it uses the growing view).
    ctx.get("view_at", {}).pop(path, None)
    if isinstance(ctx.get("viewed_versions"), dict):
        ctx["viewed_versions"].pop(path, None)
    _rng = ", ".join(f"{s}-{e}" for s, e in norm)
    return (f"✓ keep: keeping lines {_rng} of {path}; the rest of its view is dropped from your "
            f"context (read_file a range to bring any of it back).")


_BATCHABLE = {"read_file", "search_text", "find_refs", "file_purpose",
              "semantic_search", "depends_on", "list_dir"}
# ckpt-182 (flag-gated experiment, JARVIS_BATCH_ONLY): when batch is the ONLY exposed tool, it
# must carry the action tools too — edits/keep/run_code/finish — not just lookups.
_BATCH_ALL = _BATCHABLE | {"edit_file", "create_file", "replace_lines", "keep", "run_code", "finish"}
_BATCH_ONLY = os.environ.get("JARVIS_BATCH_ONLY", "0") == "1"


async def _do_batch(args: dict, ctx: dict) -> str:
    """Run several tool ops in ONE round (ckpt-181): native tool-calling does one call per round
    (gpt-oss emitted 1/round in 128/128), so a multi-op step wasted a round-trip per op. `calls`
    = [{tool, args}, …]; dispatch each IN ORDER via _dispatch and concatenate. By default only
    READ-ONLY lookups (edits have order dependencies). Under JARVIS_BATCH_ONLY (batch is the sole
    tool) ALL ops are allowed incl. edit/keep/run_code/finish — a finish op ends the loop via the
    normal __FINISH__ path (after the batch's other ops have run). Never raises."""
    calls = args.get("calls")
    if not isinstance(calls, list) or not calls:
        return ("✗ batch: `calls` must be a non-empty list of {\"tool\": <name>, \"args\": {...}}.")
    _allowed = _BATCH_ALL if os.environ.get("JARVIS_BATCH_ONLY", "0") == "1" else _BATCHABLE
    _capnote = ""
    if len(calls) > 10:
        _capnote = f"\n\n(ran the first 10 of {len(calls)} ops — batch ≤10 at a time.)"
        calls = calls[:10]
    out = []
    _ok = 0
    _finish_summary = None
    for i, c in enumerate(calls, 1):
        if not isinstance(c, dict):
            out.append(f"── op[{i}] ✗ each op must be {{\"tool\": <name>, \"args\": {{...}}}} "
                       f"(got {type(c).__name__}).")
            continue
        tname = c.get("tool") or c.get("name") or ""
        targs = c.get("args")
        if not isinstance(targs, dict):
            # tolerate the flat form {tool, path, ...} — treat the non-tool keys as args
            targs = {k: v for k, v in c.items() if k not in ("tool", "name", "args")}
        if tname == "batch" or tname not in _allowed:
            out.append(f"── op[{i}] {tname or '?'}: ✗ not allowed here — batch runs "
                       f"{'these ops' if _BATCH_ONLY else 'read-only lookups'} "
                       f"({', '.join(sorted(_allowed))}); can't nest batch.")
            continue
        try:
            r = await _dispatch(tname, targs, ctx)
        except Exception as e:
            r = f"✗ {tname} failed internally: {str(e)[:140]}"
        if isinstance(r, tuple) and r and r[0] == "__FINISH__":
            # a finish op — run the REST of the batch first, then signal finish to the loop.
            _finish_summary = r[1] or "done"
            _ok += 1
            out.append(f"── op[{i}] finish: marked done.")
            continue
        if isinstance(r, str) and not r.startswith("✗"):
            _ok += 1
        _lbl = targs.get("path") or targs.get("symbol") or targs.get("pattern") or targs.get("query") or targs.get("tag") or ""
        out.append(f"── op[{i}] {tname}({str(_lbl)[:70]}) ──\n{r}")
    joined = "\n\n".join(out) + _capnote
    # A finish op anywhere in the batch → end the loop via the standard __FINISH__ path (the
    # batch's edits/etc. already ran and updated ctx; the loop's finish gate checks files_changed).
    if _finish_summary is not None:
        return ("__FINISH__", _finish_summary)
    # If EVERY op failed, return a ✗-prefixed result so the loop's reject-counter / identical-call
    # fallover engages (else a repeated all-failing batch could burn the budget — review ckpt-181b).
    if _ok == 0:
        return (f"✗ batch: all {len(calls)} op(s) failed — fix the calls "
                f"({', '.join(sorted(_allowed))}) and resend.\n{joined}")
    return joined


def _parse_json_ops(text):
    """JSON-ops mode (ckpt-183): parse the coder's text content — a sequence of FLAT JSON ops,
    one per line — into (ops, done, summary). Tolerant by design (weak model): strips <think>…
    </think> reasoning, ```json fences, and prose between objects; reads concatenated objects;
    flattens a stray nested {"ops":[…]} if the model reverts; honours done via {"tool":"done",…}
    or a top-level {"done":true,...}. Each op = {"tool": <name>, "args": {...}}. Never raises.

    Two-pass: first the VISIBLE content (with <think>…</think> reasoning stripped — the clean
    path). gpt-oss intermittently leaks its WHOLE response into the reasoning channel (the
    harmony empty-turn), so if the visible pass finds nothing, SALVAGE by re-scanning the full
    text incl. reasoning — better a leaked op than a dead round. (Mirrors the native-loop salvage.)"""
    if not isinstance(text, str) or not text:
        return [], False, ""
    visible = re.sub(r'<think>.*?</think>', '', text, flags=re.S)   # drop hidden CoT
    ops, done, summary = _scan_json_ops(visible)
    if not ops and not done:                       # nothing in visible → salvage from reasoning
        ops, done, summary = _scan_json_ops(text)
    return ops, done, summary


def _scan_json_ops(text):
    """Single-pass balanced-brace scan of `text` → (ops, done, summary). Never raises."""
    ops, done, summary = [], False, ""
    if not isinstance(text, str) or not text:
        return ops, done, summary
    def _consume(obj):
        nonlocal done, summary
        if not isinstance(obj, dict):
            return
        # a stray nested batch/ops list → flatten it
        inner = obj.get("ops") or obj.get("calls")
        if isinstance(inner, list):
            for it in inner:
                _consume(it)
        if obj.get("done") is True:
            done = True
            summary = obj.get("summary") or (obj.get("args") or {}).get("summary") or summary
        tool = obj.get("tool") or obj.get("name")
        if not tool:
            return
        if tool == "done":
            done = True
            summary = (obj.get("args") or {}).get("summary") or obj.get("summary") or summary
            return
        args = obj.get("args")
        if not isinstance(args, dict):
            args = {k: v for k, v in obj.items() if k not in ("tool", "name", "args")}
        if len(ops) < 24:                       # bound a runaway op list
            ops.append({"tool": tool, "args": args})
    # SINGLE linear pass (O(n), never quadratic): track brace depth + string
    # state; each time depth returns to 0 from >0, that span is a candidate
    # top-level {...} — try to parse it. A nested-brace storm (`{{{{…`) that
    # never balances is just skipped at EOF, not re-scanned per `{` (the old
    # two-level scan was O(n²): `'{'*30000` → 32 s, run TWICE via salvage).
    depth, start, instr, esc = 0, -1, False, False
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if c == '\\':
            esc = True
            continue
        if c == '"':
            instr = not instr
            continue
        if instr:
            continue
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        _consume(json.loads(text[start:i + 1]))
                    except Exception:
                        pass            # prose/garbage span — skip, keep scanning
                    start = -1
    return ops, done, summary


def _coalesce_edit_ops(ops):
    """Merge multiple FLAT `edit_file` ops on the SAME path within one round into a single
    edit_file op carrying an `edits` list, so all hunks apply together — the dangling-ref /
    parse gate then sees the WHOLE refactor at once (removing a def + fixing its callers in
    one consolidated diff, never a half-done one that reverts).

    WHY: gpt-oss hand-writes JSON in this mode and reliably MIS-BALANCES the nested
    `edits:[{old:[...],new:[...]}]` closing brackets (`]}}}` / `]}}]}}` instead of `]}]}}`)
    → json.loads fails → 0 ops (ckpt-183 fresh12: 4a5d2a7d, f327). So the MODEL emits FLAT
    single-hunk ops ({path, old, new} — only `}}` to close, no arrays) and the HARNESS does the
    nesting here. (Project principle: harness computes the global structure, model acts local.)
    Order-preserving: a merged op sits at the position of the FIRST edit to its path; non-edit
    ops and edits with no usable path pass through untouched. _do_edit already accepts string
    or list `old`/`new` per hunk, so no applier change is needed."""
    out, edit_idx = [], {}
    for op in ops:
        if not isinstance(op, dict) or op.get("tool") != "edit_file":
            out.append(op)
            continue
        a = op.get("args") if isinstance(op.get("args"), dict) else {}
        path = a.get("path")
        # this op's hunks: an explicit (well-formed) edits list wins; else the flat old/new
        _edits = a.get("edits")
        if isinstance(_edits, list) and _edits:
            hunks = [h for h in _edits if isinstance(h, dict)]
        else:
            h = {}
            if a.get("old") is not None:
                h["old"] = a.get("old")
            if a.get("new") is not None:
                h["new"] = a.get("new")
            if a.get("start_line") is not None:
                h["start_line"] = a.get("start_line")
            hunks = [h] if h else []
        if path and path in edit_idx:
            out[edit_idx[path]]["args"]["edits"].extend(hunks)
        else:
            merged = {"tool": "edit_file", "args": {"path": path, "edits": list(hunks)}}
            if path:
                edit_idx[path] = len(out)
            out.append(merged)
    return out


async def _do_refs(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_refs_searches
    sym, _e = _str_or_err(args, "symbol", "find_refs")
    if _e:
        return _e
    return await _run_refs_searches([sym], ctx.get("project_root", ""))


async def _do_callers(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_dependency_lookup
    tag, _e = _str_or_err(args, "tag", "find_callers")
    if _e:
        return _e + " (a #tag from an `|appears N (#tag)` annotation)."
    return await _run_dependency_lookup([tag], ctx.get("project_root", ""))


async def _do_search(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_code_searches
    pat, _e = _str_or_err(args, "pattern", "search_text")
    if _e:
        return _e
    return await _run_code_searches([pat], ctx.get("project_root", ""))


def _do_purpose(args: dict, ctx: dict) -> str:
    from core.tool_call import _run_purpose_lookups
    path, _e = _str_or_err(args, "path", "file_purpose")
    if _e:
        return _e
    return _run_purpose_lookups([path], ctx.get("purpose_map") or "",
                                ctx.get("project_root", ""))


async def _do_semantic(args: dict, ctx: dict) -> str:
    # Embedding search over the CODE (same path as the text loop) — no purpose map.
    from tools.code_index import _maps_dir, _load_all_code
    from tools.embeddings import semantic_retrieve
    q, _e = _str_or_err(args, "query", "semantic_search")
    if _e:
        return _e
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
    # The raw unavailable message may be prefixed with `✗ `, `(`, or nothing, and may
    # carry TEXT-protocol [SEARCH:]/[REFS:]/[PURPOSE:] advice the NATIVE coder can't use
    # (ckpt-164: the old check only matched the `(`-prefixed form, so the ✗ form leaked
    # through with wrong-tool advice). Catch any prefix and hand back the NATIVE tools.
    _stripped = low.lstrip("✗ ()").strip()
    if _stripped.startswith("semantic search unavailable") or _stripped.startswith("no code to search"):
        return ("✗ semantic_search is unavailable (the embedding index couldn't be built). "
                "Do NOT retry it — use search_text (exact text/regex), find_refs (a known "
                "symbol name), or file_purpose (a file's API) instead.")
    if "no " in low and ("match" in low or "result" in low) and len(out.strip()) < 80:
        return (f"✗ semantic_search: {out.strip()} — 0 matches. Try search_text with "
                f"an exact term, or rephrase the concept.")
    return out


def _do_dependson(args: dict, ctx: dict) -> str:
    from core.exploration_tools import extract_dependencies
    sym, _e = _str_or_err(args, "symbol", "depends_on")
    if _e:
        return _e
    return extract_dependencies(sym, ctx.get("project_root", ""))


def _do_run(args: dict, ctx: dict) -> str:
    """Run a shell command against the coder's EDITED sandbox and return the
    output — so the coder can OBSERVE its change's runtime behaviour instead of
    SIMULATING it in its head (the static gates prove a patch parses + names
    resolve; only running proves it DOES the right thing). cwd = the sandbox dir
    where edits land; the bwrap sandbox is read-only/no-net with a STDLIB-ONLY
    python (3rd-party packages and test plugins are NOT installed — the full test
    suite cannot run here; that's the Docker grader's job). Use `python -c …` to
    smoke-check standard-library-only logic. A ModuleNotFoundError for any 3rd-party
    module is an ENVIRONMENT limit, NOT a bug in the edit — see the branch below."""
    from core.safe_exec import run_sandboxed
    _cmdv = args.get("command")
    cmd = _cmdv.strip() if isinstance(_cmdv, str) else ""
    if not cmd:
        return ("✗ run_code needs a `command` STRING — e.g. python -c \"<a check that "
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
    # Keep the TAIL, not the head: a Python traceback's exception line AND
    # pytest's PASS/FAIL summary both live at the END — head-truncation would
    # drop exactly the verdict. A small model scans the top, so also lift the
    # single most useful last line up front.
    _MAX = 3500
    shown = out if len(out) <= _MAX else "…(earlier output trimmed)…\n" + out[-_MAX:]
    last = next((l.strip() for l in reversed(out.splitlines()) if l.strip()), "")
    if code == 0:
        if not out:
            # A passing check is SILENT (assert raised nothing) — say so plainly,
            # or a small model reads "no output" as "nothing happened / failed".
            return ("✓ ran in your edited sandbox — exit 0, NO error raised: your "
                    "command SUCCEEDED (every assert/check passed). This is your "
                    "edit's real behaviour. To SEE a value rather than just pass/fail, "
                    "add a print(...) to your command and run again.")
        return (f"✓ ran in your edited sandbox — exit 0 (success). This output IS your "
                f"edit's real behaviour:\n{shown}")
    # A ModuleNotFoundError/ImportError is, AFTER the venv site-packages bind
    # (ckpt-166), almost always a genuinely-absent 3rd-party dep (jinja2, PyQt5,
    # web.py, django …) — an ENVIRONMENT limit of this smoke-check box, NOT a bug
    # in the edit. Two ckpt-165 night-run instances were LOST when the coder misread
    # this and rabbit-holed (edited prod import lines, created shim modules like a
    # root-level yaml.py). Name the module and steer hard away from "fixing" it,
    # while still allowing the rare real case (a module the coder itself just made).
    import re as _re
    _imp = _re.search(r"No module named ['\"]([\w.]+)['\"]", out)
    if _imp:
        _mod = _imp.group(1).split('.')[0]
        # Record it so create_file can REFUSE a shim for this module (ckpt-167): the
        # weak coder ignores the prose below and tries to manufacture a stub package
        # (web/__init__.py, infogami/…) which pollutes the patch and breaks `git apply`
        # in the real env (4a5d2a7d ✓→broken on ckpt-166). The applier-side block is
        # the safety net the prose alone didn't provide.
        ctx.setdefault("_failed_imports", set()).add(_mod)
        return (f"⚠ run_code: `import {_mod}` failed (ModuleNotFoundError). The smoke-check "
                f"sandbox runs a MINIMAL python — the standard library only; third-party packages "
                f"and test plugins are NOT installed. If `{_mod}` is a third-party dependency (i.e. NOT a file you "
                f"just created or renamed in this task), this is an ENVIRONMENT limitation of the "
                f"smoke box, NOT a bug in your edit — the real test environment has `{_mod}`.\n"
                f"DO NOT: edit/remove import statements in the project, create stub/shim modules, "
                f"wrap imports in try/except, or change your design to dodge `{_mod}`. Those are all "
                f"wrong — they corrupt the patch. Reason about `{_mod}` statically and move on.\n"
                f"ONLY if `{_mod}` is a module YOU just created/renamed is this a real bug to fix.\n"
                f"--- output (tail) ---\n{shown or '(no output)'}")
    return (f"✗ ran in your edited sandbox — exit {code}{timed}. This is YOUR EDIT'S "
            f"real runtime behaviour, NOT a tool error.\n"
            f"WHAT WENT WRONG (last line): {last[:200] or '(no output)'}\n"
            f"--- full output (tail) ---\n{shown or '(no output)'}")


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
            _payload = (args.get("command")            # run_code
                        or args.get("new_content")      # replace_lines
                        or args.get("content")          # create_file
                        or (str(args.get("hunks")) if args.get("hunks") else "")  # edit_file
                        or "")
            f.write(f"\n{'='*70}\nTOOL {tool}  args: start={args.get('start_line')} "
                    f"end={args.get('end_line')} path={args.get('path')}\n"
                    f"--- payload (command/new_content/hunks) ---\n{_payload}\n"
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
    if name == "list_dir":
        return _do_list_dir(args, ctx)
    if name == "keep":
        return _do_keep(args, ctx)
    if name == "batch":
        return await _do_batch(args, ctx)
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
        res = await asyncio.get_event_loop().run_in_executor(None, _do_run, args, ctx)
        _debug_edit_trace("run_code", args, res)
        return res
    if name == "trace_to_test":
        from core.exploration_tools import build_trace_template
        return build_trace_template(args.get("target", ""))
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
        "trace": "trace_to_test", "design_test": "trace_to_test",
        "plan": "trace_to_test", "think": "trace_to_test",
        "done": "finish", "stop": "finish", "complete": "finish", "end": "finish",
    }
    suggestion = _ALIAS.get((name or "").strip().lower().lstrip("[").rstrip("]:"))
    hint = (f" Did you mean '{suggestion}'?" if suggestion else "")
    return (f"✗ Unknown tool '{name}'.{hint} Available: read_file, find_refs, "
            f"find_callers, search_text, file_purpose, semantic_search, depends_on, "
            f"edit_file, create_file, run_code, finish.")


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
                                 per_provider_retries: int = 4,
                                 tool_choice: str = "required"):
    """Call the model's native tool API, cycling its gpt-oss endpoints. For each
    provider: retry the SAME endpoint on transient errors, skip to the next
    provider on a permanent error. Only raises once EVERY provider is exhausted —
    so the workflow switches to a different MODEL only after gpt-oss has had every
    endpoint. Non-gpt-oss models keep the single-endpoint behavior.
    tool_choice defaults to "required": gpt-oss (and mistral/medium) emit their
    plan in the harmony `analysis`/reasoning channel and STOP at the
    analysis→commentary boundary without emitting the tool call (finish_reason=stop,
    no tool_calls = the "empty-turn"). "required" forces a tool call every turn so
    the model can't end on reasoning alone. call_nvidia_tools falls back to "auto"
    if a provider 400s on "required"."""
    from clients.nvidia import call_nvidia_tools
    short = model_id.split('/')[-1]
    providers = [_GPT_OSS_ENDPOINT[short]] if short in _GPT_OSS_ENDPOINT else [""]
    last = None
    for pi, provider in enumerate(providers):
        for attempt in range(per_provider_retries):
            try:
                return await call_nvidia_tools(model_id, messages, tools,
                                               max_tokens=max_tokens,
                                               tool_choice=tool_choice,
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
    "FIRST, plan-adherence: re-check the STEP TEXT (in this conversation, not a file) "
    "and confirm your edit does EXACTLY "
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
    "  • VALUE-MAPPING: if the spec maps cases to specific result values (enum members, "
    "codes, statuses), check EACH branch returns the SPEC's value for that case — "
    "especially the first-run/empty/missing/None case, which is easy to leave on the "
    "OLD default (e.g. set to `equal` when the spec says `unknown`). If run_code is "
    "available, ASSERT that boundary case (construct the empty/first-run input, assert "
    "the field equals the spec's value) and RUN it — a value you ran beats one you "
    "eyeballed.\n"
    "If you find a CONCRETE problem, fix it with edit_file now. If the code is "
    "correct as written, call finish — do NOT change it just to change something."
)

# No-edit-finish guard. A coder (esp. a fallback link like mistral/medium dropped
# into a step the primary left incomplete) sometimes calls finish on its FIRST
# round having made ZERO edits — a polite bail, not a real completion. The step
# is NOT done (it was handed off precisely because no edit landed yet). Reject the
# first such finish and nudge ONCE to actually make the change; if the coder still
# finishes, accept it (fail-soft → the chain falls over to the next link). Fired
# AT MOST once per native pass. (user 2026-06-02; medium bailed on ansible-a26c325.)
_NO_EDIT_FINISH_NUDGE = (
    "⚠ You called finish but have made ZERO edits this step — nothing changed. "
    "This step was handed to you because it is NOT done yet, so finishing now "
    "delivers an empty patch. Do the work first: read the target file if you "
    "haven't, then make the edit the step requires with edit_file / replace_lines "
    "/ create_file. Only call finish AFTER an edit has actually landed (you'll see "
    "a ✓ Applied diff). If — after looking — you are certain the step needs no code "
    "change, call finish again and say why in the summary."
)

_EMPTY_TURN_NUDGE = (
    "⚠ You ended your turn with NO tool call. In this agent your reply must be a "
    "STRUCTURED tool call (the function-calling interface) — read_file, edit_file, "
    "search_text, find_refs, run_code, finish, etc. Do NOT write the call as plain "
    "text or a JSON object in your message; emit it as an actual tool call. Make "
    "your next move now as a tool call."
)

# Authoritative INDENT| write-format instruction. The native view is ALWAYS
# rendered prefix_ws (`LINENO ⇥INDENT|<real spaces>code`), so every coder that
# edits — native tool-calling AND JSON-ops (text mode) — must carry this block,
# appended LAST so it wins. Shared here so the two loops can't drift apart.
_INDENT_FORMAT_BLOCK = (
    "\n\n## INDENTATION — you DECLARE it as a number; the harness applies the spaces\n"
    "Every code line is shown as `LINENO ⇥INDENT|<real spaces>code` — e.g. "
    "`286 ⇥4|    def setvalue` means line 286, indent 4, then 4 real spaces, then the "
    "code. The bare number on the LEFT is the line; the number after the `⇥` is the "
    "INDENT — you SEE the indentation AND read its exact number (`4`).\n"
    "FOR `old`: copy the view line(s) VERBATIM — keep the whole `LINENO ⇥INDENT|` "
    "prefix (e.g. `286 ⇥4|    def setvalue`). The harness anchors on BOTH the line "
    "number (so a repeated line lands on the RIGHT one) AND the content (so a stale "
    "number self-corrects), and re-applies the indent from the number. You do NOT strip "
    "anything — just copy what you see.\n"
    "FOR `new` (new code, no line number yet): write each line as `INDENT|code` — the "
    "indent NUMBER (the one after the `⇥` in the view), a pipe, then the code WITHOUT "
    "leading spaces. The harness re-emits INDENT spaces for you, so you NEVER type or "
    "count leading spaces and can never drop them.\n"
    "  • CHANGED line → its `new` reuses the SAME `INDENT` the view shows for the line "
    "it replaces (`286 ⇥4|...` → your new line is `4|...`).\n"
    "  • NEW nested line → use a SIBLING's `INDENT`: a method `def` takes its class's "
    "method indent (look at another `def` in that class, e.g. `4|`); a body line is "
    "its header's INDENT + 4. NEVER write `0|` for something that lives inside a "
    "class/function — that ejects it (the dedent bug). A blank line is `0|`.")

# Bullet-CoT (ckpt-185, flag JARVIS_BULLET_COT — read at CALL time so tests can toggle).
# A SOFT reasoning-style nudge: same moves, terser wording. The goal is COST (gpt-oss is
# paid per reasoning token) + marginal speed — NOT fewer rounds. Deliberately NOT a rigid
# template: EDIT-COT's enforced slots tripped weak models into reject loops (ckpt-133),
# so this is framed as a style preference that correctness always overrides. The bullet
# example compresses HOW-TO-THINK move 1's own header-overwrite trace, so the two
# sections describe the SAME reasoning at two verbosities — no contradiction.
_BULLET_COT_BLOCK = (
    "\n\n## REASONING STYLE — tight bullets (form only; changes NOTHING about what you check)\n"
    "Run the same moves (TRACE → GAP → PLAN → BUILD → verify), but WRITE your reasoning as "
    "terse bullets — one bullet per fact or decision, most under ~12 words:\n"
    "  • L214: header = 'Bearer …'\n"
    "  • L230 runs unconditionally → OVERWRITES it — the bug\n"
    "  • fix: guard L230 with `if 'Authorization' not in headers:`\n"
    "  • new line lives inside the def at ⇥8 → my `new` uses 8|\n"
    "Each TRACE bullet still carries its concrete fact (line number, value, branch taken) — "
    "compress the WORDING, never the checking. Cut only narration filler: no \"Now I will…\", "
    "no re-pasting whole blocks you are NOT checking, no restating the task back to yourself. "
    "QUOTING a real line or a spec literal that you are CHECKING is NOT filler — keep it, "
    "exactly: the line-by-line trace and the char-for-char output-vs-expected diff (the move "
    "where you compare your result to the spec's example) stay verbatim however long they run. "
    "The ~12-word guide is for narration, not for these checks. If a subtle step needs full "
    "sentences to get right, use them — correctness always beats brevity; this is a style "
    "preference, never a reason to skip or shorten a check.\n"
    "(These bullets are your private REASONING/thinking. Your visible output is unchanged — "
    "keep following the output rules above exactly.)"
)

# ── SALVAGE: do the native tool-calling OURSELVES (ckpt-148) ─────────────────
# gpt-oss on a cheap provider (DeepInfra) intermittently returns finish_reason=stop
# with NO structured `tool_calls` — but the call it MEANT to make is sitting right
# there as TEXT in the harmony reasoning/commentary channel (e.g. `…view that file.
# {"path":"x.py","start_line":40,"end_line":70}`). The provider just failed to parse
# its own commentary into tool_calls. So we parse it: extract the JSON, infer the tool,
# synthesize the call, continue. Keeps the cheap provider, costs NO extra API call
# (unlike a retry), and works because the model already DID the work.
_SALVAGE_TOOL_NAMES = {t["function"]["name"] for t in CODER_TOOLS}


def _balanced_json_objs(text: str) -> list:
    """Every balanced {...} substring in `text`, string/escape aware (so a `{`/`}`
    inside a string literal — code in an old/new array — doesn't break the match)."""
    objs = []; i = 0; n = len(text)
    while i < n:
        if text[i] == '{':
            depth = 0; j = i; instr = False; esc = False; q = ''
            while j < n:
                c = text[j]
                if instr:
                    if esc: esc = False
                    elif c == '\\': esc = True
                    elif c == q: instr = False
                elif c in '"\'':
                    instr = True; q = c
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        objs.append(text[i:j + 1]); break
                j += 1
            i = j + 1
        else:
            i += 1
    return objs


def _infer_tool_from_args(d: dict):
    """Map a bare arguments object to the tool whose signature it fits — the harmony
    leak usually drops the function name and emits just the args. Ordered by specificity."""
    k = set(d.keys())
    if "old" in k or "new" in k: return "edit_file"
    if "command" in k: return "run_code"
    if "symbol" in k: return "find_refs"
    if "tag" in k: return "find_callers"
    if "query" in k: return "semantic_search"
    if "pattern" in k: return "search_text"
    if "content" in k and "path" in k: return "create_file"
    if "summary" in k and "path" not in k and "old" not in k: return "finish"
    if "path" in k: return "read_file"
    return None


def _salvage_inline_tool_call(text: str, ctr: int):
    """Parse a tool call the model leaked as TEXT and return a synthetic tool_call dict
    (or None). Tries the LAST balanced JSON object first (the leak is usually at the end
    of the reasoning). Handles both the {"name","arguments"} wrapper and a bare args
    object (tool inferred from its keys). Only returns a call for a real CODER_TOOLS tool."""
    if not text:
        return None
    for raw in reversed(_balanced_json_objs(text)):
        try:
            d = json.loads(raw)
        except Exception:
            continue
        if not isinstance(d, dict) or not d:
            continue
        nm, ar = d.get("name"), d.get("arguments")
        if isinstance(nm, str) and nm in _SALVAGE_TOOL_NAMES and ar is not None:
            a = ar if isinstance(ar, str) else json.dumps(ar)
            return {"id": f"salvage_{ctr}", "type": "function",
                    "function": {"name": nm, "arguments": a}}
        nm2 = _infer_tool_from_args(d)
        if nm2:
            return {"id": f"salvage_{ctr}", "type": "function",
                    "function": {"name": nm2, "arguments": json.dumps(d)}}
    return None


def finalize_coder_system(system: str) -> str:
    """Apply the env-gated system-prompt appends EXACTLY as call_with_native_tools does, so the
    render harness (behavioral_audit/render_prompt.py) sees the SAME final system the coder gets
    (CLAUDE.md doctrine: no drift between audited and live artifacts). Order matters — the always-on
    _INDENT_FORMAT_BLOCK is appended after the batch block and before trace/cot. Default env (no
    flags) → just system + _INDENT_FORMAT_BLOCK."""
    if _BATCH_ONLY:
        system = system + (
            "\n\n## ⚠ ONE TOOL: batch — this OVERRIDES the calling convention above\n"
            "Your ONLY callable tool is `batch`. Every tool named above (read_file, edit_file, "
            "keep, run_code, finish, …) is an OPERATION you invoke ONLY by listing it in batch's "
            "`calls` = [{\"tool\": <name>, \"args\": {...}}, …] — never call them directly. Pack "
            "the ops that go together into ONE batch each round: LOOK = batch all your read/search "
            "ops together (one round); then — after you SEE the results — batch your edit_file "
            "op(s); then batch a run_code check; then batch finish. NEVER read_file and edit_file "
            "the same file in the same batch (the edit runs before you've seen the file). Aim for "
            "~4 rounds total.")
    # ALWAYS-ON: the native view is always rendered prefix_ws (LINENO ⇥INDENT|<real spaces>code),
    # so this INDENT| write instruction must ALWAYS be appended (was a flag; dropping it re-armed
    # the col-0 dedent bug). Appended LAST of the always-on parts so it wins.
    system = system + _INDENT_FORMAT_BLOCK
    if _TRACE_MODE:
        system = system + (
            "\n\n## PROVE A SUBTLE EDIT — trace_to_test as your CLOSING step\n"
            "When a step's correctness hinges on a nuance (an order, a condition, an edge "
            "— 'X must not override Y', 'when A present, B is suppressed'), don't edit and "
            "hope. AFTER you've made the edit, call trace_to_test(target) and fill its "
            "template: trace the REAL flow (now including your edit) to the EDGE — cite "
            "real lines via read_file; imagined citations are rejected — name where correct "
            "vs naive diverges, and write a test that CATCHES the naive bug. Then run_code "
            "that test against your edit: green proves it, red shows exactly what to fix "
            "(change only the edge, nothing extra). A subtle fix you've run a discriminating "
            "test on beats one you only reasoned about.")
    if _EDIT_COT:
        system = system + (
            "\n\n## GROUND YOUR EDITS WHEN IT HELPS (optional, not enforced)\n"
            "edit_file / replace_lines / create_file accept three OPTIONAL fields — fill them to "
            "keep yourself honest, but an edit is NEVER rejected for omitting or paraphrasing them:\n"
            "  • goal   — the spec behaviour this edit makes true (1 concrete sentence).\n"
            "  • traced — what the code does NOW at the edit site (quote the real line if handy).\n"
            "  • check  — one concrete input→expected-output case your edit satisfies.\n"
            "Reason in whatever way fits the change — don't force a template. What matters is a "
            "correct edit grounded in the real code, not filled-in fields.")
    if os.environ.get("JARVIS_BULLET_COT", "0") == "1":
        system = system + _BULLET_COT_BLOCK
    return system


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
    # System-prompt finalization (env-gated appends) lives in finalize_coder_system so the render
    # harness sees the IDENTICAL final system (ckpt-197, CLAUDE.md doctrine: no audited/live drift).
    system = finalize_coder_system(system)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_content}]
    ctx.setdefault("files_changed", set())
    done = False
    final = ""
    reason = "budget-exhausted"
    rnd = 0
    _fail_counts: dict = {}   # (tool, raw_args) → consecutive-reject count (audit #46)
    _total_rejects = 0        # TOTAL ✗ this step — backstop for the varied-reject evasion
                              # of _fail_counts (a coder that tweaks args each round never
                              # trips the identical-3× stop and burns the whole budget). (M1/N1)
    _verify_nudged = False    # one forced self-check before finishing (per step)
    _empty_retries = 0        # empty-turn (stop, no tool_call) recovery attempts (ckpt-145)
    _salvage_count = 0        # inline tool-calls parsed from the reasoning channel (ckpt-148)
    _noedit_finish_nudged = False  # one push-back on a finish-with-ZERO-edits bail
    _stuck = False            # hard-stop flag: same edit rejected ≥3× → fall over
    _trace_nudges = 0         # grounding nudges spent on imagined trace citations

    def _verify_nudge_msg():
        body = _VERIFY_NUDGE.format(
            files=", ".join(sorted(ctx.get("files_changed", set()))) or "(none)")
        # Deterministic harness advisory: across the files you changed, did you DEFINE
        # an enum member the spec needs but never ASSIGN it anywhere? That member's case
        # is likely unhandled. (Harness computes the global; coder confirms locally.)
        _dead = []
        for _p in sorted(ctx.get("files_changed", set())):
            _c = ctx.get("file_contents", {}).get(_p)
            if isinstance(_c, str) and _p.endswith(".py"):
                _dead += _unassigned_enum_members(_c)
        if _dead:
            body += (
                "\n\n⚠ UNHANDLED-CASE CHECK: you DEFINED these enum member(s) but never "
                "ASSIGN them to anything: " + ", ".join(sorted(set(_dead))) + ". Each is "
                "probably a case the spec names but your code doesn't set — find the "
                "branch that should produce it and assign it there (or confirm it's "
                "genuinely unused).")
        # JARVIS_TRACE: a passive system-prompt line gets 0 adoption (the coder goes
        # straight to edit→run_code and never reaches for the optional tool). This is
        # the just-in-time moment — the coder is about to finish WITH edits — so point
        # it AT trace_to_test as the way to do the self-check. Recency makes it hard to
        # ignore; it still finishes after one pass (NOT a gate, per the chosen design).
        if _TRACE_MODE:
            body += (
                "\n\nDO THIS SELF-CHECK AS A GROUNDED TRACE — call trace_to_test(target) "
                "and fill its template: walk the REAL flow to the EDGE citing actual "
                "@file:line (imagined lines are rejected), name where the correct vs the "
                "naive impl diverges, then run_code the discriminating test that a naive "
                "impl FAILS. A test you RAN green beats a self-check you only narrated. "
                "(Skip the trace only if this edit is purely mechanical — a rename, an "
                "import, a typo.)")
        return {"role": "user", "content": body}
    for rnd in range(1, max_rounds + 1):
        ctx["round"] = rnd   # so tools can stamp diffs/views with WHEN they happened
        messages = _trim_history(messages, max_history_chars, model_id)
        try:
            _tools = _BATCH_ONLY_TOOLS if _BATCH_ONLY else CODER_TOOLS
            msg = await _call_tools_with_retry(model_id, messages, _tools, max_tokens)
        except Exception as e:
            warn(f"  [native:{short}] giving up after error: {str(e)[:120]}")
            reason = "api-error"
            break
        if not isinstance(msg, dict):
            warn(f"  [native:{short}] non-dict model message — stopping")
            reason = "api-error"
            break
        # CAPTURE THE CODER'S CoT. gpt-oss is a native tool-calling model: it puts
        # its chain-of-thought in `reasoning` (or `reasoning_content`) and leaves
        # `content` EMPTY on a tool-call turn — so the prompt's 4-move CoT, if the
        # model does it, lives there. (1) LOG it so the coder's reasoning is finally
        # visible in the thinking log (the non-streaming native path never logged it
        # before — only the streaming planner/summary phases did). (2) PERSIST a
        # FULL copy into the assistant turn so the coder builds on its OWN prior reasoning
        # across rounds — UNCAPPED within the step (point 1): the old 1500-char cap truncated
        # the conclusion of a hard multi-file CoT (the part the next round needs), forcing the
        # coder to re-derive (= re-read = redundant lookups). The f631 bloat it guarded against
        # was the repeated full-file RE-DUMPS, not reasoning text — and those are now killed by
        # the re-read short-circuit, so the freed budget pays for full reasoning. The step resets
        # the history every step, and _trim_history(400k) stays as the context-overflow backstop.
        _reason = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        _vis = (msg.get("content") or "").strip()
        if _reason:
            thought_logger.write_header(model_id, f"coder round {rnd}")
            thought_logger.write_chunk(model_id, _reason)
            # Also surface it (capped) in the instance-prefixed RUN log so the coder's
            # reasoning is observable per-instance for offline audits — the thought_logger
            # writes to separate per-model files and isn't started under swe_bench, so the
            # CoT was effectively unobservable from the run log (ckpt-135). Full text still
            # goes to thought_logger; this is the greppable tail.
            status(f"  [native:{short}] 💭 {_reason[:400].replace(chr(10), ' ')}"
                   + (" …" if len(_reason) > 400 else ""))
        _persist = _vis
        if _reason:
            _persist = (f"[my reasoning] {_reason}\n{_vis}").strip()
        # The assistant message that issued tool_calls MUST precede the tool
        # results in history, with its tool_calls intact.
        messages.append({
            "role": "assistant",
            "content": _persist,
            **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {}),
        })
        tcs = msg.get("tool_calls") or []
        # SALVAGE (ckpt-148): no structured tool_calls, but the model may have leaked the
        # call as TEXT in the reasoning/commentary channel (the harmony empty-turn). Parse
        # it ourselves — "do the native ourselves" — so we recover the step on the cheap
        # provider with no extra API call, instead of retrying or dying. Cap to avoid loops.
        if not tcs and _salvage_count < 4:
            _sal = _salvage_inline_tool_call((_reason or "") + "\n" + (_vis or ""), _salvage_count)
            if _sal:
                _salvage_count += 1
                tcs = [_sal]
                # the assistant turn we just appended MUST carry the call so its tool
                # result has a matching preceding tool_calls (API invariant).
                messages[-1]["tool_calls"] = tcs
                status(f"  [native:{short}] round {rnd}: salvaged inline "
                       f"{_sal['function']['name']} call from the reasoning channel "
                       f"(provider returned it as text, not a structured tool_call)")
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
            _fr = msg.get("_finish_reason", "")
            # EMPTY-TURN RECOVERY (ckpt-145): finish_reason=stop with no tool_call means
            # tool_choice=required was DROPPED for this call — the silent 400→auto
            # downgrade in call_nvidia_tools, or the provider ignoring it. Probes confirm
            # `required` reliably yields a tool call, so don't die (and don't salvage
            # leaked text) — nudge "emit a STRUCTURED tool call" and retry. Bounded so a
            # genuine no-op finish still ends the step. This is what cost f327 its step
            # (round 9 knew it needed read_file(40-60) but emitted it as text → stop).
            if _empty_retries < 2:
                _empty_retries += 1
                messages.append({"role": "user", "content": _EMPTY_TURN_NUDGE})
                status(f"  [native:{short}] round {rnd}: empty-turn "
                       + (f"(finish_reason={_fr}) " if _fr else "")
                       + f"— retrying with a forced tool call ({_empty_retries}/2)")
                continue
            # An empty assistant turn (no tool calls, no content) is a STALL, not
            # a finish — distinguish so the workflow can tell "model did nothing"
            # from a deliberate stop. (Audit #11/#45.)
            reason = "no-tool-call" if final.strip() else "empty-turn"
            status(f"  [native:{short}] round {rnd}: no tool call ({reason}"
                   + (f", finish_reason={_fr}" if _fr else "") + ")")
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
                    if not ctx.get("files_changed") and not _noedit_finish_nudged:
                        # Finish on a step with ZERO edits = a bail, not completion.
                        # Push back ONCE; if the coder finishes again, accept it.
                        _noedit_finish_nudged = True
                        result_str = _NO_EDIT_FINISH_NUDGE
                        status(f"  [native:{short}] round {rnd}: finish with ZERO "
                               f"edits — nudged to make the change first")
                    else:
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
            # Count ONLY EDIT-op rejects toward the stuck backstops (ckpt-201, bughunt #6). The
            # old `name != "run_code"` gate counted EVERY ✗, but lookups legitimately return ✗
            # during normal exploration (semantic_search "0 matches", search_text no-hits,
            # unavailable embeddings) — those piled up to the 8-strike _total_rejects and aborted
            # the step BEFORE any edit even landed. The real stuck signal is rejected EDITS; pure
            # lookup-spin is already bounded by max_rounds. (Mirrors the JSON-ops loop.)
            if (isinstance(result_str, str) and result_str.startswith("✗")
                    and name in ("edit_file", "replace_lines", "create_file")):
                _total_rejects += 1
                # Observability (ckpt-134/137): surface WHY each edit was rejected. The
                # reject text used to live only in the model's tool channel. ckpt-137 fix:
                # reject messages are DOUBLE-PREFIXED ("✗ edit_file NOT applied to {path}: ✗
                # edit REJECTED — {path}: {reason}"), so the old [:110] cap cut off inside the
                # redundant path wrapper and hid the real reason as "other". Strip the
                # wrapper(s) — keep the text after the LAST "REJECTED —"/"NOT applied to … :"
                # so the actual cause (parse/unreachable/duplicate/old-not-found) is logged.
                _rj = result_str.split("\n", 1)[0]
                for _mark in (" edit REJECTED — ", " NOT applied to "):
                    if _mark in _rj:
                        _rj = _rj.split(_mark)[-1]
                _rj = re.sub(r'^[^:]{0,80}\.py[^:]*:\s*', '', _rj)  # drop a leading "{path}: "
                status(f"  [native:{short}] round {rnd}: edit REJECTED — {_rj[:150]}")
                # backstop: many DIFFERENT rejected edit calls (varied args evade the
                # identical-3× check) → still stuck; fall over rather than burn the budget.
                if _total_rejects >= 8:
                    warn(f"  [native:{short}] {_total_rejects} rejected edit calls this "
                         f"step — stuck; stopping for fallover")
                    _stuck = True
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
            # SUPERSEDED marker: once an edit LANDS, mark any earlier read_file view
            # of that file in the history — but do NOT tell the coder to re-read (the
            # old blanket "⟪STALE — read it again⟫" banner is exactly what drove the
            # full re-reads that blew the context window: f631). Instead point to the
            # post-edit diff: the changed region is in the diff above; the REST of this
            # view is still accurate. So the coder copies `old` from the right place
            # without re-dumping the whole file. (Marks once; the precise mismatch case
            # is still caught by _old_not_found_msg.)
            if (name in ("edit_file", "replace_lines")
                    and isinstance(result_str, str) and result_str.startswith("✓")):
                _ep = (args.get("path") or "") if isinstance(args, dict) else ""
                if _ep:
                    _when = ctx.get("view_at", {}).get(_ep, "a later round")
                    for _m in messages[:-1]:
                        _c = _m.get("content")
                        if (_m.get("role") == "tool" and _msg_is_view_of(_c, _ep)
                                and "⟪SUPERSEDED" not in _c):
                            _m["content"] = (
                                f"⟪SUPERSEDED — {_ep} was edited after this read ({_when}). "
                                f"For the region you changed, use the diff in that edit's "
                                f"result above; the REST of this view is still accurate. "
                                f"Don't re-read the whole file — your view + the diff is "
                                f"current.⟫\n" + _c)
            # KEEP eviction (ckpt-179): the coder called keep(path, ranges) → DROP the rest of
            # that file's view from history, leaving only the kept ranges, to free context. Pure
            # content-replacement on existing tool messages (no add/remove → API pairing intact).
            if (name == "keep" and isinstance(result_str, str) and result_str.startswith("✓")):
                _kp = (args.get("path") or "") if isinstance(args, dict) else ""
                _kr = ctx.get("_kept", {}).get(_kp)
                _kc = ctx.get("file_contents", {}).get(_kp)
                if _kp and _kr and _kc:
                    _krender = _render_ranges(_kc, _kr)
                    _rngs = ", ".join(f"{s}-{e}" for s, e in _kr)
                    for _m in messages:
                        _c = _m.get("content")
                        if (_m.get("role") == "tool" and _msg_is_view_of(_c, _kp)
                                and "⟪KEPT" not in _c):
                            _m["content"] = (
                                f"⟪KEPT only lines {_rngs} of {_kp} (you called keep); the rest "
                                f"of this view was dropped to save context — read_file a range "
                                f"to bring any of it back.⟫\n{_krender}")
            # READ replace (ckpt-196): a read_file of a >cap file returns the ONE growing view of
            # that file (def-index + every revealed range) — strictly MORE complete than any earlier
            # view of it. Collapse every earlier copy to a pointer so there's exactly one live view
            # per file: no duplicate dumps, the coder reads ONE coherent picture. Content-only swap.
            if (name == "read_file" and isinstance(result_str, str) and "=== VIEW:" in result_str
                    and ("GROWING VIEW" in result_str or "TOO LARGE" in result_str)):
                _rp = (args.get("path") or "") if isinstance(args, dict) else ""
                if _rp:
                    _supersede_prior_file_views(messages, _rp)
        # TRACE grounding (JARVIS_TRACE): if the coder filled a trace this round,
        # check its `@ file:line | code` citations against the REAL files. An
        # imagined flow gets a concrete re-trace nudge — the SAME enforcement the
        # planner's text loop already has, which was ABSENT here (the template was
        # a dead-end otherwise: nothing received or verified the filled trace).
        # Injected AFTER the tool results (API forbids a user turn between an
        # assistant-with-tool_calls and its results); capped so it can't spam an
        # uncooperative model.
        if _TRACE_MODE and _trace_nudges < 2:
            _ac = msg.get("content") or ""
            if "@" in _ac and ":" in _ac:   # cheap prefilter for the `@file:line` form
                from core.exploration_tools import verify_trace_lines
                _gw = verify_trace_lines(_ac, ctx.get("project_root", ""))
                if _gw:
                    _trace_nudges += 1
                    messages.append({"role": "user", "content": _gw})
                    status(f"  [native:{short}] round {rnd}: trace citations not "
                           f"grounded — asked to re-cite real lines")
                    continue
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
                # reset reason too: `finished` was set when the finish fired, but we're
                # un-finishing for the self-check. If the coder never re-finishes (hits the
                # round cap), reason must reflect that — leaving it `finished` would log a
                # budget-truncated step as a clean finish and suppress the warning. (pass-6 M1.)
                reason = "budget-exhausted"
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


# ══════════════════════════════════════════════════════════════════════════════
# JSON-OPS coder (ckpt-183, flag JARVIS_JSON_OPS) — gpt-oss in TEXT mode (no native
# tool-calling), STREAMING, emitting FLAT JSON-line ops it terminates with a `done`
# op. Native tool-calling makes gpt-oss emit ONE call/round (128/128) and it can't
# build the nested batch array (empty `calls`). Flat single ops are its STRONGEST
# skill — so we let it emit MANY of them as plain text (text mode = no tool_call to
# end the turn), parse them, run them all, results next round. Reuses every _do_*.
# ══════════════════════════════════════════════════════════════════════════════
_JSON_OPS_PROMPT = (
    "\n\n## ⚠ OUTPUT FORMAT — JSON OPS (this OVERRIDES the tool-calling convention above)\n"
    "You do NOT make function/tool calls. You RESPOND WITH OPERATIONS as a sequence of FLAT "
    "JSON objects, ONE PER LINE, nothing else — no prose, no markdown fences, no array wrapper. "
    "Each line is exactly:\n"
    '  {\"tool\": \"<name>\", \"args\": { ... }}\n'
    "Emit every op you want THIS round, one per line, then stop. The harness runs them IN ORDER "
    "and sends every result back in the next message — you do NOT see a result mid-response, so "
    "gather first, act next round. THE OPS:\n"
    '  {\"tool\":\"read_file\",\"args\":{\"path\":\"p\"[,\"start_line\":N,\"end_line\":M]}}\n'
    '  {\"tool\":\"list_dir\",\"args\":{\"path\":\"dir\"}}   {\"tool\":\"search_text\",\"args\":{\"pattern\":\"re\"}}\n'
    '  {\"tool\":\"find_refs\",\"args\":{\"symbol\":\"name\"}}   {\"tool\":\"find_callers\",\"args\":{\"tag\":\"#t\"}}\n'
    '  {\"tool\":\"file_purpose\",\"args\":{\"path\":\"p\"}}   {\"tool\":\"semantic_search\",\"args\":{\"query\":\"q\"}}\n'
    '  {\"tool\":\"depends_on\",\"args\":{\"symbol\":\"name\"}}\n'
    '  {\"tool\":\"edit_file\",\"args\":{\"path\":\"p\",\"old\":\"286 ⇥4|    def foo\",\"new\":\"4|def foo2\"}}\n'
    '  {\"tool\":\"create_file\",\"args\":{\"path\":\"p\",\"content\":\"...\"}}\n'
    '  {\"tool\":\"keep\",\"args\":{\"path\":\"p\",\"ranges\":[[40,60]]}}   {\"tool\":\"run_code\",\"args\":{\"command\":\"...\"}}\n'
    '  {\"tool\":\"done\",\"args\":{\"summary\":\"one line: what you changed\"}}  ← emit ONLY when the edit is complete AND verified\n'
    "RULES: (1) one JSON object per line, FLAT — never nest ops in an array. (2) edit_file is FLAT: "
    "`old` and `new` are STRINGS, not arrays. For several lines, join them with \\n inside the "
    "string — e.g. \"new\":\"4|def foo():\\n8|return 1\" (NO leading spaces after the pipe — the "
    "harness re-emits the indent from the number). `old`/`new` use the "
    "`LINENO ⇥INDENT|code` / `INDENT|code` format from the INDENTATION section. Do NOT use an "
    "`edits` array and do NOT nest `[...]` — that is the one structure you keep mis-closing. "
    "(3) Do NOT read_file and edit_file the SAME file in one round — you must SEE it (this round) "
    "before you edit it (next round). (4) ONE edit_file op = ONE change. To change a file in "
    "SEVERAL places (e.g. remove a def AND fix its callers), emit SEVERAL edit_file ops for that "
    "file in the SAME round — the harness applies them together as one diff. (5) Ideal shape: "
    "round 1 = all your read/search ops together; round 2 = your edit_file op(s); round 3 = a "
    "run_code check; then a `done` op. (6) Emit ONLY the JSON lines — no commentary.\n"
    "TWO THINGS THIS REPLACES from the toolkit above: there is NO `batch` tool here — emitting "
    "several ops (one per line) in a round IS the batch, so never wrap them. And wherever the "
    "instructions above say \"call finish\" / `finish(summary)`, you instead emit the `done` op."
)
_JSON_OPS_NUDGE = (
    "⚠ I couldn't find any valid JSON op in your reply. Respond with ONLY flat JSON ops, one per "
    "line — e.g. `{\"tool\":\"read_file\",\"args\":{\"path\":\"x.py\"}}` — and nothing else. When "
    "the edit is done and verified, emit `{\"tool\":\"done\",\"args\":{\"summary\":\"…\"}}`."
)
_JSON_OPS_VERIFY = (
    "Before you finish: re-trace your edit on the spec's hardest case (or run_code a quick check). "
    "If it's correct, emit `{\"tool\":\"done\",\"args\":{\"summary\":\"…\"}}` again; otherwise emit "
    "the fixing edit_file op."
)
_JSON_OPS_NO_EDIT = (
    "⚠ You emitted `done` but NOTHING has changed — no edit landed (you made none, or your "
    "edit_file op was rejected with a ✗). This step is NOT done; finishing now ships an empty "
    "patch. Make the change first: if you haven't seen the target file, read_file it; then emit "
    "an edit_file op (all hunks for a file in ONE op) and confirm you get a ✓ applied diff in the "
    "results. Only emit `done` AFTER an edit has actually landed. If — after looking — you are "
    "certain the step needs no code change, emit `done` again and say why in the summary."
)


async def call_with_json_ops(model_id: str, system: str, user_content: str,
                             ctx: dict, max_rounds: int = 40,
                             max_history_chars: int = 400_000) -> dict:
    """JSON-ops coder loop. Streams gpt-oss in TEXT mode (no tools), parses the FLAT JSON-line ops
    it emits, runs each via _dispatch (every _do_* reused), feeds results back, loops. `done` op
    (with a verify gate) ends it. Same return contract as call_with_native_tools."""
    from clients.nvidia import call_nvidia_stream
    short = model_id.split('/')[-1]
    # Same authoritative INDENT block the native loop appends (edits use the SAME
    # `INDENT|code` write-format), THEN the JSON-ops protocol override LAST so it
    # wins on HOW to emit (flat JSON lines, not function calls).
    system = (system or "") + _INDENT_FORMAT_BLOCK + _JSON_OPS_PROMPT
    if os.environ.get("JARVIS_BULLET_COT", "0") == "1":
        # ckpt-185 experiment: tight-bullet reasoning style (cost lever; soft, never a
        # gate). Appended after the protocol override — it's style-only, no protocol words.
        system = system + _BULLET_COT_BLOCK
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_content}]
    ctx.setdefault("files_changed", set())
    done = False
    final = ""
    reason = "budget-exhausted"
    rnd = 0
    _empty = 0
    _verify_nudged = False
    _no_edit_nudged = False
    _total_rejects = 0       # EDIT-op rejects only (the real stuck signal)
    for rnd in range(1, max_rounds + 1):
        ctx["round"] = rnd
        messages = _trim_history(messages, max_history_chars, model_id)
        try:
            # NO stop_check: a bare "tool":"done" substring scan would truncate the
            # stream mid-op whenever an edit/create's CONTENT contains that text (and
            # would chop the done op's own summary). The model ends its turn after its
            # ops naturally; max_tokens bounds any post-done rambling. (review P0-2.)
            content = await call_nvidia_stream(
                model_id, prompt="", system="", messages_override=messages,
                stop_check=None, max_tokens=16384,
                log_label=f"json-coder round {rnd}")
        except Exception as e:
            warn(f"  [json:{short}] round {rnd}: api error — {str(e)[:120]}")
            reason = "api-error"
            break
        ops, want_done, summary = _parse_json_ops(content)
        ops = _coalesce_edit_ops(ops)   # flat same-file edits → one edits batch (refactor coherence)
        messages.append({"role": "assistant", "content": (content or "(no output)")[:24000]})
        if not ops and not want_done:
            _empty += 1
            if _empty <= 2:
                messages.append({"role": "user", "content": _JSON_OPS_NUDGE})
                status(f"  [json:{short}] round {rnd}: no JSON ops parsed — nudging ({_empty}/2)")
                continue
            reason = "no-ops"
            status(f"  [json:{short}] round {rnd}: still no ops — stopping")
            break
        status(f"  [json:{short}] round {rnd}: {len(ops)} op(s) [{','.join(o['tool'] for o in ops)}]"
               + (" +done" if want_done else ""))
        results = []
        for k, op in enumerate(ops, 1):
            tname, targs = op["tool"], op["args"]
            try:
                r = await _dispatch(tname, targs, ctx)
            except Exception as e:
                r = f"✗ {tname} failed internally: {str(e)[:140]}"
            if isinstance(r, tuple) and r and r[0] == "__FINISH__":   # a stray finish op
                want_done = True
                summary = r[1] or summary
                continue
            # Only EDIT-op rejects signal a stuck loop. A lookup returning "✗ no
            # results" is normal exploration (search_text/find_refs/semantic_search
            # return ✗ on no hits) — counting those would trip the stuck-bail on one
            # big speculative LOOK round, before any edit is even attempted. (P1-2.)
            if (isinstance(r, str) and r.startswith("✗")
                    and tname in ("edit_file", "replace_lines", "create_file")):
                _total_rejects += 1
            _lbl = targs.get("path") or targs.get("symbol") or targs.get("pattern") or targs.get("query") or ""
            results.append(f"── op[{k}] {tname}({str(_lbl)[:70]}) ──\n{r}")
        if len(ops) >= 24:        # parser caps a runaway op list — say so, don't drop silently (P1-5)
            results.append("⚠ NOTE: only the first 24 ops this round were run; "
                           "emit fewer ops per round (a focused LOOK, then EDIT).")
        # Only feed a RESULTS turn back when ops actually ran. A pure-`done` round
        # (or one whose only op was a finish, which we `continue` past) produces no
        # results — appending an empty "RESULTS of your ops" turn just before we
        # break is noise the model never sees act on, so skip it.
        if results:
            messages.append({"role": "user", "content":
                             "RESULTS of your ops (in order):\n\n" + "\n\n".join(results)
                             + "\n\nEmit your next ops, or `{\"tool\":\"done\",\"args\":{\"summary\":\"…\"}}` "
                               "when the edit is complete AND verified."})
        if _total_rejects >= 12:
            warn(f"  [json:{short}] {_total_rejects} rejected ops — stuck; stopping for fallover")
            reason = "stuck-repeating"
            break
        if want_done:
            # NO-EDIT done: the coder asked to finish but nothing has landed (no edit
            # made, or its edit got rejected this same round → files_changed empty).
            # An empty patch is not "done"; nudge ONCE to actually make the change,
            # then accept a second done (fail-soft → chain falls over). (P0-3/P1-1.)
            if not ctx.get("files_changed") and not _no_edit_nudged:
                _no_edit_nudged = True
                messages.append({"role": "user", "content": _JSON_OPS_NO_EDIT})
                status(f"  [json:{short}] round {rnd}: done with ZERO edits — nudging to edit first")
                continue
            if ctx.get("files_changed") and not _verify_nudged:
                _verify_nudged = True
                messages.append({"role": "user", "content": _JSON_OPS_VERIFY})
                status(f"  [json:{short}] round {rnd}: done requested — one verify pass first")
                continue
            done = True
            final = summary or "done"
            reason = "finished"
            break
    files = sorted(ctx.get("files_changed", set()))
    if not done and not files:
        warn(f"  [json:{short}] stopped ({reason}) with ZERO edits — step produced nothing.")
    return {"answer": final, "done": done, "files_changed": files,
            "rounds": rnd, "reason": reason}
