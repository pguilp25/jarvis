"""
Tool Call Loop — shared by all workflows.

Any AI can pause mid-thought to search:
  [SEARCH: pattern]    → ripgrep code search (coding agent)
  [WEBSEARCH: query]   → web search (research, chat)

JARVIS detects the tags, runs the searches, feeds results back,
and the AI continues from where it left off. Up to 5 rounds.
"""

import asyncio
import os
import re
from core.retry import call_with_retry
from core.cli import step, status, warn


# In-flight locks: prevent duplicate lookups across parallel AI calls.
# When two coders both request [REFS: foo] at the same time, only one
# actually runs the search — the other waits and gets the cached result.
_inflight_locks: dict[str, asyncio.Lock] = {}


# ─── Tag Patterns ────────────────────────────────────────────────────────────

SEARCH_TAG = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)
WEBSEARCH_TAG = re.compile(r'\[WEBSEARCH:\s*(.+?)\]', re.IGNORECASE)
DETAIL_TAG = re.compile(r'\[DETAIL:\s*(.+?)\]', re.IGNORECASE)
CODE_TAG = re.compile(r'\[CODE:\s*(.+?)\]', re.IGNORECASE)
REFS_TAG = re.compile(r'\[REFS:\s*(.+?)\]', re.IGNORECASE)
PURPOSE_TAG = re.compile(r'\[PURPOSE:\s*(.+?)\]', re.IGNORECASE)
SEMANTIC_TAG = re.compile(r'\[SEMANTIC:\s*(.+?)\]', re.IGNORECASE)
LSP_TAG = re.compile(r'\[LSP:\s*(.+?)\]', re.IGNORECASE)
KNOWLEDGE_TAG = re.compile(r'\[KNOWLEDGE:\s*(.+?)\]', re.IGNORECASE)
# KEEP strips a previously-loaded [CODE:] result to only the specified line
# ranges, removing the full file from context.  Format:
#   [KEEP: filepath 10-50, 80-120]
KEEP_TAG = re.compile(r'\[KEEP:\s*(.+?)\]', re.IGNORECASE)
# VIEW reads a slice of a LARGE file directly from disk (no prior [CODE:]
# required). Used when [CODE: bigfile] returns only a SKELETON (file >
# model's context cap) — the model picks a line number from the skeleton
# and uses [VIEW: path lineN] to see actual surrounding content.
#
# Single-line input expands to ~200 lines centered on the line. Range
# input is kept as-is up to a 600-line cap. Auto-extends to the enclosing
# def/class so the model always sees a complete logical unit.
#
# REJECTED on small files (the whole file fits in [CODE:] with a 20k-token
# thinking reserve — use [CODE: path] instead).
VIEW_TAG = re.compile(r'\[VIEW:\s*(.+?)\]', re.IGNORECASE)
# STOP signals "execute my tool calls now, then let me continue thinking."
# Robust two-tag combination — both halves must appear in order, separated
# by only whitespace/newlines. The CONFIRM_STOP half is a unique token
# that has no other reason to appear in any prose, code, or example: the
# model literally cannot produce it except by deliberate intent.
#
# This robustness matters when the model is editing its OWN codebase and
# constantly discusses signal tags in prose ("how does [STOP] vs [DONE]
# work?"). A single [STOP] alone — anywhere, in any context — does NOT
# fire. Only the full ordered combination does.
#
# Fires (case-insensitive, optional whitespace and one or more newlines
# between the two halves):
#   [STOP]\n[CONFIRM_STOP]        ← canonical: separate lines
#   [STOP][CONFIRM_STOP]          ← also fires: adjacent
#   [STOP]  [CONFIRM_STOP]        ← also fires: same line, spaces
#
# Does NOT fire:
#   [STOP]                              ← bare tag, anywhere
#   "discussion of [STOP] tag"          ← anywhere in prose
#   `[STOP]`                            ← in backticks
#   [STOP] then I'll [CONFIRM_STOP]     ← arbitrary text between halves
#   [CONFIRM_STOP] then [STOP]          ← wrong order
STOP_TAG = re.compile(r'\[STOP\]\s*\[CONFIRM_STOP\]', re.IGNORECASE)
# DONE signals the model is completely finished — apply edits and exit.
# Same two-tag-combination robustness as STOP.
DONE_TAG = re.compile(r'\[DONE\]\s*\[CONFIRM_DONE\]', re.IGNORECASE)
# FORCE DONE — used by the coder when the step requirement is ALREADY MET
# in the file and no edits are needed. Plain [DONE] without any edits
# triggers a retry (the coder might just have forgotten to emit edits);
# [FORCE DONE] is the explicit "I am intentionally producing no edits"
# escape hatch. Same two-tag protocol.
FORCE_DONE_TAG = re.compile(
    r'\[FORCE\s+DONE\]\s*\[CONFIRM_FORCE_DONE\]', re.IGNORECASE,
)
# CONTINUE signals "I'm not done writing my output but I have no tool
# calls — give me another round so I can finish." Used when a long plan,
# review, or analysis would overflow a single response. The runtime
# loops without firing any tool processing and feeds back a CONTINUATION
# banner so the model picks up where it stopped.
# Two-tag protocol identical to STOP/DONE.
CONTINUE_TAG = re.compile(r'\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]', re.IGNORECASE)
# Bare-tag detectors — fire when the model wrote one half of the signal
# but not the other. Used to inject a correction so the model learns the
# combined form instead of looping silently.
_BARE_STOP = re.compile(r'(?<!\[)\[STOP\](?!\s*\[CONFIRM_STOP\])', re.IGNORECASE)
_BARE_DONE = re.compile(r'(?<!\[)\[DONE\](?!\s*\[CONFIRM_DONE\])', re.IGNORECASE)
_BARE_CONTINUE = re.compile(r'(?<!\[)\[CONTINUE\](?!\s*\[CONFIRM_CONTINUE\])', re.IGNORECASE)
_BARE_FORCE_DONE = re.compile(
    r'(?<!\[)\[FORCE\s+DONE\](?!\s*\[CONFIRM_FORCE_DONE\])', re.IGNORECASE,
)
_BARE_PLAN_DONE = re.compile(
    r'(?<!\[)\[PLAN\s+DONE\](?!\s*\[CONFIRM_PLAN_DONE\])', re.IGNORECASE,
)
# DISCARD removes a previously-loaded tool result by its #label.
# Format: [DISCARD: #label]
DISCARD_TAG = re.compile(r'\[DISCARD:\s*#(\w+)\]', re.IGNORECASE)

# ─── PLAN tool — incremental plan drafting ───────────────────────────────────
# The planner uses these blocks to write and refine a plan that persists
# across rounds. The plan is rendered back in [YOUR PLAN] each round with
# line numbers so the planner can surgically edit it.
#
# === PLAN ===  …content…  === END PLAN ===
#   Writes (or rewrites) the entire plan body.
# === PLAN_EDIT ===  [REPLACE LINES N-M] new content [/REPLACE]
#                    [INSERT AFTER LINE N] new content [/INSERT]
#                    === END PLAN_EDIT ===
#   Surgical edits using the SAME line-anchored primitives the coder uses
#   on files. Multiple ops per block are applied bottom-up (highest line
#   first) so earlier line numbers stay valid.
# [PLAN DONE] [CONFIRM_PLAN_DONE]
#   Finalize: return the current plan as the planner's answer and break.
_PLAN_BLOCK = re.compile(
    r'===\s*PLAN\s*===\s*\n?(.*?)\n?===\s*END\s+PLAN\s*===',
    re.DOTALL | re.IGNORECASE,
)
_PLAN_EDIT_BLOCK = re.compile(
    r'===\s*PLAN_EDIT\s*===\s*\n?(.*?)\n?===\s*END\s+PLAN_EDIT\s*===',
    re.DOTALL | re.IGNORECASE,
)
_PLAN_REPLACE_LINES = re.compile(
    r'\[REPLACE\s+LINES\s+(\d+)\s*-\s*(\d+)\]\s*\n?(.*?)\n?\[/REPLACE\]',
    re.DOTALL | re.IGNORECASE,
)
_PLAN_INSERT_AFTER = re.compile(
    r'\[INSERT\s+AFTER\s+LINE\s+(\d+)\]\s*\n?(.*?)\n?\[/INSERT\]',
    re.DOTALL | re.IGNORECASE,
)
PLAN_DONE_TAG = re.compile(
    r'\[PLAN\s+DONE\]\s*\[CONFIRM_PLAN_DONE\]', re.IGNORECASE,
)

# Both forms count as inline reasoning:
#   <think>...</think>     — what the streaming clients wrap reasoning_content in
#   [think]...[/think]     — a bracketed equivalent the model can emit directly
# A model that lacks a reasoning channel (or whose channel is being lost across
# rounds) can use [think]...[/think] and get the same handling: visible in the
# stream, stripped from final plan body, never dispatched as a tool.
_THINK_BLOCK = re.compile(
    r'(?:<think>.*?</think>|\[think\].*?\[/think\])',
    re.DOTALL | re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> and [think]...[/think] blocks so reasoning
    never lands in the plan body.

    The streaming clients wrap reasoning_content in <think>...</think> as it
    arrives, and models without a reasoning channel can emit [think]...[/think]
    directly. Either form gets stripped here so the plan body the coder
    consumes stays clean — reasoning belongs in the reasoning channel, not
    interleaved with REQUIREMENTS / STEPS.
    """
    return _THINK_BLOCK.sub('', text)


# A real plan is a few KB; a salvaged `<think>` dump beyond this is not a plan
# and must never win plan-selection on length (pylint-4551: a 71,104-char think
# blob became "the richest plan" and pointed at the wrong file).
_SALVAGE_MAX_CHARS = 8000

# Markers that signal "the plan starts here" inside otherwise-prose text.
_PLAN_BODY_MARKER = re.compile(
    r'(===\s*PLAN\s*===|##\s*TASK\s*SHAPE|##\s*GOAL|###?\s*STEP\s*\d)',
    re.IGNORECASE,
)


def _salvage_plan_from_think(text: str) -> str:
    """Last-resort recovery for a planner/merger that did its WHOLE plan inside
    the thinking channel and emitted a thin/empty visible body.

    A reasoning model (e.g. glm-5.1) can reason its way to the correct plan
    entirely inside native `<think>` (or a `[think]` block), then stop — leaving
    nothing visible. _strip_think would then zero it and the run would discard a
    correct plan and fall back to a weaker draft (observed on pylint-4551). When
    that happens, the plan IS in the reasoning — pull it back out rather than
    throwing it away. Prefer the slice from the first plan-structure marker
    (=== PLAN ===, ## GOAL, ### STEP, …); else return the full reasoning text.
    Returns '' if there's no reasoning content to salvage."""
    blocks = re.findall(
        r'(?:<think>(.*?)</think>|\[think\](.*?)\[/think\])',
        text, re.DOTALL | re.IGNORECASE,
    )
    reasoning = "\n".join((a or b) for a, b in blocks).strip()
    if not reasoning:
        return ""
    m = _PLAN_BODY_MARKER.search(reasoning)
    out = reasoning[m.start():].strip() if m else reasoning
    # CAP: a 70K think-dump is not a plan. Keep the TAIL (a model's conclusion
    # sits at the end of its reasoning), so an over-long salvage can't win
    # plan-selection on raw length nor flood the coder.
    if len(out) > _SALVAGE_MAX_CHARS:
        out = ("(salvaged from an over-long reasoning dump — tail kept)\n...\n"
               + out[-_SALVAGE_MAX_CHARS:])
    return out


def _apply_plan_edits(current_plan: str, edit_body: str) -> tuple[str, list[str]]:
    """Apply REPLACE LINES / INSERT AFTER ops from a PLAN_EDIT body to the
    current plan. Returns (new_plan, log_lines). Ops are sorted bottom-up
    so earlier line numbers stay valid as later changes shift content.

    Edits target lines in the CURRENT plan. Out-of-range targets are
    skipped with a log entry (the model sees what failed in the manifest).
    """
    lines = current_plan.split('\n') if current_plan else []
    n = len(lines)
    ops: list[tuple[str, int, int, str]] = []  # (kind, start, end, content)
    for m in _PLAN_REPLACE_LINES.finditer(edit_body):
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        ops.append(("replace", a, b, m.group(3)))
    for m in _PLAN_INSERT_AFTER.finditer(edit_body):
        a = int(m.group(1))
        ops.append(("insert", a, a, m.group(2)))
    # Apply bottom-up: largest start first
    ops.sort(key=lambda op: op[1], reverse=True)
    logs: list[str] = []
    for kind, a, b, content in ops:
        new_content_lines = content.split('\n') if content else ['']
        if kind == "replace":
            if a < 1 or a > n or b < 1 or b > n:
                logs.append(
                    f"REPLACE LINES {a}-{b} out of range (plan has {n} lines) — skipped"
                )
                continue
            lines = lines[:a - 1] + new_content_lines + lines[b:]
            logs.append(f"REPLACE LINES {a}-{b}: replaced {b - a + 1} → {len(new_content_lines)} lines")
        else:  # insert
            if a < 0 or a > n:
                logs.append(
                    f"INSERT AFTER LINE {a} out of range (plan has {n} lines) — skipped"
                )
                continue
            lines = lines[:a] + new_content_lines + lines[a:]
            logs.append(f"INSERT AFTER LINE {a}: inserted {len(new_content_lines)} line(s)")
        n = len(lines)
    return '\n'.join(lines), logs


def _render_plan_with_line_numbers(plan: str) -> str:
    """Format the plan with right-aligned 3-digit line numbers for the
    [YOUR PLAN] section. Matches the readability of the iN|code format
    the coder already sees in [CODE:] output.
    """
    if not plan:
        return ""
    lines = plan.split('\n')
    width = max(3, len(str(len(lines))))
    return '\n'.join(f"{i + 1:>{width}}: {ln}" for i, ln in enumerate(lines))
# Label suffix on tool calls — optional #label at the end of the argument.
# E.g. [REFS: process_turn #ref1] — the #ref1 is the label.
_LABEL_SUFFIX = re.compile(r'\s+#(\w+)\s*$')


def _strip_label(tag_arg: str) -> tuple[str, str | None]:
    """Strip optional #label from a tool argument. Returns (clean_arg, label_or_None)."""
    m = _LABEL_SUFFIX.search(tag_arg)
    if m:
        return tag_arg[:m.start()].strip(), m.group(1)
    return tag_arg.strip(), None


def _norm_key(tag_type: str, clean_arg: str) -> str:
    """Build a stable cache/manifest key from (tag_type, clean_arg).

    Trailing whitespace, leading `./`, mixed slash directions, case
    differences, whitespace around range dashes/commas, and label
    suffixes all used to produce DISTINCT keys for the same semantic
    tag — so the model could re-issue `[CODE: foo.py]` and
    `[CODE: ./foo.py]` and have neither hit the cache. Normalization
    collapses every variation we've seen models actually produce.

    Idempotent: `_norm_key(t, _norm_key(t, x)[len(t)+1:]) == _norm_key(t, x)`
    Tested by `_NORM_KEY_SELF_TEST` at module load — if it ever stops
    being idempotent, JARVIS refuses to start.
    """
    s = clean_arg.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    # Re-strip after `./` removal — the path after the prefix may have
    # had leading whitespace (e.g. `./ foo.py` from a model that double-
    # spaced after the dot-slash). Without this, `./ foo.py` keyed
    # differently from `foo.py`, breaking the equivalence class.
    s = s.strip()
    # Collapse runs of internal whitespace to a single space so
    # `[KEEP: foo.py  10-20]` and `[KEEP: foo.py 10-20]` key the same.
    s = re.sub(r'\s+', ' ', s)
    # Normalise spacing around range/list separators inside numeric
    # specs. Models occasionally write `10 - 20` or `10 , 20`; the
    # canonical form has neither leading nor trailing whitespace on
    # `-` and `,` so all variants hash the same.
    s = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', s)
    s = re.sub(r'\s*,\s*', ',', s)
    return f"{tag_type}:{s.lower()}"


# Self-test: every shape we ever expect to see must hash to a fixed-point
# and equivalent inputs must collide. Runs once at import — fast-fails
# loudly if a future edit silently breaks normalization.
_NORM_KEY_SELF_TEST = [
    # (tag_type, [equivalent inputs that MUST share a key])
    ("CODE", ["foo.py", "./foo.py", " foo.py ", "FOO.PY", "foo.py "]),
    ("CODE", ["a/b.py", "a\\b.py", "./a/b.py"]),
    ("VIEW", ["foo.py 100-200", "foo.py  100-200", "foo.py 100 - 200"]),
    ("KEEP", ["foo.py 10-20,30-40", "foo.py 10-20, 30-40", "foo.py 10-20 , 30-40"]),
    ("REFS", ["my_func", "MY_FUNC"]),
]
def _run_norm_key_self_test() -> None:
    for tt, variants in _NORM_KEY_SELF_TEST:
        keys = {_norm_key(tt, v) for v in variants}
        assert len(keys) == 1, (
            f"_norm_key not collapsing equivalents for {tt}: "
            f"{variants} → {keys}"
        )
        # Idempotence: re-applying _norm_key on the stripped arg of a
        # key must yield the same key.
        for v in variants:
            once = _norm_key(tt, v)
            prefix = tt + ":"
            twice = _norm_key(tt, once[len(prefix):])
            assert once == twice, (
                f"_norm_key not idempotent: {v!r} → {once!r} → {twice!r}"
            )

_run_norm_key_self_test()


# Markers our deep-think preambles use, by canonical section name.
# Each entry: (display_name, regex_pattern). When round 1 completes,
# we scan _round_texts[0] against these and collect the ones present —
# the continuation prompt then lists them as "already done, don't redo."
_PREAMBLE_MARKERS = [
    ("DEEP THINK preamble",        re.compile(r'^\s*#{1,3}\s*DEEP\s+THINK\b', re.IGNORECASE | re.MULTILINE)),
    ("REAL GOAL / INTENT section", re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?REAL\s+(?:GOAL|INTENT)\b', re.IGNORECASE | re.MULTILINE)),
    ("HARDEST UNKNOWN section",    re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?HARDEST\s+UNKNOWN\b', re.IGNORECASE | re.MULTILINE)),
    ("PRE-MORTEM section",         re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?PRE-?MORTEM\b', re.IGNORECASE | re.MULTILINE)),
    ("APPROACHES / ARCHITECTURES", re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:\d+-?\d*\s+)?(?:APPROACHES|ARCHITECTURES|SUBSTANTIVELY\s+DIFFERENT)\b', re.IGNORECASE | re.MULTILINE)),
    ("BLIND SPOT section",         re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?BLIND\s+SPOT\b', re.IGNORECASE | re.MULTILINE)),
    ("OPEN QUESTIONS list",        re.compile(r'^\s*#{1,3}\s*OPEN\s+QUESTIONS\b', re.IGNORECASE | re.MULTILINE)),
    ("INTEGRATION CHECKLIST",      re.compile(r'^\s*#{1,4}\s*\d?\.?\s*INTEGRATION\s+CHECKLIST\b', re.IGNORECASE | re.MULTILINE)),
    ("REQUIREMENT restatement",    re.compile(r'^\s*#{1,4}\s*\d?\.?\s*REQUIREMENT\b', re.IGNORECASE | re.MULTILINE)),
    ("PLAN OF EDITS",              re.compile(r'^\s*#{1,4}\s*\d?\.?\s*PLAN\s+OF\s+EDITS\b', re.IGNORECASE | re.MULTILINE)),
    ("WHAT COULD GO WRONG",        re.compile(r'^\s*#{1,4}\s*\d?\.?\s*WHAT\s+COULD\s+GO\s+WRONG\b', re.IGNORECASE | re.MULTILINE)),
    ("WHAT MUST BE TRUE",          re.compile(r'^\s*#{1,4}\s*\d?\.?\s*WHAT\s+MUST\s+BE\s+TRUE\b', re.IGNORECASE | re.MULTILINE)),
    ("EVIDENCE PLAN",              re.compile(r'^\s*#{1,4}\s*\d?\.?\s*EVIDENCE\s+PLAN\b', re.IGNORECASE | re.MULTILINE)),
]


def _detect_preamble_sections(text: str) -> list[str]:
    """Return the display names of every preamble section detected in `text`.

    Called once after round 1 to remember which deep-think sections the
    model has already completed. The continuation prompt quotes the list
    back as "✓ already done — do NOT redo these" so the model resumes
    instead of restarting its reasoning.
    """
    if not text:
        return []
    found = []
    for name, pat in _PREAMBLE_MARKERS:
        if pat.search(text):
            found.append(name)
    return found


def _build_continue_prompt(
    base_prompt: str,
    round_history_texts: list[str],
    round_num: int,
    max_rounds: int,
    preamble_done: list[str],
) -> str:
    """Construct the prompt for the round AFTER a [CONTINUE][CONFIRM_CONTINUE]
    signal. The model needs to keep writing its output (plan, review, etc.)
    without firing tools and without restarting its reasoning.

    The prompt design:
      1. A loud "CONTINUATION MODE" banner at the very top — this comes
         BEFORE the original task prompt so the model reads it first.
      2. The list of preamble sections already completed in round 1.
      3. The round-by-round history so the model sees where it left off.
      4. An explicit "resume from the last sentence" instruction.

    No CONTEXT MANIFEST, no RESULTS YOU REQUESTED — there were no tools.
    """
    rounds_left = max_rounds - round_num
    # Flow prior output as one continuous stream — no round labels.
    # The model reads its own previous response and continues writing
    # from the last sentence. The horizontal rule between rounds is
    # minimal: a signal that streaming paused/resumed, not a banner.
    history_block = "\n\n────────\n\n".join(round_history_texts)

    # The system prompt is kept intact so the model still has every
    # rule, role description, and signal definition it needs — only the
    # framing around its own prior output changes. The work-so-far is
    # streamed as one continuous narrative (no round labels) so it
    # feels like one ongoing response, not a series of restarts.
    return f"""{base_prompt}

══════════════════════════════════════════════════════════════════════
YOUR WORK SO FAR (continuous — you signaled [CONTINUE] for more space)
══════════════════════════════════════════════════════════════════════
{history_block}

──────────────────────────────────────────────────────────────────────
↓ Continue writing from where you stopped. Same response, same thought
  stream — just more space. No tools this round (you said you don't
  need any). When you finish, end with [DONE][CONFIRM_DONE], or signal
  another [CONTINUE][CONFIRM_CONTINUE] if you still need more space.
  Budget: {rounds_left} round(s) remain.
{("  (Already-written sections — don't restate, but revise if needed: "
  + ", ".join(preamble_done[:3])
  + ("…" if len(preamble_done) > 3 else "") + ")") if preamble_done else ""}
──────────────────────────────────────────────────────────────────────"""


# ─── Quote / Edit-block masking ─────────────────────────────────────────────
# Models often DISCUSS tool tags in prose ("I'll then [KEEP: file 50-80]")
# or copy them inside fenced code blocks while explaining a plan. The naive
# regex extractors used to fire on those, sending the model into a loop where
# every round it explained that it was about to call a tool, and the system
# went and called it again. The model never gets a chance to think.
#
# We mask out tool-tag-shaped substrings that appear inside:
#   1. Backtick-quoted spans (`...` and ```...```)
#   2. `=== EDIT: ... === ... [/REPLACE]` (or [/INSERT]) blocks — tags inside
#      an open edit block are file CONTENT being inserted, not tool calls.
#   3. Lines explicitly marked with the literal escape "\["  (model
#      convention: write `\[KEEP: ...]` to mention without invoking).
#
# Mask = replace every '[' with '\x00' so tag regexes don't match.  We never
# show the masked text to anyone — just feed it through extractors.

_FENCED_CODE_BLOCK = re.compile(r'```.*?```', re.DOTALL)
_INLINE_BACKTICK = re.compile(r'`[^`\n]+`')
# Matches both reasoning forms — see _strip_think comment above for rationale.
_THINK_BLOCK = re.compile(
    r'(?:<think>.*?</think>|\[think\].*?\[/think\])',
    re.DOTALL | re.IGNORECASE,
)
# Deliberate tool-use blocks: [tool use]...[/tool use]
# When ANY such block is present in the response, ONLY tags inside these
# blocks are executed — everything outside is treated as explanatory text.
# This prevents accidental/hallucinated tool calls.
_TOOL_USE_BLOCK = re.compile(r'\[tool use\](.*?)\[/tool use\]', re.DOTALL | re.IGNORECASE)
# Edit/code-writing blocks — content inside is CODE, not tool calls.
# Each pattern covers one form of code writing the coder can produce.
# All are masked so tool tags inside written code never fire accidentally.

# === FILE: ... === ... === END FILE === (full file creation).
# Body MUST end at the literal `=== END FILE ===` terminator. The previous
# cross-section fallback `(?===\s*(?:EDIT|FILE):)` was a string-literal
# trap: when the coder writes a new file that legitimately CONTAINS the
# literal text `=== EDIT:` or `=== FILE:` inside a string (e.g. JARVIS
# source rewriting its own prompts), the fallback terminated the block
# early, and every tool tag after that point in the response could fire
# spuriously.
#
# Trade-off: an UNTERMINATED `=== FILE:` block now masks the rest of the
# response. We surface that condition explicitly via
# `_detect_unterminated_blocks` so the model sees a loud warning and
# correction nudge instead of silent damage.
_EDIT_FILE_SPAN = re.compile(
    r'===\s*FILE:.*?===\s*END\s+FILE\s*===',
    re.DOTALL | re.IGNORECASE,
)
# [SEARCH]...[/SEARCH] — code the coder is searching for
_SEARCH_BLOCK = re.compile(r'\[SEARCH[^\]]*\](.*?)\[/SEARCH\]', re.DOTALL | re.IGNORECASE)
# [REPLACE]...[/REPLACE] — replacement code
_REPLACE_BLOCK = re.compile(r'\[REPLACE[^\]]*\](.*?)\[/REPLACE\]', re.DOTALL | re.IGNORECASE)
# [INSERT AFTER LINE N]...[/INSERT] — inserted code
_INSERT_BLOCK = re.compile(r'\[INSERT[^\]]*\](.*?)\[/INSERT\]', re.DOTALL | re.IGNORECASE)
# === EDIT/FILE: ... <terminator>. Terminator is one of:
#   [/edit]           — primary (v11), closes an `[edit]…[/edit]` block
#   === END EDIT ===  — v11 envelope close for `=== EDIT:`
#   [/REPLACE]        — closes a SEARCH/REPLACE edit block (legacy/fallback)
#   [/INSERT]         — closes an INSERT AFTER block
#   === END FILE ===  — closes a `=== FILE:` new-file body
# The cross-section fallback `(?===\s*(?:EDIT|FILE):)` was REMOVED because
# string-literal occurrences of `=== EDIT:`/`=== FILE:` inside an edit
# body terminated the span early.
_EDIT_BLOCK_SPAN = re.compile(
    r'===\s*(?:EDIT|FILE):.*?'
    r'(?:\[/edit\]|===\s*END\s+EDIT\s*===|\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*===)',
    re.DOTALL | re.IGNORECASE,
)
# Whole format-B `=== EDIT: … === END EDIT ===` envelope, masked so a signal /
# tool tag quoted as FILE CONTENT inside an `[edit]` body can't fire. The END
# EDIT is line-anchored (re.M) so a mid-line `[/edit]` / `=== END EDIT ===` in
# the content can't truncate the mask early (the bug _EDIT_BLOCK_SPAN alone has).
_EDIT_ENVELOPE_SPAN = re.compile(
    r'===\s*EDIT:.*?^[ \t]*===\s*END\s+EDIT\s*===',
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
# Detectors for unterminated FILE / EDIT blocks — used to warn the model
# when the strict terminator regexes above masked the rest of the response.
_FILE_HEADER = re.compile(r'===\s*FILE:\s*(\S+)', re.IGNORECASE)
_EDIT_HEADER = re.compile(r'===\s*EDIT:\s*(\S+)', re.IGNORECASE)
_FILE_TERMINATOR = re.compile(r'===\s*END\s+FILE\s*===', re.IGNORECASE)
_EDIT_TERMINATOR = re.compile(
    r'\[/edit\]|===\s*END\s+EDIT\s*===|\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*===',
    re.IGNORECASE,
)
_BACKSLASH_BRACKET = re.compile(r'\\\[')

# Plan-body spans — used by signal masking so that a planner writing a
# `=== PLAN === ... === END PLAN ===` body that documents the JARVIS
# signal protocol cannot accidentally fire its own [PLAN DONE] from
# inside that body. The plan body is data (the artifact handed to the
# coder); signals belong OUTSIDE it. Matched non-greedily and applied
# in `_mask_quoted_tags_core` alongside the EDIT/FILE block masks.
_PLAN_BLOCK_SPAN = re.compile(
    r'===\s*PLAN\s*===.*?===\s*END\s+PLAN\s*===',
    re.DOTALL | re.IGNORECASE,
)
_PLAN_EDIT_BLOCK_SPAN = re.compile(
    r'===\s*PLAN_EDIT\s*===.*?===\s*END\s+PLAN_EDIT\s*===',
    re.DOTALL | re.IGNORECASE,
)
_PLAN_OPEN = re.compile(r'===\s*PLAN\s*===', re.IGNORECASE)
_PLAN_CLOSE = re.compile(r'===\s*END\s+PLAN\s*===', re.IGNORECASE)
_PLAN_EDIT_OPEN = re.compile(r'===\s*PLAN_EDIT\s*===', re.IGNORECASE)
_PLAN_EDIT_CLOSE = re.compile(r'===\s*END\s+PLAN_EDIT\s*===', re.IGNORECASE)

# PLAN_DONE context validation. A PLAN_DONE pair fires only when it
# appears in one of these structurally valid positions:
#
#   1. After `=== END PLAN ===`     (signal terminates a closed plan body)
#   2. After a canonical terminal   (e.g. ## VERIFICATION) — the model wrote
#      section header                  the conventional "I'm done" section
#   3. After a closed [think] /     (model reasoned about an early commit
#      <think> block                  and is explicitly justifying it)
#
# Everything else is treated as a stray signal: PLAN_DONE written mid-
# investigation, written inside prose that's documenting the protocol,
# or emitted by accident. The runtime injects a one-shot correction
# explaining the rule and gives the model another round.
#
# The terminal-section list covers the canonical names used by:
#   - PLAN_COT_EXISTING       → ## VERIFICATION
#   - PLAN_COT_NEW            → ## CONFIDENCE GATE / ## TEST CRITERIA
#   - MERGE_PROMPT_TEMPLATE   → ## PRE-MORTEM RESOLUTION
#   - informal / merger tail  → ## FINAL NOTES, ## SUMMARY
# Add a new name here when a workflow introduces a new terminal section.
_PLAN_TERMINAL_SECTION = re.compile(
    r'^[ \t]*##[ \t]*'
    r'(?:VERIFICATION'
    r'|CONFIDENCE[ \t]+GATE'
    r'|PRE[-\s]?MORTEM[ \t]+RESOLUTION'
    r'|TEST[ \t]+CRITERIA'
    r'|FINAL[ \t]+NOTES'
    r'|SUMMARY)'
    r'\b',
    re.MULTILINE | re.IGNORECASE,
)
_PLAN_END_BLOCK = re.compile(r'===\s*END\s+PLAN\s*===', re.IGNORECASE)
_THINK_CLOSE_TAG = re.compile(r'</think>|\[/think\]', re.IGNORECASE)
# Generous look-back windows — the model may write a couple of paragraphs
# between the canonical marker and the actual signal (e.g. "the plan is
# complete; the user will observe X; ready to commit" + [PLAN DONE]).
_PLAN_DONE_LONG_LOOKBACK = 2000   # for END PLAN / terminal section
_PLAN_DONE_SHORT_LOOKBACK = 800   # for [/think] (kept tighter because
                                  # the model should think THEN commit
                                  # immediately, not write more prose)
# Below this many chars, a visible "plan" is empty/boilerplate — try to salvage
# the real plan from the thinking channel before falling back to a weaker draft.
_PLAN_SALVAGE_THRESHOLD = 200


# Backtrack-in-response directive. Models can write
#
#   [continue from: -N]
#
# on its own line to erase the N lines IMMEDIATELY PRECEDING the
# directive (plus the directive's own line) before downstream processing
# sees the response. Use case: the model wrote a wrong edit or a wrong
# plan step, then in [think] realized the mistake; instead of explaining
# the wrong content in its visible output it backtracks and rewrites.
#
# Directives inside masked contexts (code fences, backticks, [think] /
# <think> blocks) are treated as documentation and NOT applied — quoting
# the syntax in a prompt or example doesn't fire it. Anywhere else, the
# directive fires.
#
# N is a positive integer. N=0 or N>500 is treated as invalid and the
# directive is stripped (but no content erased). Multiple directives in
# one response are applied in document order; each operates on the state
# produced by the previous one.
_CONTINUE_FROM_RE = re.compile(
    r'\[continue\s+from:\s*-(\d+)\s*\]',
    re.IGNORECASE,
)


def _apply_continue_from(text: str) -> str:
    """Apply [continue from: -N] backtrack directives to `text`.

    Each application:
      1. Locates the first directive in the masked view of the text
         (so directives inside fences / backticks / think blocks are
         skipped — they're documentation, not commands).
      2. Walks back N newlines in the ORIGINAL text from the directive
         position, finding the start of the N-th line above the
         directive's own line. Position 0 if N exceeds available
         lines (erases everything before the directive).
      3. Removes the range [cut_at, directive_end_plus_trailing_newline)
         from the original text. The directive's line vanishes; the
         lines above it are gone.
      4. Loops to handle the next directive.

    Returns the rewritten text. Idempotent on text with no directives.
    """
    while True:
        masked = _mask_for_signals(text)
        m = _CONTINUE_FROM_RE.search(masked)
        if not m:
            return text
        try:
            n = int(m.group(1))
        except ValueError:
            n = 0
        if n <= 0 or n > 500:
            # Malformed or absurd backtrack — strip the directive,
            # keep surrounding text. The 500-line ceiling is a sanity
            # guard against pathological models.
            text = text[:m.start()] + text[m.end():]
            continue

        before = text[:m.start()]
        newline_positions = [i for i, ch in enumerate(before) if ch == '\n']
        # To erase N content lines plus the directive's own line prefix,
        # we cut starting right after the (n+1)-th-from-end newline.
        # Example: text = "A\nB\nC\n[continue from: -2]\nD"
        #          before = "A\nB\nC\n"      newlines at [1, 3, 5]
        #          n=2 -> needed=3 -> newline_positions[-3]=1 -> cut_at=2
        #          (start of "B")
        # If we don't have enough newlines, cut from 0 (erase all prefix).
        if len(newline_positions) >= n + 1:
            cut_at = newline_positions[-(n + 1)] + 1
        else:
            cut_at = 0

        end = m.end()
        # Consume the directive's trailing newline so we don't leave a
        # blank gap where it used to live.
        if end < len(text) and text[end] == '\n':
            end += 1
        text = text[:cut_at] + text[end:]


def _plan_done_context_kind(text: str, signal_start: int) -> "str | None":
    """Return a short name for the valid context preceding a PLAN_DONE
    match, or None if no valid context is present.

    Context names are also used as diagnostics in the runtime logs so a
    debugger can see WHY a particular [PLAN DONE] was honored.
    """
    long_window = text[max(0, signal_start - _PLAN_DONE_LONG_LOOKBACK):signal_start]
    if _PLAN_END_BLOCK.search(long_window):
        return "end-plan-block"
    if _PLAN_TERMINAL_SECTION.search(long_window):
        return "terminal-section"
    short_window = text[max(0, signal_start - _PLAN_DONE_SHORT_LOOKBACK):signal_start]
    if _THINK_CLOSE_TAG.search(short_window):
        return "post-think"
    return None


def _detect_unterminated_blocks(text: str) -> list[tuple[str, str]]:
    """Return a list of (kind, filepath) for every `=== FILE:` or `=== EDIT:`
    header that has no matching terminator after it.

    Used to surface a loud warning to the model when the strict masking
    regexes (`_EDIT_FILE_SPAN`, `_EDIT_BLOCK_SPAN`) failed to find a
    closer, which would otherwise silently mask every tool tag after the
    unterminated header.
    """
    issues: list[tuple[str, str]] = []
    # FILE blocks
    for m in _FILE_HEADER.finditer(text):
        after = text[m.end():]
        if not _FILE_TERMINATOR.search(after):
            issues.append(('FILE', m.group(1).strip()))
    # EDIT blocks
    for m in _EDIT_HEADER.finditer(text):
        after = text[m.end():]
        if not _EDIT_TERMINATOR.search(after):
            issues.append(('EDIT', m.group(1).strip()))
    return issues


def _mask_quoted_tags_core(text: str, enforce_tool_use_blocks: bool) -> str:
    """Inner mask: applies backtick / fenced / think / edit-block / escape
    masking to `text`. Optionally also applies the [tool use] block
    enforcement (mask every `[` outside [tool use]...[/tool use] regions).

    The reason this is split into a parameter:
      • For TAG EXTRACTION ([CODE:], [REFS:], etc.), the [tool use] block
        enforcement is correct — only deliberately wrapped tags fire.
      • For SIGNAL DETECTION ([STOP][CONFIRM_STOP] et al.), the [tool use]
        enforcement is WRONG — the signal is supposed to go OUTSIDE the
        block (right after [/tool use]). Masking outside-of-block `[`
        chars hides the signal from the runtime, and the model thinks
        nothing fired. Use enforce_tool_use_blocks=False for signals.
    """
    if not text or '[' not in text:
        return text

    masked = list(text)

    def _blank(start: int, end: int) -> None:
        for i in range(start, min(end, len(masked))):
            if masked[i] == '[':
                masked[i] = '\x00'

    # 0. <think>...</think> blocks — model's internal reasoning.
    for m in _THINK_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 0b. UNCLOSED <think>...EOT — during streaming the closing tag
    # hasn't arrived yet. Without this, a model that mentions
    # [STOP][CONFIRM_STOP] inside its still-open thinking (e.g. while
    # discussing the protocol) triggers stop_check, the stream aborts
    # mid-thought, and the model never returns to write its real
    # response. Observed before this fix: qwen-3.5 / minimax-m2.7
    # rounds that ended on a [CONFIRM_STOP] inside an unterminated
    # <think> block, with no visible content after. Mask from the
    # FIRST unclosed <think> to end of text.
    _think_opens = [m.start() for m in re.finditer(r'<think>', text, re.IGNORECASE)]
    _think_closes = [m.start() for m in re.finditer(r'</think>', text, re.IGNORECASE)]
    if len(_think_opens) > len(_think_closes):
        # The (len(closes))th open (0-indexed) is the first one unclosed.
        _blank(_think_opens[len(_think_closes)], len(text))

    # 1. Fenced code blocks (```...```)
    for m in _FENCED_CODE_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 1b. UNCLOSED fenced code (```...EOT) — same streaming risk.
    # If the model opens ``` and mentions a signal inside before the
    # closing ``` arrives, the signal would fire prematurely. Count
    # the ``` markers; if odd, the last one is unclosed.
    _fence_positions = [m.start() for m in re.finditer(r'```', text)]
    if len(_fence_positions) % 2 == 1:
        _blank(_fence_positions[-1], len(text))

    # 2. Inline backtick spans (`...`)
    for m in _INLINE_BACKTICK.finditer(text):
        _blank(m.start(), m.end())

    # 2b. UNCLOSED inline backticks (`...EOL or `...EOT) — during
    # streaming the closing ` may not have arrived yet. Observed in
    # 20260513_173120 glm-5.1: model was streaming
    #   "Two-tag protocol: `[STOP][CONFIRM_STOP]`, `[DONE][CONFIRM_DONE]…"
    # and the stop_check ran when [CONFIRM_DONE] arrived — at that
    # moment the second ` was still unclosed (it would have arrived in
    # a later delta). The regex above didn't match the unclosed span,
    # so [DONE][CONFIRM_DONE] wasn't masked → DONE_TAG fired → stream
    # aborted → the closing ` never arrived. Mirror the unclosed-think
    # / unclosed-fence fixes: walk the text, find any ` that isn't
    # paired (next ` on the same line), mask from there to end-of-line
    # (or end-of-text if no newline).
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '`' and masked[i] == '`':  # not already masked
            # Skip if this is part of a triple-fence already handled.
            if i + 2 < n and text[i + 1] == '`' and text[i + 2] == '`':
                i += 3
                continue
            # Find the next single ` on the same line.
            j = i + 1
            while j < n and text[j] != '`' and text[j] != '\n':
                j += 1
            if j < n and text[j] == '`':
                # Found closing — regex above already handled it. Move past.
                i = j + 1
            else:
                # Unclosed. Mask from i to end-of-line (or end-of-text).
                _blank(i, j)
                i = j
        else:
            i += 1

    # 3. Code-writing blocks — mask all forms where the model writes actual code.
    for pattern in (_EDIT_ENVELOPE_SPAN, _EDIT_FILE_SPAN, _SEARCH_BLOCK,
                    _REPLACE_BLOCK, _INSERT_BLOCK, _EDIT_BLOCK_SPAN):
        for m in pattern.finditer(text):
            _blank(m.start(), m.end())

    # 3b. Plan / plan-edit body spans — these are the planner's artifact.
    # Signals (PLAN_DONE, STOP, DONE, CONTINUE) written INSIDE a
    # `=== PLAN === ... === END PLAN ===` body are data, not commands —
    # the model is documenting the protocol or quoting an example. The
    # real signal lives AFTER `=== END PLAN ===`. Masking the body span
    # prevents the documented-signal-fires-itself failure mode.
    for pattern in (_PLAN_BLOCK_SPAN, _PLAN_EDIT_BLOCK_SPAN):
        for m in pattern.finditer(text):
            _blank(m.start(), m.end())

    # 3c. UNCLOSED plan / plan-edit blocks — during streaming the closing
    # `=== END PLAN ===` may not have arrived yet. Without this, the
    # model could write a signal pair inside the still-open plan body
    # and stop_check would fire mid-stream, aborting the plan before it
    # completes. Same shape as the unclosed-think / unclosed-fence
    # guards above. We use the LAST unclosed opener as the mask start.
    for open_re, close_re in (
        (_PLAN_OPEN, _PLAN_CLOSE),
        (_PLAN_EDIT_OPEN, _PLAN_EDIT_CLOSE),
    ):
        opens = [m.start() for m in open_re.finditer(text)]
        closes = [m.start() for m in close_re.finditer(text)]
        if len(opens) > len(closes):
            # The (len(closes))th open (0-indexed) is the first unclosed.
            _blank(opens[len(closes)], len(text))

    # 4. Explicit escape: `\[TAG: ...]` → mask just the leading `[`
    for m in _BACKSLASH_BRACKET.finditer(text):
        idx = m.end() - 1
        if 0 <= idx < len(masked) and masked[idx] == '[':
            masked[idx] = '\x00'

    # 5. [tool use]...[/tool use] enforcement (only when requested).
    if enforce_tool_use_blocks:
        tool_use_blocks = list(_TOOL_USE_BLOCK.finditer(text))
        if tool_use_blocks:
            inside = set()
            for m in tool_use_blocks:
                inside.update(range(m.start(1), m.end(1)))
            for i in range(len(masked)):
                if masked[i] == '[' and i not in inside:
                    masked[i] = '\x00'

    return ''.join(masked)


def _mask_quoted_tags(text: str) -> str:
    """FULL mask for tag extraction — applies all rules including
    [tool use] block enforcement. Only deliberately-wrapped tool tags
    survive this mask. Use this when extracting [CODE:], [REFS:], etc.
    """
    return _mask_quoted_tags_core(text, enforce_tool_use_blocks=True)


def _mask_for_signals(text: str) -> str:
    """Signal-detection mask — applies backtick / fenced / escape rules
    so the model can SAFELY discuss [STOP][CONFIRM_STOP] in prose, but
    does NOT apply [tool use] block enforcement. That second rule was
    causing two-tag signals written OUTSIDE [tool use] blocks (the
    canonical position — right after [/tool use]) to be masked out,
    which made the runtime miss the signal entirely and the model
    hallucinate tool results that never came.
    """
    return _mask_quoted_tags_core(text, enforce_tool_use_blocks=False)


# Module-level: these patterns never change — compile once, not per call.
# 1. Pure line-range patterns like "339-342" — [SEARCH: N-M] anchored edit
# 2. File paths like "ui/index.html" — [SEARCH: filepath] edit reference
# Routing these to ripgrep produces garbage and loops the model.
_SEARCH_LINE_RANGE = re.compile(r'^\d+\s*-\s*\d+$')
_SEARCH_FILE_PATH = re.compile(r'\.\w{1,5}$')

# Tag-arg validators. Models occasionally produce malformed args like
# `[CODE: I want to read workflows/code.py please]` because their training
# data has prose-shaped tool calls. We reject anything that doesn't fit
# the expected shape so a single misformed tag can't blow up the loop.
#
# Path-shaped arg: letters/digits/_/./-//+ plus optional trailing line
# spec ("file.py 100-200", "file.py 4849"). Up to one whitespace block
# separating path from spec. No commas inside a single tag's arg — the
# model writes multiple tags for multiple paths.
_PATH_ARG_RE = re.compile(
    r'^[\w./\-+]+(?:\s+\d+(?:\s*-\s*\d+)?'
    r'(?:\s*,\s*\d+\s*-\s*\d+)*)?\s*$'
)
# Identifier-shaped arg: REFS/LSP target a single symbol. Allow dots
# (e.g. `module.func`) but not whitespace or punctuation.
_IDENT_ARG_RE = re.compile(r'^[\w.]+$')


def _arg_looks_path(arg: str) -> bool:
    return bool(_PATH_ARG_RE.match(arg.strip()))


def _arg_looks_ident(arg: str) -> bool:
    return bool(_IDENT_ARG_RE.match(arg.strip()))


# Per-round cap on the number of tags we'll FIRE. The model has written
# 12+ tags in a single block in practice (deepseek's 8 KEEPs); this lets
# legitimate exploration through but rejects pathological 50-tag dumps
# that would overflow the prompt budget on the next round.
MAX_TAGS_PER_ROUND = 15


def extract_search_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    results = []
    for q in SEARCH_TAG.findall(masked):
        clean, _ = _strip_label(q)
        stripped = clean.strip()
        if _SEARCH_LINE_RANGE.match(stripped):
            continue  # anchored edit syntax [SEARCH: 45-49]
        if _SEARCH_FILE_PATH.search(stripped) and ' ' not in stripped:
            continue  # file path like "ui/index.html", not a search query
        results.append(q)
    return results

def extract_websearch_tags(text: str) -> list[str]:
    return WEBSEARCH_TAG.findall(_mask_quoted_tags(text))

def extract_detail_tags(text: str) -> list[str]:
    return DETAIL_TAG.findall(_mask_quoted_tags(text))

def extract_code_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    out = []
    for raw in CODE_TAG.findall(masked):
        clean, _lbl = _strip_label(raw)
        # Validate: must look like a file path, not prose. Without this,
        # `[CODE: I want to see workflows/code.py]` would route the whole
        # sentence into the path resolver and fail in confusing ways.
        if _arg_looks_path(clean):
            out.append(raw)
    return out

def extract_refs_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    out = []
    for raw in REFS_TAG.findall(masked):
        clean, _lbl = _strip_label(raw)
        # REFS args are symbol names — no whitespace, no slashes.
        if _arg_looks_ident(clean):
            out.append(raw)
    return out

def extract_purpose_tags(text: str) -> list[str]:
    return PURPOSE_TAG.findall(_mask_quoted_tags(text))

def extract_semantic_tags(text: str) -> list[str]:
    return SEMANTIC_TAG.findall(_mask_quoted_tags(text))

def extract_lsp_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    out = []
    for raw in LSP_TAG.findall(masked):
        clean, _lbl = _strip_label(raw)
        if _arg_looks_ident(clean):
            out.append(raw)
    return out

def extract_knowledge_tags(text: str) -> list[str]:
    return KNOWLEDGE_TAG.findall(_mask_quoted_tags(text))

def extract_keep_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    out = []
    for raw in KEEP_TAG.findall(masked):
        clean, _lbl = _strip_label(raw)
        if _arg_looks_path(clean):
            out.append(raw)
    return out

def extract_view_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    out = []
    for raw in VIEW_TAG.findall(masked):
        clean, _lbl = _strip_label(raw)
        if _arg_looks_path(clean):
            out.append(raw)
    return out

def extract_discard_tags(text: str) -> list[str]:
    """Extract #labels from [DISCARD: #label] tags."""
    return DISCARD_TAG.findall(_mask_quoted_tags(text))

def has_tool_tags(text: str) -> bool:
    masked = _mask_quoted_tags(text)
    return bool(SEARCH_TAG.search(masked) or WEBSEARCH_TAG.search(masked)
                or DETAIL_TAG.search(masked) or CODE_TAG.search(masked)
                or REFS_TAG.search(masked) or PURPOSE_TAG.search(masked)
                or SEMANTIC_TAG.search(masked)
                or LSP_TAG.search(masked) or KNOWLEDGE_TAG.search(masked)
                or KEEP_TAG.search(masked) or VIEW_TAG.search(masked)
                or DISCARD_TAG.search(masked))


# ─── Tool Runners ────────────────────────────────────────────────────────────

async def _run_code_searches(patterns: list[str], project_root: str) -> str:
    """Run ripgrep code searches. Returns formatted results."""
    from tools.codebase import search_code, format_search_results

    output_parts = []
    for pattern in patterns:
        status(f"    Code search: {pattern}")
        results = search_code(pattern, project_root)
        if results:
            output_parts.append(f"\n=== Code search: '{pattern}' ===")
            output_parts.append(format_search_results(results))
        else:
            output_parts.append(f"\n=== Code search '{pattern}': no matches ===")
    return "\n".join(output_parts)


async def _run_web_searches(queries: list[str]) -> str:
    """Run web searches. Returns formatted results."""
    output_parts = []
    for query in queries:
        status(f"    Web search: {query}")
        try:
            from tools.search import web_search
            results = await web_search(query, max_results=3)
            if results:
                output_parts.append(f"\n=== Web search: '{query}' ===")
                for r in results:
                    title = r.get("title", "")
                    content = r.get("content", "")[:500]
                    url = r.get("url", "")
                    output_parts.append(f"  {title}")
                    if url:
                        output_parts.append(f"  URL: {url}")
                    if content:
                        output_parts.append(f"  {content}")
                    output_parts.append("")
            else:
                output_parts.append(f"\n=== Web search '{query}': no results ===")
        except Exception as e:
            warn(f"Web search failed for '{query}': {e}")
            output_parts.append(f"\n=== Web search '{query}': error — {e} ===")
    return "\n".join(output_parts)


# ─── Detail Lookup ───────────────────────────────────────────────────────────

def _run_detail_lookups(section_names: list[str], detailed_map: str,
                        project_root: str | None = None) -> str:
    """Look up sections from the detailed code map.

    When `detailed_map` is present (full JARVIS pipeline, Phase-1 ran),
    serve sections from it. Otherwise (CLI / exploration mode / no
    detailed_map built), fall back to the exploration-mode DETAIL: a
    deep dive on a single named symbol via AST + ripgrep.
    """
    from tools.code_index import get_detail_section
    from core.exploration_tools import extract_detail

    output_parts = []
    for name in section_names:
        status(f"    Detail lookup: {name}")
        if detailed_map:
            section = get_detail_section(detailed_map, name)
            if section.startswith("(no section found"):
                # Unavailable subject — list what IS in the map so the model
                # can pick a real one instead of re-guessing the same name.
                try:
                    from tools.code_index import list_sections
                    avail = list_sections(detailed_map)
                except Exception:
                    avail = []
                avail_str = (", ".join(avail[:20]) + ("" if len(avail) <= 20 else f" (+{len(avail)-20} more)")) if avail else "(none)"
                output_parts.append(
                    f"\n⚠ DETAIL: no subject '{name}' in the code map. "
                    f"Available subjects: {avail_str}. "
                    f"Use one of those exactly, or [SEARCH: {name}] / [REFS: {name}] to locate it."
                )
            else:
                output_parts.append(f"\n=== Detail: '{name}' ===\n{section}")
        elif project_root:
            output_parts.append("\n" + extract_detail(name, project_root))
        else:
            output_parts.append(
                f"\n=== Detail: '{name}' — no detailed_map or project_root ===\n"
            )
    return "\n".join(output_parts)


# ─── Code File Reader ───────────────────────────────────────────────────────

def _diff_source_tag(sandbox_path: str, project_path: str, source: str | None) -> str:
    """Produce a header annotation ONLY when it's load-bearing.

    Reading mode (subagent feedback): "from sandbox" on every CODE/VIEW
    result is noise when the sandbox content equals disk. Suppress it
    in that case. Surface "from sandbox (edited)" only when the two
    actually differ, so a developer mid-edit still gets the signal.
    """
    import os as _os
    if source == "project":
        # Disk path was used directly (no sandbox copy yet). Quiet —
        # that's the normal first-read state.
        return ""
    if not source:
        return ""
    # source == "sandbox": compare with disk.
    try:
        if not _os.path.isfile(sandbox_path):
            return ""
        if not _os.path.isfile(project_path):
            return "from sandbox (created)"
        sb_size = _os.path.getsize(sandbox_path)
        pj_size = _os.path.getsize(project_path)
        if sb_size != pj_size:
            return "from sandbox (edited)"
        # Cheap byte compare for small/medium files. Skip for very large.
        if sb_size > 2_000_000:
            return ""
        with open(sandbox_path, "rb") as a, open(project_path, "rb") as b:
            if a.read() != b.read():
                return "from sandbox (edited)"
    except Exception:
        pass
    return ""


def _parse_code_arg(raw: str) -> tuple[str, list[tuple[int, int]] | None]:
    """Parse a [CODE: ...] argument into (filepath, optional_line_ranges).

    Handles:
      [CODE: ui/server.py]           → ("ui/server.py", None)
      [CODE: ui/server.py 87-95]     → ("ui/server.py", [(87, 95)])
      [CODE: ui/server.py 87-95, 200-250]  → ("ui/server.py", [(87, 95), (200, 250)])
      [CODE: main.py 390-505]        → ("main.py", [(390, 505)])
    """
    raw = raw.strip()
    # Match a trailing sequence of "N-M" ranges (optionally comma-separated)
    # after the filepath.  The filepath itself never contains digits-dash-digits
    # as a trailing token.
    range_pat = re.compile(r'\s+((?:\d+\s*-\s*\d+)(?:\s*,\s*\d+\s*-\s*\d+)*)\s*$')
    m = range_pat.search(raw)
    if not m:
        return raw, None
    filepath = raw[:m.start()].strip()
    ranges = []
    for rng in re.findall(r'(\d+)\s*-\s*(\d+)', m.group(1)):
        ranges.append((int(rng[0]), int(rng[1])))
    return filepath, ranges if ranges else None


# Patterns for skeleton extraction — matches top-level structural lines
# across the languages we usually see. Each entry: (regex, label).
# The regex must capture an identifier or a useful signature fragment.
# We anchor on column 0 (top-level) plus 1 level of indent (4 spaces or
# a tab) to also pick up methods and class-level functions.
_SKELETON_PATTERNS = [
    # Python
    (re.compile(r'^(?:    |\t)?(?:async\s+)?def\s+(\w+)\s*\(', re.MULTILINE), 'def'),
    (re.compile(r'^(?:    |\t)?class\s+(\w+)', re.MULTILINE), 'class'),
    (re.compile(r'^([A-Z_][A-Z0-9_]{2,})\s*=', re.MULTILINE), 'CONST'),
    # JavaScript / TypeScript
    (re.compile(r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', re.MULTILINE), 'function'),
    (re.compile(r'^(?:export\s+)?class\s+(\w+)', re.MULTILINE), 'class'),
    (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=', re.MULTILINE), 'const'),
    # Markdown / reST headers
    (re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE), 'header'),
]


def _build_file_skeleton(
    all_lines: list[str], max_items: int = 200, filename: str = "",
) -> str:
    """Return a compact skeleton of a file: top-level / one-indent
    definitions with their line numbers. Used when [CODE:] is called on
    a file too large to return in full — the skeleton lets the model
    decide which line ranges to ask for via [KEEP:].

    Output format (one per line):
      LNNNN  def function_name
      LNNNN  class ClassName
      LNNNN  CONST_NAME
      LNNNN  ## Section Header

    For Python files we parse with `ast` so docstrings, comments, and
    string-literal occurrences of `def foo()` / `class Bar` don't pollute
    the skeleton. For non-Python files we fall back to regex but skip any
    match whose containing line starts with a comment-prefix char (`#`,
    `//`, `*`) — covers the most common pollution.

    Caps the number of items at `max_items` to keep the skeleton tiny —
    the goal is to fit in <2k tokens even for a 10k-line file.
    """
    items: list[tuple[int, str]] = []  # (line_number, label_text)

    # Python: AST parse handles docstring / string-literal false positives.
    if filename.lower().endswith(".py"):
        try:
            import ast
            src = "\n".join(all_lines)
            tree = ast.parse(src, filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    items.append((node.lineno, f"def {node.name}"))
                elif isinstance(node, ast.ClassDef):
                    items.append((node.lineno, f"class {node.name}"))
                elif isinstance(node, ast.Assign):
                    # Module-level UPPER_CASE constants only
                    if node.col_offset != 0:
                        continue
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id.isupper() and len(tgt.id) >= 3:
                            items.append((node.lineno, f"CONST {tgt.id}"))
        except SyntaxError:
            # Fall through to regex path — partial / broken syntax still
            # deserves SOMETHING in the skeleton.
            items = []

    if not items:
        # Non-Python or AST-parse failed: regex pass with a comment guard.
        content = "\n".join(all_lines)
        # Build per-line lookup so we can quickly check the containing
        # line's leading non-whitespace char.
        line_starts = [0]
        for i, ch in enumerate(content):
            if ch == '\n':
                line_starts.append(i + 1)

        def _line_of(pos: int) -> int:
            # Binary search for the line containing `pos`.
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= pos:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        def _line_text(line_idx: int) -> str:
            return all_lines[line_idx] if 0 <= line_idx < len(all_lines) else ""

        def _is_comment_line(txt: str) -> bool:
            stripped = txt.lstrip()
            return stripped.startswith(('#', '//', '*'))

        for pattern, kind in _SKELETON_PATTERNS:
            for m in pattern.finditer(content):
                line_no = content.count('\n', 0, m.start()) + 1
                line_idx = line_no - 1
                # Skip matches inside line-comments. (String literals are
                # harder to detect without a real parser; the comment guard
                # alone catches the dominant pollution source.)
                if _is_comment_line(_line_text(line_idx)):
                    continue
                if kind == 'header':
                    level, text = m.group(1), m.group(2).strip()
                    items.append((line_no, f"{level} {text[:80]}"))
                elif kind == 'CONST':
                    items.append((line_no, f"CONST {m.group(1)}"))
                else:
                    items.append((line_no, f"{kind} {m.group(1)}"))

    # De-duplicate by (line_no, label) and sort by line number
    seen: set[tuple[int, str]] = set()
    unique = []
    for ln, lbl in sorted(items, key=lambda t: t[0]):
        key = (ln, lbl)
        if key in seen:
            continue
        seen.add(key)
        unique.append((ln, lbl))

    if len(unique) > max_items:
        # Sample evenly across the file so the user sees structure end-to-end
        step = len(unique) / max_items
        unique = [unique[int(i * step)] for i in range(max_items)]

    if not unique:
        return "(no top-level definitions detected — request a [KEEP:] range)"

    return "\n".join(f"  L{ln:<6} {lbl}" for ln, lbl in unique)


def _is_binary_path(path: str) -> "tuple[bool, str]":
    """Return (is_binary, reason) for path.

    v8.15: shared binary guard used by CODE / VIEW / KEEP. Checks both
    the IGNORE_EXTENSIONS suffix list AND a first-4KB NUL-byte sniff
    (catches files that lie about their extension, or have no extension).
    """
    import os
    from tools.codebase import IGNORE_EXTENSIONS as _IGN_EXT
    if not os.path.isfile(path):
        return False, ""
    ext = os.path.splitext(path)[1].lower()
    if ext in _IGN_EXT:
        return True, f"BINARY FILE ({ext}) — skipped"
    try:
        with open(path, "rb") as _fb:
            head = _fb.read(4096)
        if b"\x00" in head:
            return True, "BINARY FILE (null bytes in first 4KB) — skipped"
    except Exception:
        pass
    return False, ""


async def _run_code_reads(
    filepaths: list[str], project_root: str,
    viewed_versions: "dict[str, str] | None" = None,
    display_mode: str = "prefix",
) -> str:
    """Read source code files from the sandbox.

    Always reads from .jarvis_sandbox/ — that's the working copy where
    all edits are applied. The real project is untouched.

    Supports optional line-range arguments:
      [CODE: path N-M]        → return only lines N through M
      [CODE: path N-M, A-B]   → return multiple ranges

    If `viewed_versions` is provided, the content of every successfully-read
    file is recorded there (keyed by filepath). This is what the model just
    saw, so any [REPLACE LINES X-Y] edits the model writes after this read
    have line numbers relative to THIS content. The on_stop callback uses
    that snapshot as the basis for line edits, instead of whatever the
    file looks like at apply time (which may have changed via earlier
    mid-stream edits in the same response).
    """
    import os
    from tools.codebase import read_file, norm_path, add_line_numbers, file_uses_tabs

    KEEP_HINT_THRESHOLD = 1500  # lines — informational note above this
    KEEP_FORCE_THRESHOLD = 8000 # lines — REQUIRE KEEP above this; full-file
                                # [CODE:] returns a skeleton view instead.
                                # Why: workflows/code.py (~12k lines, ~120k tokens)
                                # would overflow a 200k context when stacked with
                                # the prompt + history. The 8000-line threshold
                                # keeps small/medium files un-truncated for the
                                # exploration workflow (subagent feedback) while
                                # still protecting the truly oversized files.
    sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")

    output_parts = []
    for raw_fpath in filepaths:
        # Parse optional line ranges from the argument
        fpath, line_ranges = _parse_code_arg(raw_fpath)
        fpath = norm_path(fpath.strip())
        if line_ranges:
            range_str = ", ".join(f"{a}-{b}" for a, b in line_ranges)
            status(f"    Reading code: {fpath} (lines {range_str})")
        else:
            status(f"    Reading code: {fpath}")

        content = None
        source = None        # "sandbox" or "project" — tracked for the header
        sandbox_exists = False

        # The SANDBOX is the canonical post-edit state. We always read it
        # first. Silently falling back to the project root file when the
        # sandbox state looks "weird" (empty, starts with `[`) was the
        # bug behind the 19-round step-3 loop: bad edits truncated the
        # sandbox file → fallback served the original 84-line project
        # file → model thought "edit didn't apply" → retried forever.
        # Now we ONLY fall back when the sandbox file genuinely does
        # not exist, and we surface every other condition as an explicit
        # status the model can act on.
        sandbox_path = os.path.join(sandbox_dir, fpath)
        project_path = os.path.join(project_root, fpath)
        # v8.7 fix: if the sandbox file is OLDER than the project file,
        # the project was edited outside the sandbox (e.g. dev work with
        # the user's editor, or branch switching). The stale sandbox view
        # then under-reports line counts, causing VIEW/REPLACE-LINES to
        # fail against real line numbers. Re-copy from project.
        # In production (SWE-bench), each instance gets a fresh sandbox,
        # so this never fires there — but it makes dev/test runs match
        # disk truth.
        if os.path.isfile(sandbox_path) and os.path.isfile(project_path):
            try:
                if os.path.getmtime(project_path) > os.path.getmtime(sandbox_path):
                    import shutil as _shutil
                    _shutil.copy2(project_path, sandbox_path)
            except Exception:
                pass  # mtime check is best-effort
        if os.path.isfile(sandbox_path):
            sandbox_exists = True
            # v8.15 binary guard via shared helper.
            _is_bin, _reason = _is_binary_path(sandbox_path)
            if _is_bin:
                output_parts.append(
                    f"\n=== Code: {fpath} — {_reason}. ===\n"
                )
                continue
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                source = "sandbox"
            except PermissionError as e:
                # v8.15 fix (B20): branch the error so the model gets the
                # correct diagnosis. Permission-denied is NOT encoding
                # corruption; the "[REVERT FILE]" recovery is destructive
                # for that case.
                output_parts.append(
                    f"\n=== Code: {fpath} — PERMISSION DENIED: {e} ===\n"
                    f"The sandbox copy at {sandbox_path} cannot be read "
                    f"due to file permissions / ownership.\n"
                    f"This is NOT an encoding corruption — do NOT use "
                    f"[REVERT FILE]. Check file mode + ownership of the "
                    f"sandbox path, or escalate to the user.\n"
                )
                continue
            except UnicodeDecodeError as e:
                output_parts.append(
                    f"\n=== Code: {fpath} — ENCODING ERROR: {e} ===\n"
                    f"The sandbox copy cannot be decoded as UTF-8. The "
                    f"file may be a non-UTF text encoding or a binary "
                    f"that slipped past the binary guard.\n"
                )
                continue
            except Exception as e:
                # Read failure on the sandbox file is unusual and worth
                # surfacing — don't mask it by serving the original.
                output_parts.append(
                    f"\n=== Code: {fpath} — SANDBOX READ ERROR: {e} ===\n"
                    f"The sandbox copy at {sandbox_path} exists but cannot be read.\n"
                    f"This usually means a prior edit corrupted the file's encoding.\n"
                    f"Recovery: write [REVERT FILE: {fpath}] to restore the pre-edit\n"
                    f"snapshot, then plan the correct edit from clean state.\n"
                )
                continue
        else:
            # Sandbox doesn't have the file. Lazy-load it from project_root
            # INTO the sandbox so subsequent rounds always read from the same
            # path — previously we fell back to project_root inline, which
            # meant R1 read from project, R2 from sandbox (if a later step
            # copied it), and a transient path-resolution failure between
            # those reads showed up as "FILE NOT FOUND" for a file that
            # exists. Pulling into sandbox once keeps every subsequent read
            # consistent.
            project_path = os.path.join(project_root, fpath)
            if os.path.isfile(project_path):
                # v8.15 fix: subagent r15 (B9) found lazy-load branch
                # bypassed binary guards. Check BEFORE copying — saves
                # disk + matches the sibling branch's behavior.
                _is_bin, _reason = _is_binary_path(project_path)
                if _is_bin:
                    output_parts.append(
                        f"\n=== Code: {fpath} — {_reason}. ===\n"
                    )
                    continue
                try:
                    os.makedirs(os.path.dirname(sandbox_path), exist_ok=True)
                    import shutil as _shutil
                    _shutil.copy2(project_path, sandbox_path)
                    sandbox_exists = True
                    with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    source = "sandbox"
                except Exception:
                    # Copy failed (permissions, disk, etc.) — read project
                    # directly so the round still produces useful content.
                    content = read_file(project_path)
                    source = "project"
            else:
                # File doesn't exist in either place — emit a focused FILE NOT
                # FOUND with a list of similarly-named files so the model can
                # fix the path instead of looping on a typo.
                content = None
                source = None

        # Empty-sandbox-file guard: a sandbox file that is exactly empty
        # almost always indicates a destructive edit (the model wrote a
        # `=== FILE: ... ===` that produced no output, or a SEARCH/REPLACE
        # that obliterated the body). Surface this LOUDLY instead of
        # falling back to the project file — the model needs to know the
        # damage so it can REVERT.
        if sandbox_exists and content is not None and content == "":
            # Only cry "destructive edit" if the file HAD content before (a
            # non-empty project version). A brand-new empty file (no prior
            # content, nothing to revert to) gets a neutral message — otherwise
            # the model panics and tries to [REVERT FILE:] a file it just made.
            _proj = os.path.join(project_root, fpath)
            _had_content = os.path.isfile(_proj) and os.path.getsize(_proj) > 0
            if _had_content:
                output_parts.append(
                    f"\n=== Code: {fpath} — SANDBOX FILE IS EMPTY (0 bytes) ===\n"
                    f"⛔ The sandbox copy of {fpath} is now empty, but the original\n"
                    f"   had content. This is almost certainly a destructive edit (a\n"
                    f"   `=== FILE:` rewrite that produced no body, or a SEARCH/REPLACE\n"
                    f"   that matched the whole file and replaced it with nothing).\n"
                    f"\n"
                    f"RECOVERY OPTIONS:\n"
                    f"  1. [REVERT FILE: {fpath}]   — restore the pre-edit snapshot.\n"
                    f"     Do this BEFORE your next [STOP][CONFIRM_STOP], then plan\n"
                    f"     the correct edit from the restored state.\n"
                    f"  2. If the emptying was intentional (rare), write fresh\n"
                    f"     content with a new === FILE: {fpath} === block.\n"
                    f"\n"
                    f"The sandbox is canonical — what you see is what the next\n"
                    f"step's coder will see.\n"
                )
            else:
                output_parts.append(
                    f"\n=== Code: {fpath} — EMPTY (0 bytes, no prior content) ===\n"
                    f"This file exists but has no content yet — it wasn't emptied,\n"
                    f"so there's nothing to revert. Write its content with a\n"
                    f"`=== FILE: {fpath} === … === END FILE ===` block.\n"
                )
            continue

        # Binary / unreadable files return a [... — skipped] string — treat as missing.
        # Only the literal `[BINARY` / `[READ ERROR` / `[FILE NOT FOUND` prefixes from
        # the read_file helper count as failures; legitimate files whose content
        # happens to start with `[` (JSON arrays, TOML arrays of tables, Lua tables,
        # etc.) must NOT be rejected — that was the silent "FILE NOT FOUND" bug on
        # any list-shaped JSON.
        _READ_FAIL_PREFIXES = ("[BINARY", "[READ ERROR", "[FILE NOT FOUND", "[ERROR")
        if content and any(content.startswith(p) for p in _READ_FAIL_PREFIXES):
            output_parts.append(f"\n=== Code: {fpath} — {content.strip()} ===")
            continue

        # `content is None` means the file genuinely doesn't exist anywhere.
        # `content == ""` means the file exists but is empty — that is a
        # legitimate state and must NOT be reported as FILE NOT FOUND
        # (which used to send the model on a hunt for a file that's right
        # there, just empty).
        is_empty_project_file = (
            content == "" and source == "project"
            and not any((content or "").startswith(p) for p in _READ_FAIL_PREFIXES)
        )
        if is_empty_project_file:
            if viewed_versions is not None:
                viewed_versions[fpath] = ""
            output_parts.append(
                f"\n=== Code: {fpath} (0 lines — empty file, from project) ===\n"
                f"(file exists but is empty)"
            )
            continue
        if content is not None and content != "" and not any(
            content.startswith(p) for p in _READ_FAIL_PREFIXES
        ):
            # Record the FULL file version the model is about to see — line
            # numbers in any subsequent [REPLACE LINES] are relative to this.
            if viewed_versions is not None:
                viewed_versions[fpath] = content

            all_lines = content.split('\n')
            # v8.7 fix: subagent found CODE reported 248 lines for a
            # 247-line file (off-by-one). split('\n') on content ending
            # with '\n' produces an extra empty trailing element. Match
            # `wc -l` semantics: count actual logical lines.
            total_lines = len(all_lines)
            if all_lines and all_lines[-1] == '' and content.endswith('\n'):
                total_lines -= 1
                all_lines = all_lines[:-1]

            if line_ranges:
                # Return only the requested line ranges with correct numbering.
                # Format matches add_line_numbers — `LINENO|INDENT|content`
                # (prefix) or `LINENO|<spaces><content>` (whitespace).
                selected_parts = []
                for start, end in line_ranges:
                    start = max(1, start)
                    end = min(total_lines, end)
                    slice_lines = all_lines[start - 1:end]
                    renumbered = []
                    for i, line in enumerate(slice_lines):
                        expanded = line.expandtabs(4)
                        stripped = expanded.lstrip(' ')
                        n_indent = len(expanded) - len(stripped)
                        if display_mode == "whitespace":
                            renumbered.append(f"{start + i}:{' ' * n_indent}{stripped}")
                        else:
                            # v10: line# uses ':' so view lines paste verbatim.
                            renumbered.append(f"{start + i}:{n_indent}|{stripped}")
                    selected_parts.append('\n'.join(renumbered))

                range_str = ", ".join(f"{a}-{b}" for a, b in line_ranges)
                combined = '\n'.join(selected_parts)
                source_tag = _diff_source_tag(
                    sandbox_path=os.path.join(sandbox_dir, fpath),
                    project_path=os.path.join(project_root, fpath),
                    source=source,
                )
                header_suffix = f" — {source_tag}" if source_tag else ""
                output_parts.append(
                    f"\n=== Code: {fpath} (lines {range_str} of {total_lines}{header_suffix}) ===\n{combined}"
                )
            else:
                if total_lines > KEEP_FORCE_THRESHOLD:
                    # Huge file — return a skeleton instead of the full body.
                    # Loading the full file would blow the model's context.
                    skeleton = _build_file_skeleton(all_lines, filename=fpath)
                    output_parts.append(
                        f"\n=== Code: {fpath} ({total_lines} lines — SKELETON ONLY) ===\n"
                        f"⛔ This file is too large to return in full "
                        f"({total_lines} lines > {KEEP_FORCE_THRESHOLD} threshold). "
                        f"Loading the entire file would overflow the model's "
                        f"context window. Below is the file's SKELETON — "
                        f"top-level definitions with their line numbers.\n\n"
                        f"To READ ACTUAL CONTENT around a line in the skeleton,\n"
                        f"use VIEW with a line number (this is the right tool here):\n"
                        f"  [tool use] [VIEW: {fpath} LINE_NUMBER] [/tool use]\n"
                        f"    → ~80 lines centered on LINE_NUMBER (±40). Use this\n"
                        f"      to explore when you don't yet know the exact range.\n"
                        f"  [tool use] [VIEW: {fpath} START-END] [/tool use]\n"
                        f"    → an explicit range (max 600 lines) once you know it.\n"
                        f"Do NOT use [KEEP:] to read this file — KEEP is only for\n"
                        f"NARROWING a file you ALREADY read with [CODE:]; on a file\n"
                        f"this size, VIEW by line number above is the correct tool.\n"
                        f"(KEEP still works for pinning a range right before a\n"
                        f"[REPLACE LINES N-M] edit, but VIEW first to find the lines.)\n"
                        f"Each call must be followed by [STOP][CONFIRM_STOP].\n"
                        f"\n{skeleton}\n"
                    )
                    # Don't record skeleton as viewed_versions — line-anchored
                    # edits against a skeleton view would be wrong. Force the
                    # model to KEEP first, which DOES record the real content.
                    if viewed_versions is not None:
                        viewed_versions.pop(fpath, None)
                else:
                    numbered = add_line_numbers(content, display_mode=display_mode)
                    large_note = ""
                    if total_lines > KEEP_HINT_THRESHOLD:
                        large_note = (
                            f"(Large file — {total_lines} lines. For surgical "
                            f"edits, [KEEP: {fpath} X-Y, A-B] pins specific "
                            f"ranges.)\n"
                        )
                    if file_uses_tabs(content):
                        large_note = (
                            "(⚠ File uses TAB indentation. The renderer shows "
                            "tabs as 4 spaces for display; write REPLACE bodies "
                            "matching the file's real tab indentation, not "
                            "spaces, to avoid mixing.)\n"
                        ) + large_note
                    # Annotate the source ONLY when sandbox content actually
                    # differs from disk — the "from sandbox" tag was edit-
                    # workflow chrome that confused reading-mode users.
                    # Reading the sandbox when it equals disk is just "reading
                    # the file"; no need to mention the cache layer.
                    source_tag = _diff_source_tag(
                        sandbox_path=os.path.join(sandbox_dir, fpath),
                        project_path=os.path.join(project_root, fpath),
                        source=source,
                    )
                    header_suffix = f" — {source_tag}" if source_tag else ""
                    output_parts.append(
                        f"\n=== Code: {fpath} ({total_lines} lines{header_suffix}) ===\n"
                        f"{large_note}"
                        f"{numbered}"
                    )
        else:
            # A directory where a file path was expected — say so plainly
            # instead of the misleading "FILE NOT FOUND" (the dir clearly
            # exists; the model just needs to name a file inside it).
            _dir_hit = next((d for d in (os.path.join(sandbox_dir, fpath),
                                         os.path.join(project_root, fpath))
                             if os.path.isdir(d)), None)
            if _dir_hit:
                output_parts.append(
                    f"\n=== Code: {fpath} — IS A DIRECTORY, NOT A FILE ===\n"
                    f"Name a specific file, e.g. [CODE: {fpath.rstrip('/')}/<file>.py].\n"
                    f"See [PROJECT FILES] in the prompt for what's inside {fpath}."
                )
                continue
            # File doesn't exist anywhere — show available files near that
            # name so the model has a concrete next move instead of looping
            # on the same wrong path. We search the SANDBOX (the canonical
            # post-edit state) and, when sandbox has nothing yet, the
            # project root.
            suggestions: list[str] = []
            try:
                basename = os.path.basename(fpath).lower()
                stem = basename.split('.')[0] if '.' in basename else basename
                search_root = sandbox_dir if os.path.isdir(sandbox_dir) else project_root
                # Walk once, cap to keep latency bounded on huge repos.
                seen = 0
                for dp, dirs, files in os.walk(search_root):
                    # Skip common heavy dirs that never contain source.
                    dirs[:] = [d for d in dirs if d not in (
                        ".git", "__pycache__", "node_modules",
                        ".venv", "venv", "dist", "build", ".jarvis_sandbox",
                    )]
                    for fn in files:
                        seen += 1
                        if seen > 5000:
                            break
                        if stem and stem in fn.lower():
                            rel = os.path.relpath(os.path.join(dp, fn), search_root)
                            suggestions.append(rel)
                    if seen > 5000 or len(suggestions) >= 30:
                        break
                suggestions = sorted(set(suggestions))[:15]
            except Exception:
                suggestions = []

            if suggestions:
                sug_block = "\n".join(f"  • {s}" for s in suggestions)
                output_parts.append(
                    f"\n=== Code: {fpath} — FILE NOT FOUND ===\n"
                    f"This path does not exist in the sandbox or the project root.\n"
                    f"Files with a similar name (search root: "
                    f"{'sandbox' if os.path.isdir(sandbox_dir) else 'project'}):\n"
                    f"{sug_block}\n"
                    f"Pick the actual path and re-issue [CODE: <path>] — do NOT\n"
                    f"retry the same path; the file genuinely is not there."
                )
            else:
                output_parts.append(
                    f"\n=== Code: {fpath} — FILE NOT FOUND ===\n"
                    f"This path does not exist in the sandbox or the project root,\n"
                    f"and no file with a similar name was found. Check the path\n"
                    f"against [PROJECT FILES] in the prompt — do NOT retry the\n"
                    f"same path; the file genuinely is not there."
                )

    # ── Dependency-index annotation ───────────────────────────────────────
    # For each successfully-read file, append `|appears N (#tag)` markers
    # to def lines of symbols that have >=threshold refs project-wide.
    # Uses LSP for precise counts where available; falls back to AST
    # upper-bound if LSP is down. Surfaces blast-radius signal inline so
    # the model can see "this function is called everywhere" without an
    # extra round-trip. Safe to no-op on error.
    try:
        from core.dependency_index_cache import annotate_code_output_async
        annotated_parts = []
        for part in output_parts:
            m = re.search(r'=== Code: (\S+?) ', part)
            if m and re.search(r'^i\d+\|', part, re.MULTILINE):
                fpath_in_part = m.group(1)
                annotated_parts.append(
                    await annotate_code_output_async(sandbox_dir, fpath_in_part, part)
                )
            else:
                annotated_parts.append(part)
        return "\n".join(annotated_parts)
    except Exception:
        return "\n".join(output_parts)


# ─── KEEP Handler ────────────────────────────────────────────────────────────

async def _run_keep(
    keep_args: list[str], project_root: str,
    persistent_lookups: dict[str, str],
    research_cache: dict | None = None,
    viewed_versions: "dict[str, str] | None" = None,
    on_keep_seen: "Callable[[str, str], None] | None" = None,
    display_mode: str = "prefix",
) -> str:
    """Process [KEEP: filepath X-Y, A-B] tags.

    1. Parse filepath + line ranges from the tag argument
    2. Find the original file content (from persistent_lookups or disk)
    3. Build filtered view with preserved line numbers
    4. Run auto-RAG on kept lines
    5. REPLACE the CODE entry in persistent_lookups with the filtered view
    6. Return the filtered view + dependency summary

    on_keep_seen, if provided, is called with (canonical_key, raw_arg) for
    each KEEP that actually fires. The caller uses it to register the KEEP
    in its manifest / re-read counter so loop detection works for KEEP
    just like it does for CODE.
    """
    import os
    from tools.codebase import read_file, norm_path
    from workflows.code import _parse_keep_ranges, _filter_by_ranges, _auto_rag

    output_parts = []

    def _persist_keep_failure(arg_raw: str, message: str) -> None:
        """Persist a KEEP failure into the next prompt (mirror of
        _persist_view_failure). Without this, KEEP error messages went to
        round_output, which is never rendered — so the model re-issued the
        same bad KEEP every round with no way to learn why."""
        try:
            stripped_arg, _lbl = _strip_label(arg_raw.strip())
            key = _norm_key("KEEP", stripped_arg)
            persistent_lookups[key] = message
            if on_keep_seen is not None:
                try:
                    on_keep_seen(key, arg_raw)
                except Exception:
                    pass
        except Exception:
            pass

    for arg in keep_args:
        arg = arg.strip()

        # Strip optional #label suffix BEFORE range parsing — without this,
        # `[KEEP: foo.py 10-20 #lbl]` would feed "10-20 #lbl" into the range
        # parser and the label survives into the filepath. The label is
        # purely for DISCARD identification; the canonical KEEP key never
        # includes it.
        arg_no_label, _kept_label = _strip_label(arg)

        # Parse: "filepath X-Y, A-B" or "filepath X-Y A-B"
        # The filepath is everything before the first digit-dash-digit pattern.
        # BUT — if the filepath itself contains a `N-M` segment (e.g.
        # `tools/v2-3/foo.py 50-80`), naive parsing splits in the wrong place.
        # Heuristic: also look for an explicit whitespace boundary between
        # filename and ranges; prefer that when present.
        ws_split_match = re.search(
            r'^(\S+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue|lua|sh))\s+(.+)$',
            arg_no_label, re.IGNORECASE,
        )
        if ws_split_match:
            filepath = ws_split_match.group(1).strip()
            ranges_text = ws_split_match.group(2).strip()
        else:
            range_match = re.search(r'(\d+)\s*-\s*(\d+)', arg_no_label)
            if not range_match:
                _msg = (f"⚠ KEEP: '{arg}' has no line range. KEEP pins sub-ranges "
                        f"of a file — use [KEEP: {arg_no_label.strip() or 'file.py'} 10-50] "
                        f"or [KEEP: file.py 10-50, 80-120]. For the WHOLE file use [CODE: file.py].")
                output_parts.append(_msg)
                _persist_keep_failure(arg, _msg)
                continue
            filepath = arg_no_label[:range_match.start()].strip()
            ranges_text = arg_no_label[range_match.start():]
        filepath = norm_path(filepath)

        status(f"    KEEP: {filepath}")

        # Find original content — check persistent_lookups first.
        # Build the CODE key via `_norm_key` so it matches exactly what
        # `_store("CODE", ...)` would have produced. The previous ad-hoc
        # `f"CODE:{filepath.lower()}"` could mismatch if the stored key
        # was normalized differently (e.g. with `./` stripped).
        original_content = None
        norm_key = filepath.strip().lower()
        code_key = _norm_key("CODE", filepath)

        # Search persistent_lookups for a matching CODE entry.
        # Match order, MOST-SPECIFIC FIRST:
        #   1. exact key match
        #   2. path is a proper suffix of the KEEP target (the KEEP wrote
        #      the full relative path, the CODE used the basename)
        #   3. path is a proper SUFFIX of the existing key (KEEP used the
        #      basename, CODE used the full path)
        # Bidirectional `endswith` without a SLASH guard used to pick the
        # wrong file when basenames collided (e.g. `foo/bar.py` and
        # `qux/bar.py` both end with `bar.py`). We now require the
        # boundary to fall on a path separator so partial-token matches
        # like `lib.py` ↔ `mylib.py` cannot collide.
        def _suffix_with_sep(longer: str, shorter: str) -> bool:
            if longer == shorter:
                return True
            if not longer.endswith(shorter):
                return False
            cut = len(longer) - len(shorter)
            return cut == 0 or longer[cut - 1] in '/\\'

        matched_key = None
        # Pass 1: exact match.
        if code_key in persistent_lookups:
            matched_key = code_key
        # Pass 2: existing key has the KEEP target as a path-bounded suffix.
        if matched_key is None:
            for key in persistent_lookups:
                if not key.startswith("CODE:"):
                    continue
                key_path = key[5:]
                if _suffix_with_sep(key_path, norm_key):
                    matched_key = key
                    break
        # Pass 3: KEEP target has the existing key as a path-bounded suffix.
        if matched_key is None:
            for key in persistent_lookups:
                if not key.startswith("CODE:"):
                    continue
                key_path = key[5:]
                if _suffix_with_sep(norm_key, key_path):
                    matched_key = key
                    break

        # Read from sandbox first, then fall back to project root
        sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")
        sandbox_path = os.path.join(sandbox_dir, filepath)
        project_path = os.path.join(project_root, filepath)
        # v8.15 fix: same mtime-refresh as _run_code_reads / _run_view.
        # Subagent r10 found KEEP could also serve stale snapshots.
        if os.path.isfile(sandbox_path) and os.path.isfile(project_path):
            try:
                if os.path.getmtime(project_path) > os.path.getmtime(sandbox_path):
                    import shutil as _shutil
                    _shutil.copy2(project_path, sandbox_path)
            except Exception:
                pass
        raw_content = None
        # v8.15: binary guard for KEEP.
        _bin_target = sandbox_path if os.path.isfile(sandbox_path) else project_path
        _is_bin, _reason = _is_binary_path(_bin_target)
        if _is_bin:
            output_parts.append(
                f"=== KEEP: '{filepath}' — {_reason}. ==="
            )
            continue
        if os.path.isfile(sandbox_path):
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
            except Exception:
                raw_content = None

        # Only the read_file failure prefixes count as a miss. Legitimate
        # files whose content starts with `[` (JSON arrays, TOML arrays,
        # Lua tables …) MUST NOT trip the rejection — that produced the
        # silent "file not found" loop on list-shaped JSON.
        _READ_FAIL_PREFIXES = ("[BINARY", "[READ ERROR", "[FILE NOT FOUND", "[ERROR")
        def _looks_like_read_failure(s: "str | None") -> bool:
            return bool(s) and any(s.startswith(p) for p in _READ_FAIL_PREFIXES)

        if not raw_content:
            full_path = os.path.join(project_root, filepath)
            raw_content = read_file(full_path)
        # No CWD-relative fallback — the previous `read_file(filepath)` last-ditch
        # used the process CWD, which is not guaranteed to be project_root. That
        # path silently resolved to wrong content or hid genuine failures.

        if not raw_content or _looks_like_read_failure(raw_content):
            _msg = (f"⚠ KEEP: file not found '{filepath}'. Check the path/spelling, "
                    f"or [CODE: {filepath}] first to confirm it exists.")
            output_parts.append(_msg)
            _persist_keep_failure(arg, _msg)
            continue

        # Parse KEEP ranges FIRST — before recording the file as "seen".
        # Recording viewed_versions for a KEEP that has invalid/missing
        # ranges used to imprint the FULL file as the model's anchor
        # snapshot, which then misled subsequent [REPLACE LINES] edits
        # into thinking they were anchored on a narrow window when they
        # were actually anchored on the whole file.
        ranges = _parse_keep_ranges(ranges_text, filepath)
        if not ranges:
            # Try parsing the full arg
            ranges = _parse_keep_ranges(arg, filepath)

        if not ranges:
            _msg = (f"⚠ KEEP: no valid line ranges in '{arg}'. Use digit-dash-digit, "
                    f"e.g. [KEEP: {filepath} 10-50] or [KEEP: {filepath} 10-50, 80-120].")
            output_parts.append(_msg)
            _persist_keep_failure(arg, _msg)
            continue

        # v8.15 fix (B16): detect descending pairs in input string that
        # _parse_keep_ranges silently drops. Surface them so a model
        # passing `50-60, 30-20, 80-85` knows the middle was rejected.
        import re as _re_b16
        _all_pairs = _re_b16.findall(r'(\d+)\s*-\s*(\d+)', ranges_text or arg)
        _descending = []
        for a, b in _all_pairs:
            ai, bi = int(a), int(b)
            if ai > 0 and bi > 0 and bi < ai:
                _descending.append((ai, bi))
        if _descending:
            output_parts.append(
                "⚠ KEEP dropped descending range(s) "
                + ", ".join(f"{a}-{b}" for a, b in _descending)
                + " — start must be ≤ end. Other ranges processed."
            )

        # v8.12 fix: clamp ranges to file bounds and reject ranges that
        # are entirely past EOF with a clear error. Subagent r8 found
        # `KEEP file 300-400` on a 247-line file used to crash with a
        # raw Python IndexError; the underlying scope-extender is now
        # safe, but the higher-level UX is still better with a clean
        # rejection than a silent "KEPT 101/247" with zero visible lines.
        _kc_total_lines = raw_content.count('\n')
        if not raw_content.endswith('\n'):
            _kc_total_lines += 1
        clamped_ranges = []
        dropped_ranges = []
        clamped_warnings = []
        for s, e in ranges:
            if s > _kc_total_lines:
                dropped_ranges.append((s, e))
                continue  # entirely past EOF — drop
            if e > _kc_total_lines:
                clamped_warnings.append((s, e, _kc_total_lines))
                e = _kc_total_lines
            clamped_ranges.append((s, e))
        if not clamped_ranges:
            _msg = (
                f"⚠ KEEP: '{arg}' — every requested range is past end-of-file "
                f"(file '{filepath}' has only {_kc_total_lines} lines). "
                f"Re-request within 1-{_kc_total_lines}, e.g. "
                f"[KEEP: {filepath} 1-{min(60, _kc_total_lines)}]."
            )
            output_parts.append(_msg)
            _persist_keep_failure(arg, _msg)
            continue
        # v8.15 fix: subagent r13 noted PARTIAL past-EOF was silent. VIEW
        # warns in this case; KEEP should too for symmetry.
        keep_warnings: list[str] = []
        for s, e in dropped_ranges:
            keep_warnings.append(
                f"⚠ Range {s}-{e} dropped — entirely past EOF "
                f"(file has {_kc_total_lines} lines)."
            )
        for s, e, t in clamped_warnings:
            keep_warnings.append(
                f"⚠ Range {s}-{e} clamped to {s}-{t} (file has {t} lines)."
            )
        ranges = clamped_ranges

        # Record what the model is about to see — KEEP preserves real line
        # numbers, so any subsequent [REPLACE LINES X-Y] anchors to THIS
        # snapshot (not whatever the file looks like at apply time after
        # mid-stream edits). Only imprint after ranges are confirmed valid.
        if viewed_versions is not None:
            viewed_versions[filepath] = raw_content

        # Build filtered view
        filtered = _filter_by_ranges(raw_content, ranges, filepath, display_mode=display_mode)
        kept_lines = sum(e - s + 1 for s, e in ranges)
        # v8.7 fix: same off-by-1 as add_line_numbers — files ending
        # with '\n' should not be counted as N+1 lines.
        total_lines = raw_content.count('\n')
        if not raw_content.endswith('\n'):
            total_lines += 1

        # Auto-RAG: find dependencies in kept code
        deps = await _auto_rag(filtered, filepath, project_root, research_cache)

        # Build the replacement result
        _hdr = (
            f"\n=== Code: {filepath} (KEPT {kept_lines}/{total_lines} lines, "
            f"line numbers accurate for [REPLACE LINES]) ===\n"
        )
        # KEEP-not-in-context note: KEEP is meant to NARROW a file already
        # read with [CODE:]. If this file was never in context, KEEP read it
        # from disk — which works, but for a big file you haven't read,
        # [VIEW: file N-M] is the direct tool. Surface this once.
        if matched_key is None and total_lines > 1500:  # large-file hint threshold
            _hdr += (
                f"⚠ '{filepath}' was not in context (no prior [CODE:]). KEEP "
                f"narrows an already-read file; for a large file you haven't "
                f"read, [VIEW: {filepath} N-M] reads a range directly.\n"
            )
        if keep_warnings:
            _hdr += "\n".join(keep_warnings) + "\n"
        replacement = f"{_hdr}{filtered}\n"
        if deps:
            replacement += f"\n{deps}\n"

        # Build the canonical KEEP key — one entry PER (file, ranges) pair.
        # Previously every KEEP for the same file overwrote the single
        # CODE:filepath entry, so a model that wrote 7 KEEPs in one round
        # would only see the LAST range in its TOOL RESULTS — and re-request
        # the others next round in a tight loop. Observed: a planner
        # requested `KEEP: workflows/code.py 7168-7518` 6 rounds in a row
        # because every subsequent KEEP wiped it out.
        #
        # Key by the model's AS-WRITTEN tag (via `_norm_key`) so the manifest
        # entry, the persistent_lookups entry, and `_norm_tag_key(round_keys)`
        # all share the same string. If we keyed by the PARSED canonical
        # ranges (e.g. `10-20,30-40` no-space) but the model wrote them
        # with a space (`10-20, 30-40`), the keys mismatch and the manifest
        # / re-read detection silently fails. Same bug that hit VIEW.
        canonical_keep_key = _norm_key("KEEP", arg_no_label)
        # Redundant cross-check: the key we store under MUST equal what
        # `call_with_tools` will compute for `round_keys` from the model's
        # raw tag (label-and-all). The chain is:
        #   model tag → _strip_label → _norm_key
        # `_norm_tag_key` in call_with_tools does this exact sequence.
        # If our storage key ever drifts from this, manifest lookups fail
        # silently — which is the VIEW bug we just fixed.
        _round_keys_key = _norm_key("KEEP", _strip_label(arg)[0])
        if canonical_keep_key != _round_keys_key:
            warn(
                f"  KEEP key drift: stored={canonical_keep_key!r} "
                f"round_keys would compute={_round_keys_key!r} "
                f"arg={arg!r} — manifest lookups will MISS"
            )

        # Drop the full CODE entry on first KEEP for this file — that's the
        # whole point of KEEP (narrow the context). After this drops, every
        # subsequent KEEP for the same file accumulates as its own entry,
        # all visible to the model side-by-side.
        if matched_key and matched_key.startswith("CODE:"):
            persistent_lookups.pop(matched_key, None)
            status(f"    KEEP: dropped full CODE:{filepath} from context — "
                   f"narrowed via KEEP")

        persistent_lookups[canonical_keep_key] = replacement
        status(f"    KEEP: {filepath}: {kept_lines}/{total_lines} lines kept")

        # Notify the caller so it can register this KEEP in its manifest /
        # re-read counter using the SAME key we just stored under. Without
        # the keys matching, `_annotate_entry` couldn't find the manifest
        # entry for a KEEP result and showed the model the wrong tag header.
        if on_keep_seen is not None:
            try:
                on_keep_seen(canonical_keep_key, arg)
            except Exception:
                pass

        output_parts.append(replacement)

    return "\n".join(output_parts)


# ─── VIEW — read a slice of a large file by line number ──────────────────────

# Window sizing: 200 lines centered on a single-line input, 600 lines max
# for an explicit range. Bigger than KEEP because the model uses VIEW to
# UNDERSTAND code (whole functions, neighboring helpers), whereas KEEP is
# for surgical edits where you already know the precise range.
_VIEW_DEFAULT_WINDOW = 80       # lines, ±40 around a single-line target.
                                 # v8.9 fix: was 200; subagent found a
                                 # 1-line request returning 169 lines was
                                 # too broad for surgical drilling.
_VIEW_MAX_RANGE = 600           # lines, hard cap for explicit ranges
_VIEW_THINKING_RESERVE = 20_000 # tokens reserved for model output/thinking


async def _run_view(
    view_args: list[str], project_root: str,
    persistent_lookups: dict[str, str],
    model_id: str | None = None,
    research_cache: dict | None = None,
    viewed_versions: "dict[str, str] | None" = None,
    on_view_seen: "Callable[[str, str], None] | None" = None,
    display_mode: str = "prefix",
) -> str:
    """Read a slice of a large file by line number.

    Reject + redirect on small files: if the file content + a 20k-token
    thinking reserve fits inside the model's context window, the model
    should be using `[CODE: path]` instead — VIEW's purpose is to navigate
    files that DON'T fit. The check uses tiktoken (cl100k_base) for an
    accurate token count, with a regex fallback when tiktoken isn't
    importable.

    Args:
      view_args: ['path lineN', 'path N-M', ...] from extract_view_tags.
      model_id: the calling model's full ID (e.g. 'nvidia/deepseek-v4-pro').
                Used to look up the context window from config.MODELS.
                Falls back to a permissive 128k assumption when unknown.

    Storage: one entry per (file, range) pair under canonical key
    `VIEW:filepath start-end`. Multiple VIEWs on the same file coexist
    in persistent_lookups (mirrors the KEEP fix from earlier — single-key
    storage would have each VIEW silently overwriting the previous one).
    """
    import os
    from tools.codebase import read_file, norm_path
    from workflows.code import _filter_by_ranges
    from core.tokens import count_tokens

    # Look up model window from config; default to a moderate value so an
    # unknown model_id doesn't accidentally let VIEW serve gigantic files.
    try:
        from config import MODELS
        model_window = (
            MODELS.get(model_id or "", {}).get("window", 128_000)
            if model_id else 128_000
        )
    except Exception:
        model_window = 128_000

    output_parts = []

    def _persist_view_failure(arg_raw: str, message: str) -> None:
        """Store a VIEW failure under its canonical key + register in the
        manifest so the model sees the failure reason in TOOL RESULTS
        next round instead of silently re-issuing the same failing call.
        Earlier the failure went to round_output (never read), so loops
        like deepseek-v4-pro R3-R8 in 20260512_165633 spent 5 rounds
        re-emitting the same VIEW with no way to learn why it wasn't
        working. Persisting closes that loop.
        """
        try:
            stripped_arg, _lbl = _strip_label(arg_raw.strip())
            key = _norm_key("VIEW", stripped_arg)
            persistent_lookups[key] = message
            if on_view_seen is not None:
                try:
                    on_view_seen(key, arg_raw)
                except Exception:
                    pass
        except Exception:
            pass  # purely diagnostic — never break the loop

    for arg in view_args:
        arg = arg.strip()
        arg_no_label, _label = _strip_label(arg)

        # Parse: "path LINE" or "path N-M". Same filepath-splitting rule
        # as _run_keep so paths containing N-M (e.g. v2-3/foo.py) don't
        # break on the first dash.
        ws_split = re.search(
            r'^(\S+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue|lua|sh))\s+(.+)$',
            arg_no_label, re.IGNORECASE,
        )
        if ws_split:
            filepath = ws_split.group(1).strip()
            spec = ws_split.group(2).strip()
        else:
            # Fallback: split before the first digit
            digit_match = re.search(r'\d', arg_no_label)
            if not digit_match:
                _vp = arg_no_label.strip() or "file.py"
                _msg = (
                    f"⚠ VIEW: '{arg}' has no line range. VIEW reads a slice — "
                    f"use [VIEW: {_vp} 120] (±40 lines) or [VIEW: {_vp} 100-160]. "
                    f"For the WHOLE file use [CODE: {_vp}]."
                )
                output_parts.append(_msg)
                _persist_view_failure(arg, _msg)
                continue
            filepath = arg_no_label[:digit_match.start()].strip()
            spec = arg_no_label[digit_match.start():].strip()

        filepath = norm_path(filepath)

        # If parsing failed and we already emitted the invalid-format
        # rejection above, ensure it's also persisted so the model sees
        # the same diagnosis next round. (See _persist_view_failure.)

        # Read the file. Prefer sandbox version when present so VIEW sees
        # the post-edit state the model is actually working with.
        sandbox_path = os.path.join(project_root, ".jarvis_sandbox", filepath)
        project_path = os.path.join(project_root, filepath)
        # v8.15 fix: subagent r10 caught VIEW returning a stale 331-line
        # snapshot when the disk file was 415 lines. The mtime-refresh
        # was only on _run_code_reads — VIEW and KEEP also need it.
        if os.path.isfile(sandbox_path) and os.path.isfile(project_path):
            try:
                if os.path.getmtime(project_path) > os.path.getmtime(sandbox_path):
                    import shutil as _shutil
                    _shutil.copy2(project_path, sandbox_path)
            except Exception:
                pass
        raw_content = None
        sandbox_read_succeeded = False
        # v8.15: binary guard for VIEW. Use whichever copy will be read.
        _bin_target = sandbox_path if os.path.isfile(sandbox_path) else project_path
        _is_bin, _reason = _is_binary_path(_bin_target)
        if _is_bin:
            output_parts.append(
                f"=== VIEW: '{filepath}' — {_reason}. ==="
            )
            continue
        if os.path.isfile(sandbox_path):
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
                sandbox_read_succeeded = True
            except Exception:
                raw_content = None
        # v8.15 fix: 0-byte file (legitimate empty `__init__.py`) returned
        # `""` which is falsy → `if not raw_content:` triggered, fell
        # through to the FAIL prefix check, then "FILE NOT FOUND". CODE
        # handled this with a SANDBOX-FILE-IS-EMPTY banner; VIEW/KEEP
        # contradicted by saying not-found. Distinguish "file exists but
        # is empty" from "file truly doesn't exist".
        if not sandbox_read_succeeded and (raw_content is None or raw_content == ""):
            raw_content = read_file(project_path)

        _READ_FAIL_PREFIXES = ("[BINARY", "[READ ERROR", "[FILE NOT FOUND", "[ERROR")
        # Only treat as failure if read_file returned a fail prefix OR
        # the file genuinely doesn't exist on either side.
        if raw_content is None or any(
            (raw_content or "").startswith(p) for p in _READ_FAIL_PREFIXES
        ):
            _msg = f"=== VIEW: file not found '{filepath}' ==="
            output_parts.append(_msg)
            _persist_view_failure(arg, _msg)
            continue
        # v8.15: empty-file case — give an honest message instead of
        # masquerading as not-found.
        if raw_content == "":
            output_parts.append(
                f"=== VIEW: '{filepath}' is EMPTY (0 bytes / 0 lines). "
                f"No content to display. ==="
            )
            continue

        # ── Small-file gate REMOVED ────────────────────────────────────
        # We used to REJECT VIEW when the full file fit in the model's
        # window with a 20k-token reserve ("just use [CODE:] instead").
        # That created a deadlock with [CODE:]'s `KEEP_FORCE_THRESHOLD =
        # 1500 lines` skeleton path:
        #   • CODE: workflows/code.py (7684 lines) → SKELETON, "use VIEW
        #     for actual content"
        #   • VIEW: workflows/code.py 4820 → REJECTED, "file fits in your
        #     window, use CODE instead"
        # File: 85k tokens. deepseek-v4-pro window: 1M tokens. 85k + 20k
        # fits → VIEW rejected. CODE returns skeleton → no real content
        # ever reaches the model.
        # The rejection ALSO never persisted across rounds (it went into
        # round_output, which is never read), so the model saw no reason
        # for the failure and kept re-issuing the same VIEW. Observed in
        # 20260512_165633 deepseek-v4-pro R3-R8 (same VIEW set 5 rounds
        # in a row, byte-identical responses R6=R7=R8).
        # New behaviour: always serve the slice. If the model genuinely
        # had the full file via [CODE:], it can use that — calling VIEW
        # anyway is harmless (the slice is small, fits in budget).
        file_tokens = count_tokens(raw_content)

        # ── Parse the line/range spec ──────────────────────────────────
        # v8.7 fix: same off-by-1 as add_line_numbers / _run_keep.
        total_lines = raw_content.count('\n')
        if not raw_content.endswith('\n'):
            total_lines += 1
        range_match = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', spec)
        single_match = re.match(r'^\s*(\d+)\s*$', spec)
        # v8.4 fix: detect out-of-bounds requests BEFORE clamping so we
        # can surface a warning in the header. Previously OOB requests
        # silently degraded to the file's last function, with no signal
        # that the model asked for the wrong area.
        oob_warning = None
        if range_match:
            req_start = int(range_match.group(1))
            req_end = int(range_match.group(2))
            # v8.15 fix: subagent r11 noted backwards range (e.g. 100-50)
            # was silently flipped. Make the flip explicit so the model
            # doesn't think it's getting what it asked for.
            if req_start > req_end:
                oob_warning = (
                    f"⚠ Backwards range L{req_start}-{req_end} — "
                    f"interpreted as L{req_end}-{req_start} (start "
                    f"must be ≤ end)."
                )
            elif req_start > total_lines or req_end > total_lines:
                oob_warning = (
                    f"⚠ Requested L{req_start}-{req_end} is past EOF "
                    f"(file has {total_lines} lines). Returning available range."
                )
        elif single_match:
            req_line = int(single_match.group(1))
            # v8.15 fix: subagent r13 noted VIEW with line 0 was silently
            # accepted (returned lines 1-40). Be consistent with negative
            # rejection — line numbers are 1-based.
            if req_line == 0:
                _msg = (
                    f"=== VIEW: INVALID line spec '{spec}' for "
                    f"'{filepath}' — line numbers are 1-based; use "
                    f"line 1 or higher. ==="
                )
                output_parts.append(_msg)
                _persist_view_failure(arg, _msg)
                continue
            if req_line > total_lines:
                oob_warning = (
                    f"⚠ Requested L{req_line} is past EOF "
                    f"(file has {total_lines} lines). Returning end of file."
                )

        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                start, end = end, start
            span = end - start + 1
            if span > _VIEW_MAX_RANGE:
                _msg = (
                    f"=== VIEW: REJECTED for '{filepath} {spec}' — "
                    f"range {span} lines exceeds {_VIEW_MAX_RANGE}-line cap ===\n"
                    f"  • Split into multiple [VIEW:] calls of "
                    f"≤ {_VIEW_MAX_RANGE} lines each.\n"
                    f"  • A request this large usually means you should "
                    f"narrow your investigation: pick the function you\n"
                    f"    actually need to read from the skeleton, then VIEW "
                    f"around just that line.\n"
                )
                output_parts.append(_msg)
                _persist_view_failure(arg, _msg)
                continue
        elif single_match:
            line_no = int(single_match.group(1))
            half = _VIEW_DEFAULT_WINDOW // 2
            start = max(1, line_no - half)
            end = min(total_lines, line_no + half)
        else:
            _msg = (
                f"=== VIEW: invalid line spec '{spec}' for '{filepath}' — "
                f"expected a single line (e.g. 4849) or range (e.g. 4810-4910) ==="
            )
            output_parts.append(_msg)
            _persist_view_failure(arg, _msg)
            continue

        # Clamp to file bounds.
        start = max(1, min(start, total_lines))
        end = max(start, min(end, total_lines))

        # v9.1: no auto-extension to enclosing def/class. Past-EOF clamp
        # above is accurate; def-anchor extension was surprising — the
        # model asks for line N, gets lines 30-N+30, breaks the contract
        # that line numbers in tool output are usable directly for
        # SEARCH/REPLACE LINES. (User feedback: "i don't want warning,
        # i want accuracy".)

        if (end - start + 1) > _VIEW_MAX_RANGE:
            # Center the cap on the original target so the user's request
            # stays the focus. For a range input, center on its midpoint.
            mid = (start + end) // 2
            half = _VIEW_MAX_RANGE // 2
            start = max(1, mid - half)
            end = min(total_lines, mid + half)

        filtered = _filter_by_ranges(raw_content, [(start, end)], filepath, display_mode=display_mode)

        # Strip leading/trailing `(N lines hidden: A-B)` markers from a VIEW
        # output. _filter_by_ranges emits them for every gap including the
        # ones BEFORE the requested range and AFTER it. For VIEW (always one
        # contiguous slice with a header that already says "lines X-Y of Z"),
        # those flanking markers are pure noise — they make a clean slice
        # look truncated and trigger the model's "let me re-VIEW with a
        # wider range" reflex. We strip them plus any blank padding; middle
        # markers (rare multi-range case) are preserved.
        #
        # Real source blank lines render as `i0| {lineno}` (never bare ''),
        # so anything empty in the output is noise we can safely drop.
        _filt_lines = filtered.split('\n')
        while _filt_lines:
            ln = _filt_lines[0].lstrip()
            if ln.startswith('·') or ln == '':
                _filt_lines.pop(0)
            else:
                break
        while _filt_lines:
            ln = _filt_lines[-1].lstrip()
            if ln.startswith('·') or ln == '':
                _filt_lines.pop()
            else:
                break
        filtered = '\n'.join(_filt_lines)

        # v8.4 fix: prepend OOB warning when the requested line(s) exceeded
        # the file's actual length. Without this, an out-of-bounds request
        # silently returns the file's last function with no signal that
        # the model asked for the wrong area.
        header_warning = (oob_warning + "\n") if oob_warning else ""
        replacement = (
            f"\n=== VIEW: {filepath} (lines {start}-{end} of {total_lines}, "
            f"line numbers accurate for [REPLACE LINES]) ===\n"
            f"{header_warning}"
            f"{filtered}\n"
        )

        # CRITICAL: key by the model's AS-WRITTEN tag arg, NOT by the
        # post-scope-extension range. The runtime's _norm_tag_key + manifest
        # comparisons use the model's literal text (e.g. "4796-5000"). If
        # we key by the extended range (e.g. "4796-5050") the model's
        # subsequent re-request of the same tag won't match the manifest
        # entry, so:
        #   • _reread_count never increments → stall guard never fires
        #   • the manifest lookup misses → model thinks the call didn't
        #     fire and re-issues it.
        # Observed in kimi-k2.6 R3-R6: same VIEW re-issued 4× before the
        # generic stall ratio finally tripped. Using _norm_key on the
        # model's raw arg ensures every downstream key check matches.
        canonical_view_key = _norm_key("VIEW", arg_no_label)
        # Redundant cross-check (see KEEP for the rationale) — `round_keys`
        # uses `_strip_label → _norm_key` on the model's raw tag, our
        # storage key MUST match exactly or manifest lookups silently miss.
        _round_keys_key = _norm_key("VIEW", _strip_label(arg)[0])
        if canonical_view_key != _round_keys_key:
            warn(
                f"  VIEW key drift: stored={canonical_view_key!r} "
                f"round_keys would compute={_round_keys_key!r} "
                f"arg={arg!r} — manifest lookups will MISS"
            )
        persistent_lookups[canonical_view_key] = replacement
        status(
            f"    VIEW: {filepath} {start}-{end} "
            f"({end - start + 1}/{total_lines} lines)"
        )

        # Record the snapshot so subsequent [REPLACE LINES] anchors to the
        # version we just showed the model — same mechanism KEEP uses.
        if viewed_versions is not None:
            viewed_versions[filepath] = raw_content

        if on_view_seen is not None:
            try:
                on_view_seen(canonical_view_key, arg)
            except Exception:
                pass

        output_parts.append(replacement)

    # Dependency-index annotation — LSP-precise (same async path as
    # _run_code_reads). Uses sym refs from LSP where available; falls
    # back to AST upper bound when LSP is down.
    try:
        from core.dependency_index_cache import annotate_code_output_async
        sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")
        annotated_parts = []
        for part in output_parts:
            m = re.search(r'=== VIEW: (\S+?) \(lines ', part)
            if m and re.search(r'^i\d+\|', part, re.MULTILINE):
                fpath_in_part = m.group(1)
                annotated_parts.append(
                    await annotate_code_output_async(sandbox_dir, fpath_in_part, part)
                )
            else:
                annotated_parts.append(part)
        return "\n".join(annotated_parts)
    except Exception:
        return "\n".join(output_parts)


# ─── Reference Search ───────────────────────────────────────────────────────

async def _run_dependency_lookup(tags: list[str], project_root: str) -> str:
    """Resolve [DEPENDENCY: #TAG] tool calls (LSP-precise when available).

    Each tag was registered in the dependency index when the agent's first
    VIEW/CODE call surfaced it inline as `|appears N (#tag)`. Drilling in
    via this tool returns the definition location and the full list of
    file:line references for that symbol — resolved through LSP so the
    count matches what the model actually edits, not the AST upper bound.
    """
    import os
    from core.dependency_index_cache import lookup_dependency_async
    sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")
    parts = []
    for tag in tags:
        parts.append(f"\n=== DEPENDENCY: #{tag.lstrip('#')} ===")
        parts.append(await lookup_dependency_async(sandbox_dir, tag))
    return "\n".join(parts)


async def _run_refs_searches(names: list[str], project_root: str) -> str:
    """Ripgrep word-boundary search for all references to a name."""
    from tools.codebase import search_refs

    output_parts = []
    for name in names:
        name = name.strip()
        status(f"    Refs search: {name}")
        result = search_refs(name, project_root)
        output_parts.append(result)
    return "\n".join(output_parts)


async def _run_lsp_searches(names: list[str], project_root: str) -> str:
    """LSP semantic search — finds dependencies, types, indirect references."""
    output_parts = []
    for name in names:
        name = name.strip()
        status(f"    LSP search: {name}")
        try:
            from tools.lsp import lsp_find_references, get_lsp_client
            # v8.7 fix: distinguish "server unavailable" from "couldn't find
            # the symbol's definition". Previously both returned None and
            # said "no LSP server available", which was actively misleading
            # when the symbol was simply mistyped or the bare-name lookup
            # didn't match a top-level def.
            client = await get_lsp_client(project_root)
            if not client:
                output_parts.append(
                    f"=== LSP for '{name}': no LSP server installed "
                    f"(install pylsp via `pip install python-lsp-server` or "
                    f"pyright); use [REFS: {name}] instead ==="
                )
                continue
            result = await lsp_find_references(name, project_root)
            if result:
                output_parts.append(result)
            else:
                output_parts.append(
                    f"=== LSP for '{name}': server is available but couldn't "
                    f"find a definition for that symbol (typo? not a "
                    f"top-level def/class?). Try [REFS: {name}] to grep "
                    f"all mentions, or pass the qualified name. ==="
                )
        except Exception as e:
            output_parts.append(f"=== LSP for '{name}': failed ({str(e)[:80]}), use [REFS: {name}] instead ===")
    return "\n".join(output_parts)


def _run_purpose_lookups(categories: list[str], purpose_map: str, project_root: str) -> str:
    """Look up purpose categories and return code snippets with context.

    When `purpose_map` is present (full JARVIS pipeline, Phase-1 ran),
    serve snippets from it. Otherwise (CLI / exploration mode), each
    `category` is treated as a filepath and the new PURPOSE tool
    returns the module's docstring + public symbol docstrings.
    """
    from tools.code_index import get_purpose_snippets
    from core.exploration_tools import extract_purpose

    output_parts = []
    for cat in categories:
        status(f"    Purpose lookup: {cat}")
        if purpose_map:
            result = get_purpose_snippets(purpose_map, cat, project_root)
        else:
            result = extract_purpose(cat, project_root)
        output_parts.append(result)
    return "\n".join(output_parts)


def _run_semantic_lookups(queries: list[str], project_root: str,
                          purpose_map: str | None = None) -> str:
    """Concept search across the project. Always uses the keyword-rank
    explorer (the embedding-based path lives in the planner closure
    inside run_with_tools and isn't exposed here)."""
    from core.exploration_tools import semantic_search

    output_parts = []
    for q in queries:
        status(f"    Semantic lookup: {q}")
        output_parts.append(semantic_search(q, project_root))
    return "\n".join(output_parts)


def _run_knowledge_lookups(topics: list[str]) -> str:
    """Look up knowledge topics."""
    from knowledge import get_knowledge

    output_parts = []
    for topic in topics:
        status(f"    Knowledge: {topic}")
        result = get_knowledge(topic.strip())
        output_parts.append(result)
    return "\n".join(output_parts)


# ─── Tool Tag Detection (for stream early-stop) ─────────────────────────────

_ALL_TAGS = re.compile(
    r'\[(SEARCH|WEBSEARCH|DETAIL|CODE|REFS|PURPOSE|LSP|KNOWLEDGE|KEEP|DISCARD):\s*.+?\]'
    r'|\[STOP\]\s*\[CONFIRM_STOP\]'
    r'|\[DONE\]\s*\[CONFIRM_DONE\]'
    r'|\[FORCE\s+DONE\]\s*\[CONFIRM_FORCE_DONE\]'
    r'|\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]',
    re.IGNORECASE,
)


def _text_has_complete_tag(text: str) -> bool:
    """Return True if text contains at least one complete tool tag or
    fully-formed two-tag signal. Bare `[STOP]` alone NO LONGER counts —
    the two-tag protocol means bare halves are inert text."""
    return bool(_ALL_TAGS.search(text))


_TOOL_USE_OPEN  = re.compile(r'\[tool use\]',  re.IGNORECASE)
_TOOL_USE_CLOSE = re.compile(r'\[/tool use\]', re.IGNORECASE)


def _autocomplete_tool_blocks(text: str) -> tuple[str, int]:
    """Close any unclosed [tool use] blocks by inserting a [/tool use] tag
    BEFORE the NEXT [tool use] open (or before [STOP]/[DONE]/[CONTINUE], or
    at end of text) for each orphaned opener.

    Walks opens and closes in document order so each missing closer lands at
    the correct boundary instead of being collapsed onto a single position.

    Returns (fixed_text, number_of_blocks_fixed).
    A non-zero count means the model wrote [tool use] but forgot [/tool use].
    """
    opens  = list(_TOOL_USE_OPEN.finditer(text))
    closes = list(_TOOL_USE_CLOSE.finditer(text))
    if len(opens) <= len(closes):
        return text, 0

    # Build interleaved list of events: open / close / signal-or-stop boundary.
    # Each event has (pos, kind). Walk left→right pairing opens to closes;
    # any open without a close gets a synthetic [/tool use] inserted at the
    # first boundary AFTER the open (next open / next signal / end-of-text).
    BoundaryRe = re.compile(
        r'\[tool use\]|\[/tool use\]|\[STOP\]\s*\[CONFIRM_STOP\]'
        r'|\[DONE\]\s*\[CONFIRM_DONE\]'
        r'|\[FORCE\s+DONE\]\s*\[CONFIRM_FORCE_DONE\]'
        r'|\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]',
        re.IGNORECASE,
    )
    insertions: list[int] = []
    open_stack: list[int] = []
    for m in BoundaryRe.finditer(text):
        tok = m.group(0).lower()
        if tok == '[tool use]':
            if open_stack:
                # Previous open never closed — synthesize close right before
                # THIS open (i.e. at m.start()).
                insertions.append(m.start())
                open_stack.pop()
            open_stack.append(m.start())
        elif tok == '[/tool use]':
            if open_stack:
                open_stack.pop()
        else:
            # signal: any open before this should be closed before the signal.
            while open_stack:
                insertions.append(m.start())
                open_stack.pop()

    # Any remaining unclosed opens close at end of text.
    while open_stack:
        insertions.append(len(text))
        open_stack.pop()

    if not insertions:
        return text, 0

    # Insert from right to left so earlier offsets stay valid.
    insertions.sort()
    fixed = text
    closer = '[/tool use]'
    for pos in reversed(insertions):
        # Ensure we sit on its own boundary — surround with newlines if needed.
        prefix = '' if (pos == 0 or fixed[pos - 1] in '\n ') else '\n'
        suffix = '' if (pos >= len(fixed) or fixed[pos] in '\n ') else '\n'
        fixed = fixed[:pos] + f"{prefix}{closer}{suffix}" + fixed[pos:]
    return fixed, len(insertions)


def _describe_tool_mode(result: str) -> str:
    """Return a short string describing whether block or fallback mode is active."""
    blocks = list(_TOOL_USE_BLOCK.finditer(result))
    if blocks:
        return f"block mode ({len(blocks)} [tool use] block(s))"
    return "bare-tag fallback (model omitted [tool use] wrapper)"


def _tag_summary(
    code_tags, web_tags, detail_tags, file_tags, refs_tags,
    purpose_tags, semantic_tags, lsp_tags, knowledge_tags, keep_tags,
    view_tags, dependency_tags=(),
    research_cache: dict | None = None,
    persistent_lookups: dict | None = None,
) -> str:
    """Build a one-line summary of tags found this round with cache annotations."""
    parts = []
    def _note(label: str, tags: list[str], type_key: str) -> None:
        if not tags:
            return
        hits = 0
        if research_cache is not None and type_key not in ("CODE", "KEEP", "VIEW"):
            for t in tags:
                clean, _ = _strip_label(t)
                k = f"{type_key}:{clean.strip().lower()}"
                if k in research_cache or k in persistent_lookups:
                    hits += 1
        hit_str = f" ({hits} cached)" if hits else ""
        parts.append(f"{label}×{len(tags)}{hit_str}")

    _note("CODE",     file_tags,     "CODE")
    _note("REFS",     refs_tags,     "REFS")
    _note("SEARCH",   code_tags,     "SEARCH")
    _note("WEB",      web_tags,      "WEBSEARCH")
    _note("DETAIL",   detail_tags,   "DETAIL")
    _note("PURPOSE",  purpose_tags,  "PURPOSE")
    _note("SEMANTIC", semantic_tags, "SEMANTIC")
    _note("LSP",      lsp_tags,      "LSP")
    _note("KNOW",     knowledge_tags,"KNOWLEDGE")
    _note("KEEP",     keep_tags,     "KEEP")
    _note("VIEW",     view_tags,     "VIEW")
    _note("DEPENDENCY", dependency_tags, "DEPENDENCY")
    return ", ".join(parts) if parts else "(none)"


# ─── Main Tool Call Loop ────────────────────────────────────────────────────

def _past_thinking_keep_from(round_texts: list, cap_tokens: int, count_fn) -> int:
    """Index of the first past-thinking round to KEEP so the NEWEST rounds fit
    within cap_tokens. Returns 0 when everything fits (keep all). Always keeps at
    least the last round — its tool results + reject feedback are what the model
    needs most. Bounds the only unbounded-growth term the runtime controls, so a
    long loop can't silently overflow the model's context window. (audit #4)"""
    if len(round_texts) <= 1 or count_fn("\n".join(round_texts)) <= cap_tokens:
        return 0
    running = 0
    keep_from = len(round_texts)
    for j in range(len(round_texts) - 1, -1, -1):
        running += count_fn(round_texts[j]) + 50          # ~marker overhead
        if running > cap_tokens and j < len(round_texts) - 1:
            break
        keep_from = j
    return keep_from


async def call_with_tools(
    model: str,
    prompt: str,
    project_root: str | None = None,
    max_tokens: int = 16384,   # proven default; 32768 was too slow (planner+implement timeouts)
    max_rounds: int = 20,
    enable_code_search: bool = True,
    enable_web_search: bool = True,
    detailed_map: str | None = None,
    purpose_map: str | None = None,
    research_cache: dict | None = None,
    log_label: str = "",
    on_stop: "Callable[[str], str | None] | None" = None,
    has_pending_edits: "Callable[[str], bool] | None" = None,
    allow_run: bool = False,
    run_cwd: "str | None" = None,
    viewed_versions: "dict[str, str] | None" = None,
    stop_on_tool_block: bool = False,
    cache_file_reads: bool = False,
    read_only_role: bool = False,
) -> dict:
    """
    Call a model with mid-thought tool use.

    The AI wraps tool calls in [tool use]...[/tool use] then [STOP].
    Only tags INSIDE [tool use] blocks execute — tags outside are ignored.
    JARVIS runs ALL requested lookups at once and feeds results back.

    Signals:
      [STOP]   → execute tool calls + on_stop callback, continue thinking
      [DONE]   → apply final edits, model is completely finished

    has_pending_edits: optional non-mutating predicate, called with the
             full response when [DONE] fires. Returns True if the turn carries
             edit blocks that haven't been applied yet. When True, the verify
             gate converts the [DONE] into an apply-and-show-diff round so the
             model verifies the diff before finishing (it must NOT mutate any
             dedup state — on_stop does the real apply).

    on_stop: optional callback called with full_response when [STOP] fires.
             Used by coders to apply pending edit blocks before tool lookups,
             so [CODE:] reads return the post-edit state.
             The callback MAY return a feedback string describing what
             happened to the edits (which applied, which were skipped and
             why). When it does, the string is prepended to the next-round
             prompt as an "EDIT APPLICATION RESULTS" block, so the model
             gets explicit signal instead of having to infer success by
             re-reading the file. Returning None means "no feedback to add."

    viewed_versions: optional dict updated whenever the model reads a file via
             [CODE: path]. Maps filepath → content the model just saw. Used by
             on_stop to anchor [REPLACE LINES] edits to the version the model
             was actually looking at, instead of whatever the file currently
             is on disk. Without this, a model that views V0, writes [REPLACE
             LINES 22-24], then writes more edits after a mid-stream [STOP]
             would have its V0-relative line numbers applied to the post-STOP
             file (which has different line numbers).

    Tool tags:
      [SEARCH: pattern]       → code search
      [WEBSEARCH: query]      → web search
      [DETAIL: section name]  → detailed code map lookup
      [CODE: path/to/file]    → read actual source code file
      [REFS: name]            → find all definitions, imports, usages
      [PURPOSE: category]     → all code serving a purpose (exact/fuzzy category name)
      [SEMANTIC: description] → vector-embedding search over the CODE (functions/classes), returns the top 10 matching `file:line` units
      [DISCARD: #label]       → remove a labeled result from context

    research_cache: shared dict that accumulates all lookup results across
    multiple AI calls. Same tag won't re-run if cached.

    Returns {"model": str, "answer": str, "done": bool, "force_done": bool,
    "research": {tag_key: result}}.
    "done" is True when the model explicitly wrote [DONE] — it is NOT present in
    "answer" (stripped before return), so callers must check this flag, not the text.
    "force_done" is True when the model wrote [FORCE DONE][CONFIRM_FORCE_DONE] —
    the coder's escape hatch for "step requirement already met, no edits needed".
    When force_done is True, done is also True. The IMPLEMENT loop uses
    force_done to distinguish "I am intentionally producing no edits" from
    "I forgot to produce edits", which otherwise trigger a retry.
    """
    full_response = ""
    _done_signaled = False
    _force_done_signaled = False
    current_prompt = prompt

    # [SEARCH:] and [REFS:] use ripgrep on project_root, but ripgrep respects
    # .gitignore and .jarvis_sandbox is in .gitignore — so edits applied to the
    # sandbox are invisible to those tools. When the sandbox exists, search it
    # directly instead, since it contains the live (post-edit) file state.
    _sandbox_dir = os.path.join(project_root, ".jarvis_sandbox") if project_root else None
    _search_root = (
        _sandbox_dir
        if (_sandbox_dir and os.path.isdir(_sandbox_dir))
        else project_root
    )

    # Track this call's research (also writes to shared cache if provided)
    local_research: dict[str, str] = {}
    # Persistent lookup results — survives across rounds. Keyed by "TYPE:arg".
    # When [KEEP:] fires, it REPLACES the corresponding [CODE:] entry, removing
    # the full file from context and inserting only the kept ranges.
    persistent_lookups: dict[str, str] = {}
    # Maps #label → list of TYPE:arg keys, for [DISCARD: #label] support.
    _label_to_keys: dict[str, list[str]] = {}
    # Stall guard: if the model issues only ALREADY-CACHED tools for two
    # rounds in a row, it's spinning — break and let it commit. Tracks the
    # set of tag-keys requested per round.
    _last_round_keys: set[str] = set()
    _stall_rounds: int = 0

    # ── Context manifest — tracks what this model has actually received ──────
    # {key: {"round": int, "tag_type": str, "arg": str}}
    # Only contains tools this model ran or whose shared-cache results it got.
    _manifest: dict[str, dict] = {}

    # Re-read tracker — counts how many times the model has re-issued a CODE
    # or KEEP for the SAME argument across rounds. CODE/KEEP can legitimately
    # re-fire (the file may have changed), but if the model re-requests the
    # IDENTICAL ranges multiple rounds in a row it's stuck in a "let me verify
    # one more time" loop. We escalate warnings to break the loop.
    # Maps "CODE:path" or "KEEP:path 10-20" → count.
    _reread_count: dict[str, int] = {}

    # Per-round response text — used to build tagged round history in prompt.
    _round_texts: list[str] = []  # _round_texts[i] = text produced in round i+1
    # Round numbers (1-indexed) where the stream aborted on scaffold
    # hallucination — i.e., the model started fabricating a fake
    # `────── ROUND N — your tool result ──────` block inside its own
    # response. The next round's prompt prepends a ⚠ notice to that
    # round's tool-result section so the model sees: "you were inventing
    # — here are the real results."
    _hallucinated_rounds: set[int] = set()

    # ── PLAN state ───────────────────────────────────────────────────────
    # The planner can use === PLAN === / === PLAN_EDIT === blocks to draft
    # a plan incrementally. `current_plan` is the live text of the plan;
    # it persists across rounds and is rendered back in [YOUR PLAN] each
    # round. `plan_version` increments on every modification so the
    # manifest can show "v3 (47 lines)".
    current_plan: str = ""
    plan_version: int = 0
    # Per-round log of plan ops applied (op kind + result). Surfaced into
    # the manifest so the model sees what its plan ops did and didn't do.
    _plan_op_log: list[str] = []

    # Track which DEEP THINK preamble sections the model has already
    # completed in round 1, so the continuation prompt in rounds 2+ can
    # tell the model "you already wrote these — don't redo them."
    # Populated after round 1 by scanning _round_texts[0] for known
    # section markers.
    _preamble_done: list[str] = []

    # Edit-apply feedback captured from the on_stop callback. Each entry
    # is the string returned by on_stop in a round, recording which edit
    # blocks applied and which were skipped. The most recent entry is
    # injected into the next-round prompt as "EDIT APPLICATION RESULTS"
    # so the model sees explicit success/failure instead of having to
    # infer it from re-reading the file. This was the dominant cause
    # of the step-3 19-round loop on domains/prompts.py: the model
    # wrote 19 nearly-identical edits with no feedback that one of them
    # had already landed, then accidentally landed several duplicates.
    _last_edit_feedback: str | None = None

    # Track edit attempts per file across rounds for stall detection.
    # Each entry: list[bool] of round-by-round "did any edit on this
    # file apply successfully?" When the recent N attempts on a file
    # are all failures, we surface a stop-flailing nudge.
    _edit_attempts_per_file: dict[str, list[bool]] = {}

    # Terms whose SEARCH / REFS / LSP lookup returned NO MATCHES. When the
    # model re-requests one of these (or the cache surfaces it again on
    # subsequent rounds), `_annotate_entry` prepends a "NOT IN CODEBASE"
    # header so the model stops chasing a phantom symbol. The kimi-k2.6
    # planning loop on `handle_deepcode` (an entry that exists in the
    # stale code map but not in the actual project) is what motivated
    # this — kimi re-issued the search 3 rounds in a row before the stall
    # guard force-broke the loop.
    _not_found_terms: set[str] = set()

    def _stop_check(accumulated: str) -> bool:
        # STOP, DONE, and CONTINUE are all two-tag signals.
        # Use _mask_for_signals (NOT _mask_quoted_tags) — the signal
        # canonical position is OUTSIDE [tool use] blocks (right after
        # [/tool use]). The full _mask_quoted_tags masks every '[' outside
        # [tool use] blocks when any block is present — which used to
        # eat the signal and make the runtime miss [STOP][CONFIRM_STOP]
        # entirely. The lighter mask still protects against backtick /
        # fenced-block discussion of the syntax.
        masked = _mask_for_signals(accumulated)
        if FORCE_DONE_TAG.search(masked):
            return True
        if DONE_TAG.search(masked):
            return True
        if STOP_TAG.search(masked):
            return True
        if CONTINUE_TAG.search(masked):
            return True

        # `stop_on_tool_block` is accepted for back-compat with call sites
        # that still pass it, but it intentionally does NOTHING here.
        # Stopping on `[/tool use]` alone used to cut the stream BEFORE the
        # model finished writing `[CONFIRM_STOP]`, causing tool execution
        # without a real signal. The two-tag pair is now the ONLY trigger.
        return False

    # Temporary instrumentation (env-gated, zero-cost when unset): trace WHY the
    # tool loop exits, to root-cause the merger stopping after round 1.
    _DBG_LOOP = bool(os.environ.get("JARVIS_DEBUG_TOOLLOOP"))
    def _dbg(msg):
        if _DBG_LOOP:
            warn(f"  [toolloop:{log_label}] R{round_num}: {msg}")

    _empty_streak = 0
    _dead_tool_nudges = 0   # bounded rescues for unrecognized/disabled tool calls
    for round_num in range(1, max_rounds + 1):
        # ── Dump the exact prompt sent to this model this round ──────────
        # Diagnostic only — writes `{session_dir}/prompts/{model}__{label}__R{N}.md`
        # so a debugger can read EXACTLY what the model saw for any round.
        # Loops, parroting, and "model ignored the manifest" failures are
        # all visible only when you can compare prompt vs response side-
        # by-side. The label slug disambiguates parallel call_with_tools
        # invocations of the same model in different roles (e.g. planning
        # vs improving the same plan).
        try:
            from core import thought_logger
            _sd = thought_logger.session_dir()
            if _sd is not None:
                _prompts_dir = _sd / "prompts"
                _prompts_dir.mkdir(parents=True, exist_ok=True)
                _short_model = model.split("/")[-1]
                _safe_label = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in (log_label or "call")
                )[:40]
                _path = _prompts_dir / f"{_short_model}__{_safe_label}__R{round_num}.md"
                _path.write_text(current_prompt, encoding="utf-8")
        except Exception:
            pass  # diagnostic-only — never break the model loop on log failures

        result = await call_with_retry(
            model, current_prompt, max_tokens=max_tokens,
            stop_check=_stop_check,
            log_label=f"{log_label} — R{round_num}" if log_label else f"R{round_num}",
        )

        # Apply [continue from: -N] backtrack directives BEFORE any other
        # processing — every downstream consumer (signal detection,
        # masking, plan extraction, tool extraction, edit extraction)
        # should see the rewritten response, never the discarded content.
        # The live stream log still shows the original (handy for
        # debugging), but artifacts only carry the clean version.
        _result_pre_backtrack = result
        result = _apply_continue_from(result)
        if result != _result_pre_backtrack:
            status(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"[continue from: -N] backtrack applied "
                f"({len(_result_pre_backtrack):,} → {len(result):,} chars)"
            )

        # ── Empty-response guard ─────────────────────────────────────
        # Some models return an empty string under load (no reasoning, no
        # visible content). Without this, the loop kept calling the same
        # model 30 times for nothing — every call writes a header to the
        # log and burns rate-limit. Two empty responses in a row → break.
        if not result or not result.strip():
            _empty_streak += 1
            if _empty_streak >= 2:
                warn(f"  [{model.split('/')[-1]}] round {round_num}: empty response "
                     f"× {_empty_streak} — breaking tool loop")
                break
            warn(f"  [{model.split('/')[-1]}] round {round_num}: empty response "
                 f"(streak {_empty_streak}) — nudging model")
            current_prompt = (
                current_prompt
                + "\n\nNote: your previous response was empty. "
                  "Please write your answer now."
            )
            continue
        _empty_streak = 0

        # ── Cross-round signal detection ──────────────────────────────
        # When streaming gets cut mid-signal (e.g. max_tokens hits at
        # `[STOP]\n[CONFIRM_` and the rest arrives on the next call),
        # checking only the current round's text would miss the signal.
        # To handle this, we check signals against the BRIDGE between
        # the previous round's tail and the current round's head, then
        # fall back to the current round on its own.
        #
        # Algorithm:
        #   1. Compute `bridge` = last 64 chars of prev round + current
        #      result. 64 chars is enough to span any signal pair (the
        #      longest is [CONTINUE][CONFIRM_CONTINUE] at ~30 chars).
        #   2. Detect signals on the masked bridge.
        #   3. If a signal spans the boundary, mark that and strip the
        #      cross-boundary fragments from both round texts.
        #   4. Also detect signals fully inside current `result` (normal case).
        prev_idx = len(_round_texts) - 1 if _round_texts else -1

        def _current_bridge() -> tuple[str, int, str]:
            """Recompute (masked_bridge, bridge_offset, prev_tail) from the
            CURRENT state of `result` and `_round_texts`. Must be called
            every time `_signal_in_bridge` is consulted — `result` and the
            prev round's text both mutate while we strip signals."""
            tail = ""
            if prev_idx >= 0:
                tail = _round_texts[prev_idx][-64:]
            return _mask_for_signals(tail + result), len(tail), tail

        def _signal_in_bridge(pattern):
            """Return the FIRST bridge-relative match for `pattern`, or None.

            Always recomputes the bridge from the current `result` and
            `_round_texts[prev_idx]` so consecutive calls reflect prior
            consumes. The earlier implementation cached `_masked_bridge`
            at round start, which made `while _signal_in_bridge(STOP_TAG)`
            an infinite loop whenever two signals appeared in the bridge:
            the regex kept returning the same match against a stale mask.
            """
            masked, _offset, _tail = _current_bridge()
            return pattern.search(masked)

        def _consume_bridge_signal(pattern) -> bool:
            """Strip ONE signal (the leftmost match) that may span the
            prev-round / current-round boundary. Updates `_round_texts[-1]`
            and current `result` so the signal text is removed from both
            halves. Returns True if anything was stripped.

            Three cases:
              (1) signal entirely in prev_tail  (s, e <= bridge_offset)
              (2) signal entirely in current    (s >= bridge_offset)
              (3) signal straddles boundary     (s < bridge_offset < e)

            Each call recomputes the bridge first so repeated consumes
            converge — caller may loop on `while _signal_in_bridge(...)`.
            """
            nonlocal result
            masked, bridge_offset, prev_tail = _current_bridge()
            m = pattern.search(masked)
            if not m:
                return False
            s, e = m.start(), m.end()
            if s < bridge_offset and prev_idx >= 0:
                # Signal at least starts in prev_tail.
                prev_text = _round_texts[prev_idx]
                tail_start_abs = len(prev_text) - len(prev_tail)
                prev_start_abs = tail_start_abs + s
                # The signal extends in prev up to min(e, bridge_offset)
                prev_end_in_bridge = min(e, bridge_offset)
                prev_end_abs = tail_start_abs + prev_end_in_bridge
                prev_end_abs = min(prev_end_abs, len(prev_text))
                _round_texts[prev_idx] = (
                    prev_text[:prev_start_abs] + prev_text[prev_end_abs:]
                )
                # Strip the remainder from current result if any.
                if e > bridge_offset:
                    cur_end_in_bridge = e - bridge_offset
                    result = result[cur_end_in_bridge:]
            else:
                # Whole signal inside current result.
                cur_s = s - bridge_offset
                cur_e = e - bridge_offset
                result = result[:cur_s] + result[cur_e:]
            return True

        # Verify-gate keep-alive: set when a [DONE] arrives in the SAME turn
        # as fresh edits. We then DON'T finish — we apply the edits, show the
        # model the diff, and make it verify before re-issuing [DONE]. Reset
        # each round so it only holds for the one turn it's needed.
        _gate_keep_alive = False

        # ── FORCE DONE: step requirement already met, no edits needed ─
        # Must be checked BEFORE DONE_TAG because `[FORCE DONE]` shouldn't
        # also be eaten by the DONE handler. Implies _done_signaled — the
        # round terminates either way — but ALSO sets _force_done_signaled
        # so the IMPLEMENT loop knows not to retry on "zero edits".
        if _signal_in_bridge(FORCE_DONE_TAG):
            _consume_bridge_signal(FORCE_DONE_TAG)
            while _signal_in_bridge(DONE_TAG):
                _consume_bridge_signal(DONE_TAG)
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
            while _signal_in_bridge(CONTINUE_TAG):
                _consume_bridge_signal(CONTINUE_TAG)
            result = result.rstrip()
            _round_texts.append(result)
            full_response += result
            _done_signaled = True
            _force_done_signaled = True
            break

        # ── DONE: finish the loop ─────────────────────────────────────
        if _signal_in_bridge(DONE_TAG):
            _consume_bridge_signal(DONE_TAG)
            # Also remove any trailing STOP/CONTINUE in the bridge.
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
            while _signal_in_bridge(CONTINUE_TAG):
                _consume_bridge_signal(CONTINUE_TAG)
            # ── VERIFY GATE ──────────────────────────────────────────
            # A [DONE] written in the SAME turn as fresh edits never saw the
            # result — exactly the premature-[DONE]→NO_PATCH failure. Convert
            # it to an "end edit section": apply the edits, show the diff,
            # and require a verify pass. [DONE] is honored only when the turn
            # carries NO new edits (already verified, or a clean no-op). The
            # check is non-mutating; on_stop (below) does the real apply +
            # marks the blocks seen, so next round's [DONE] carries nothing
            # new and is honored.
            if (has_pending_edits is not None and on_stop is not None
                    and has_pending_edits(full_response + result)):
                _gate_keep_alive = True
                status(
                    f"  [{model.split('/')[-1]}] round {round_num}: "
                    f"[DONE] held — applying edits, then showing the diff to "
                    f"verify before finishing"
                )
                # fall through (NO break): on_stop applies + returns the diff,
                # the continuation prompt frames the verify step.
            else:
                result = result.rstrip()
                _round_texts.append(result)
                full_response += result
                _done_signaled = True
                break

        # ── CONTINUE: keep writing without tools ──────────────────────
        if _signal_in_bridge(CONTINUE_TAG):
            _consume_bridge_signal(CONTINUE_TAG)
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
            result = result.rstrip()
            _round_texts.append(result)
            full_response += result
            current_prompt = _build_continue_prompt(
                base_prompt=prompt,
                round_history_texts=_round_texts,
                round_num=round_num,
                max_rounds=max_rounds,
                preamble_done=_preamble_done,
            )
            status(f"  [{model.split('/')[-1]}] round {round_num}: "
                   f"[CONTINUE] signal — resuming output without tool processing")
            continue

        # ── STOP: apply tools + continue thinking ─────────────────────
        has_stop = bool(_signal_in_bridge(STOP_TAG))
        if has_stop:
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)

        # ── Auto-complete unclosed [tool use] blocks ──────────────────
        # Models like minimax write [tool use] [CODE:...] but forget [/tool use].
        # Without the closing tag, _mask_quoted_tags never activates block
        # enforcement, so tags in explanatory text can fire accidentally.
        # We close them BEFORE recording the round text so subsequent rounds'
        # YOUR WORK SO FAR shows the auto-closed form (matching what the tag
        # extractor actually fires on this round).
        result, n_autoclosed = _autocomplete_tool_blocks(result)
        if n_autoclosed:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"auto-closed {n_autoclosed} unclosed [tool use] block(s) "
                f"— model forgot [/tool use]"
            )

        # ── Near-verbatim response loop detector ─────────────────────────
        # Run AFTER all post-processing (signal strip, auto-close, leak
        # trim) so we compare apples-to-apples against the same form that
        # `_round_texts[-1]` stored last round.
        #
        # Use a WHITESPACE-NORMALISED comparison rather than `==`. Streaming
        # output legitimately varies in trailing newlines / blank-line
        # density across rounds, so byte-equality misses obvious loops
        # like 20260512_171644 kimi-k2.6 R4/R5 (identical content, differed
        # only by a single blank line). Collapsing whitespace runs catches
        # those without affecting any real content difference.
        def _norm_response(s: str) -> str:
            return re.sub(r'\s+', ' ', s).strip()

        if _round_texts and _norm_response(result) == _norm_response(_round_texts[-1]):
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: response "
                f"NEAR-IDENTICAL to round {round_num - 1} ({len(result)} chars, "
                f"whitespace-normalised match) — model is pattern-looping. "
                f"Forcing commit."
            )
            try:
                from core import workflow_log
                workflow_log.phase_event(
                    f"R{round_num} [{model.split('/')[-1]}/{log_label or 'call'}] DUP_RESPONSE",
                    chars=len(result),
                    action="forced-commit",
                )
            except Exception:
                pass
            # Don't re-append (already in _round_texts from prior round),
            # don't re-run identical tool calls (pure waste), break out.
            break

        # ── Scaffold-hallucination trim ──────────────────────────────────
        # If the model wrote `────── ROUND` inside its response, it was
        # fabricating a fake tool-result section. The stream guard already
        # aborted, but the partial text up to (and possibly including) the
        # scaffold marker is still in `result`. Trim from the marker
        # onward — the model's real reasoning + tool calls before that
        # point are KEPT (so the tools fire normally based on placement);
        # only the invented content is dropped. Flag the round so the
        # next prompt prepends a targeted notice in this round's tool
        # result section.
        _hallu_idx = result.find("────── ROUND")
        if _hallu_idx >= 0:
            _hallucinated_rounds.add(round_num)
            n_trimmed = len(result) - _hallu_idx
            result = result[:_hallu_idx].rstrip()
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"scaffold-hallucination — trimmed {n_trimmed:,} chars of "
                f"fabricated tool-result content"
            )

        _round_texts.append(result)

        # ── Process PLAN / PLAN_EDIT blocks ───────────────────────────────
        # The planner can write its plan incrementally using top-level
        # `=== PLAN === … === END PLAN ===` blocks (full write/rewrite) and
        # `=== PLAN_EDIT === [REPLACE LINES N-M]…[/REPLACE]
        #                    [INSERT AFTER LINE N]…[/INSERT]
        #                    === END PLAN_EDIT ===` blocks (surgical edits).
        # The current plan lives in `current_plan` and is rendered back in
        # [YOUR PLAN] each round with line numbers so the model can target
        # specific lines for edits.
        #
        # Multiple PLAN blocks per round → applied in document order; the
        # last one wins for full writes. PLAN + PLAN_EDIT mix → ops are
        # applied in document order across both kinds.
        _plan_op_log = []  # reset per round; surfaced into the manifest
        _plan_changed = False
        # Find all plan blocks in document order
        _plan_ops_with_pos: list[tuple[int, str, str]] = []
        for m in _PLAN_BLOCK.finditer(result):
            _plan_ops_with_pos.append((m.start(), "write", m.group(1)))
        for m in _PLAN_EDIT_BLOCK.finditer(result):
            _plan_ops_with_pos.append((m.start(), "edit", m.group(1)))
        _plan_ops_with_pos.sort(key=lambda t: t[0])
        for _pos, _kind, _body in _plan_ops_with_pos:
            if _kind == "write":
                current_plan = _strip_think(_body).rstrip()
                plan_version += 1
                _line_count = current_plan.count('\n') + 1 if current_plan else 0
                _plan_op_log.append(
                    f"v{plan_version}: PLAN written ({_line_count} lines)"
                )
                # Empty/structure-less plan guard (error-audit F1): a plan with
                # no `### STEP N:` header gives the coder nothing to implement,
                # and otherwise ships SILENTLY. Surface a loud per-round note.
                if not re.search(r'(?m)^\s*###\s*STEP\s*\d+', current_plan):
                    _plan_op_log.append(
                        "⚠ PLAN has NO `### STEP N:` headers — the coder cannot "
                        "implement it. Add at least one `### STEP 1: <name>` with a "
                        "`FILES:` line and WHAT-TO-DO before [PLAN DONE]."
                    )
                _plan_changed = True
            else:  # edit
                if not current_plan:
                    _plan_op_log.append(
                        "PLAN_EDIT skipped — no plan exists yet. "
                        "Use === PLAN === first to create the plan."
                    )
                    continue
                _new_plan, _edit_logs = _apply_plan_edits(current_plan, _strip_think(_body))
                if _new_plan != current_plan:
                    current_plan = _new_plan
                    plan_version += 1
                    _plan_op_log.append(
                        f"v{plan_version}: PLAN_EDIT applied — "
                        + "; ".join(_edit_logs)
                    )
                    _plan_changed = True
                else:
                    _plan_op_log.append(
                        "PLAN_EDIT no-op — " + "; ".join(_edit_logs)
                    )

        # ── PLAN DONE: finalize and return the plan as the answer ─────────
        # Two-tag signal mirroring [STOP][CONFIRM_STOP] / [DONE][CONFIRM_DONE].
        # When fires, the runtime returns the planner's final answer and
        # breaks out of the loop.
        #   • Preferred path: model used === PLAN === to build `current_plan`
        #     incrementally → use that as the answer.
        #   • Backward-compat path: model wrote a plan in raw prose (## GOAL,
        #     ## REQUIREMENTS, etc.) WITHOUT the === PLAN === wrapper, then
        #     ended with [PLAN DONE][CONFIRM_PLAN_DONE]. Observed in qwen-3.5:
        #     wrote a full plan + [PLAN DONE], but `current_plan` was empty
        #     so the prior runtime just warned and continued — qwen's plan
        #     was silently discarded and the next round began. Now we fall
        #     back to using the model's current-round response as the answer.
        # In both cases: strip the [PLAN DONE]/[CONFIRM_PLAN_DONE]/[STOP]/
        # [CONFIRM_STOP] tags from the answer so they don't leak downstream.
        # ── PLAN_DONE detection: masking + context validation ─────────────
        # Two-layer filter so only a *genuine* terminal PLAN_DONE fires:
        #
        #   Layer A (mask): a pair written INSIDE a code fence, [think]
        #     block, `=== PLAN === ... === END PLAN ===` body, edit body,
        #     etc. is data, not a signal. _mask_for_signals blanks the
        #     leading `[` of those instances so the regex can't match.
        #
        #   Layer B (context): of the candidates that survive masking,
        #     only those that follow a recognized "I just finished a plan"
        #     marker count. See _plan_done_context_kind for the list:
        #       - `=== END PLAN ===` in the last 2000 chars
        #       - `## VERIFICATION` / `## CONFIDENCE GATE` / etc. in 2000
        #       - closed [/think] or </think> in the last 800 chars
        #
        # If at least one candidate is in a valid context, the LAST such
        # one fires (terminal signals are by nature near EOF, and writing
        # multiple PLAN_DONE pairs is itself a protocol error — we honor
        # the final one so the model can't accidentally commit twice).
        #
        # If candidates exist but NONE are in a valid context, the model
        # likely emitted PLAN_DONE mid-investigation or while writing
        # prose that documented the protocol. We fall through to the
        # "rejected PLAN_DONE" branch below, which surfaces a structured
        # correction next round.
        _signal_masked = _mask_for_signals(result)
        _pd_matches = list(PLAN_DONE_TAG.finditer(_signal_masked))
        _pd_valid: list[tuple[int, str]] = []
        for _m in _pd_matches:
            kind = _plan_done_context_kind(result, _m.start())
            if kind is not None:
                _pd_valid.append((_m.start(), kind))

        if _pd_valid:
            _signal_pos, _ctx_kind = _pd_valid[-1]
            if current_plan:
                _answer = current_plan
                _src = f"=== PLAN === (v{plan_version}, {current_plan.count(chr(10)) + 1} lines)"
            else:
                # Backward-compat: use the raw response as the plan.
                _answer = _strip_think(result)
                for _tag in (PLAN_DONE_TAG, STOP_TAG, DONE_TAG, CONTINUE_TAG):
                    _answer = _tag.sub('', _answer)
                # Strip stray half-signals too (model may have written
                # [PLAN DONE] alone if half-arrived during streaming).
                _answer = re.sub(
                    r'\[(?:PLAN\s+DONE|CONFIRM_PLAN_DONE|STOP|CONFIRM_STOP|'
                    r'DONE|CONFIRM_DONE|FORCE\s+DONE|CONFIRM_FORCE_DONE|'
                    r'CONTINUE|CONFIRM_CONTINUE)\]',
                    '', _answer, flags=re.IGNORECASE,
                ).strip()
                _src = "raw-prose plan (no === PLAN === block used)"
                # SALVAGE: if the visible body is also thin, the model did its
                # plan inside the thinking channel and emitted nothing usable.
                # Recover the plan from the reasoning rather than discarding it
                # (the correct plan is in there — see pylint-4551). Belt to the
                # prompt's "think then EXIT and WRITE" suspenders.
                if len(_answer) < _PLAN_SALVAGE_THRESHOLD:
                    _salvaged = _salvage_plan_from_think(result)
                    if len(_salvaged) > len(_answer):
                        _answer = _salvaged
                        _src = ("salvaged from the thinking channel (plan was "
                                "written inside think, not as visible output)")
                        warn(
                            f"  [{model.split('/')[-1]}] round {round_num}: "
                            f"[PLAN DONE] with empty visible plan — SALVAGED "
                            f"{len(_answer):,} chars from the reasoning channel "
                            f"(emit the plan as VISIBLE text next time)"
                        )
                if "salvaged" not in _src:
                    warn(
                        f"  [{model.split('/')[-1]}] round {round_num}: "
                        f"[PLAN DONE] with empty === PLAN === — falling back "
                        f"to raw-prose response ({len(_answer):,} chars)"
                    )
            status(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"[PLAN DONE] in context '{_ctx_kind}' — "
                f"finalizing plan from {_src}"
            )
            full_response = _answer
            _done_signaled = True
            break

        # PLAN_DONE pair written but NOT in a valid context — reject it
        # and queue a structured correction for the next round. Mirrors
        # the bare-signal correction path below; we set a flag here and
        # the injection happens in the same block (so we never double-
        # inject when both conditions are true on the same round).
        _suspected_invalid_plan_done = bool(_pd_matches) and not _pd_valid

        # ── Capture which preamble sections were written in round 1 ──────
        # Once round 1 finishes, scan _round_texts[0] for the section
        # markers our prompts use. The continuation prompt builder will
        # quote this list back to the model in subsequent rounds as
        # "you already completed these — do not redo them."
        if round_num == 1 and not _preamble_done:
            _preamble_done = _detect_preamble_sections(_round_texts[0])

        # ── Bare-tag correction: model wrote [STOP] / [DONE] / [CONTINUE] alone ──
        # In the two-tag signal protocol, a bare half is just text. But the
        # model may have INTENDED it as a signal and missed the protocol.
        # Detect the situation and inject a one-shot correction below.
        # Only fires when NO real signal fired this round.
        _suspected_bare_signal = False
        if not has_stop and not _done_signaled:
            _mask_bare = _mask_quoted_tags(result)
            # Strip real two-tag signal matches first so we don't
            # double-count [STOP] inside [STOP][CONFIRM_STOP] as bare.
            _mask_bare = STOP_TAG.sub('', _mask_bare)
            _mask_bare = DONE_TAG.sub('', _mask_bare)
            _mask_bare = FORCE_DONE_TAG.sub('', _mask_bare)
            _mask_bare = CONTINUE_TAG.sub('', _mask_bare)
            if (_BARE_STOP.search(_mask_bare)
                    or _BARE_DONE.search(_mask_bare)
                    or _BARE_FORCE_DONE.search(_mask_bare)
                    or _BARE_CONTINUE.search(_mask_bare)):
                _suspected_bare_signal = True


        # ── Detect tool tags via the bulletproof TagDetector ──────────────
        # Single source of truth. TagDetector runs two independent extraction
        # passes (DOTALL regex + bracket-scan), classifies every match with
        # an explicit rejection reason, and is covered by a self-test that
        # fails JARVIS startup if any case regresses. See core/tool_detector.py.
        from core.tool_detector import TagDetector
        _detector = TagDetector(result)
        code_tags     = _detector.valid_args("SEARCH")    if enable_code_search else []
        web_tags      = _detector.valid_args("WEBSEARCH") if enable_web_search else []
        detail_tags   = _detector.valid_args("DETAIL")    if detailed_map else []
        file_tags     = _detector.valid_args("CODE")      if project_root else []
        refs_tags     = _detector.valid_args("REFS")      if project_root else []
        purpose_tags  = _detector.valid_args("PURPOSE")   if purpose_map else []
        # SEMANTIC now indexes the CODE itself (AST chunks) — it no longer needs
        # a purpose map, so it's a first-class tool gated only on project_root.
        semantic_tags = _detector.valid_args("SEMANTIC")  if project_root else []
        lsp_tags      = _detector.valid_args("LSP")       if project_root else []
        knowledge_tags = _detector.valid_args("KNOWLEDGE")
        # RUN is extracted OUTSIDE the detector (bracket-balanced) — its free-form
        # shell command routinely contains `]` and needs no [tool use] wrapper.
        from core.review_verify import extract_run_cmds
        run_tags      = extract_run_cmds(result)          if allow_run else []
        keep_tags     = _detector.valid_args("KEEP")      if project_root else []
        view_tags     = _detector.valid_args("VIEW")      if project_root else []
        dependency_tags = _detector.valid_args("DEPENDENCY") if project_root else []
        # DISCARD args are `#label` form — keep the legacy extractor (it
        # has bespoke regex that the new detector's validator would over-
        # tighten). DISCARD never collides with content/position issues.
        discard_tags = extract_discard_tags(result)

        # ── Per-round tag-count cap ──────────────────────────────────────
        # When the model goes off the rails (e.g. after a degeneration
        # near-miss) it can dump 30+ tags in one block. Honouring all of
        # them would explode the next round's context. Cap by truncating
        # each tag list to the FIRST N (preserving order so the model's
        # primary intent is kept) until the sum fits MAX_TAGS_PER_ROUND.
        _all_tag_lists = [
            code_tags, web_tags, detail_tags, file_tags, refs_tags,
            purpose_tags, semantic_tags, lsp_tags, knowledge_tags,
            keep_tags, view_tags, dependency_tags, run_tags,
        ]
        _total_tags = sum(len(lst) for lst in _all_tag_lists)
        if _total_tags > MAX_TAGS_PER_ROUND:
            # Pro-rata trim: each list keeps ⌈cap * its_share⌉ tags.
            scale = MAX_TAGS_PER_ROUND / _total_tags
            trimmed_total = 0
            for lst in _all_tag_lists:
                keep_n = max(1, int(len(lst) * scale)) if lst else 0
                del lst[keep_n:]
                trimmed_total += len(lst)
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"{_total_tags} tags exceeds cap {MAX_TAGS_PER_ROUND} — "
                f"trimmed to {trimmed_total}"
            )

        has_tags = bool(code_tags or web_tags or detail_tags or file_tags
                        or refs_tags or purpose_tags or semantic_tags or lsp_tags
                        or knowledge_tags or keep_tags or view_tags
                        or dependency_tags or discard_tags)
        _dbg(f"has_tags={has_tags} has_stop={has_stop} "
             f"purpose={len(purpose_tags)} file={len(file_tags)} code={len(code_tags)} "
             f"refs={len(refs_tags)} view={len(view_tags)} detail={len(detail_tags)} "
             f"result_len={len(result)}")

        # ── Dropped-tag detection (visibility for the model) ─────────
        # The full _mask_quoted_tags enforces [tool use] blocks: any tag
        # written OUTSIDE [tool use]...[/tool use] gets masked away and
        # silently ignored. That's correct enforcement, but invisible to
        # the model — which then thinks all its tool calls fired and
        # hallucinates results for the ones that didn't.
        #
        # Detect this case: re-run tag extraction with the LIGHTER mask
        # (no [tool use] enforcement) and compare. Tags that appear in
        # the light extraction but NOT in the strict extraction are
        # "dropped" — they were detected but never fired. Surface those
        # to the model so it knows what's missing and can re-wrap them.
        # ── Dropped-tag list (from the detector's rejection set) ─────────
        # The detector classifies every tag found in the response with an
        # explicit rejection reason. We surface the ones that LOOK like
        # genuine tool calls but didn't fire because of placement issues
        # (outside [tool use], or no [tool use] at all). We DON'T surface
        # rejections for masked-by-* reasons (the tag was inside a code
        # block / think / edit — clearly not a real call) or for the
        # edit-syntax overloads (`[SEARCH: 45-49]`, `[SEARCH: file.py]`).
        _SURFACE_REJECTIONS = {
            "outside-tool-use-block",
            "no-tool-use-block-in-response",
        }
        dropped_tags: list[tuple[str, str]] = []
        _dropped_seen: set[str] = set()
        # Malformed-arg / unknown-tool rejections: tags the model clearly
        # MEANT as tool calls but that can't fire (empty arg, prose instead
        # of a path, unknown tool name, …). Previously dropped silently — the
        # model then hallucinated results. Collect them to surface explicitly.
        bad_tags: list[tuple[str, str, str]] = []   # (tag_type, arg, reason)
        _bad_seen: set[str] = set()
        for t in _detector.rejected_tags():
            _reason = t.rejection_reason or ""
            if _reason.startswith("malformed-arg") or _reason == "unknown-tag-type":
                bkey = _norm_key(t.tag_type, t.clean_arg)
                if bkey not in _bad_seen:
                    _bad_seen.add(bkey)
                    detail = _reason.split(":", 1)[1].strip() if ":" in _reason else _reason
                    bad_tags.append((t.tag_type, t.clean_arg, detail))
                continue
            if t.rejection_reason not in _SURFACE_REJECTIONS:
                continue
            # Edit-syntax overloads of [SEARCH:] are not real tool calls;
            # validate_arg already rejects them as "malformed-arg" but
            # the placement check fires FIRST. Filter the same shapes
            # here so we don't tell the model "your edit anchor didn't
            # fire as a search."
            if t.tag_type == "SEARCH":
                if re.match(r'^\d+\s*-\s*\d+$', t.clean_arg):
                    continue
                if (re.search(r'\.\w{1,5}$', t.clean_arg)
                        and ' ' not in t.clean_arg):
                    continue
            key = _norm_key(t.tag_type, t.clean_arg)
            if key in _dropped_seen:
                continue
            _dropped_seen.add(key)
            dropped_tags.append((t.tag_type, t.clean_arg))

        # ── Bare-signal correction injection ─────────────────────────
        # Model wrote a lone [STOP] or [DONE] (no CONFIRM half) AND no
        # real tool tags this round. Most likely they MEANT to signal but
        # used the old single-tag form. Inject a one-shot note teaching
        # the two-tag protocol so they can correct themselves next round.
        # If real tool tags ARE present, the round is doing work — skip
        # the nag, the next round's results will let them try again.
        if _suspected_bare_signal and not has_tags:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"bare [STOP]/[DONE] without CONFIRM half — "
                f"injecting two-tag protocol reminder"
            )
            full_response += result
            correction = (
                "\n\n[SYSTEM NOTE: You wrote [STOP] or [DONE] without its "
                "CONFIRM half — that does NOT fire the signal. The runtime "
                "uses a TWO-TAG combination to prevent accidental signals "
                "from prose mentions.\n\n"
                "To execute pending tool calls and continue thinking, write:\n"
                "  [STOP]\n  [CONFIRM_STOP]\n\n"
                "To finalize edits and end the loop (coders/reviewers only), write:\n"
                "  [DONE]\n  [CONFIRM_DONE]\n\n"
                "Both halves must appear in order, separated only by "
                "whitespace/newlines. A bare [STOP] alone — anywhere, "
                "in any context, including the end of your response — is "
                "treated as plain text and the loop continues.]\n\n"
                "Continue your response with the correct two-tag signal "
                "if that's what you meant.\n"
            )
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response + correction
            )
            full_response = ""
            continue

        # ── PLAN_DONE rejected (invalid context) ─────────────────────
        # Model emitted [PLAN DONE][CONFIRM_PLAN_DONE] but it didn't
        # follow any recognized termination marker. The signal was
        # *intentional* (both halves present, both unmasked), just
        # placed wrong. Tell the model what we expected so it can
        # correct itself in the next round. Skip the injection when
        # real tool tags ARE present — the round is doing work and
        # the lookups will themselves push the model toward the
        # canonical terminal section.
        if _suspected_invalid_plan_done and not has_tags:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"[PLAN DONE][CONFIRM_PLAN_DONE] rejected — not in a "
                f"recognized termination context. Injecting protocol "
                f"reminder."
            )
            full_response += result
            correction = (
                "\n\n[SYSTEM NOTE: You emitted [PLAN DONE][CONFIRM_PLAN_DONE] "
                "but the runtime REJECTED it — the signal was not in a "
                "valid termination context, so the plan was NOT finalized "
                "and the loop continues.\n\n"
                "PLAN_DONE only fires when it appears in one of these "
                "structural positions:\n"
                "  1. AFTER `=== END PLAN ===` — your final `=== PLAN ===` "
                "block was properly closed and the signal terminates it.\n"
                "  2. AFTER a canonical terminal section header — one of "
                "`## VERIFICATION`, `## CONFIDENCE GATE`, "
                "`## PRE-MORTEM RESOLUTION`, `## TEST CRITERIA`, "
                "`## FINAL NOTES`, or `## SUMMARY`. Write this section "
                "as the last part of your plan; the runtime treats it as "
                "the conventional 'I'm done' marker.\n"
                "  3. AFTER a closed `[think]...[/think]` or "
                "`<think>...</think>` block — reserved for the case where "
                "you have genuine reason to commit early (e.g. the user's "
                "task is a trivial fix that doesn't warrant a ## VERIFICATION "
                "section). The [think] block must explain WHY ending early "
                "is correct.\n\n"
                "What to do next:\n"
                "  • Normal case: write the canonical terminal section "
                "(## VERIFICATION) describing how the user will observe the "
                "change working, then re-issue [PLAN DONE][CONFIRM_PLAN_DONE] "
                "on its own at the end of your response.\n"
                "  • Early-commit case: open a [think]...[/think] block "
                "explaining why the canonical section doesn't apply, then "
                "re-issue [PLAN DONE][CONFIRM_PLAN_DONE] immediately after "
                "the closing [/think].\n\n"
                "Your plan and any progress so far are preserved in YOUR "
                "PLAN / YOUR PAST THINKING above — no work was lost. "
                "Continue from where you left off.]\n"
            )
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response + correction
            )
            full_response = ""
            continue

        # ── Verbatim-restatement detection ────────────────────────────
        # In round 2+, if the model re-writes preamble sections AND the
        # content is substantially the same as round 1 (no new conclusion
        # named), it's wasting tokens. We only intervene when the redo is
        # pure restatement — REVISION is welcome and shouldn't be flagged.
        # Heuristic: section header present AND no REINFORCE/REVISE/DEEPER
        # marker AND content overlap with round 1 is high (≥60% of lines
        # appear in round 1 verbatim).
        if round_num >= 2 and _preamble_done:
            this_round_sections = _detect_preamble_sections(result)
            redone = [s for s in this_round_sections if s in _preamble_done]
            if redone:
                # Check for revision markers — if present, allow it.
                has_revision_marker = bool(re.search(
                    r'\b(REINFORCE|REVISE|REVISING|GO DEEPER|on reflection'
                    r'|new evidence|update(?:s|d)?\s+(?:my|the)\s+(?:plan|approach))\b',
                    result, re.IGNORECASE,
                ))
                # Check overlap with round 1 — if the lines mostly match
                # what's already there, it's pure restatement.
                round1_text = _round_texts[0] if _round_texts else ""
                round1_lines = set(
                    ln.strip() for ln in round1_text.splitlines()
                    if len(ln.strip()) > 20  # ignore short / boilerplate lines
                )
                this_lines = [
                    ln.strip() for ln in result.splitlines()
                    if len(ln.strip()) > 20
                ]
                if this_lines:
                    overlap = sum(1 for ln in this_lines if ln in round1_lines)
                    overlap_ratio = overlap / len(this_lines)
                else:
                    overlap_ratio = 0.0

                is_pure_restatement = (
                    not has_revision_marker and overlap_ratio >= 0.6
                )

                if is_pure_restatement:
                    warn(
                        f"  [{model.split('/')[-1]}] round {round_num}: "
                        f"verbatim restatement of {len(redone)} preamble "
                        f"section(s) (overlap {overlap_ratio:.0%}) — nudging"
                    )
                    full_response += result
                    redo_list = "\n".join(f"    ✓ {s}" for s in redone)
                    correction = (
                        f"\n\n[SYSTEM NOTE: you just repeated these sections "
                        f"from earlier in your response with no new conclusion:\n"
                        f"{redo_list}\n\n"
                        f"That content stands as written — it's in YOUR WORK "
                        f"SO FAR above. You can REVISE any of it explicitly "
                        f"('REVISE: Approach B is now better because new "
                        f"evidence X shows Y'). If you have nothing new to "
                        f"add to those sections, move on — integrate the tool "
                        f"results with one of REINFORCE / REVISE / GO DEEPER, "
                        f"then take the next concrete action. This is the "
                        f"same response — pick up at the next sentence.]\n\n"
                    )
                    current_prompt = (
                        current_prompt + "\n\nASSISTANT: " + full_response + correction
                    )
                    full_response = ""
                    continue

        # ── "Partial view" hallucination guard ────────────────────────
        # Self-check coders sometimes look at a small file (e.g. 66 lines)
        # and hallucinate that "the output only showed 2 lines, appears
        # to be a partial view" — then re-read the same file 4-5 rounds
        # in a loop. The file is whole; the model is wrong. Detect those
        # phrases and inject a one-shot correction that quotes the real
        # line count from the manifest so the model can see its own error.
        _PARTIAL_VIEW_PHRASES = re.compile(
            r'(?:'
            r'appears?\s+to\s+be\s+a\s+partial(?:/filtered)?\s+view'
            r'|this\s+can.?t\s+be\s+the\s+whole\s+file'
            r'|output\s+(?:seems|looks|appears)\s+(?:filtered|truncated|incomplete)'
            r'|only\s+\d+\s+lines\s+were\s+returned'
            r'|the\s+view\s+is\s+incomplete'
            r'|only\s+showed\s+\d+\s+lines'
            r')',
            re.IGNORECASE,
        )
        _has_partial_view_claim = bool(_PARTIAL_VIEW_PHRASES.search(result))
        # Only nag if the model ALSO re-requested a CODE/KEEP this round
        # (it's not just musing — it's about to loop). And only if NO
        # truncation header is actually present in persistent_lookups.
        _has_legit_truncation = any(
            "SKELETON ONLY" in v or "KEPT " in v or "VIEW: " in v
            for v in persistent_lookups.values()
        )
        if (_has_partial_view_claim and (file_tags or keep_tags or view_tags)
                and not _has_legit_truncation):
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"detected 'partial view' hallucination — injecting "
                f"line-count anchor"
            )
            # Pull the actual line counts from persistent_lookups so the
            # injected note quotes the truth back to the model.
            line_facts = []
            for k, v in persistent_lookups.items():
                if not k.startswith("CODE:"):
                    continue
                m = re.search(r'\((\d+) lines\)', v)
                if m:
                    line_facts.append(f"  • {k[5:]}: {m.group(1)} lines (header is authoritative)")
            facts_block = "\n".join(line_facts) if line_facts else (
                "  • (no [CODE:] reads recorded yet)"
            )
            full_response += result
            correction = (
                "\n\n[SYSTEM NOTE: You wrote a phrase claiming the [CODE:] "
                "output was 'partial' / 'truncated' / 'incomplete'. That is a "
                "HALLUCINATION. The runtime always names the total line count "
                "in the header — `=== Code: <path> (N lines) ===` — and that "
                "number is authoritative.\n\n"
                "Your actual reads this session:\n"
                f"{facts_block}\n\n"
                "If the header says N lines and you see N numbered lines, the "
                "file is COMPLETE. Short files are short, not partial. The "
                "only legitimate truncation markers are 'SKELETON ONLY' and "
                "'KEPT N/M lines' — neither is present in your current reads.\n\n"
                "Do NOT re-request the same file. Reason from the content you "
                "have. If you genuinely needed a different file, name THAT "
                "file in your next tool call.]\n\n"
                "Continue your verification using the content you already have."
            )
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response + correction
            )
            full_response = ""
            continue

        # ── Plan-completion guard ────────────────────────────────────
        # If the response contains BOTH tool tags AND plan-format headers
        # (## GOAL, ## REQUIREMENTS, ## IMPLEMENTATION STEPS, BEST: Plan #N),
        # the model has effectively committed to writing the plan. The tool
        # tags are stray — likely from "let me also check..." mid-plan. If we
        # honor them, the next round asks the model to "continue" and it
        # rewrites the entire plan from scratch (observed in deepseek-v4-pro
        # logs: R2 wrote a plan with stray tags, R3 rewrote it verbatim).
        # Treat the response as final: keep everything, skip tool processing.
        _PLAN_HEADERS = re.compile(
            r'(?m)^(?:#{1,3}\s*(?:GOAL|REQUIREMENTS|IMPLEMENTATION\s+STEPS'
            r'|SHARED\s+INTERFACES|EDGE\s+CASES|VERIFICATION'
            r'|TEST\s+CRITERIA)\b'
            r'|BEST:\s*Plan\s*#?\d+'
            r'|###\s*STEP\s*\d+:)',
            re.IGNORECASE,
        )
        _plan_committed = bool(_PLAN_HEADERS.search(result))
        if _plan_committed and has_tags:
            _dbg("EXIT: plan-headers + tool tags → treat as final")
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"plan headers + stray tool tags detected — "
                f"treating response as final (no tool execution this round)"
            )
            full_response += result
            break

        # Detect the specific mistake: model wrote [SEARCH: N-M] expecting a
        # tool result, but that's edit syntax not a tool.  The filter removed
        # it from code_tags so has_tags is False — which would break the loop
        # and return a partial response with no edit content.
        # Only trigger when [SEARCH: N-M] appears WITHOUT a preceding === EDIT:
        # line (i.e., as a standalone tool call, not inside an edit block).
        _LINE_RANGE_TAG = re.compile(r'\[SEARCH:\s*\d+\s*-\s*\d+\s*\]')
        _EDIT_BLOCK = re.compile(r'===\s*EDIT:', re.IGNORECASE)
        has_misused_search = (
            bool(_LINE_RANGE_TAG.search(result))
            and not bool(_EDIT_BLOCK.search(result))
            and not bool(_EDIT_BLOCK.search(full_response[-500:]))  # not recently in an edit block
        )
        if has_misused_search and not has_tags:
            warn("  ⚠️  Model used [SEARCH: N-M] as a tool call — injecting correction")
            correction = (
                "\n\n[SYSTEM NOTE: [SEARCH: N-M] is EDIT SYNTAX, not a tool call. "
                "It belongs inside an === EDIT: file.py === block like this:\n"
                "=== EDIT: file.py ===\n"
                "[SEARCH: 45-49]\n"
                "exact code to find\n"
                "[/SEARCH]\n"
                "[REPLACE]\n"
                "new code\n"
                "[/REPLACE]\n"
                "Continue writing your edit blocks now. Do NOT write [SEARCH: N-M] "
                "as a standalone tag expecting a result.]\n\n"
            )
            full_response += result
            current_prompt = current_prompt + "\n\nASSISTANT: " + full_response + correction + "\n\nContinue:"
            result = ""
            full_response = ""
            continue

        # UNRECOGNIZED / DISABLED TOOL CALL (pylint-4551 merger bug): the model
        # wrote a tool tag we did NOT extract — e.g. [PURPOSE: file] when this
        # call has no purpose_map, so purpose_tags is gated to [] (~line 3519).
        # has_tags is then False even though the model clearly intended to
        # investigate and continue (it emitted [STOP] expecting a result next
        # round). The "no tags → treat as final" break below would take that
        # mid-investigation deferral as the FINAL answer and silently quit —
        # exactly how the merger died after one round with a "let me read
        # inspector.py first" non-plan. Instead, tell the model the tool isn't
        # available and keep the loop alive so it produces its answer from what it
        # already has. Bounded so a model re-requesting a dead tool can't spin.
        if (not has_tags and not _gate_keep_alive
                and has_tool_tags(result) and _dead_tool_nudges < 2
                and (has_pending_edits is None
                     or not has_pending_edits(full_response + result))):
            _dead_tool_nudges += 1
            _dbg(f"unrecognized/disabled tool call — rescue nudge "
                 f"{_dead_tool_nudges}/2 (not treating as final)")
            warn(f"  [{model.split('/')[-1]}] round {round_num}: requested tool not "
                 f"available in this step — asking it to proceed with what it has "
                 f"(nudge {_dead_tool_nudges}/2)")
            full_response += result
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response
                + "\n\n[SYSTEM NOTE: The tool you requested is not available in "
                "this step, so it returned nothing. Do NOT request files or tools "
                "again — proceed with the information you already have and write "
                "your complete answer now.]\n\nContinue:")
            result = ""
            full_response = ""
            continue

        if has_tags:
            # Trim result to end at the last tag — anything the model
            # wrote after the last ] is speculation without results.
            last_bracket = result.rfind(']')
            if last_bracket >= 0:
                result = result[:last_bracket + 1]

        full_response += result

        # A [STOP] that ends an edit section (no tool tags, but fresh edits to
        # apply) must keep the round alive so on_stop applies the edits and the
        # diff/verify block is shown — otherwise the model never sees the diff
        # the prompt promised it. Scoped to roles that pass has_pending_edits
        # (coder/self-check/reviewer); other roles keep the old behavior.
        _stop_with_pending = (
            has_stop and has_pending_edits is not None
            and has_pending_edits(full_response)
        )
        # EDIT-WITHOUT-SIGNAL (bug-hunt #1): the model wrote a complete edit
        # block but emitted NO [STOP]/[DONE]/tool call. Previously the loop
        # broke here and the edit either applied silently at loop-end (no diff
        # shown) or vanished — and the model, seeing only "(no tool results)",
        # looped for rounds, confused. Treat fresh-edits-without-a-signal as an
        # implicit [STOP]: keep the round alive so on_stop applies them and the
        # model SEES the verify diff. Dedup makes this safe — a re-written
        # identical edit yields no NEW pending edits, so the loop breaks cleanly.
        _pending_no_signal = (
            not has_stop and not _done_signaled and not _gate_keep_alive
            and has_pending_edits is not None
            and has_pending_edits(full_response)
        )
        if (not has_tags and not _gate_keep_alive
                and not _stop_with_pending and not _pending_no_signal):
            _dbg("EXIT: no tags + nothing pending → break (treat as final)")
            break  # No tool requests and nothing to apply — done
        # When _gate_keep_alive is set, a [DONE] carried fresh edits: keep the
        # round alive (even with no tool tags) so on_stop applies them and the
        # diff/verify block is shown before the model re-issues [DONE].

        # ── Apply pending edits BEFORE tool lookups ──────────────────
        # When a coder writes edit blocks then [STOP] + [CODE: file],
        # they want to verify their edits. on_stop applies the edits
        # to the sandbox so [CODE:] reads return the post-edit state.
        # The callback may return a feedback string describing which
        # edits applied vs were skipped — captured here and surfaced
        # in the next-round prompt so the model sees explicit results.
        if on_stop is not None:
            try:
                feedback = on_stop(full_response)
                if feedback and isinstance(feedback, str) and feedback.strip():
                    _last_edit_feedback = feedback.strip()
                    if _pending_no_signal:
                        # bug-hunt #1: teach the model the signal it skipped,
                        # while still showing it the diff (already applied).
                        _last_edit_feedback = (
                            "⚠ You wrote an edit block with NO [STOP][CONFIRM_STOP] "
                            "signal. Edits are applied and the verify-diff is shown "
                            "only when you end the turn with [STOP][CONFIRM_STOP] (or "
                            "[DONE][CONFIRM_DONE] when finished). I applied them for "
                            "you this time — verify the diff below — but ALWAYS emit "
                            "the signal yourself; signals written inside [think] are "
                            "INERT and do nothing.\n\n" + _last_edit_feedback
                        )
                    # Parse the feedback to update per-file attempt history.
                    # Lines starting with "✓" = success on that file; "✗" = miss.
                    # We use this for stall detection below.
                    #
                    # Feedback line shapes we must support:
                    #   ✓ CREATED  path/to/file.py (N lines written)
                    #   ✓ MODIFIED path/to/file.py (A → B lines)
                    #   ✗ REJECTED edit on path/to/file.py: SEARCH anchor …
                    #   ✗ REJECTED SEARCH starting with 'def foo' had 3 matches…
                    #   ↺ REVERTED path/to/file.py to prior snapshot
                    # The first token after the marker is a verb; the filepath
                    # is the first token-that-LOOKS-like-a-path on the line.
                    _path_like = re.compile(
                        r'(?<![\w/.])([\w./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue|lua|sh))(?![\w/.])'
                    )
                    for ln in _last_edit_feedback.splitlines():
                        s = ln.strip()
                        if not s or s[0] not in '✓✗↺':
                            continue
                        m = _path_like.search(s)
                        if not m:
                            continue
                        fp = m.group(1).rstrip(':,')
                        applied = s.startswith('✓')
                        _edit_attempts_per_file.setdefault(fp, []).append(applied)
                else:
                    _last_edit_feedback = None
            except Exception as e:
                warn(f"  on_stop callback error: {e}")
                _last_edit_feedback = None

        # ── Edit-flailing detection ──────────────────────────────────
        # If 3 consecutive rounds have attempted edits on the SAME file
        # and ALL failed, the model is flailing — it'll keep writing
        # variations of the same broken SEARCH forever. Surface a
        # strong correction telling it to take a different approach
        # (REPLACE LINES, REVERT then redo, or give up cleanly).
        # Observed: 19-round step-3 loop on domains/prompts.py where
        # the model kept tweaking SEARCH anchors that never matched.
        for fp, history in _edit_attempts_per_file.items():
            if len(history) >= 3 and not any(history[-3:]):
                _last_edit_feedback = (
                    (_last_edit_feedback + "\n\n") if _last_edit_feedback else ""
                ) + (
                    f"🛑 EDIT-FLAILING DETECTED on {fp}\n"
                    f"  Edits on {fp} failed the last 3 rounds. STOP repeating the "
                    f"same anchors. CHOOSE ONE recovery path:\n"
                    f"    1. [VIEW: {fp} A B] the target region, then copy the lines "
                    f"you anchor on VERBATIM with their CURRENT numbers + INDENT "
                    f"counts. Anchor on two DISTINCTIVE code lines — never a blank, "
                    f"never a line that repeats in the file.\n"
                    f"    2. Use [REPLACE LINES N-M] (inside === EDIT: {fp} ===) with "
                    f"numbers from your latest read. Close with [/REPLACE] (NOT "
                    f"[/REPLACE LINES]); body = ONLY the new lines in INDENT|code "
                    f"form, no context lines:\n"
                    f"         === EDIT: {fp} ===\n"
                    f"         [REPLACE LINES N-M]\n"
                    f"         8|new code here\n"
                    f"         [/REPLACE]\n"
                    f"         === END EDIT ===\n"
                    f"         [STOP][CONFIRM_STOP]\n"
                    f"    3. If a prior edit landed wrong, [REVERT FILE: {fp}] then "
                    f"redo from clean.\n"
                    f"    4. If truly stuck: [DONE][CONFIRM_DONE] with what you have."
                )
                # Reset the history so we don't fire again next round on
                # the same trigger — the model gets one strong nudge per
                # 3-failure streak, not repeated nags.
                _edit_attempts_per_file[fp] = []

        # Run requested lookups — check cache first
        round_output = ""  # results from THIS round only (for logging)

        # Filter out tags this model already ran in a previous round.
        # NOTE: we only check local_research (this model's own history), NOT
        # the shared research_cache. Results from other parallel models are NOT
        # "cached" from this model's perspective — it must request them itself.
        # This prevents false stall detection and stops the model from thinking
        # it has seen content it never actually requested.
        def _cached_or_run(tag_type: str, tags: list[str]) -> tuple[list[str], str]:
            """Returns (new_tags_to_run, cached_output_for_already_run_tags).

            CODE and KEEP normally re-run because the file may have been
            edited mid-conversation (coder paths). When `cache_file_reads`
            is True (planner / reviewer paths that don't write edits) we
            also cache CODE/KEEP — re-asking for the same path returns the
            stored content with a notice instead of burning a tool round.
            This was the #1 source of planners exhausting their budget:
            kimi-k2.6 re-issued [CODE: separable.py] in R1, R2, R7 because
            the runtime kept obliging.
            """
            if tag_type in ("CODE", "KEEP") and not cache_file_reads:
                return tags, ""
            cached_out = ""
            new_tags = []
            for tag in tags:
                clean_tag, label = _strip_label(tag)
                key = _norm_key(tag_type, clean_tag)
                if key in local_research:
                    # This model ran it in an earlier round — show cached result
                    rn = _manifest[key]["round"] if key in _manifest else "?"
                    cached_out += (
                        f"\n[CACHED — you already ran {tag_type}: {clean_tag} "
                        f"in round {rn}. Result is unchanged — do not re-request.]\n"
                        + local_research[key]
                    )
                    persistent_lookups[key] = local_research[key]
                    if label:
                        _label_to_keys.setdefault(label, []).append(key)
                else:
                    new_tags.append(tag)
            return new_tags, cached_out

        def _store(tag_type: str, tag: str, result: str):
            """Store a result in local_research, shared cache, and persistent lookups."""
            clean_tag, label = _strip_label(tag)
            key = _norm_key(tag_type, clean_tag)
            local_research[key] = result
            persistent_lookups[key] = result
            if research_cache is not None:
                research_cache[key] = result
            if label:
                _label_to_keys.setdefault(label, []).append(key)
            _manifest[key] = {"round": round_num, "tag_type": tag_type, "arg": clean_tag}

            # Flag empty / no-match results so re-requests get a strong
            # "NOT IN CODEBASE" header instead of the same empty body.
            # Only meaningful for symbol-style lookups — file reads can
            # legitimately return empty bodies for empty files.
            if tag_type in ("SEARCH", "REFS", "LSP"):
                lower = (result or "").lower()
                if (
                    not result.strip()
                    or "no matches" in lower
                    or "no matches found" in lower
                    or "no references found" in lower
                    or "no results" in lower
                ):
                    _not_found_terms.add(key)
                else:
                    # Defensive: if the same key was previously empty and
                    # now isn't (file content changed, fixed typo, etc.),
                    # clear the flag so the cached annotation reflects truth.
                    _not_found_terms.discard(key)

        async def _locked_lookup(tag_type: str, tag: str, run_fn) -> str:
            """Run a lookup with a per-key lock to prevent duplicate concurrent executions.
            For non-CODE/KEEP types: if the result is already in the shared cache
            (from another parallel model), return it directly and record it as seen
            by this model — no re-execution, no wasted API call."""
            clean_tag, label = _strip_label(tag)
            key = _norm_key(tag_type, clean_tag)
            if key not in _inflight_locks:
                _inflight_locks[key] = asyncio.Lock()
            lock = _inflight_locks[key]

            async with lock:
                if tag_type not in ("CODE", "KEEP"):
                    if research_cache is not None and key in research_cache:
                        cached = research_cache[key]
                        # Record as seen by this model (local_research + manifest)
                        local_research[key] = cached
                        persistent_lookups[key] = cached
                        if label:
                            _label_to_keys.setdefault(label, []).append(key)
                        _manifest[key] = {"round": round_num,
                                          "tag_type": tag_type, "arg": clean_tag}
                        # Annotate the result so the model can tell a
                        # parallel-model cache hit apart from a freshly-run
                        # lookup. Without this the model can't distinguish
                        # results it asked for in this round from results
                        # another planner already produced. Cosmetic but
                        # affects trust in the cache.
                        return (
                            f"\n[CACHED — {tag_type}: {clean_tag} was already "
                            f"looked up by a parallel model. Result follows.]\n"
                            + cached
                        )
                try:
                    result = run_fn(clean_tag)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as e:
                    # A tool executor blowing up must degrade to a VISIBLE error
                    # in the round output, never abort the whole run (stability
                    # audit #5). The model sees it and can try another approach.
                    # Not stored, so a transient failure can be retried next round
                    # (the stall detector guards against endless repeats).
                    warn(f"  [{tag_type}: {clean_tag[:60]}] tool error: {str(e)[:120]}")
                    return (f"\n✗ {tag_type}: {clean_tag} failed — {str(e)[:160]}. "
                            f"Try a different lookup, a narrower input, or proceed "
                            f"with what you have.\n")
                _store(tag_type, tag, result)
                return result

        # ── Handle [DISCARD: #label] — remove labeled results from context ──
        if discard_tags:
            for label in discard_tags:
                if label in _label_to_keys:
                    for key in _label_to_keys[label]:
                        persistent_lookups.pop(key, None)
                        local_research.pop(key, None)
                    status(f"  Discarded #{label} ({len(_label_to_keys[label])} results)")
                    del _label_to_keys[label]
                else:
                    warn(f"  [DISCARD: #{label}] — label not found, ignoring")

        total = (
            len(code_tags) + len(web_tags) + len(detail_tags) + len(file_tags)
            + len(refs_tags) + len(purpose_tags) + len(semantic_tags)
            + len(lsp_tags) + len(knowledge_tags) + len(keep_tags)
            + len(view_tags) + len(dependency_tags)
        )
        mode = _describe_tool_mode(result)
        tags_desc = _tag_summary(
            code_tags, web_tags, detail_tags, file_tags, refs_tags,
            purpose_tags, semantic_tags, lsp_tags, knowledge_tags, keep_tags,
            view_tags, dependency_tags,
            research_cache, persistent_lookups,
        )
        status(f"  [{model.split('/')[-1]}] tool round {round_num}/{max_rounds}: "
               f"{total} lookup(s) — {mode}")
        status(f"    tags: {tags_desc}")

        # ── Stall detection ────────────────────────────────────────────
        # Build a stable key set for THIS round's tag requests so we can
        # tell whether the model is making progress or just re-requesting
        # already-cached lookups. If two consecutive rounds request only
        # tools whose results are already in persistent_lookups, the model
        # is spinning — break out and let it commit.
        def _norm_tag_key(tag_type: str, tag_arg: str) -> str:
            clean, _ = _strip_label(tag_arg)
            return _norm_key(tag_type, clean)

        round_keys: set[str] = set()
        for t in code_tags:    round_keys.add(_norm_tag_key("SEARCH", t))
        for t in web_tags:     round_keys.add(_norm_tag_key("WEBSEARCH", t))
        for t in detail_tags:  round_keys.add(_norm_tag_key("DETAIL", t))
        for t in file_tags:    round_keys.add(_norm_tag_key("CODE", t))
        for t in refs_tags:    round_keys.add(_norm_tag_key("REFS", t))
        for t in purpose_tags: round_keys.add(_norm_tag_key("PURPOSE", t))
        for t in lsp_tags:     round_keys.add(_norm_tag_key("LSP", t))
        for t in knowledge_tags: round_keys.add(_norm_tag_key("KNOWLEDGE", t))
        for t in keep_tags:    round_keys.add(_norm_tag_key("KEEP", t))
        for t in view_tags:    round_keys.add(_norm_tag_key("VIEW", t))
        for t in dependency_tags: round_keys.add(_norm_tag_key("DEPENDENCY", t))
        for t in run_tags:     round_keys.add(_norm_tag_key("RUN", t))

        # A round is "stalled" if every key was already run by this model.
        # We do NOT check the shared research_cache — results from other parallel
        # models are new to this model and must not trigger false stall detection.
        # CODE/KEEP/VIEW exception: file content can change, so a single re-read
        # is legitimate. But ANY repeat is always a loop signal (threshold 1).
        def _is_cached(k: str) -> bool:
            tt = k.split(':', 1)[0]
            if tt in ('CODE', 'KEEP', 'VIEW'):
                # Any repeat is a loop signal. The previous threshold of >=2
                # let the model re-issue the same CODE/KEEP twice silently
                # before stall detection fired — by which point the model
                # had already wasted 2 rounds reading content it already had.
                return _reread_count.get(k, 0) >= 1
            return k in local_research

        # Update re-read counters BEFORE stall detection. We now track
        # ALL tag types, not just CODE/KEEP/VIEW. Previously the counter
        # only fired for those three, so when the model re-issued
        # `[SEARCH: "deepcode"]` or `[REFS: foo]` round after round, the
        # manifest never showed the "⛔ RE-READ ×N — DO NOT request this
        # again" warning. Observed 20260512_171644 kimi-k2.6 R4/R5: same
        # SEARCH set twice, manifest showed no RE-READ marker, model had
        # no in-prompt signal it was looping. Tracking every tag fixes that.
        for k in round_keys:
            if k in _manifest:
                _reread_count[k] = _reread_count.get(k, 0) + 1

        # ── Stall detection ─────────────────────────────────────────────
        # Three triggers, each independently increments `_stall_rounds`:
        #   (a) ALL keys this round are cached / repeated.
        #   (b) Exact same key set as last round.
        #   (c) ≥50% of this round's keys are repeats. Pure "all_cached"
        #       was easy to evade — the model just adds one new tag per
        #       round to look like it's making progress, while really
        #       spinning on the same 5 repeats. The ratio check catches
        #       that pattern. CODE/KEEP repeats count toward the ratio
        #       (via _reread_count); non-CODE/KEEP repeats count via the
        #       local_research membership check.
        if round_keys:
            n_repeats = sum(
                1 for k in round_keys
                if (
                    k.split(':', 1)[0] in ('CODE', 'KEEP', 'VIEW')
                    and _reread_count.get(k, 0) >= 1
                ) or (
                    k.split(':', 1)[0] not in ('CODE', 'KEEP', 'VIEW')
                    and k in local_research
                )
            )
            repeat_ratio = n_repeats / len(round_keys)
        else:
            repeat_ratio = 0.0

        if round_keys and all(_is_cached(k) for k in round_keys):
            _stall_rounds += 1
        elif round_keys == _last_round_keys and round_keys:
            _stall_rounds += 1
        elif round_keys and repeat_ratio >= 0.5:
            _stall_rounds += 1
        else:
            _stall_rounds = 0
        _last_round_keys = round_keys

        # ── Per-round diagnostic log to workflow.log ─────────────────────
        # One line per (model, round). Captures what the model wrote,
        # which tags fired, how many were dropped (rejected by the
        # detector for placement/shape reasons), the stall streak so far,
        # and the manifest/persistent-lookups footprint. Reading this
        # log in sequence lets you tell at a glance whether a loop is
        # "same tags re-issued", "all tags being dropped", "stall guard
        # not catching the loop", or "response truncated/empty".
        try:
            from core import workflow_log
            short_model = model.split("/")[-1]
            _tags_summary = ", ".join(sorted(round_keys)) or "(none)"
            if len(_tags_summary) > 200:
                _tags_summary = _tags_summary[:197] + "..."
            _dropped_summary = (
                ", ".join(f"{tt}:{a[:30]}" for tt, a in dropped_tags[:5])
                if dropped_tags else "0"
            )
            workflow_log.phase_event(
                f"R{round_num} [{short_model}/{log_label or 'call'}]",
                resp_chars=len(result),
                tags_fired=_tags_summary,
                dropped=_dropped_summary,
                stall=f"{_stall_rounds}/2",
                manifest=len(_manifest),
                lookups_kb=sum(len(v) for v in persistent_lookups.values()) // 1024,
            )
        except Exception:
            pass  # diagnostic-only — never break the model loop on log failures

        if _stall_rounds >= 2:
            warn(
                f"  Stall: round {round_num} repeated already-cached lookups. "
                f"Forcing the model to commit instead of looping."
            )
            # Inject a hard commit instruction and let the loop run ONE more
            # turn so the model can emit its plan/code, then break.
            stall_note = (
                "\n\n══════════════════════════════════════════════════════════════════════\n"
                "🛑 STOP INVESTIGATING — COMMIT NOW\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "You have spent multiple rounds re-requesting the SAME tool results.\n"
                "Investigation is over. Write your final answer NOW using only what\n"
                "you already know. Do NOT use any more tool tags. Do NOT write [STOP].\n"
                "If you are a planner: WRITE THE PLAN.\n"
                "If you are a coder: WRITE THE EDIT BLOCKS, then [DONE].\n"
                "══════════════════════════════════════════════════════════════════════\n"
            )
            current_prompt = current_prompt + stall_note
            # Force one final round with no tool processing
            try:
                final_result = await call_with_retry(
                    model, current_prompt, max_tokens=max_tokens,
                    stop_check=None,  # no early stop — let it write everything
                    log_label=log_label + " (commit)",
                )
                # Strip any remaining signals — BOTH the full two-tag pairs
                # AND the bare halves. Without the bare-half pass, a forced
                # commit can leak `[CONFIRM_DONE]` / `[STOP]` text into the
                # final answer, which downstream regex (e.g. _extract_code_blocks
                # consumed_spans) doesn't expect.
                for _pat in (FORCE_DONE_TAG, DONE_TAG, STOP_TAG, CONTINUE_TAG):
                    final_result = _pat.sub('', final_result)
                for _half in (
                    r'\[CONFIRM_FORCE_DONE\]', r'\[FORCE\s+DONE\]',
                    r'\[CONFIRM_DONE\]', r'\[DONE\]',
                    r'\[CONFIRM_STOP\]', r'\[STOP\]',
                    r'\[CONFIRM_CONTINUE\]', r'\[CONTINUE\]',
                ):
                    final_result = re.sub(_half, '', final_result, flags=re.IGNORECASE)
                final_result = final_result.rstrip()
                full_response += "\n" + final_result
            except Exception as e:
                warn(f"  Forced-commit round failed: {e}")
            break

        # Lookup runners — defined ONCE outside the loop so each closure
        # captures fresh `tag` via its parameter only, never via the loop
        # variable. The previous inline-`async def` form was correct only
        # because we awaited every result before the next iteration; making
        # it explicit removes the foot-gun.
        # v12: roles read in PREFIX mode — `LINENO:INDENT|content` (INDENT = a
        # leading-space COUNT). Reverted from whitespace: the coder kept
        # mis-indenting NEW block bodies (e.g. an `else:` body at the keyword's
        # level) when it had to type real spaces. With the explicit count the
        # model writes `16|raise TypeError(other)` and the runtime expands `16|`
        # into 16 spaces — it only has to get the NUMBER right, not type spaces.
        _display_mode = "prefix"
        def _run_search(tag): return _run_code_searches([tag], _search_root)
        def _run_web(tag):    return _run_web_searches([tag])
        def _run_detail(tag): return _run_detail_lookups([tag], detailed_map, project_root=project_root)
        async def _run_code(tag):
            return await _run_code_reads(
                [tag], project_root, viewed_versions=viewed_versions,
                display_mode=_display_mode,
            )
        def _run_refs(tag):   return _run_refs_searches([tag], _search_root)
        def _run_purpose(tag): return _run_purpose_lookups([tag], purpose_map, project_root)
        async def _run_semantic(tag):
            from tools.code_index import _maps_dir, _load_all_code
            from tools.embeddings import semantic_retrieve
            maps_dir = _maps_dir(project_root)
            _, file_hash = _load_all_code(project_root)
            # Semantic search now indexes the CODE itself (AST chunks) — no
            # purpose map required, so it works map-free.
            return await semantic_retrieve(
                tag, project_root, maps_dir, file_hash, top_n=10
            )
        def _run_lsp(tag):
            return (
                f"=== LSP for '{tag}': folded into REFS / DEPENDENCY ===\n"
                f"  • [REFS: {tag}] — DEFINED / IMPORTED / USED classification, ripgrep-fast\n"
                f"  • [DEPENDENCY: #tag] — type-resolved callers (LSP-precise, follows aliases)\n"
                f"Prefer DEPENDENCY when REFS misses sites due to import aliases or type indirection.\n"
            )
        def _run_knowledge(tag): return _run_knowledge_lookups([tag])

        if code_tags and project_root:
            new_tags, cached = _cached_or_run("SEARCH", code_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("SEARCH", t, _run_search)
                round_output += r

        if web_tags:
            new_tags, cached = _cached_or_run("WEBSEARCH", web_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("WEBSEARCH", t, _run_web)
                round_output += r

        if detail_tags and detailed_map:
            new_tags, cached = _cached_or_run("DETAIL", detail_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("DETAIL", t, _run_detail)
                round_output += r

        if file_tags and project_root:
            new_tags, cached = _cached_or_run("CODE", file_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("CODE", t, _run_code)
                round_output += r

        if refs_tags and project_root:
            new_tags, cached = _cached_or_run("REFS", refs_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("REFS", t, _run_refs)
                round_output += r

        if purpose_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("PURPOSE", purpose_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("PURPOSE", t, _run_purpose)
                round_output += r

        if semantic_tags and project_root:
            new_tags, cached = _cached_or_run("SEMANTIC", semantic_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("SEMANTIC", t, _run_semantic)
                round_output += r

        if lsp_tags and project_root:
            new_tags, cached = _cached_or_run("LSP", lsp_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("LSP", t, _run_lsp)
                round_output += r

        if knowledge_tags:
            new_tags, cached = _cached_or_run("KNOWLEDGE", knowledge_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("KNOWLEDGE", t, _run_knowledge)
                round_output += r

        if dependency_tags and project_root:
            async def _run_dep(tag):
                return await _run_dependency_lookup([tag], project_root)
            new_tags, cached = _cached_or_run("DEPENDENCY", dependency_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("DEPENDENCY", t, _run_dep)
                round_output += r

        # ── RUN handler — diagnostic command in the fail-proof sandbox ──
        # Planner / reviewer only (allow_run). Runs in core.safe_exec: read-only
        # filesystem (no edit/delete possible), no network, no privilege, HOME
        # hidden, ephemeral /tmp, resource-limited. The model runs commands to
        # OBSERVE (run the app, call a function, inspect behavior) — it cannot
        # mutate anything.
        if run_tags and allow_run:
            def _run_command(tag):
                from core.safe_exec import run_sandboxed
                cmd = (tag or "").strip()
                if not cmd:
                    return ("=== RUN — NO COMMAND ===\n"
                            "[RUN:] needs a command to run, e.g. "
                            "[RUN: python -c \"import pkg.m as m; print(m.f(3))\"]. "
                            "A blank command does nothing.\n")
                base = run_cwd or project_root or os.getcwd()
                res = run_sandboxed(cmd, cwd=base, project_root=base, timeout=60)
                # 1. BLOCKED by policy (or sandbox unavailable / fail-safe deny)
                if res["blocked"]:
                    if res.get("sandbox") == "none":
                        return (f"=== RUN — UNAVAILABLE: {cmd} ===\n{res['reason']}\n")
                    return (f"=== RUN — ✗ BLOCKED: {cmd} ===\n{res['reason']}\n"
                            f"Nothing ran. This is a read-only, no-network, "
                            f"no-privilege diagnostic sandbox — run a command "
                            f"that OBSERVES (prints/inspects/runs the code), "
                            f"not one that edits, deletes, installs, or reaches "
                            f"the network.\n")
                code = res["exit_code"]
                out = res["output"] or "(the command produced no output on stdout or stderr)"
                # 2. TIMED OUT (hang / fork-bomb guard)
                if res["timed_out"]:
                    return (f"=== RUN — ⏱ TIMED OUT (>60s, killed): {cmd} ===\n"
                            f"The command ran too long and was killed. Make it "
                            f"terminate quickly (smaller input, add a limit, or "
                            f"avoid waiting on something).\n{out}\n")
                # 3. RAN OK (exit 0)
                if code == 0:
                    return (f"=== RUN — ✓ ok (exit 0): {cmd} ===  "
                            f"[read-only/no-net sandbox]\n{out}\n")
                # 4. RAN but FAILED (non-zero exit). Distinguish a genuine code
                #    failure from a network attempt the sandbox blocked — the
                #    latter shows up as a DNS/connection error and is EXPECTED,
                #    not a bug, so don't send the model chasing its own code.
                _net = any(s in out for s in (
                    "name resolution", "Name or service not known",
                    "Temporary failure in name resolution", "Connection refused",
                    "Network is unreachable", "ECONNREFUSED", "ENETUNREACH",
                    "getaddrinfo", "nodename nor servname", "No route to host",
                ))
                if _net:
                    return (f"=== RUN — ✗ FAILED (exit {code}): {cmd} ===  "
                            f"[read-only/no-net sandbox]\n"
                            f"The error below is a NETWORK error — this sandbox has "
                            f"NO network access by design, so it's expected, not a "
                            f"bug in the code. Verify with a command that doesn't "
                            f"need the network.\n{out}\n")
                return (f"=== RUN — ✗ FAILED (exit {code}): {cmd} ===  "
                        f"[read-only/no-net sandbox]\n"
                        f"The command RAN but exited non-zero — read the output "
                        f"below (traceback / error message) to see WHY. If it "
                        f"mentions network/DNS/connection, that's the sandbox (no "
                        f"network); otherwise it's the code's own failure.\n{out}\n")
            new_tags, cached = _cached_or_run("RUN", run_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("RUN", t, _run_command)
                round_output += r

        # ── KEEP handler — replaces CODE entries in persistent_lookups ──
        if keep_tags and project_root:
            def _on_keep_seen(canonical_key: str, raw_arg: str) -> None:
                # Register the KEEP in the manifest + local_research so
                # the loop detector and cache annotations work for KEEP
                # the same way they do for CODE. KEEP content itself
                # already lives in persistent_lookups (replacing the
                # corresponding CODE entry).
                local_research[canonical_key] = persistent_lookups.get(
                    canonical_key, ""
                )
                _manifest[canonical_key] = {
                    "round": round_num,
                    "tag_type": "KEEP",
                    "arg": raw_arg.strip(),
                }
            keep_result = await _run_keep(
                keep_tags, project_root,
                persistent_lookups, research_cache,
                viewed_versions=viewed_versions,
                on_keep_seen=_on_keep_seen,
                display_mode=_display_mode,
            )
            round_output += keep_result

        # ── VIEW handler — read a slice of a large file by line number ──
        if view_tags and project_root:
            def _on_view_seen(canonical_key: str, raw_arg: str) -> None:
                # Mirror _on_keep_seen so loop detection + annotation work
                # the same way for VIEW as for CODE/KEEP.
                local_research[canonical_key] = persistent_lookups.get(
                    canonical_key, ""
                )
                _manifest[canonical_key] = {
                    "round": round_num,
                    "tag_type": "VIEW",
                    "arg": raw_arg.strip(),
                }
            view_result = await _run_view(
                view_tags, project_root,
                persistent_lookups,
                model_id=model,
                research_cache=research_cache,
                viewed_versions=viewed_versions,
                on_view_seen=_on_view_seen,
                display_mode=_display_mode,
            )
            round_output += view_result

        # `past_thinking` is built BELOW after the budget logic decides
        # which results to include. Don't render `round_history` here —
        # we now interleave thinking + results round by round.

        # Rebuild search_output from ALL persistent lookups.
        # This is the key mechanism: if KEEP replaced a CODE entry,
        # the full file is gone — only the kept ranges remain.
        # CONTEXT BUDGET — if the cumulative tool-results would exceed
        # ~80k chars (≈ 20k tokens), drop the LEAST-recent entries to
        # stay within the model context. The KEEP mechanism normally
        # keeps things small, but a coder reading 8 large files in one
        # session can still pile on. We always keep the entries touched
        # THIS round so the model never loses the result of what it
        # just asked for.
        def _annotate_entry(k: str, v: str) -> str:
            # Prefix each tool result with a header that tells the model
            # exactly which tag produced it and which round it was first
            # fired in. Without this, the model can't tell a fresh lookup
            # from a stale one — every round it just sees the same dump of
            # `persistent_lookups.values()` and re-issues tools that have
            # already run. The header is short; the result body follows
            # verbatim so existing line-count parsing still works.
            info = _manifest.get(k)
            if not info:
                # Defensive fallback: an entry exists in persistent_lookups
                # but has no manifest. This SHOULDN'T happen — every
                # storage path (`_store`, `_on_keep_seen`, `_on_view_seen`)
                # writes the manifest at the same time. If we see it, log
                # loudly and synthesise a header from the key itself so
                # the model still gets SOMETHING informative instead of a
                # bare result body.
                warn(
                    f"  manifest miss for persistent_lookups key {k!r} — "
                    f"key chain broken; synthesising header"
                )
                _tt_arg = k.split(":", 1)
                if len(_tt_arg) == 2:
                    return f"\n[← {_tt_arg[0]}: {_tt_arg[1]} — (manifest-missing)]\n{v}"
                return v
            tt = info["tag_type"]
            arg = info["arg"]
            rn = info["round"]
            # NOT-FOUND header: when a symbol-style lookup returned zero
            # matches, flag the term loudly so the model stops looking for
            # it across rounds. Observed in practice: kimi-k2.6 searched
            # `handle_deepcode` 3 rounds in a row because the stale code
            # map listed it but the actual file doesn't define it.
            if k in _not_found_terms:
                return (
                    f"\n[← {tt}: {arg} — from R{rn} — ⛔ NOT IN CODEBASE]\n"
                    f"This term returned ZERO matches. It does NOT exist "
                    f"in the project. STOP searching for it. If you saw it "
                    f"named in the code map / DETAIL output, the map is "
                    f"stale — trust the SEARCH/REFS result, not the map.\n"
                    f"{v}"
                )
            stale = "" if rn == round_num else " — DO NOT re-request"
            return f"\n[← {tt}: {arg} — from R{rn}{stale}]\n{v}"

        TOOL_OUTPUT_BUDGET = 80_000  # chars
        this_round_keys = round_keys  # set built earlier this round
        all_entries = list(persistent_lookups.items())
        total_chars = sum(len(v) for _, v in all_entries)
        if total_chars > TOOL_OUTPUT_BUDGET and len(all_entries) > 1:
            # ── Recency scoring ─────────────────────────────────────────
            # Bump every entry the model has been REFERENCING in its
            # recent prose (by tag argument substring match against
            # _round_texts[-1]). Without this, a CODE result the model
            # keeps citing across 6 rounds loses to fresh lookups even
            # though it's actively used. We treat "named in last round"
            # as equivalent to "just looked up this round".
            recent_text = _round_texts[-1] if _round_texts else ""
            recent_text_lower = recent_text.lower()
            def _entry_score(k: str) -> tuple[int, int]:
                # This-round entries are SACRED — they were JUST asked for.
                if k in this_round_keys:
                    return (10_000_000, round_num)
                info = _manifest.get(k)
                base_round = info["round"] if info else 0
                # Boost if the model mentioned the arg by name in last round
                bump = 0
                if info:
                    arg = (info.get("arg") or "").strip().lower()
                    if arg and len(arg) >= 3 and arg in recent_text_lower:
                        bump = round_num  # "as if it were touched this round"
                return (base_round + bump, base_round)
            all_entries.sort(key=lambda kv: _entry_score(kv[0]), reverse=True)
            kept_entries: list[tuple[str, str]] = []
            dropped_entries: list[tuple[str, str]] = []
            running = 0
            for k, v in all_entries:
                # HARD CAP — every entry (including this-round ones) is
                # subject to the budget. Previously this-round entries were
                # exempted, which let a single round of 7 large KEEPs blow
                # the model's input-token limit and return HTTP 400 "0
                # output tokens." The model can ALWAYS re-issue a dropped
                # tag if needed; an API failure ends the whole pipeline.
                if running + len(v) > TOOL_OUTPUT_BUDGET:
                    dropped_entries.append((k, v))
                    continue
                kept_entries.append((k, v))
                running += len(v)
            if dropped_entries:
                warn(
                    f"  [{model.split('/')[-1]}] round {round_num}: "
                    f"tool-results over {TOOL_OUTPUT_BUDGET:,}-char budget — "
                    f"dropped {len(dropped_entries)} lookup(s) from prompt"
                )
                # Surface the drop to the model so it stops re-issuing the
                # dropped tags blindly. The previous behaviour `warn()`'d to
                # stderr only — the model never saw it, so it kept
                # requesting the same KEEPs round after round, never knowing
                # the runtime had silently stripped them.
                dropped_summary_lines = []
                for k, v in dropped_entries:
                    info = _manifest.get(k)
                    if info:
                        dropped_summary_lines.append(
                            f"  ⛔ [{info['tag_type']}: {info['arg']}] "
                            f"({len(v):,} chars) — TOO LARGE to include"
                        )
                    else:
                        dropped_summary_lines.append(
                            f"  ⛔ {k} ({len(v):,} chars) — TOO LARGE to include"
                        )
                _budget_drop_block = (
                    "══════════════════════════════════════════════════════════════════════\n"
                    "BUDGET OVERFLOW — these results were DROPPED from your context\n"
                    "══════════════════════════════════════════════════════════════════════\n"
                    "Your cumulative tool results exceed the runtime's context cap.\n"
                    "The following lookups RAN but their content is NOT in TOOL\n"
                    "RESULTS below. You can SEE them in the manifest but you CANNOT\n"
                    "read their content this round.\n\n"
                    + "\n".join(dropped_summary_lines) + "\n\n"
                    "FIX in your next round:\n"
                    "  • Use [DISCARD: #label] to free results you no longer need.\n"
                    "  • Request NARROWER KEEPs (≤ 80 lines per range).\n"
                    "  • Don't request all the dropped tags again at once — pick\n"
                    "    one or two, do your work, then move on.\n"
                    "══════════════════════════════════════════════════════════════════════\n\n"
                )
            else:
                _budget_drop_block = ""
            _kept_keys = {k for k, _ in kept_entries}
        else:
            _kept_keys = set(persistent_lookups.keys())
            _budget_drop_block = ""

        # ── Build [YOUR PAST THINKING] — round-by-round interleaved view ──
        # Each round = "What you thought" (model's prose + tool calls) THEN
        # "What your tools returned" (results that arrived in that round).
        # Reads as a chronological narrative: you thought X, the tools
        # answered Y, then you thought Z, the tools answered W, ...
        #
        # Replaces the previous split between [YOUR TOOL RESULTS] (all
        # results lumped together) and [YOUR PRIOR TURNS] (all prose
        # lumped together) — those framings hid the temporal connection
        # between what the model asked and what came back.
        #
        # Entries dropped for budget (k not in _kept_keys) DON'T appear
        # here. The model still sees what it called in its own prose, and
        # the [BUDGET OVERFLOW] block above (if any) tells it those
        # specific results were stripped.
        _past_thinking_parts: list[str] = []
        if _round_texts:
            # Bucket manifest entries by the round they were last
            # registered (KEEP's _on_keep_seen updates the round to the
            # KEEP round, so a CODE→KEEP shows under the KEEP round which
            # is correct — that's the current state).
            _by_round: dict[int, list[tuple[str, dict]]] = {}
            for _k, _info in _manifest.items():
                if _k not in _kept_keys:
                    continue
                _by_round.setdefault(_info["round"], []).append((_k, _info))

            # Bound past-thinking growth: keep the NEWEST rounds within a token
            # budget vs the model's context window, so a long loop never grows
            # the prompt into a silent HTTP-400 context overflow. The newest
            # rounds (latest tool results + ✗ reject feedback) are always kept;
            # older rounds are elided (their tool results still live in [YOUR
            # TOOL INDEX]). Mechanical — count_tokens only, no LLM, no latency.
            # (stability audit #4.)
            from core.tokens import count_tokens as _ctok
            from config import MODELS as _MODELS
            _win = _MODELS.get(model, {}).get("window", 128_000)
            _pt_cap = max(8_000, _win - 30_000 - 20_000 - (max_tokens // 4))
            _keep_from = _past_thinking_keep_from(_round_texts, _pt_cap, _ctok)
            if _keep_from > 0:
                _past_thinking_parts.append(
                    f"────── (rounds 1–{_keep_from} elided to stay within the "
                    f"context budget — their tool results remain in [YOUR TOOL "
                    f"INDEX] above) ──────"
                )

            for _i, _text in enumerate(_round_texts):
                if _i < _keep_from:
                    continue
                _rn = _i + 1
                _past_thinking_parts.append(
                    f"────── ROUND {_rn} — your thinking ──────"
                )
                _past_thinking_parts.append(_text.rstrip())
                _round_entries = _by_round.get(_rn, [])
                if _round_entries:
                    _past_thinking_parts.append("")
                    _past_thinking_parts.append(
                        f"────── ROUND {_rn} — your tool result ──────"
                    )
                    # Targeted hallucination notice — if the round's
                    # stream aborted because the model started faking a
                    # tool result, tell the model right here, attached to
                    # this round's results, that what they imagined is
                    # NOT what's below. Avoids both the model trusting
                    # its own fabrication AND repeating it next round.
                    if _rn in _hallucinated_rounds:
                        _past_thinking_parts.append(
                            "⚠ In your previous response you started "
                            "writing a fake `────── ROUND N — your tool "
                            "result ──────` block and inventing content "
                            "after it. The runtime aborted that stream. "
                            "What you imagined is GONE. The REAL results "
                            "from the tools you actually called are listed "
                            "below — quote and reason from these, not "
                            "from whatever you started to fabricate."
                        )
                        _past_thinking_parts.append("")
                    for _k, _info in _round_entries:
                        _tt = _info["tag_type"]
                        _arg = _info["arg"]
                        _content = persistent_lookups.get(_k, "").rstrip()
                        # Mark NOT-IN-CODEBASE results loudly inline.
                        _flag = (
                            "  ⛔ NO MATCHES IN CODEBASE"
                            if _k in _not_found_terms else ""
                        )
                        _past_thinking_parts.append(
                            f"\n[{_tt}: {_arg}]{_flag}\n{_content}"
                        )
                else:
                    if _rn in _hallucinated_rounds:
                        _past_thinking_parts.append("")
                        _past_thinking_parts.append(
                            f"────── ROUND {_rn} — your tool result ──────"
                        )
                        _past_thinking_parts.append(
                            "⚠ Your stream aborted because you began "
                            "fabricating a fake tool-result block. No "
                            "real tools fired in this round (or all "
                            "were dropped). Don't reason from your "
                            "imagined results — make NEW tool calls if "
                            "you still need the info."
                        )
                    else:
                        _past_thinking_parts.append(
                            "\n(no tool results from this round — either "
                            "no tools fired, or all results were dropped "
                            "for budget; see [BUDGET OVERFLOW] above if "
                            "shown)"
                        )
                _past_thinking_parts.append("")  # blank line between rounds

        past_thinking = "\n".join(_past_thinking_parts).rstrip()
        if not past_thinking:
            past_thinking = (
                "(This is round 1 — no past thinking yet. Begin by orienting "
                "to the [USER REQUEST] above and either calling tools or "
                "writing your plan.)"
            )

        # ── Build [YOUR PLAN] section ───────────────────────────────────
        # Only shown once a plan exists. Line-numbered so the model can
        # edit specific lines via PLAN_EDIT. The header tells the model
        # this is HIS draft — JARVIS preserves it; he refines it.
        if current_plan:
            _plan_numbered = _render_plan_with_line_numbers(current_plan)
            _line_count = current_plan.count('\n') + 1
            _op_log_str = ""
            if _plan_op_log:
                _op_log_str = (
                    "\n\nPlan operations applied this round:\n"
                    + "\n".join(f"  • {l}" for l in _plan_op_log)
                )
            plan_section = (
                "\n══════════════════════════════════════════════════════════════════════\n"
                f"[YOUR PLAN] — your draft (v{plan_version}, {_line_count} lines)\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "Your plan-in-progress, persisted across rounds. Line numbers shown\n"
                "for editing. Refine it with === PLAN_EDIT === [REPLACE LINES N-M]…\n"
                "[/REPLACE] or [INSERT AFTER LINE N]…[/INSERT]. Rewrite it from\n"
                "scratch with another === PLAN ===. When complete, signal\n"
                "[PLAN DONE][CONFIRM_PLAN_DONE] to finalize and submit.\n"
                "\n"
                f"{_plan_numbered}"
                f"{_op_log_str}\n"
            )
        else:
            plan_section = ""

        # Build continuation prompt — escalating budget pressure that
        # tells the model to commit a DRAFT plan, not just "wrap up
        # investigation." Models read "commit" as "lock in your approach
        # in prose" rather than "write === PLAN ===" — so the cue is
        # explicit about the action. Past halfway: start the draft.
        # Past 2/3: stop investigating, draft now (refine later via
        # === PLAN_EDIT ===). Past budget: no more tools, period.
        #
        # Observed (20260513_131849 minimax-m2.7): the model wrote 6
        # rounds of "## ORIENT (updated after RN results)" refining its
        # approach in prose, never emitted === PLAN ===, then in R6
        # degenerated into empty `[tool use]…[/tool use]` blocks. The
        # weaker prior wording let "commit" mean anything.
        budget_msg = ""
        rounds_used = round_num
        rounds_left = max_rounds - rounds_used
        # Role-aware budget nudge (bug-hunt #7): an edit-writing role (coder /
        # self-check / reviewer — they pass has_pending_edits) must be told to
        # FINISH ITS EDITS, never to "write a === PLAN ===" (that planner-only
        # text was being shown to stuck coders, derailing them mid-edit).
        _edit_role = has_pending_edits is not None
        if rounds_left <= 0:
            if _edit_role:
                budget_msg = (
                    f"\n⛔ Round {rounds_used}/{max_rounds} — NO ROUNDS LEFT. FINAL "
                    "response: make your remaining edit(s) now and end with "
                    "[STOP][CONFIRM_STOP], or [DONE][CONFIRM_DONE] if the change is "
                    "complete. No more tool calls or investigation."
                )
            else:
                budget_msg = (
                    f"\n⛔ Round {rounds_used}/{max_rounds} — NO ROUNDS LEFT. This is your "
                    "FINAL response. Write your COMPLETE plan inside `=== PLAN === ... "
                    "=== END PLAN ===` THEN `[PLAN DONE][CONFIRM_PLAN_DONE]`. Do NOT "
                    "use any more tool tags."
                )
        elif rounds_used >= max(3, max_rounds * 2 // 3):
            if _edit_role:
                budget_msg = (
                    f"\n⛔ Round {rounds_used}/{max_rounds} — {rounds_left} round(s) left. "
                    "STOP INVESTIGATING. Make your edit(s) THIS round and end with "
                    "[STOP][CONFIRM_STOP] (or [DONE][CONFIRM_DONE] if already done). "
                    "Tool calls only for one SPECIFIC remaining gap."
                )
            else:
                budget_msg = (
                    f"\n⛔ Round {rounds_used}/{max_rounds} — {rounds_left} round(s) left. "
                    "STOP INVESTIGATING. Write your `=== PLAN === ... === END PLAN ===` "
                    "block this round — even if not perfect, you can refine with "
                    "`=== PLAN_EDIT ===` later. Tool calls only for one SPECIFIC "
                    "remaining gap (and only if naming it as one question)."
                )
        elif rounds_used >= max(2, max_rounds // 2):
            if _edit_role:
                budget_msg = (
                    f"\n⚠ Round {rounds_used}/{max_rounds} — past halfway, {rounds_left} "
                    "round(s) left. Make your FIRST edit now (don't keep investigating): "
                    "write the [edit] block and [STOP][CONFIRM_STOP] to see the diff, "
                    "then refine. Landing an edit beats more reading."
                )
            else:
                budget_msg = (
                    f"\n⚠ Round {rounds_used}/{max_rounds} — past halfway, {rounds_left} "
                    "round(s) left. START YOUR DRAFT PLAN NOW: open "
                    "`=== PLAN === ... === END PLAN ===` and write what you have so far, "
                    "even if incomplete. Refine in later rounds with `=== PLAN_EDIT ===` "
                    "or with one more focused tool call. Refining a draft beats "
                    "polishing your approach in prose."
                )

        # ── Build context manifest ────────────────────────────────────
        def _manifest_line(k: str, v: dict) -> str:
            arg = v["arg"]
            tt  = v["tag_type"]
            rn  = v["round"]
            result_text = persistent_lookups.get(k, "")
            lines_hint = ""
            m = re.search(r'\((\d+) lines\)', result_text)
            if m:
                lines_hint = f" — {m.group(1)} lines"
            elif "KEPT" in result_text:
                m2 = re.search(r'KEPT (\d+/\d+ lines)', result_text)
                if m2:
                    lines_hint = f" — {m2.group(1)} kept"
            return f"  [{tt}: {arg}] (R{rn}{lines_hint})"

        manifest_lines = ["[YOUR TOOL INDEX] — every tool call you've fired and what it returned:"]
        if _manifest:
            for k, v in _manifest.items():
                base = _manifest_line(k, v)
                rcount = _reread_count.get(k, 0)
                if rcount >= 1:
                    base += f"  ⛔ RE-READ {rcount}× — DO NOT request this again"
                manifest_lines.append(base)
        else:
            manifest_lines.append("  (no tool results yet)")
        # If the model just re-issued a CODE/KEEP for an identical key this
        # round, surface that as a separate top-level warning — easy to miss
        # buried in the manifest list.
        # bug-hunt #7: re-reading a file you just EDITED is legitimate
        # verification, not a loop — the content genuinely changed. Exclude
        # files that have had edit activity this session from the loop nag.
        _edited_files = set(_edit_attempts_per_file.keys())
        repeat_offenders = [k for k in round_keys
                            if k.split(':', 1)[0] in ('CODE', 'KEEP', 'VIEW')
                            and _reread_count.get(k, 0) >= 1
                            and not any(fp and fp in k for fp in _edited_files)]
        if repeat_offenders:
            manifest_lines.append(
                "🛑 LOOP DETECTED: you just re-requested " +
                ", ".join(f"[{k}]" for k in repeat_offenders) +
                " — and the file has NOT changed since your last read. "
                "STOP re-investigating. Use what you already have and COMMIT."
            )
        manifest_lines.append(
            "⚠ HARD RULES for [YOUR TOOL INDEX]:\n"
            "  • Anything NOT listed above is UNKNOWN to you — do NOT reference,\n"
            "    quote, or reason about content you haven't seen.\n"
            "  • Need a file NOT listed → call the tool to fetch it.\n"
            "  • File IS listed → do NOT re-call. Reason from [YOUR TOOL RESULTS]."
        )
        manifest_str = "\n".join(manifest_lines)

        # ── Build the "dropped tool calls" block ──────────────────────
        # When the model wrote tool tags OUTSIDE [tool use] blocks, the
        # strict masker drops them. We detected those in `dropped_tags`
        # above. Show the model exactly what fired vs. what didn't so it
        # doesn't hallucinate results for the dropped ones. This was the
        # bug behind the "writes 3 [DETAIL:] tags, only 1 fires, model
        # invents results for the other 2" failure.
        if dropped_tags:
            dropped_lines = []
            for tt, arg in dropped_tags:
                dropped_lines.append(f"  ✗ [{tt}: {arg}]")
            dropped_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                "TOOL CALLS THAT DID NOT FIRE — must be inside [tool use]...[/tool use]\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "These tags appeared in your response but were NOT executed,\n"
                "because they were written OUTSIDE a [tool use]...[/tool use]\n"
                "block. The runtime only fires tags that are deliberately\n"
                "wrapped, to prevent accidental execution of tags discussed in\n"
                "prose or in examples.\n\n"
                + "\n".join(dropped_lines) + "\n\n"
                "If you wanted these to fire, wrap EACH tool call in its own\n"
                "[tool use]...[/tool use] block, then signal once at the end:\n"
                "  [tool use]\n"
                "  [DETAIL: Ensemble]\n"
                "  [/tool use]\n"
                "  [tool use]\n"
                "  [DETAIL: Debate]\n"
                "  [/tool use]\n"
                "  [tool use]\n"
                "  [DETAIL: Synthesizer]\n"
                "  [/tool use]\n"
                "  [STOP]\n"
                "  [CONFIRM_STOP]\n\n"
                "OR put all calls in ONE [tool use] block:\n"
                "  [tool use]\n"
                "  [DETAIL: Ensemble]\n"
                "  [DETAIL: Debate]\n"
                "  [DETAIL: Synthesizer]\n"
                "  [/tool use]\n"
                "  [STOP]\n"
                "  [CONFIRM_STOP]\n\n"
                "⚠ Do NOT assume the dropped calls produced results — they did\n"
                "  not. The TOOL RESULTS section below contains ONLY what fired.\n"
                "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            dropped_block = ""

        # ── Malformed / unknown tool calls ───────────────────────────
        # Tags the model meant as tool calls but that can't fire. Tell it
        # exactly what was wrong and the corrected form, so it never has to
        # guess (and never hallucinates a result for a call that didn't run).
        if bad_tags:
            _ALIAS = {  # common wrong tool names → the right one
                "READ": "CODE", "OPEN": "CODE", "CAT": "CODE", "GET": "CODE",
                "GREP": "SEARCH", "FIND": "SEARCH", "RG": "SEARCH",
                "LS": "SEARCH", "LIST": "SEARCH", "GLOB": "SEARCH",
                "WRITE": "=== FILE:", "EDIT": "=== EDIT:",
            }
            def _bad_hint(tt: str, arg: str) -> str:
                ttu = tt.upper()
                if ttu in _ALIAS:
                    return f"no such tool. Did you mean [{_ALIAS[ttu]} {arg or 'path'}]?"
                if ttu in ("CODE", "VIEW", "KEEP", "PURPOSE", "DETAIL"):
                    if not arg.strip():
                        return f"empty argument — give a path, e.g. [{ttu}: path/to/file.py]."
                    return (f"that doesn't look like a path. Write just the path: "
                            f"[{ttu}: path/to/file.py]"
                            + (" N-M] for a range" if ttu in ("VIEW", "KEEP") else "]") + ".")
                if ttu == "REFS":
                    return "REFS takes ONE symbol (no spaces). Use [REFS: symbol_name], or [SEARCH: free text] for phrases."
                if ttu == "DEPENDENCY":
                    return "expects a #hex tag from a `|appears N (#tag)` annotation, e.g. [DEPENDENCY: #3df] — not a name. Use [REFS: name] to find usages."
                if ttu in ("SEARCH", "SEMANTIC"):
                    return f"empty/invalid query. Use [{ttu}: your query here]."
                return "malformed — check the tool syntax in the TOOL TABLE above."
            _bad_lines = [f"  ✗ [{tt}: {arg}] — {_bad_hint(tt, arg)}" for tt, arg, _ in bad_tags]
            bad_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                "MALFORMED TOOL CALLS — these did NOT run\n"
                "══════════════════════════════════════════════════════════════════════\n"
                + "\n".join(_bad_lines) + "\n\n"
                "Fix the syntax and re-issue inside [tool use]...[/tool use]. Do NOT\n"
                "assume these produced results — they did not.\n"
                "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            bad_block = ""

        # ── Build the "edit application results" block ────────────────
        # The dominant cause of multi-round edit loops is the model
        # writing an edit, then [CODE:] verifying it, but never seeing
        # explicit "edit applied" or "edit skipped" feedback from the
        # runtime. Without that signal the model has to infer success
        # by reading the file diff — which it consistently gets wrong
        # (see the 19-round step-3 loop on domains/prompts.py). Here
        # we inject the on_stop feedback at the TOP of the continuation
        # prompt so it's the first thing the model reads next round.
        if _last_edit_feedback:
            # The feedback may lead with a "DIFF\n<unified diff>" section
            # (LINENO:+ added / LINENO:- removed / LINENO: context) followed
            # by the per-file ✓/✗ summary. Split them so the diff is framed
            # as the thing to VERIFY and the summary reads as status.
            _fb = _last_edit_feedback
            _diff_part, _summary_part = "", _fb
            if _fb.startswith("DIFF\n"):
                _rest = _fb[len("DIFF\n"):]
                # summary lines start with the ✓/✗/↺ markers; the diff is
                # everything before the first such line.
                _m = re.search(r'(?m)^\s*[✓✗↺]', _rest)
                if _m:
                    _diff_part = _rest[:_m.start()].rstrip()
                    _summary_part = _rest[_m.start():].rstrip()
                else:
                    _diff_part, _summary_part = _rest.rstrip(), ""
            _has_reject = ('✗' in _summary_part) or ('REJECTED' in _summary_part)
            _has_applied = ('✓' in _summary_part) or bool(_diff_part)
            # Accurate header (bug-hunt #2): don't claim "EDITS WERE APPLIED"
            # when some were rejected — that contradiction (header says applied,
            # ✗ lines say rejected) made the model believe nothing landed.
            if _has_applied and _has_reject:
                _hdr = ("SOME EDITS APPLIED, SOME REJECTED — "
                        "VERIFY THE DIFF, THEN FIX THE REJECTED ONES")
            elif _has_reject:
                _hdr = "YOUR EDIT WAS REJECTED — IT DID NOT APPLY (see ✗ below)"
            else:
                _hdr = "YOUR EDITS WERE APPLIED — VERIFY THIS DIFF BEFORE YOU FINISH"
            edit_results_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                f"{_hdr}\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "This is THIS ROUND's result — the ACTUAL before/after of files you changed.\n"
                "(Any ✗ REJECTED lines from EARLIER rounds in the history above may have\n"
                " since been fixed — trust THIS block + the file view, not stale history.)\n"
                "  `N:+ …`  a line you ADDED       (N = its new line number)\n"
                "  `N:- …`  a line you REMOVED     (N = its old line number)\n"
                "  `N:  …`  unchanged context\n\n"
                + (f"{_diff_part}\n\n" if _diff_part else "")
                + (f"{_summary_part}\n\n" if _summary_part else "")
                + "▶ VERIFY, line by line, before [DONE]:\n"
                "  1. Every `:-` line — did you MEAN to remove it? An unexpected\n"
                "     `:-` is a deletion you didn't intend (a swallowed line, a\n"
                "     dropped blank). If so, the edit is WRONG — fix it.\n"
                "  2. Every `:+` line — right indentation? right place? complete?\n"
                "  3. Does the change actually do what the step asked?\n\n"
                "▶ THEN decide:\n"
                "  • Diff is correct and the step is satisfied → write\n"
                "      [DONE]\n      [CONFIRM_DONE]\n"
                "    (The edits are already on disk — do NOT re-issue them; a\n"
                "     second copy of the same edit DUPLICATES the lines.)\n"
                "  • Diff is wrong / incomplete → write a CORRECTIVE [edit]\n"
                "    block now (anchor the lines as they are AFTER this diff —\n"
                "    use the new line numbers above). Do not repeat the edit\n"
                "    that already applied.\n"
                + ("  • An edit was REJECTED (see ✗ above) — it did NOT apply;\n"
                   "    the file is unchanged for that one. The reason says what\n"
                   "    to fix. Common: anchor text doesn't match the file (re-read\n"
                   "    with [CODE:] and copy the exact line), or a blank/trivial\n"
                   "    anchor (use a distinctive code line as your top & bottom).\n"
                   if _has_reject else "")
                + "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            edit_results_block = ""

        # ── Unterminated FILE / EDIT block detection ──────────────────
        # Now that the masking regexes (`_EDIT_FILE_SPAN`, `_EDIT_BLOCK_SPAN`)
        # require explicit terminators (`=== END FILE ===`, `[/REPLACE]`,
        # `[/INSERT]`), an unterminated header silently masks every tool
        # tag after it for the rest of the response. We surface that loudly
        # so the model can write the missing terminator next round instead
        # of wondering why its post-header tool calls went nowhere.
        unterminated = _detect_unterminated_blocks(result)
        if unterminated:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"{len(unterminated)} unterminated edit block(s) — "
                f"tool tags after the header were masked"
            )
            unterm_lines = []
            for kind, fp in unterminated:
                if kind == 'FILE':
                    unterm_lines.append(
                        f"  ✗ `=== FILE: {fp}` is missing `=== END FILE ===`"
                    )
                else:
                    unterm_lines.append(
                        f"  ✗ `=== EDIT: {fp}` is missing its close "
                        f"(`[/edit]` then `=== END EDIT ===`)"
                    )
            unterminated_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                "UNTERMINATED EDIT BLOCK(S) — tool tags after these were ignored\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "You wrote a `=== FILE:` or `=== EDIT:` header without its closing\n"
                "terminator. The runtime masks everything after such a header to\n"
                "prevent stray tool tags inside half-written code from firing — but\n"
                "that also masked any TOOL CALLS you wrote after the header. They\n"
                "did NOT fire.\n\n"
                + "\n".join(unterm_lines) + "\n\n"
                "FIX in your next round:\n"
                "  • For `=== FILE:` — close the body with a literal\n"
                "      === END FILE ===\n"
                "    on its own line. Anything between the header and that line\n"
                "    is treated as file content.\n"
                "  • For `=== EDIT:` — close the `[edit]` block with `[/edit]`,\n"
                "    then end the envelope with `=== END EDIT ===`, then your\n"
                "    `[STOP][CONFIRM_STOP]` on its own line. (Legacy SEARCH/REPLACE\n"
                "    and INSERT blocks close with [/REPLACE] / [/INSERT].)\n"
                "    A new `=== EDIT:` header does NOT close the previous one.\n\n"
                "Tool calls you wanted to fire — re-issue them in a NEW round\n"
                "after closing the unterminated block.\n"
                "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            unterminated_block = ""

        # The "preamble already done" reminder is now woven into the
        # final "continue your work" cue at the bottom of the prompt
        # rather than appearing as its own banner. A separate banner
        # made the prompt feel like a fresh round; embedding the
        # reminder in the continue cue makes it feel like a single
        # ongoing thought stream.
        preamble_block = ""

        # The continuation prompt is structured to FEEL like one
        # continuous call to the model, not a series of restarts:
        #
        #   [Full system prompt — kept intact, the model needs it every
        #    round so it doesn't forget its role, rules, and tools.]
        #
        #   [USER REQUEST — also kept intact for the same reason.]
        #
        #   [Edit feedback / manifest — only when relevant, framed as
        #    "since you last wrote, here's what changed in the world."]
        #
        #   [YOUR WORK SO FAR — the model's own prior output streamed
        #    together with no round labels, like a Claude turn that just
        #    happens to have used tools partway through.]
        #
        #   [TOOL RESULTS — cumulative, deduped via persistent_lookups.]
        #
        #   [A single-line "continue from your last sentence" cue.]
        #
        # The system prompt and USER REQUEST stay full every round so the
        # model never forgets its instructions; only the framing around
        # the model's own previous work changes to emphasize continuity.

        # Preamble continuity cue, embedded in the continue line below.
        # If we tracked completed orient/preamble sections, we mention
        # them in passing — no banner, just an inline reminder.
        if _preamble_done:
            preamble_cue = (
                f"  (You already did your initial orient — "
                f"{', '.join(_preamble_done[:3])}"
                f"{'…' if len(_preamble_done) > 3 else ''}. Revise these only "
                f"if new evidence demands; don't restate them.)"
            )
        else:
            preamble_cue = ""

        # NEW ORDER: results + manifest come BEFORE the model's own
        # work-so-far. Putting results AFTER work-so-far meant the model
        # finished reading its own narrative and then had to mentally
        # back-track to a separate section to find the answers — leading
        # to "I haven't seen the result yet" re-issues. With results
        # FIRST, the model reads them as context, then reads its own
        # work, then continues writing — natural flow, no back-tracking.
        # Continuation prompt — order matters for PREFIX CACHING.
        #
        # The {prompt} block (= [SYSTEM] + [USER REQUEST] + [PROJECT
        # CONTEXT]) is STABLE across rounds for a given task. Putting it
        # FIRST means the token prefix is identical on every round; vLLM
        # (which NVIDIA NIM uses under the hood) caches the KV for that
        # prefix automatically. Round 2's prefill skips most of the work
        # Round 1 already did. The 4 parallel planners on the same host
        # also share the cache.
        #
        # Anything that CHANGES between rounds (per-round warning banners,
        # tool index, past thinking, plan section, the next-turn cue) goes
        # AFTER the stable prefix. Earlier the warning banners were
        # PREPENDED to {prompt}, which killed every cache hit: even a
        # single-round "BUDGET OVERFLOW" appearance shifted every later
        # token. Now those banners attach AFTER the stable block.
        #
        # Section ownership (already covered in [SYSTEM]'s PROMPT
        # STRUCTURE explainer): [SYSTEM]/[USER REQUEST]/[PROJECT CONTEXT]
        # = JARVIS or HUMAN; [YOUR ...] sections = YOU; [WRITE YOUR NEXT
        # TURN BELOW] is where the new response goes.
        # ROLE-AWARE NEXT-TURN CUE (root-cause fix 2026-05-27): the old scaffold
        # told EVERY role "Have enough → start writing the plan … [PLAN DONE]" —
        # planner language. A coder (esp. a fallback model) landing on that
        # re-entered PLANNING mode every round: re-derived the bug instead of
        # emitting the edit, burning rounds with 0 edits (django-14792 audit).
        # Edit-writing roles (coder/self-check/reviewer pass has_pending_edits)
        # now get an ACT-NOW cue; the planner keeps the plan cue.
        if has_pending_edits is not None:
            _next_turn_cue = (
                "Your analysis in [YOUR PAST THINKING] STANDS — do NOT re-derive "
                "it. Read the most recent result (a diff, a ✗ REJECTED, or a tool "
                "result), then ACT:\n\n"
                "  • Edit applied + diff looks correct → [DONE][CONFIRM_DONE].\n"
                "  • Edit ✗ REJECTED → emit a CORRECTIVE edit NOW. Anchor on the "
                "CURRENT file line shown in the reject / [YOUR TOOL INDEX] — NOT "
                "the line you WISH were there (an `[edit]` keeps lines that are "
                "ALREADY in the file). If anchors keep failing, use "
                "[REPLACE LINES N-M] … [/REPLACE] — it bypasses anchoring.\n"
                "  • Still missing one fact → ONE [tool use]…[/tool use] + "
                "[STOP][CONFIRM_STOP].\n\n"
                "You are the CODER — do NOT write `=== PLAN ===` or [PLAN DONE]. "
                "Don't re-explain the bug; emit the edit. Keep [think] short — and "
                "the edit/signal must be OUTSIDE [think] (tags inside [think] are "
                "inert)."
            )
        else:
            _next_turn_cue = (
                "You're building on [YOUR PAST THINKING] above. Read the most "
                "recent round's results — what changed? Then either:\n\n"
                "  • Need more info → reason, then NEW tool calls (not already in "
                "[YOUR TOOL INDEX]) in [tool use]…[/tool use] + [STOP][CONFIRM_STOP].\n"
                "  • Have enough → start writing the plan with === PLAN === … "
                "=== END PLAN === (or refine an existing plan with === PLAN_EDIT "
                "=== …). When the plan is complete, signal [PLAN DONE][CONFIRM_PLAN_DONE]."
            )
        current_prompt = f"""{prompt}

{unterminated_block}{_budget_drop_block}{dropped_block}{bad_block}{edit_results_block}══════════════════════════════════════════════════════════════════════
[YOUR TOOL INDEX] — every tool call you've made so far
══════════════════════════════════════════════════════════════════════
{manifest_str}
{budget_msg}

══════════════════════════════════════════════════════════════════════
[YOUR PAST THINKING] — your previous rounds, oldest first
══════════════════════════════════════════════════════════════════════
Your own past work, in order. Each round shows YOUR THINKING (prose
+ tool calls you made) and the TOOL RESULT (the content the runtime
returned for those calls). Build on it. Do not repeat it.

{past_thinking}
{plan_section}
══════════════════════════════════════════════════════════════════════
[WRITE YOUR NEXT TURN BELOW]
══════════════════════════════════════════════════════════════════════
{_next_turn_cue}
{preamble_cue}
──────────────────────────────────────────────────────────────────────"""

    return {
        "model": model,
        "answer": full_response,
        "done": _done_signaled,
        "force_done": _force_done_signaled,
        "research": local_research,
        # `persistent_lookups` reflects the FINAL view the model had: CODE
        # entries replaced by their KEEP-filtered version, DISCARDed entries
        # removed. Callers that want to know exactly what the model was
        # looking at when it produced `answer` should read this, not
        # `research` (which only holds NEW results from this run).
        "persistent_lookups": dict(persistent_lookups),
    }
