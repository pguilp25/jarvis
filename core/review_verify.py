"""
review_verify — standalone parsers for reviewer routing decisions and
[VERIFY:] command tags. No third-party dependencies.

Two responsibilities:
  - extract_verify_cmd: pull a shell command out of a [VERIFY: <cmd>] tag.
  - parse_route: decode a reviewer's routing decision ([GO TO PLAN], [GO TO
    STEP N], [APPROVED]) into a structured RouteDecision.
"""

import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    kind: str                 # 'approved' | 'step' | 'plan' | 'none'
    step_num: "int | None"    # set when kind == 'step', else None
    message: str              # the reviewer's explanation / instructions (may be '')


# Tag PREFIXES. We match only the opener (`[VERIFY:`, `[GO TO PLAN:`, …) with a
# regex, then capture the body with a BRACKET-DEPTH scan up to the matching `]`.
# A plain non-greedy `.*?]` truncated at the first `]`, which silently mangled
# the most realistic bodies — list indexing `[0]`, dict access, pytest IDs
# `test[param]`, regex char-classes `[a-z]` — turning a valid command into a
# broken one. Depth-matching preserves nested `[...]`.
_VERIFY_PREFIX = re.compile(r"\[\s*VERIFY\s*:\s*", re.IGNORECASE)
_RUN_PREFIX = re.compile(r"\[\s*RUN\s*:\s*", re.IGNORECASE)

# Bodies that are FILE CONTENT, not directives — a `[RUN:]` / `[VERIFY:]` the
# model writes INSIDE an edit/new-file body (e.g. editing code/docs that quote
# JARVIS's own syntax) must NOT fire. Blank those bodies (preserving length so
# nothing else shifts) before scanning for RUN/VERIFY. A real directive lives
# OUTSIDE any edit body and is untouched.
_PROTOCOL_BODY_RES = [
    re.compile(r"(===\s*EDIT:[^\n]*\n)(.*?)(\n[ \t]*===\s*END\s+EDIT\s*===)", re.S | re.I),
    re.compile(r"(===\s*FILE:[^\n]*\n)(.*?)(\n[ \t]*===\s*END\s+FILE\s*===)", re.S | re.I),
    re.compile(r"(\[edit(?::\s*\d+)?\])(.*?)(\[/edit\])", re.S | re.I),
]


def _blank_protocol_bodies(text: str) -> str:
    """Replace edit/new-file body CONTENT with blanks so directives quoted as
    file content don't fire; keeps the surrounding markers and the text length."""
    if not text:
        return text
    for pat in _PROTOCOL_BODY_RES:
        text = pat.sub(lambda m: m.group(1) + re.sub(r"[^\n]", " ", m.group(2)) + m.group(3), text)
    return text
# Colon is OPTIONAL: the reviewer prompt's prose shows the bare `[GO TO STEP]` /
# `[GO TO PLAN]` form, so a literal-minded reviewer can emit it without a colon — and a
# colon-required parser silently dropped that rejection (a lost FAIL ships a bug).
_PLAN_PREFIX = re.compile(r"\[\s*GO\s+TO\s+PLAN\s*:?\s*", re.IGNORECASE)
_STEP_PREFIX = re.compile(r"\[\s*GO\s+TO\s+STEP\s*(\d+)?\s*:?\s*", re.IGNORECASE)

# The explicit `[APPROVED]` tag matches anywhere; the reviewer's bare
# `APPROVED`/`APPROVED — …` vocabulary matches ONLY at a line start, so prose
# ("don't write APPROVED early") can't trigger a false approval.
_APPROVED_RE = re.compile(r"\[\s*APPROVED\s*\]|(?:^|\n)\s*APPROVED\b", re.IGNORECASE)


def _capture_balanced(text: str, content_start: int) -> str:
    """Capture from content_start to the `]` that closes the tag whose opening
    `[` has already been consumed (depth starts at 1). Nested `[...]` inside the
    body are balanced and kept. If the bracket is never closed, fall back to the
    rest of the current line (so a missing `]` still yields the body, not '')."""
    depth = 1
    i = content_start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[content_start:i]
        i += 1
    nl = text.find("\n", content_start)
    return text[content_start:(nl if nl != -1 else n)]


def extract_verify_cmd(text: str) -> "str | None":
    """Return the shell command inside a [VERIFY: <cmd>] tag, or None.

    Case-insensitive; the command may span newlines and contain balanced
    brackets (indexing, pytest IDs, regex classes). If several [VERIFY:] tags
    appear, the LAST one is returned.
    """
    if not text:
        return None
    text = _blank_protocol_bodies(text)   # ignore [VERIFY:] quoted as file content
    last = None
    for m in _VERIFY_PREFIX.finditer(text):
        body = _capture_balanced(text, m.end()).strip()
        if body:
            last = body
    return last


def extract_run_cmds(text: str) -> list[str]:
    """Return every shell command inside [RUN: <cmd>] tags, in document order,
    deduplicated. Like [VERIFY:], RUN commands are free-form shell that routinely
    contain `]` (list/dict literals, indexing `a[0]`, slices, pytest IDs, regex
    classes), so RUN gets its own bracket-balanced extractor instead of going
    through the path-shaped tag detector — which would (a) reject RUN as an
    'unknown tag type', (b) require a [tool use] wrapper the prompt doesn't teach,
    and (c) truncate the command at the first `]`. A bare `[RUN:]` with an empty
    body yields one "" entry so the handler can return its NO COMMAND nudge."""
    if not text:
        return []
    text = _blank_protocol_bodies(text)   # ignore [RUN:] quoted as file content
    seen: set[str] = set()
    out: list[str] = []
    saw_empty = False
    for m in _RUN_PREFIX.finditer(text):
        body = _capture_balanced(text, m.end()).strip()
        if not body:
            saw_empty = True
            continue
        if body not in seen:
            seen.add(body)
            out.append(body)
    if saw_empty and not out:
        out.append("")        # surface the NO COMMAND feedback
    return out


def parse_route(text: str) -> RouteDecision:
    """Parse a reviewer's routing decision from free text.

    Recognizes one of:
        [GO TO PLAN: <message>]
        [GO TO STEP <N>: <message>]   (N optional → step_num=None)
        [APPROVED]   (or bare `APPROVED`/`APPROVED — …` at a line start)

    Precedence when multiple appear: PLAN > STEP > APPROVED > none. Messages may
    contain balanced brackets. Returns RouteDecision(kind, step_num, message).
    """
    if not text:
        return RouteDecision(kind="none", step_num=None, message="")
    text = _blank_protocol_bodies(text)   # a routing tag quoted as file content isn't a verdict

    # PLAN wins over everything.
    plan = _PLAN_PREFIX.search(text)
    if plan:
        msg = _capture_balanced(text, plan.end()).strip()
        return RouteDecision(kind="plan", step_num=None, message=msg)

    # STEP next.
    step = _STEP_PREFIX.search(text)
    if step:
        num_str = step.group(1)
        step_num = int(num_str) if num_str is not None else None
        msg = _capture_balanced(text, step.end()).strip()
        return RouteDecision(kind="step", step_num=step_num, message=msg)

    # APPROVED last.
    if _APPROVED_RE.search(text):
        return RouteDecision(kind="approved", step_num=None, message="")

    return RouteDecision(kind="none", step_num=None, message="")
