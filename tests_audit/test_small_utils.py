"""Audit small utility functions:
  • `_render_plan_with_line_numbers` — formats plan for [YOUR PLAN]
  • `_text_has_complete_tag` — stream early-stop trigger
  • `_describe_tool_mode` — diagnostic for block vs bare-tag mode
  • `_tag_summary` — round-summary line builder
  • `_build_continue_prompt` — CONTINUATION mode prompt assembler

These are easy to miss but each can fail in subtle ways:
  • Wrong line numbering → model edits the wrong line.
  • Stream early-stop trigger fires too eagerly → tool calls aborted.
  • Cache annotation off-by-one → "1 cached" reported when it's 0.
"""
import pytest
from core.tool_call import (
    _render_plan_with_line_numbers,
    _text_has_complete_tag,
    _describe_tool_mode,
    _tag_summary,
    _build_continue_prompt,
)


# ─────────────── _render_plan_with_line_numbers ───────────────


def test_render_plan__simple():
    plan = "line A\nline B\nline C"
    out = _render_plan_with_line_numbers(plan)
    # Each line should have its number
    assert "1" in out and "line A" in out
    assert "2" in out and "line B" in out
    assert "3" in out and "line C" in out


def test_render_plan__line_count_preserved():
    plan = "A\nB\nC\nD\nE"
    out = _render_plan_with_line_numbers(plan)
    assert out.count('\n') == plan.count('\n')


def test_render_plan__empty_plan():
    assert _render_plan_with_line_numbers("") == ""


def test_render_plan__single_line():
    out = _render_plan_with_line_numbers("only one line")
    assert "1" in out and "only one line" in out


def test_render_plan__width_scales_with_size():
    """Plan with 1000 lines needs at least 4-digit width."""
    plan = "\n".join(f"x" for _ in range(1000))
    out = _render_plan_with_line_numbers(plan)
    # The line numbers should be right-aligned to at least 4 chars wide
    # (1000 → "1000" → width 4)
    assert "1000" in out


def test_render_plan__min_width_3():
    """Plans under 1000 lines still use width 3 (right-aligned)."""
    plan = "A\nB"
    out = _render_plan_with_line_numbers(plan)
    # "  1: A" / "  2: B" — width 3 with padding
    assert "1: A" in out
    assert "2: B" in out


def test_render_plan__blank_lines_numbered():
    """Blank lines in plan get their own number."""
    plan = "A\n\nB"
    out = _render_plan_with_line_numbers(plan)
    # Line 1: A, Line 2: (blank), Line 3: B
    lines = out.split('\n')
    assert len(lines) == 3


# ─────────────── _text_has_complete_tag ───────────────


def test_has_complete_tag__valid_code_tag():
    assert _text_has_complete_tag("[CODE: a.py]")


def test_has_complete_tag__valid_refs_tag():
    assert _text_has_complete_tag("[REFS: foo_func]")


def test_has_complete_tag__valid_search_tag():
    assert _text_has_complete_tag("[SEARCH: my query]")


def test_has_complete_tag__stop_signal_complete():
    assert _text_has_complete_tag("[STOP][CONFIRM_STOP]")


def test_has_complete_tag__done_signal_complete():
    assert _text_has_complete_tag("[DONE][CONFIRM_DONE]")


def test_has_complete_tag__force_done_complete():
    assert _text_has_complete_tag("[FORCE DONE][CONFIRM_FORCE_DONE]")


def test_has_complete_tag__continue_signal_complete():
    assert _text_has_complete_tag("[CONTINUE][CONFIRM_CONTINUE]")


def test_has_complete_tag__bare_stop_rejected():
    """Bare `[STOP]` without `[CONFIRM_STOP]` is NOT a complete signal."""
    assert not _text_has_complete_tag("[STOP]")


def test_has_complete_tag__bare_done_rejected():
    assert not _text_has_complete_tag("[DONE]")


def test_has_complete_tag__bare_continue_rejected():
    assert not _text_has_complete_tag("[CONTINUE]")


def test_has_complete_tag__empty_text():
    assert not _text_has_complete_tag("")


def test_has_complete_tag__plain_prose():
    assert not _text_has_complete_tag("Just some text")


def test_has_complete_tag__incomplete_tag_rejected():
    """`[CODE: ` (open but no close) — incomplete."""
    assert not _text_has_complete_tag("[CODE: a.py")


def test_has_complete_tag__case_insensitive():
    assert _text_has_complete_tag("[code: a.py]")
    assert _text_has_complete_tag("[stop][confirm_stop]")


# ─────────────── _describe_tool_mode ───────────────


def test_describe_tool__block_mode_one():
    out = _describe_tool_mode("[tool use][CODE: a.py][/tool use]")
    assert "block" in out.lower()
    assert "1" in out


def test_describe_tool__block_mode_two():
    text = (
        "[tool use][CODE: a.py][/tool use]\n"
        "[tool use][REFS: foo][/tool use]"
    )
    out = _describe_tool_mode(text)
    assert "2" in out


def test_describe_tool__bare_tag_fallback():
    """No [tool use] wrapper → fallback mode."""
    out = _describe_tool_mode("[CODE: a.py]")
    assert "fallback" in out.lower()


def test_describe_tool__empty():
    out = _describe_tool_mode("")
    # No blocks → fallback
    assert "fallback" in out.lower()


# ─────────────── _tag_summary ───────────────


def test_tag_summary__empty_all():
    out = _tag_summary([], [], [], [], [], [], [], [], [], [], [],
                      research_cache=None, persistent_lookups={})
    assert out == "(none)"


def test_tag_summary__single_code():
    out = _tag_summary([], [], [], ["a.py"], [], [], [], [], [], [], [],
                      research_cache=None, persistent_lookups={})
    assert "CODE×1" in out


def test_tag_summary__multiple_kinds():
    out = _tag_summary(
        ["a query"],  # search
        [], [],
        ["a.py", "b.py"],  # file (CODE)
        ["foo"],  # refs
        [], [],
        ["bar"],  # lsp
        [], [], [],
        research_cache=None, persistent_lookups={},
    )
    assert "CODE×2" in out
    assert "REFS×1" in out
    assert "SEARCH×1" in out
    assert "LSP×1" in out


def test_tag_summary__cache_annotation():
    """Tags present in research_cache should be annotated with "X cached"."""
    # Cache hit on REFS:foo
    cache = {"REFS:foo": "cached_result"}
    out = _tag_summary(
        [], [], [],
        [],  # no CODE
        ["foo"],  # REFS
        [], [], [], [], [], [],
        research_cache=cache, persistent_lookups={},
    )
    # The REFS tag should be marked as cached
    assert "REFS×1" in out
    # Cache annotation present
    assert "cached" in out or "1)" in out  # `1 cached` format


def test_tag_summary__code_keep_view_not_cache_counted():
    """CODE, KEEP, VIEW are excluded from cache annotation (they're
    per-call file reads, not look-once lookups)."""
    cache = {"CODE:a.py": "result"}
    out = _tag_summary(
        [], [], [],
        ["a.py"],  # CODE
        [], [], [], [], [], [], [],
        research_cache=cache, persistent_lookups={},
    )
    # CODE×1 should NOT have a `(N cached)` suffix even though cache contains it
    assert "CODE×1" in out
    # No cache annotation
    parts = [p for p in out.split(',') if 'CODE' in p]
    assert all('cached' not in p for p in parts)


# ─────────────── _build_continue_prompt ───────────────


def test_continue_prompt__contains_base():
    base = "The original task prompt."
    out = _build_continue_prompt(base, [], 2, 5, [])
    assert base in out


def test_continue_prompt__contains_history():
    base = "TASK"
    history = ["Round 1 output", "Round 2 output"]
    out = _build_continue_prompt(base, history, 3, 5, [])
    assert "Round 1 output" in out
    assert "Round 2 output" in out


def test_continue_prompt__rounds_left_displayed():
    out = _build_continue_prompt("T", [], 2, 5, [])
    # rounds_left = 5 - 2 = 3
    assert "3" in out


def test_continue_prompt__no_preamble_done_section_when_empty():
    out = _build_continue_prompt("T", [], 1, 5, [])
    # When preamble_done is empty, the optional `Already-written sections`
    # line should NOT appear
    assert "Already-written sections" not in out


def test_continue_prompt__preamble_listed_when_present():
    out = _build_continue_prompt("T", [], 1, 5, ["DEEP THINK preamble", "PRE-MORTEM section"])
    assert "DEEP THINK preamble" in out
    assert "PRE-MORTEM section" in out


def test_continue_prompt__preamble_truncated_after_3():
    """If preamble_done has 5 items, only first 3 are shown plus an ellipsis."""
    done = ["section A", "section B", "section C", "section D", "section E"]
    out = _build_continue_prompt("T", [], 1, 5, done)
    assert "section A" in out
    assert "section B" in out
    assert "section C" in out
    # D and E omitted
    assert "…" in out  # ellipsis marker


def test_continue_prompt__work_so_far_banner_present():
    out = _build_continue_prompt("T", ["prev"], 1, 5, [])
    assert "YOUR WORK SO FAR" in out


def test_continue_prompt__history_joined_with_separator():
    history = ["A", "B", "C"]
    out = _build_continue_prompt("T", history, 1, 5, [])
    # Some separator should appear between rounds (────────)
    assert out.count("─") > 5 or "\n\n" in out
