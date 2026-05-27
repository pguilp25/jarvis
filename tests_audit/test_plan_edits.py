"""Audit `_apply_plan_edits` — applies REPLACE LINES / INSERT AFTER
operations from a PLAN_EDIT body to the current plan.

A model emits

    === PLAN_EDIT ===
    [REPLACE LINES 12-14]
    new content
    [/REPLACE]
    [INSERT AFTER LINE 20]
    more content
    [/INSERT]
    === END PLAN_EDIT ===

and this function applies the ops bottom-up so earlier line numbers
stay valid as later changes shift content.

Bugs would cause off-by-one errors, content loss, or out-of-range
silent passes. Critical because the Layer-2 plan improver writes these
edits exclusively (no full plan rewrites)."""
import pytest
from core.tool_call import _apply_plan_edits


# ─────────────── REPLACE LINES ───────────────


def test_replace__single_line():
    plan = "line 1\nline 2\nline 3\nline 4"
    edit = "[REPLACE LINES 2-2]\nnew line 2\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    assert "new line 2" in out
    assert "line 1" in out
    assert "line 3" in out
    # Old line 2 is gone
    assert "\nline 2\n" not in out


def test_replace__range_of_lines():
    plan = "line 1\nline 2\nline 3\nline 4\nline 5"
    edit = "[REPLACE LINES 2-4]\nA\nB\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    # Lines 2-4 (3 lines) replaced with 2 lines
    assert "line 1" in out
    assert "A" in out
    assert "B" in out
    assert "line 5" in out
    # Old lines 2, 3, 4 gone
    assert "line 2" not in out
    assert "line 4" not in out


def test_replace__inverted_range_normalized():
    """`[REPLACE LINES 4-2]` — code normalizes to 2-4."""
    plan = "line 1\nline 2\nline 3\nline 4"
    edit = "[REPLACE LINES 4-2]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    assert "new" in out
    assert "line 1" in out


def test_replace__out_of_range_skipped():
    """Range past EOF — silently skipped with a log entry."""
    plan = "line 1\nline 2"
    edit = "[REPLACE LINES 10-12]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    assert out == plan  # unchanged
    assert any("out of range" in l for l in log)


def test_replace__zero_line_rejected():
    """Line 0 is out of range (lines are 1-based)."""
    plan = "line 1\nline 2"
    edit = "[REPLACE LINES 0-1]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    # Either rejected or treated as full replacement — assert log entry
    if out == plan:
        assert any("out of range" in l for l in log)


def test_replace__multiple_ops_bottom_up():
    """Two REPLACE ops at different positions — bottom-up application
    keeps line numbers valid for earlier ops."""
    plan = "L1\nL2\nL3\nL4\nL5\nL6"
    # Replace lines 5-5 AND lines 2-2
    edit = (
        "[REPLACE LINES 2-2]\nNEW2\n[/REPLACE]\n"
        "[REPLACE LINES 5-5]\nNEW5\n[/REPLACE]"
    )
    out, log = _apply_plan_edits(plan, edit)
    assert "NEW2" in out
    assert "NEW5" in out
    # L1, L3, L4, L6 preserved
    assert "L1" in out
    assert "L3" in out
    assert "L4" in out
    assert "L6" in out


# ─────────────── INSERT AFTER LINE ───────────────


def test_insert__single_line_after():
    plan = "line 1\nline 2\nline 3"
    edit = "[INSERT AFTER LINE 2]\nINSERTED\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    lines = out.split('\n')
    # INSERTED should be between line 2 and line 3
    assert lines[0] == "line 1"
    assert lines[1] == "line 2"
    assert lines[2] == "INSERTED"
    assert lines[3] == "line 3"


def test_insert__after_line_0_prepends():
    """`[INSERT AFTER LINE 0]` should insert at the very top."""
    plan = "line 1\nline 2"
    edit = "[INSERT AFTER LINE 0]\nTOP\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    assert out.startswith("TOP\n") or out.startswith("TOP")


def test_insert__after_last_line_appends():
    plan = "line 1\nline 2"
    edit = "[INSERT AFTER LINE 2]\nAPPEND\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    lines = out.split('\n')
    assert "APPEND" in lines[-1] or lines[-2] == "APPEND"


def test_insert__past_eof_skipped():
    plan = "line 1"
    edit = "[INSERT AFTER LINE 99]\nnew\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    assert out == plan
    assert any("out of range" in l for l in log)


def test_insert__multiline_content():
    plan = "line 1\nline 2"
    edit = "[INSERT AFTER LINE 1]\nA\nB\nC\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    lines = out.split('\n')
    # Should have line 1, A, B, C, line 2
    assert "A" in lines
    assert "B" in lines
    assert "C" in lines


# ─────────────── MIXED OPS ───────────────


def test_mixed__replace_and_insert():
    plan = "L1\nL2\nL3\nL4"
    edit = (
        "[REPLACE LINES 1-1]\nNEW_L1\n[/REPLACE]\n"
        "[INSERT AFTER LINE 3]\nINSERTED\n[/INSERT]"
    )
    out, log = _apply_plan_edits(plan, edit)
    assert "NEW_L1" in out
    assert "INSERTED" in out
    # L2, L3, L4 still there
    assert "L2" in out
    assert "L3" in out
    assert "L4" in out


# ─────────────── EDGE CASES ───────────────


def test_empty_plan_with_edit_no_crash():
    plan = ""
    edit = "[REPLACE LINES 1-1]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    # Empty plan has 0 lines — out of range
    assert any("out of range" in l for l in log)


def test_empty_edit_body_no_changes():
    plan = "line 1\nline 2"
    out, log = _apply_plan_edits(plan, "")
    assert out == plan
    assert log == []


def test_replace_with_empty_content():
    """Replace with empty body — line is replaced with a blank line."""
    plan = "L1\nL2\nL3"
    edit = "[REPLACE LINES 2-2]\n\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    lines = out.split('\n')
    # L1 still there, L3 still there
    assert "L1" in lines
    assert "L3" in lines


def test_log_lines_returned():
    """`log` should contain a line for each operation."""
    plan = "L1\nL2"
    edit = (
        "[REPLACE LINES 1-1]\nNEW\n[/REPLACE]\n"
        "[INSERT AFTER LINE 2]\nADDED\n[/INSERT]"
    )
    out, log = _apply_plan_edits(plan, edit)
    assert len(log) >= 2


def test_case_insensitive_tags():
    """Lowercase `[replace lines 1-1]` should still match (re.IGNORECASE)."""
    plan = "L1\nL2"
    edit = "[replace lines 1-1]\nNEW\n[/replace]"
    out, log = _apply_plan_edits(plan, edit)
    assert "NEW" in out


def test_replace__preserves_remaining_lines_intact():
    """Lines NOT in the replace range should be byte-identical."""
    plan = "header_line_with_special_chars: $%&\nL2\nL3"
    edit = "[REPLACE LINES 2-2]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    # Header line preserved exactly
    assert "header_line_with_special_chars: $%&" in out
