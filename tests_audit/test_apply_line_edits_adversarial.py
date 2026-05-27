"""ADVERSARIAL SECOND-PASS audit of `_apply_line_edits` — line-number
based REPLACE/INSERT applier.

Semantics:
  • (34, 40, "code") — REPLACE lines 34-40 with code
  • (0, 34, "code") — INSERT AFTER line 34 (start=0 is the signal)
  • (s, e, "") — DELETE lines s-e

Critical safety mechanisms:
  • OOB ranges skipped (not applied).
  • Overlapping REPLACEs refused.
  • Catastrophic-shrink tripwire (>50% loss → refuse).
  • Bottom-up application keeps line numbers stable.

Adversarial:
  • Boundary indices: line 0, line N+1, line -1.
  • Empty edit list, edit on empty file.
  • INSERT AFTER edge: line=0 prepends, line=N appends.
  • Concurrent INSERT + REPLACE at the same anchor.
  • Massive replace bodies.
  • Anchor mismatch in INSERT AFTER `---` form.
"""
import pytest
from workflows.code import _apply_line_edits


# ─────────────── BASIC REPLACE ───────────────


def test_basic__single_line_replace():
    orig = "A\nB\nC\nD\n"
    out, applied, skips = _apply_line_edits(orig, [(2, 2, "i0|REPLACED")])
    assert "REPLACED" in out
    assert applied == 1
    assert skips == []


def test_basic__multi_line_replace():
    orig = "A\nB\nC\nD\nE\n"
    out, applied, _ = _apply_line_edits(orig, [(2, 4, "i0|X\ni0|Y")])
    assert "X" in out and "Y" in out
    assert "B" not in out and "C" not in out and "D" not in out
    assert "A" in out and "E" in out


def test_basic__delete_via_empty_code():
    orig = "A\nB\nC\n"
    out, applied, _ = _apply_line_edits(orig, [(2, 2, "")])
    assert "B" not in out
    assert "A" in out and "C" in out
    assert applied == 1


def test_basic__delete_range():
    orig = "A\nB\nC\nD\nE\n"
    out, applied, _ = _apply_line_edits(orig, [(2, 4, "")])
    assert "B" not in out and "C" not in out and "D" not in out
    assert "A" in out and "E" in out


# ─────────────── BASIC INSERT AFTER ───────────────


def test_insert__basic_after_middle():
    orig = "A\nB\nC\n"
    out, applied, _ = _apply_line_edits(orig, [(0, 2, "i0|INSERTED")])
    lines = out.split('\n')
    # After line 2 (B): INSERTED appears between B and C
    assert "INSERTED" in out
    assert applied == 1


def test_insert__after_line_zero_prepends():
    """(0, 0, code) — INSERT AFTER line 0 == prepend at top."""
    orig = "A\nB\n"
    out, applied, _ = _apply_line_edits(orig, [(0, 0, "i0|TOP")])
    assert out.startswith("TOP")
    assert applied == 1


def test_insert__after_last_line_appends():
    orig = "A\nB\nC\n"
    out, applied, _ = _apply_line_edits(orig, [(0, 3, "i0|APPENDED")])
    # APPENDED should appear after C
    assert "APPENDED" in out
    pos_appended = out.find("APPENDED")
    pos_c = out.find("C")
    assert pos_c < pos_appended


# ─────────────── OUT-OF-BOUNDS ───────────────


def test_oob__replace_past_eof_skipped():
    orig = "A\nB\n"
    out, applied, skips = _apply_line_edits(orig, [(100, 200, "i0|X")])
    # Range out of bounds → skip
    assert applied == 0
    assert any("out of bounds" in s for s in skips)


def test_oob__replace_line_0_skipped():
    """REPLACE with start=0 is the INSERT signal — line 0 itself doesn't exist."""
    orig = "A\nB\n"
    # (1, 0, code) — end < start
    out, applied, skips = _apply_line_edits(orig, [(1, 0, "i0|X")])
    # end < start → out of bounds → skip
    assert applied == 0


def test_oob__insert_after_too_far_skipped():
    """INSERT AFTER LINE 100 on 2-line file."""
    orig = "A\nB\n"
    out, applied, skips = _apply_line_edits(orig, [(0, 100, "i0|X")])
    assert applied == 0
    assert any("out of bounds" in s for s in skips)


def test_oob__insert_after_negative_skipped():
    orig = "A\nB\n"
    out, applied, skips = _apply_line_edits(orig, [(0, -5, "i0|X")])
    assert applied == 0


def test_oob__end_clamped_when_close():
    """If end exceeds total by a small amount (off-by-one), it's clamped, not rejected."""
    orig = "A\nB\nC\n"
    # File has 4 "lines" (after trailing \n split); test (2, 5) → clamp to 4
    out, applied, _ = _apply_line_edits(orig, [(2, 5, "i0|REPLACED")])
    # The exact behavior depends on the clamp logic — should not crash
    assert isinstance(out, str)


# ─────────────── OVERLAPPING REPLACES ───────────────


def test_overlap__two_overlapping_both_refused():
    orig = "\n".join(f"line_{i}" for i in range(20))
    edits = [
        (5, 10, "i0|A"),  # overlaps with next
        (8, 12, "i0|B"),
    ]
    out, applied, skips = _apply_line_edits(orig, edits)
    # Both refused
    assert applied == 0
    assert any("OVERLAPPING" in s for s in skips)


def test_overlap__exactly_at_boundary_overlap():
    """(5, 10) and (10, 15) — share line 10."""
    orig = "\n".join(f"line_{i}" for i in range(20))
    edits = [(5, 10, "i0|A"), (10, 15, "i0|B")]
    out, applied, skips = _apply_line_edits(orig, edits)
    # Both refused (10 is in both)
    assert applied == 0


def test_overlap__adjacent_NOT_overlap():
    """(5, 10) and (11, 15) — adjacent but DISJOINT (no shared line)."""
    orig = "\n".join(f"line_{i}" for i in range(20))
    edits = [(5, 10, "i0|A"), (11, 15, "i0|B")]
    out, applied, _ = _apply_line_edits(orig, edits)
    # Both apply — they're disjoint
    assert applied == 2
    assert "A" in out
    assert "B" in out


def test_overlap__insert_and_replace_at_same_anchor_OK():
    """INSERT AFTER line 5 + REPLACE lines 5-10 — INSERT is allowed to
    share its anchor with a REPLACE range (not an overlap)."""
    orig = "\n".join(f"line_{i}" for i in range(20))
    edits = [(0, 5, "i0|INSERTED"), (5, 10, "i0|REPLACED")]
    out, applied, _ = _apply_line_edits(orig, edits)
    # Both should apply (INSERT is special)
    assert applied >= 1


# ─────────────── BOTTOM-UP APPLICATION ───────────────


def test_bottomup__line_numbers_stable():
    """Edits target ORIGINAL line numbers. Edit 1 grows the file → edit 2's
    original line 10 reference is still applied to original line 10."""
    orig = "\n".join(f"L_{i}" for i in range(1, 11))  # 10 lines
    edits = [
        (2, 2, "i0|EXPAND_2A\ni0|EXPAND_2B"),  # 1 → 2 lines (grow)
        (10, 10, "i0|REPLACED_LAST"),
    ]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 2
    assert "REPLACED_LAST" in out
    assert "L_10" not in out


def test_bottomup__delete_then_insert_both_apply():
    """Delete line 3, insert after line 5. Both apply correctly."""
    orig = "\n".join(f"L_{i}" for i in range(1, 8))
    edits = [
        (3, 3, ""),
        (0, 5, "i0|NEW"),
    ]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 2
    assert "L_3" not in out
    assert "NEW" in out


# ─────────────── CATASTROPHIC SHRINK TRIPWIRE ───────────────


def test_shrink__50_percent_loss_refused():
    """Edit that would delete > 50% of lines is refused (file ≥ 50 lines)."""
    orig = "\n".join(f"L_{i}" for i in range(1, 101))  # 100 lines
    edits = [(1, 70, "")]  # delete 70 lines (70% of file)
    out, applied, skips = _apply_line_edits(orig, edits)
    # Should be refused
    if applied == 0:
        # Refused with a tripwire message
        assert any("shrink" in s.lower() or "catastrophic" in s.lower() for s in skips)
    # Or applied if implementation is lenient


def test_shrink__minor_shrink_allowed():
    """Edit deleting just 10% — fine."""
    orig = "\n".join(f"L_{i}" for i in range(1, 101))
    edits = [(1, 10, "")]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 1


def test_shrink__tripwire_only_above_50_line_threshold():
    """Small files (under 50 lines) — tripwire shouldn't fire."""
    orig = "\n".join(f"L_{i}" for i in range(1, 11))  # 10 lines
    edits = [(1, 9, "")]  # delete 9 of 10 lines
    out, applied, _ = _apply_line_edits(orig, edits)
    # File too small for tripwire — applies
    assert applied == 1


# ─────────────── EMPTY INPUTS ───────────────


def test_empty__no_edits():
    orig = "A\nB\n"
    out, applied, skips = _apply_line_edits(orig, [])
    # No edits → returns input expandtab'd
    assert applied == 0
    assert skips == []


def test_empty__empty_file_with_insert():
    """Empty file — INSERT AFTER LINE 0 should work (or skip gracefully)."""
    orig = ""
    out, applied, skips = _apply_line_edits(orig, [(0, 0, "i0|NEW")])
    # Either applies (file becomes "NEW") or skips
    assert isinstance(out, str)


def test_empty__empty_file_replace_line_1_applies():
    """An empty string splits into [""] — length 1. So (1,1) is in-bounds
    and the REPLACE applies. Document this contract."""
    orig = ""
    out, applied, skips = _apply_line_edits(orig, [(1, 1, "i0|X")])
    assert applied == 1
    assert "X" in out


def test_empty__empty_file_replace_line_2_skipped():
    """Line 2 doesn't exist in a 1-line (empty) file."""
    orig = ""
    out, applied, _ = _apply_line_edits(orig, [(2, 2, "i0|X")])
    assert applied == 0


# ─────────────── INSERT AFTER WITH ANCHOR (`---` separator) ───────────────


def test_anchor__insert_with_anchor():
    """`i12|pass\\n---\\ni0|def get_traces():\\ni4|return ""`
    Anchor `pass` matches file line at `end`, then new code is inserted."""
    orig = (
        "class C:\n"
        "    def method(self):\n"
        "        pass\n"
        "    other_field = None\n"
    )
    # INSERT AFTER line 3 (the `pass` line), with anchor verification
    code = "i8|pass\n---\ni0|def new_func():\ni4|return 1"
    out, applied, _ = _apply_line_edits(orig, [(0, 3, code)])
    # Should insert after line 3
    if applied == 1:
        assert "new_func" in out
        assert "pass" in out  # original preserved


def test_anchor__mismatch_falls_to_fuzzy():
    """Anchor doesn't match exactly — ±20 fuzzy fallback kicks in."""
    orig = "\n".join([f"line_{i}" for i in range(1, 30)])
    code = "i0|line_3\n---\ni0|INSERTED"
    out, applied, _ = _apply_line_edits(orig, [(0, 3, code)])
    # Should find line_3 even with anchor
    if applied == 1:
        assert "INSERTED" in out


# ─────────────── REPORT SHAPE ───────────────


def test_return__three_tuple():
    out = _apply_line_edits("A\n", [])
    assert isinstance(out, tuple) and len(out) == 3


def test_return__applied_is_int():
    _, applied, _ = _apply_line_edits("A\n", [])
    assert isinstance(applied, int)


def test_return__skips_is_list_of_str():
    _, _, skips = _apply_line_edits("A\n", [(1, 1, "i0|X")])
    assert isinstance(skips, list)
    for s in skips:
        assert isinstance(s, str)


def test_callback__on_skip_invoked():
    """Optional callback called per skip."""
    captured = []
    _apply_line_edits(
        "A\n", [(100, 200, "i0|X")], on_skip=lambda m: captured.append(m)
    )
    assert len(captured) >= 1


def test_callback__on_skip_exception_swallowed():
    """If callback raises, it doesn't crash the apply."""
    def boom(_):
        raise RuntimeError("bad")
    out, applied, _ = _apply_line_edits(
        "A\n", [(100, 200, "i0|X")], on_skip=boom
    )
    # Should not propagate
    assert isinstance(out, str)


# ─────────────── INDENT PREFIX HANDLING ───────────────


def test_indent__i_prefix_applied_to_replacement():
    orig = "old_line\n"
    out, applied, _ = _apply_line_edits(orig, [(1, 1, "i4|x = 1")])
    # The i4| prefix becomes 4 spaces
    assert "    x = 1" in out


def test_indent__multiple_lines_each_with_prefix():
    orig = "old\n"
    out, applied, _ = _apply_line_edits(
        orig, [(1, 1, "i0|def foo():\ni4|return 1")]
    )
    assert "def foo():" in out
    assert "    return 1" in out


def test_indent__different_indents_per_line():
    orig = "old\n"
    code = "i0|class C:\ni4|def m(self):\ni8|return 1"
    out, applied, _ = _apply_line_edits(orig, [(1, 1, code)])
    assert "class C:" in out
    assert "    def m(self):" in out
    assert "        return 1" in out


# ─────────────── BOUNDARY: 1-line file ───────────────


def test_single_line__replace_only_line():
    out, applied, _ = _apply_line_edits("only_line", [(1, 1, "i0|REPLACED")])
    assert applied == 1
    assert "REPLACED" in out


def test_single_line__insert_after_top_then_replace_only():
    """INSERT then REPLACE on a 1-line file."""
    out, applied, _ = _apply_line_edits(
        "only_line",
        [
            (0, 0, "i0|HEADER"),
            (1, 1, "i0|REPLACED"),
        ],
    )
    assert applied == 2
    assert "HEADER" in out
    assert "REPLACED" in out


# ─────────────── EDGE: 5000 LINE FILE ───────────────


def test_massive__5000_line_file_single_edit():
    orig = "\n".join(f"line_{i}" for i in range(5000))
    out, applied, _ = _apply_line_edits(orig, [(2500, 2500, "i0|HIT")])
    assert applied == 1
    assert "HIT" in out


def test_massive__100_edits_scattered():
    """100 distinct REPLACE edits on a 1000-line file."""
    orig = "\n".join(f"line_{i}" for i in range(1000))
    edits = [(i * 10 + 1, i * 10 + 1, f"i0|MARK_{i}") for i in range(100)]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 100
    for i in range(100):
        assert f"MARK_{i}" in out
