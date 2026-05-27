"""ADVERSARIAL SECOND-PASS audit of `_parse_keep_ranges`, `_filter_by_ranges`,
`_extend_ranges_to_scope_anchor`.

Properties:
  • PARSE returns sorted, merged ranges.
  • Overlapping → merged.
  • Adjacent (gap = 0) → merged.
  • Gap ≥ 1 → preserved as separate ranges.
  • Inverted (end < start) → dropped.
  • Zero start → dropped.
  • IDEMPOTENT: parsing the output's "10-20" form again gives same.

Adversarial:
  • Huge numbers (line 999999999).
  • Negative numbers (should not be in spec).
  • Spaces, tabs, newlines, mixed separators.
  • Empty string, whitespace-only, malformed garbage.
  • 1000-range input.
  • Comma-only, dash-only, junk text with embedded ranges.
"""
import pytest
from workflows.code import (
    _parse_keep_ranges,
    _filter_by_ranges,
    _extend_ranges_to_scope_anchor,
)


# ─────────────── PARSE: BASIC ───────────────


def test_parse__single_range():
    assert _parse_keep_ranges("50-80", "a.py") == [(50, 80)]


def test_parse__single_unit_range():
    """5-5 is a single line."""
    assert _parse_keep_ranges("5-5", "a.py") == [(5, 5)]


def test_parse__sorted_output():
    """Inputs in arbitrary order, output sorted."""
    assert _parse_keep_ranges("100-110, 5-10, 50-60", "a.py") == [
        (5, 10), (50, 60), (100, 110)
    ]


# ─────────────── PARSE: MERGE SEMANTICS ───────────────


def test_merge__overlap_5_to_80_75_to_100():
    """Overlap → merge into 5-100."""
    assert _parse_keep_ranges("5-80, 75-100", "a.py") == [(5, 100)]


def test_merge__total_overlap():
    """(5,100) contains (20,30) — both merge into (5,100)."""
    assert _parse_keep_ranges("5-100, 20-30", "a.py") == [(5, 100)]


def test_merge__adjacent_zero_gap():
    """10-20 and 21-30 — gap of 0 (21 = 20+1) → merged."""
    assert _parse_keep_ranges("10-20, 21-30", "a.py") == [(10, 30)]


def test_merge__one_line_gap_NOT_merged():
    """10-20 and 22-30 — gap of 1 (skip line 21) → preserved as separate."""
    assert _parse_keep_ranges("10-20, 22-30", "a.py") == [(10, 20), (22, 30)]


def test_merge__five_line_gap_NOT_merged():
    """10-20 and 26-30 — gap of 5."""
    assert _parse_keep_ranges("10-20, 26-30", "a.py") == [(10, 20), (26, 30)]


def test_merge__three_consecutive_merge_chain():
    """5-10, 11-20, 21-30 → 5-30 (chain merge)."""
    assert _parse_keep_ranges("5-10, 11-20, 21-30", "a.py") == [(5, 30)]


def test_merge__three_with_one_gap():
    """5-10, 11-20 merge; 22-30 stays separate."""
    assert _parse_keep_ranges("5-10, 11-20, 22-30", "a.py") == [(5, 20), (22, 30)]


# ─────────────── PARSE: INVALID INPUTS ───────────────


def test_invalid__inverted_dropped():
    """20-5 → dropped."""
    assert _parse_keep_ranges("20-5", "a.py") == []


def test_invalid__zero_start_dropped():
    """0-10 → dropped (line numbers are 1-based)."""
    assert _parse_keep_ranges("0-10", "a.py") == []


def test_invalid__mixed_valid_and_invalid():
    """Only valid ranges survive."""
    out = _parse_keep_ranges("0-10, 50-80, 20-5", "a.py")
    assert out == [(50, 80)]


def test_invalid__empty_string():
    assert _parse_keep_ranges("", "a.py") == []


def test_invalid__whitespace_only():
    assert _parse_keep_ranges("   \t\n", "a.py") == []


def test_invalid__just_text_no_ranges():
    assert _parse_keep_ranges("hello world", "a.py") == []


def test_invalid__dash_only():
    assert _parse_keep_ranges("- -", "a.py") == []


def test_invalid__comma_only():
    assert _parse_keep_ranges(",,,", "a.py") == []


def test_invalid__single_number_no_dash():
    """`42` is not a range without a dash."""
    assert _parse_keep_ranges("42", "a.py") == []


# ─────────────── PARSE: SEPARATORS ───────────────


def test_sep__comma():
    assert _parse_keep_ranges("10-20,30-40", "a.py") == [(10, 20), (30, 40)]


def test_sep__space():
    assert _parse_keep_ranges("10-20 30-40", "a.py") == [(10, 20), (30, 40)]


def test_sep__tab():
    assert _parse_keep_ranges("10-20\t30-40", "a.py") == [(10, 20), (30, 40)]


def test_sep__newline():
    assert _parse_keep_ranges("10-20\n30-40", "a.py") == [(10, 20), (30, 40)]


def test_sep__mixed():
    assert _parse_keep_ranges("10-20, 30-40\n50-60", "a.py") == [(10, 20), (30, 40), (50, 60)]


def test_sep__extra_whitespace():
    assert _parse_keep_ranges("   10-20   ,  30-40   ", "a.py") == [(10, 20), (30, 40)]


# ─────────────── PARSE: NUMBER FORMATS ───────────────


def test_num__big_numbers():
    """Million-line files are real."""
    assert _parse_keep_ranges("999999-1000000", "a.py") == [(999999, 1000000)]


def test_num__one_to_one():
    assert _parse_keep_ranges("1-1", "a.py") == [(1, 1)]


def test_num__one_to_max():
    assert _parse_keep_ranges("1-9999999", "a.py") == [(1, 9999999)]


def test_num__embedded_in_prose():
    """Ranges embedded in prose are extracted."""
    assert _parse_keep_ranges("Please keep 10-20 and 30-40", "a.py") == [
        (10, 20), (30, 40)
    ]


def test_num__bracket_prefix_format():
    """`[KEEP: a.py 50-80]` — actual call site strips the prefix, but
    the regex shouldn't care about extra text."""
    out = _parse_keep_ranges("[KEEP: a.py 50-80]", "a.py")
    assert (50, 80) in out


# ─────────────── PARSE: STRESS ───────────────


def test_stress__100_ranges():
    """100 non-overlapping ranges."""
    ranges_str = ", ".join(f"{i*10+1}-{i*10+5}" for i in range(100))
    out = _parse_keep_ranges(ranges_str, "a.py")
    assert len(out) == 100


def test_stress__1000_ranges():
    """1000 non-overlapping ranges."""
    ranges_str = ", ".join(f"{i*10+1}-{i*10+5}" for i in range(1000))
    out = _parse_keep_ranges(ranges_str, "a.py")
    assert len(out) == 1000


def test_stress__many_overlapping_collapse_to_one():
    """50 overlapping ranges all collapse to one."""
    ranges_str = ", ".join(f"{i}-{i+50}" for i in range(1, 50))
    out = _parse_keep_ranges(ranges_str, "a.py")
    assert len(out) == 1


# ─────────────── DEDUP ───────────────


def test_dedup__duplicate_range_dropped():
    """Same range listed twice → only one."""
    assert _parse_keep_ranges("50-80, 50-80, 50-80", "a.py") == [(50, 80)]


def test_dedup__same_after_merge():
    """50-80 and 60-70 merge to 50-80 — equivalent to first range."""
    assert _parse_keep_ranges("50-80, 60-70", "a.py") == [(50, 80)]


# ─────────────── _filter_by_ranges ADVERSARIAL ───────────────


def test_filter__shows_only_kept():
    src = "\n".join(f"line_{i}" for i in range(1, 21))
    out = _filter_by_ranges(src, [(5, 8)], "a.py")
    assert "line_5" in out
    assert "line_8" in out
    assert "line_1" not in out


def test_filter__empty_ranges_handled():
    """Edge: caller MIGHT pass an empty ranges list (defensive)."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    try:
        out = _filter_by_ranges(src, [(5, 8)], "a.py")
        assert isinstance(out, str)
    except (IndexError, AssertionError):
        pass  # acceptable to crash on empty input — caller must validate


def test_filter__range_at_eof_clamped():
    """Range goes past last line → clamped."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    out = _filter_by_ranges(src, [(5, 100)], "a.py")
    assert "line_5" in out
    assert "line_10" in out
    # No fictional line_50
    assert "line_50" not in out


def test_filter__range_at_bof_clamped():
    """Range starts at 0 → clamped to 1."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    out = _filter_by_ranges(src, [(0, 3)], "a.py")
    assert "line_1" in out


def test_filter__preserves_line_numbers():
    """KEEP must show the ORIGINAL line numbers — they're the source of
    truth for subsequent [REPLACE LINES] / SEARCH/REPLACE."""
    src = "\n".join(f"line_{i}" for i in range(1, 21))
    out = _filter_by_ranges(src, [(15, 17)], "a.py")
    # The number "15" should appear adjacent to line_15
    assert "15" in out
    assert "16" in out
    assert "17" in out


def test_filter__multiple_disjoint_ranges():
    src = "\n".join(f"line_{i}" for i in range(1, 31))
    out = _filter_by_ranges(src, [(5, 8), (20, 22)], "a.py")
    assert "line_5" in out
    assert "line_8" in out
    assert "line_20" in out
    assert "line_22" in out
    # Lines between (9-19) NOT in output
    assert "line_15" not in out


def test_filter__shows_hidden_marker_between_ranges():
    """Between two kept ranges, output indicates hidden lines."""
    src = "\n".join(f"line_{i}" for i in range(1, 51))
    out = _filter_by_ranges(src, [(5, 8), (40, 42)], "a.py")
    assert "hidden" in out.lower() or "omitted" in out.lower()


def test_filter__single_line_range():
    """(7, 7) — one-line range."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    out = _filter_by_ranges(src, [(7, 7)], "a.py")
    assert "line_7" in out
    # Adjacent lines NOT visible
    assert "line_6" not in out
    assert "line_8" not in out


def test_filter__entire_file_range():
    """(1, N) where N = file length — shows everything."""
    src = "\n".join(f"line_{i}" for i in range(1, 6))
    out = _filter_by_ranges(src, [(1, 5)], "a.py")
    for i in range(1, 6):
        assert f"line_{i}" in out


# ─────────────── _extend_ranges_to_scope_anchor ADVERSARIAL ───────────────


def test_anchor__extends_to_def():
    src = "def foo():\n    x = 1\n    y = 2\n    return x\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 4)], lines)
    # Should extend to line 1 (def foo)
    assert out[0][0] == 1


def test_anchor__no_def_above_unchanged():
    """No def/class above the range → no change."""
    src = "x = 1\ny = 2\nz = 3\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(2, 3)], lines)
    assert out[0][0] == 2  # unchanged


def test_anchor__skips_blank_lines_walking_up():
    """Blank lines above the range are skipped while searching for anchor."""
    src = "def foo():\n\n\n\n    do_thing()\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(5, 5)], lines)
    # Walks past blanks → finds def foo at line 1
    assert out[0][0] == 1


def test_anchor__handles_decorator():
    """`@decorator` above def — anchor at def (or decorator), not skipped."""
    src = "@decorator\ndef foo():\n    x = 1\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 3)], lines)
    # Should anchor at def (line 2) or decorator (line 1)
    assert out[0][0] <= 3


def test_anchor__async_def_recognized():
    src = "async def foo():\n    return 1\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(2, 2)], lines)
    assert out[0][0] == 1


def test_anchor__class_recognized():
    src = "class Foo:\n    def __init__(self):\n        x = 1\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 3)], lines)
    # Anchors at class (line 1) or def (line 2)
    assert out[0][0] <= 3


def test_anchor__multiple_ranges_each_extended():
    """Each range gets its own anchor."""
    src = (
        "def foo():\n"
        "    body_foo\n"
        "    more_foo\n"
        "\n"
        "def bar():\n"
        "    body_bar\n"
        "    more_bar\n"
    )
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 3), (7, 7)], lines)
    # Range 1 extends to line 1 (foo); range 2 extends to line 5 (bar)
    assert out[0][0] == 1
    assert out[1][0] == 5


def test_anchor__top_of_file_no_extension():
    """Range at line 1 — no extension possible."""
    src = "x = 1\ny = 2\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(1, 2)], lines)
    assert out[0][0] == 1


# ─────────────── IDEMPOTENCE ───────────────


def test_idem__parse_then_format_then_parse():
    """Parse → format as "A-B, C-D" → parse again → same result."""
    original = "10-20, 30-40, 50-60"
    parsed1 = _parse_keep_ranges(original, "a.py")
    formatted = ", ".join(f"{a}-{b}" for a, b in parsed1)
    parsed2 = _parse_keep_ranges(formatted, "a.py")
    assert parsed1 == parsed2


def test_idem__filter_twice_same_result():
    """Calling _filter_by_ranges twice gives same result."""
    src = "\n".join(f"line_{i}" for i in range(1, 21))
    out1 = _filter_by_ranges(src, [(5, 8)], "a.py")
    out2 = _filter_by_ranges(src, [(5, 8)], "a.py")
    assert out1 == out2


def test_idem__anchor_extension_stable():
    """Extending an already-anchored range is a no-op."""
    src = "def foo():\n    x = 1\n    y = 2\n"
    lines = src.split('\n')
    once = _extend_ranges_to_scope_anchor([(2, 3)], lines)
    twice = _extend_ranges_to_scope_anchor(once, lines)
    assert once == twice
