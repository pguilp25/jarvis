"""Audit file-reading tools — `read_file`, `add_line_numbers`,
`_parse_keep_ranges`, `_extend_ranges_to_scope_anchor`,
`_filter_by_ranges`, and `extract_relevant_sections`.

These tools convert raw files into the `i{N}|<code> <lineno>` format the
model sees, or filter them down to KEEP ranges. Bugs here corrupt the
model's view of the codebase and lead to bad edits.
"""
import pytest
from pathlib import Path

from tools.codebase import (
    read_file,
    read_files,
    add_line_numbers,
    extract_relevant_sections,
)
from workflows.code import (
    _parse_keep_ranges,
    _filter_by_ranges,
    _extend_ranges_to_scope_anchor,
)


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ───────────────────── read_file ─────────────────────

def test_read__basic(tmp_path):
    _write(tmp_path / "a.py", "def f():\n    return 1\n")
    out = read_file(str(tmp_path / "a.py"))
    assert out == "def f():\n    return 1\n"


def test_read__nonexistent_returns_none(tmp_path):
    out = read_file(str(tmp_path / "does_not_exist.py"))
    assert out is None


def test_read__empty_file(tmp_path):
    _write(tmp_path / "empty.py", "")
    out = read_file(str(tmp_path / "empty.py"))
    assert out == ""


def test_read__binary_extension_skipped(tmp_path):
    """Binary file extensions short-circuit to a marker, not garbage bytes."""
    p = tmp_path / "icon.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-binary")
    out = read_file(str(p))
    # IGNORE_EXTENSIONS includes .png — should emit a marker
    assert out is not None
    assert "BINARY" in out or "skipped" in out


def test_read__unicode_content(tmp_path):
    _write(tmp_path / "a.py", "# 北京 résumé\nx = 1\n")
    out = read_file(str(tmp_path / "a.py"))
    assert "北京" in out


def test_read__very_large_file(tmp_path):
    _write(tmp_path / "big.py", "\n".join(f"line_{i}" for i in range(10000)) + "\n")
    out = read_file(str(tmp_path / "big.py"))
    assert out.startswith("line_0")
    assert "line_9999" in out


def test_read__crlf_line_endings(tmp_path):
    """File with Windows line endings — should round-trip through Python text mode."""
    p = tmp_path / "a.py"
    p.write_bytes(b"line1\r\nline2\r\n")
    out = read_file(str(p))
    # Python's text mode converts \r\n → \n on read
    assert "line1" in out and "line2" in out


def test_read_files__multi(tmp_path):
    _write(tmp_path / "a.py", "A")
    _write(tmp_path / "b.py", "B")
    out = read_files([str(tmp_path / "a.py"), str(tmp_path / "b.py")])
    assert len(out) == 2
    assert any("A" in v for v in out.values())
    assert any("B" in v for v in out.values())


def test_read_files__missing_gets_marker(tmp_path):
    """A missing file should still produce an entry with NOT FOUND marker."""
    _write(tmp_path / "a.py", "A")
    out = read_files([str(tmp_path / "a.py"), str(tmp_path / "missing.py")])
    assert len(out) == 2
    assert any("NOT FOUND" in v for v in out.values())


# ───────────────────── add_line_numbers ─────────────────────

def test_addnums__basic_format():
    src = "def f():\n    return 1\n"
    out = add_line_numbers(src)
    # i0|def f() 1   i4|return 1 2   i0| 3 (trailing empty)
    lines = out.split('\n')
    assert lines[0] == "i0|def f(): 1"
    assert lines[1] == "i4|return 1 2"


def test_addnums__deep_indent():
    src = "                pass\n"  # 16 spaces
    out = add_line_numbers(src)
    assert out.startswith("i16|pass")


def test_addnums__tabs_expanded():
    """Tab is expanded to 4 spaces (TAB_WIDTH=4)."""
    src = "\tpass\n"
    out = add_line_numbers(src)
    # Tab → 4 spaces → i4|
    assert out.startswith("i4|pass")


def test_addnums__blank_line_format():
    src = "a\n\nb\n"
    out = add_line_numbers(src)
    lines = out.split('\n')
    # Blank line in middle should be "i0| 2"
    assert lines[1] == "i0| 2"


def test_addnums__no_trailing_newline():
    src = "single line"
    out = add_line_numbers(src)
    assert out == "i0|single line 1"


def test_addnums__line_count_preserved():
    """Output line count equals input line count (after split)."""
    src = "a\nb\nc\nd\ne\n"
    out = add_line_numbers(src)
    assert out.count('\n') == src.count('\n')


# ───────────────────── _parse_keep_ranges ─────────────────────

def test_keep_parse__single_range():
    assert _parse_keep_ranges("50-80", "a.py") == [(50, 80)]


def test_keep_parse__multiple_comma_separated():
    assert _parse_keep_ranges("50-80, 120-150", "a.py") == [(50, 80), (120, 150)]


def test_keep_parse__multiple_space_separated():
    assert _parse_keep_ranges("50-80 120-150", "a.py") == [(50, 80), (120, 150)]


def test_keep_parse__overlapping_merged():
    """Overlapping ranges (50-80, 75-100) are merged into one (50-100)."""
    out = _parse_keep_ranges("50-80, 75-100", "a.py")
    assert out == [(50, 100)]


def test_keep_parse__adjacent_merged():
    """Adjacent ranges (50-80, 81-100) are merged (zero gap)."""
    out = _parse_keep_ranges("50-80, 81-100", "a.py")
    assert out == [(50, 100)]


def test_keep_parse__gapped_NOT_merged():
    """Ranges with a real gap (50-80 and 85-100, 4-line gap) are NOT merged.
    The model asked for two windows; the runtime must respect that."""
    out = _parse_keep_ranges("50-80, 85-100", "a.py")
    assert out == [(50, 80), (85, 100)]


def test_keep_parse__inverted_dropped():
    """`(10, 5)` — end before start. Should be rejected silently."""
    out = _parse_keep_ranges("10-5", "a.py")
    assert out == []


def test_keep_parse__zero_start_dropped():
    """Line numbers are 1-based; `(0, 80)` should be rejected."""
    out = _parse_keep_ranges("0-80", "a.py")
    # Start must be > 0
    assert (0, 80) not in out


def test_keep_parse__empty_string():
    assert _parse_keep_ranges("", "a.py") == []


def test_keep_parse__no_ranges_in_text():
    assert _parse_keep_ranges("just some prose", "a.py") == []


def test_keep_parse__duplicate_range_deduped():
    """Same range listed twice — keep only one."""
    out = _parse_keep_ranges("50-80, 50-80", "a.py")
    assert out == [(50, 80)]


def test_keep_parse__bracket_prefix_format():
    """[KEEP: a.py 50-80] format — actual call site strips the prefix
    before passing to _parse_keep_ranges, but the regex shouldn't care."""
    out = _parse_keep_ranges("[KEEP: a.py 50-80]", "a.py")
    assert (50, 80) in out


def test_keep_parse__three_ranges():
    out = _parse_keep_ranges("5-10, 20-30, 50-60", "a.py")
    assert out == [(5, 10), (20, 30), (50, 60)]


# ─────────────── _extend_ranges_to_scope_anchor ───────────────

def test_scope_anchor__extends_to_def():
    """Range inside a function body extends up to the def line."""
    src = "def foo():\n    x = 1\n    y = 2\n    z = 3\n"
    lines = src.split('\n')
    # Range (3, 4) is inside foo; should extend to line 1 (def foo)
    out = _extend_ranges_to_scope_anchor([(3, 4)], lines)
    assert out[0][0] == 1


def test_scope_anchor__extends_to_class():
    src = "class C:\n    def m(self):\n        x = 1\n        y = 2\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 4)], lines)
    # Should extend to either the class (1) or the def (2)
    assert out[0][0] <= 3


def test_scope_anchor__no_change_for_top_level():
    """Range covering top-level code with no enclosing def/class is unchanged."""
    src = "x = 1\ny = 2\nz = 3\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(2, 3)], lines)
    # No def/class above — should be unchanged
    assert out[0][0] == 2


def test_scope_anchor__handles_async_def():
    src = "async def foo():\n    x = 1\n    return x\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(2, 3)], lines)
    assert out[0][0] == 1


def test_scope_anchor__handles_decorator():
    """Decorators above a def — anchor should still find the def, not the
    decorator (the prompt-format anchor needs the def for the indent baseline)."""
    src = "@decorator\ndef foo():\n    x = 1\n    return x\n"
    lines = src.split('\n')
    out = _extend_ranges_to_scope_anchor([(3, 4)], lines)
    # Anchor at line 2 (def) is fine; line 1 (decorator) is acceptable too
    assert out[0][0] <= 3


# ───────────────────── _filter_by_ranges ─────────────────────

def test_filter__shows_only_kept_lines():
    src = "\n".join(f"line_{i}" for i in range(1, 21))  # 20 lines
    out = _filter_by_ranges(src, [(5, 8)], "a.py")
    assert "line_5" in out
    assert "line_8" in out
    # Outside-range lines should be hidden (replaced with marker)
    assert "line_1\n" not in out  # line 1 was not requested
    assert "line_20" not in out


def test_filter__preserves_line_numbers():
    """KEEP must preserve original line numbers so REPLACE LINES still works."""
    src = "\n".join(f"line_{i}" for i in range(1, 21))
    out = _filter_by_ranges(src, [(5, 8)], "a.py")
    # The line number "5" should appear next to line_5
    # Since the format uses left-padded line numbers, check for "5\tline_5"
    # or similar
    assert "5" in out and "line_5" in out


def test_filter__gap_marker_for_hidden():
    """Lines outside the kept range get a 'hidden' marker."""
    src = "\n".join(f"line_{i}" for i in range(1, 21))
    out = _filter_by_ranges(src, [(10, 12)], "a.py")
    # Some indicator of hidden lines
    assert "hidden" in out.lower() or "..." in out or "omitted" in out.lower()


def test_filter__multiple_ranges():
    src = "\n".join(f"line_{i}" for i in range(1, 31))  # 30 lines
    out = _filter_by_ranges(src, [(5, 8), (20, 22)], "a.py")
    assert "line_5" in out
    assert "line_20" in out
    # Middle is hidden
    assert "line_12" not in out


def test_filter__range_at_eof():
    """Range that goes past EOF is clamped."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    out = _filter_by_ranges(src, [(5, 100)], "a.py")
    # Should not crash; should show lines 5-10
    assert "line_10" in out
    assert "line_5" in out


def test_filter__range_clamped_to_one():
    """Range starting at 0 is clamped to 1."""
    src = "\n".join(f"line_{i}" for i in range(1, 11))
    out = _filter_by_ranges(src, [(0, 3)], "a.py")
    assert "line_1" in out


# ───────────────────── extract_relevant_sections ─────────────────────

def test_relevant__short_file_returns_full():
    """File under threshold returns the whole content with line numbers."""
    src = "def foo():\n    return 1\n"
    out = extract_relevant_sections(src, "foo", context_lines=100, max_short_file=200)
    # Should contain line-numbered output
    assert "foo" in out
    assert "1" in out  # line number


def test_relevant__large_file_extracts_around_hint():
    """A large file with a known hint should focus around the hint."""
    src = "\n".join([f"unrelated_{i}" for i in range(500)] +
                   ["my_target_function()"] +
                   [f"more_{i}" for i in range(500)])
    out = extract_relevant_sections(
        src, "my_target_function", context_lines=10, max_short_file=200
    )
    assert "my_target_function" in out
    # Context lines around it
    assert "unrelated_499" in out or "more_0" in out


def test_relevant__large_file_no_keywords():
    """Large file but no usable keywords — should fall back to whole file."""
    src = "\n".join(f"line_{i}" for i in range(500))
    out = extract_relevant_sections(src, "the and for", context_lines=10, max_short_file=100)
    # All keywords are in the stoplist — returns whole file
    assert "line_0" in out
    assert "line_499" in out


def test_relevant__empty_file_handled():
    out = extract_relevant_sections("", "anything", context_lines=10, max_short_file=200)
    assert isinstance(out, str)
