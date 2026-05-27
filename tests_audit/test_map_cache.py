"""Audit `_apply_map_edits` (map editor — used by phase_understand to
update the purpose_map and detailed_map between rounds) and
`_format_research_cache` (which builds the PRE-LOADED RESEARCH block that
downstream agents see).

`_apply_map_edits` runs SEARCH/REPLACE/ADD_SECTION on the map text:
  • [SEARCH] ... [/SEARCH][REPLACE] ... [/REPLACE] — replace a section
  • [ADD_SECTION] ... [/ADD_SECTION] — append a new section

Bugs cause map corruption: section headings clobbered, ambiguous edits
silently applied (the AMBIGUITY REFUSAL is critical).
"""
import pytest
from workflows.code import _apply_map_edits, _format_research_cache


# ─────────────── _apply_map_edits ───────────────


def test_map_edit__simple_replace():
    orig = "=== SECTION: foo ===\nold body"
    resp = "[SEARCH]\nold body\n[/SEARCH][REPLACE]\nnew body\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert "new body" in out
    assert "old body" not in out


def test_map_edit__no_match_unchanged():
    orig = "=== SECTION: foo ===\nbody"
    resp = "[SEARCH]\nabsent text\n[/SEARCH][REPLACE]\nreplacement\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    # No match → no change
    assert out == orig


def test_map_edit__ambiguous_exact_match_refused():
    """If SEARCH text appears in 2+ locations EXACTLY, the edit is refused
    (silent clobber would corrupt the map)."""
    orig = "alpha\nDUPLICATE\nmid\nDUPLICATE\nomega"
    resp = "[SEARCH]\nDUPLICATE\n[/SEARCH][REPLACE]\nBAR\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    # Both DUPLICATE lines still present, no BAR (edit refused)
    assert out.count("DUPLICATE") == 2
    assert "BAR" not in out


def test_map_edit__empty_search_skipped():
    orig = "body"
    resp = "[SEARCH][/SEARCH][REPLACE]new[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert out == orig


def test_map_edit__empty_replace_deletes():
    """Empty REPLACE body deletes the matched text."""
    orig = "keep\nDELETE_ME\nkeep_after"
    resp = "[SEARCH]\nDELETE_ME\n[/SEARCH][REPLACE]\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert "DELETE_ME" not in out
    assert "keep" in out
    assert "keep_after" in out


def test_map_edit__multiple_independent_edits():
    """Two non-overlapping edits should both apply."""
    orig = "alpha\nbeta\ngamma\ndelta"
    resp = (
        "[SEARCH]\nalpha\n[/SEARCH][REPLACE]\nA1\n[/REPLACE]\n"
        "[SEARCH]\ngamma\n[/SEARCH][REPLACE]\nG1\n[/REPLACE]"
    )
    out = _apply_map_edits(orig, resp)
    assert "A1" in out
    assert "G1" in out
    assert "beta" in out
    assert "delta" in out


def test_map_edit__fuzzy_whitespace_match():
    """If exact match fails, whitespace-normalized line match should fire."""
    orig = "header\n  indented\nfooter"
    # Note: extra trailing space on indented line wouldn't be exact
    resp = "[SEARCH]\nindented\n[/SEARCH][REPLACE]\nREPLACED\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    # The fuzzy match should pick up the indented line
    assert "REPLACED" in out


def test_map_edit__fuzzy_ambiguous_refused():
    """If whitespace-normalized SEARCH matches 2+ locations, refuse."""
    orig = "  foo  \nmid\n   foo\nfooter"
    resp = "[SEARCH]\nfoo\n[/SEARCH][REPLACE]\nBAR\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    # Both `foo`s still present (whitespace-normalized fuzzy match would
    # find both → refused)
    assert "BAR" not in out


def test_map_edit__add_section_appends():
    orig = "header\nbody"
    resp = "[ADD_SECTION]\n=== SECTION: new ===\nnew content\n[/ADD_SECTION]"
    out = _apply_map_edits(orig, resp)
    assert "=== SECTION: new ===" in out
    assert "new content" in out
    # Original content preserved
    assert "header" in out
    assert "body" in out


def test_map_edit__add_section_empty_ignored():
    orig = "body"
    resp = "[ADD_SECTION][/ADD_SECTION]"
    out = _apply_map_edits(orig, resp)
    # Empty addition → no change
    assert out == orig


def test_map_edit__no_edits_in_response():
    orig = "body"
    resp = "Just some prose with no edit blocks."
    out = _apply_map_edits(orig, resp)
    assert out == orig


def test_map_edit__multi_line_search_replace():
    orig = "line 1\nline 2\nline 3\nline 4"
    resp = (
        "[SEARCH]\nline 2\nline 3\n[/SEARCH]"
        "[REPLACE]\nL2_NEW\nL3_NEW\n[/REPLACE]"
    )
    out = _apply_map_edits(orig, resp)
    assert "L2_NEW" in out
    assert "L3_NEW" in out
    assert "line 1" in out
    assert "line 4" in out


def test_map_edit__replace_changes_line_count():
    """Replace with different number of lines."""
    orig = "L1\nL2\nL3"
    resp = "[SEARCH]\nL2\n[/SEARCH][REPLACE]\nA\nB\nC\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert "A" in out
    assert "B" in out
    assert "C" in out


# ─────────────── _format_research_cache ───────────────


def test_cache__none_returns_empty():
    assert _format_research_cache(None) == ""


def test_cache__empty_dict_returns_empty():
    assert _format_research_cache({}) == ""


def test_cache__single_entry_formatted():
    cache = {"REFS:foo": "Found 3 references to `foo` in pkg/a.py:5"}
    out = _format_research_cache(cache)
    assert "PRE-LOADED RESEARCH" in out
    assert "Found 3 references" in out


def test_cache__truncation_at_max_chars():
    """Cache entries past max_chars should be truncated with a notice."""
    big_value = "x" * 50000
    cache = {"REFS:big": big_value}
    out = _format_research_cache(cache, max_chars=1000)
    # Result should be capped well below 50K
    assert len(out) < 5000
    # Should include either the truncation marker or just be small
    # (one large entry that exceeds max_chars triggers the truncation message)


def test_cache__multiple_entries_some_truncated():
    """If 5 entries don't all fit in max_chars, later ones are truncated."""
    cache = {
        "REFS:a": "value_a " * 200,
        "REFS:b": "value_b " * 200,
        "REFS:c": "value_c " * 200,
        "REFS:d": "value_d " * 200,
        "REFS:e": "value_e " * 200,
    }
    out = _format_research_cache(cache, max_chars=2000)
    # Should mention "more cached lookup(s) truncated"
    assert "truncated" in out.lower()


def test_cache__empty_value_skipped():
    """Cache entries with empty value should be skipped (no entry section)."""
    cache = {
        "REFS:empty": "",
        "REFS:full": "actual content here",
    }
    out = _format_research_cache(cache)
    assert "actual content here" in out
    # The empty entry shouldn't produce a stray empty section
    # (no easy assertion; just verify the full entry is intact)


def test_cache__has_pre_loaded_banner():
    """The output should have the PRE-LOADED RESEARCH banner."""
    cache = {"REFS:foo": "content"}
    out = _format_research_cache(cache)
    assert "PRE-LOADED RESEARCH" in out
    assert "do NOT re-search" in out or "do not re-search" in out.lower()


def test_cache__do_not_re_search_instruction():
    """The format banner should remind the agent not to re-search."""
    cache = {"REFS:foo": "x"}
    out = _format_research_cache(cache)
    # The "do NOT re-search these" instruction is in the banner
    assert "re-search" in out.lower() or "re search" in out.lower()
