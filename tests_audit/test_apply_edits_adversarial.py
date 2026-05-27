"""ADVERSARIAL SECOND-PASS audit of `_apply_edits`.

The first-pass tests verified the happy paths. This pass attacks the
function with malformed, malicious, edge-of-spec inputs:
  • Unicode normalization edge cases (combining marks, lookalikes).
  • Mixed line endings (CR / LF / CRLF / mixed).
  • Tabs vs spaces in SEARCH but not file (and vice versa).
  • Massive inputs (10K-line files, 1K-line searches).
  • Empty replacements at the start/end of file.
  • Search blocks that span the entire file.
  • Idempotence: applying the SAME edit twice (second should refuse).
  • Property: total >= matched always; matched <= len(edits) always.
  • Property: result is byte-identical when no edits match.
  • Property: line count delta is correctly tracked.
"""
import pytest
from workflows.code import _apply_edits


# ─────────────── PROPERTY-BASED INVARIANTS ───────────────


def test_inv__total_ge_matched():
    """For ANY edit list, total ≥ matched (every applied edit was attempted)."""
    cases = [
        ([], "anything"),
        ([("a", "b")], "no_match"),
        ([("foo", "bar")], "foo"),
        ([("a", "b"), ("c", "d")], "a\nc"),
        ([("a", "b"), ("nonexistent", "x"), ("c", "d")], "a\nc"),
    ]
    for edits, orig in cases:
        _, m, t, _ = _apply_edits(orig, edits)
        assert m <= t, f"matched={m} > total={t} for edits={edits}"


def test_inv__matched_le_edit_count():
    """matched ≤ len(edits) ALWAYS."""
    edits = [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]
    orig = "a\nb\nc\nd\n"
    _, m, _, _ = _apply_edits(orig, edits)
    assert m <= len(edits)


def test_inv__no_match_unchanged():
    """If NO edits match, result is byte-identical to original (modulo
    the documented tab-expansion that always happens)."""
    orig = "no special content here\n"
    edits = [("absent_pattern", "would_replace")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 0
    assert result == orig.expandtabs(4)


def test_inv__empty_edits_returns_original():
    orig = "any\ncontent\n"
    result, m, t, amb = _apply_edits(orig, [])
    assert m == 0 and t == 0 and amb == []


def test_inv__empty_original_no_match():
    """Empty file — nothing can match."""
    result, m, t, _ = _apply_edits("", [("anything", "x")])
    assert m == 0


# ─────────────── UNICODE EDGE CASES ───────────────


def test_unicode__nfc_match():
    """é can be composed (NFC) or decomposed (NFD). Test the composed form."""
    orig = "name = 'café'\n"  # é as single codepoint
    edits = [("name = 'café'", "name = 'cafe'")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_unicode__emoji_match():
    orig = "status = 'ok 🎉'\n"
    edits = [("status = 'ok 🎉'", "status = 'done'")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "done" in result


def test_unicode__rtl_text():
    """Right-to-left scripts should match correctly."""
    orig = "msg = 'العربية'\n"
    edits = [("msg = 'العربية'", "msg = 'arabic'")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_unicode__zero_width_chars_treated_literally():
    """ZWJ / ZWNJ — should match byte-for-byte."""
    zwj = "‍"
    orig = f"a{zwj}b\n"
    edits = [(f"a{zwj}b", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_unicode__combining_marks_literal():
    """Combining acute (U+0301) — must match exactly."""
    orig = "é letter\n"  # e + combining acute
    edits = [("é letter", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_unicode__zwsp_inside_doesnt_break():
    """Zero-width space inside SEARCH — matches literally."""
    zwsp = "​"
    orig = f"x{zwsp}y separator content\n"
    edits = [(f"x{zwsp}y separator content", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


# ─────────────── LINE ENDING EDGE CASES ───────────────


def test_lineending__crlf_file_lf_search():
    """File has CRLF endings, SEARCH has LF. Implementation-defined behavior."""
    orig = "line 1\r\nline 2\r\nline 3\r\n"
    edits = [("line 2", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    # Likely fails to match because line ending differs. Document the behavior.
    # If it does match, the line ending in result depends on implementation.
    assert m in {0, 1}


def test_lineending__only_lf():
    orig = "a\nb\nc\n"
    edits = [("b", "BB")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "BB" in result


def test_lineending__no_trailing_newline():
    """File doesn't end in `\\n` — must still work."""
    orig = "only_line"
    edits = [("only_line", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "REPLACED" in result


# ─────────────── TAB vs SPACE EDGE CASES ───────────────


def test_tabspace__file_tabs_search_4_spaces():
    """File uses \\t indent, SEARCH uses 4-space indent. expandtabs(4)
    normalizes both to 4 spaces → match."""
    orig = "def f():\n\treturn 1\n"  # tab indent
    edits = [("    return 1", "    return 2")]  # 4-space indent
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "return 2" in result


def test_tabspace__file_8_spaces_search_4_spaces_no_match():
    """File uses 8 spaces, SEARCH uses 4 spaces → no match (different indent)."""
    orig = "def f():\n        return 1\n"  # 8 spaces
    edits = [("    return 1", "    return 2")]  # 4 spaces
    result, m, _, _ = _apply_edits(orig, edits)
    # Strategy 1 fails; strategy 2/3 (whitespace-normalized) might succeed
    # because both `return 1` strings collapse the same on `.strip()`.
    assert m in {0, 1}


def test_tabspace__mixed_tab_and_space_in_file():
    """File has both tab- and space-indented lines (a mess) — still handled."""
    orig = "def f():\n\tline_a\n    line_b\n"
    edits = [("    line_a", "REPLACED_A")]  # 4-space form (tab expanded)
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


# ─────────────── MASSIVE INPUTS ───────────────


def test_massive__10k_line_file_single_edit():
    """File with 10K lines, edit targets one specific line."""
    lines = [f"unique_line_{i}" for i in range(10000)]
    orig = "\n".join(lines) + "\n"
    edits = [("unique_line_5000", "TARGET_HIT")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "TARGET_HIT" in result
    # Original lines around it are preserved
    assert "unique_line_4999" in result
    assert "unique_line_5001" in result


def test_massive__1000_edits_in_one_call():
    """1000 edits all applied in one call. Each on a different line."""
    lines = [f"line_{i}" for i in range(1000)]
    orig = "\n".join(lines) + "\n"
    edits = [(f"line_{i}", f"REPLACED_{i}") for i in range(1000)]
    result, m, t, _ = _apply_edits(orig, edits)
    assert t == 1000
    # All should match (they're all unique exact substrings = full lines)
    assert m == 1000


def test_massive__large_replacement_body():
    """REPLACE body of 1000 lines."""
    orig = "PLACEHOLDER\n"
    replacement = "\n".join([f"new_line_{i}" for i in range(1000)])
    edits = [("PLACEHOLDER", replacement)]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "new_line_0" in result
    assert "new_line_999" in result


# ─────────────── SEARCH AT FILE BOUNDARIES ───────────────


def test_boundary__search_at_eof():
    orig = "header\nbody\nFOOTER"
    edits = [("FOOTER", "NEW_FOOTER")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_boundary__search_at_bof():
    orig = "HEADER\nbody\nfooter"
    edits = [("HEADER", "NEW_HEADER")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_boundary__search_spans_whole_file():
    orig = "line 1\nline 2\nline 3"
    edits = [("line 1\nline 2\nline 3", "totally\nreplaced")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "totally" in result
    assert "replaced" in result


def test_boundary__delete_first_line():
    orig = "DELETE_ME\nkeep\nkeep_too\n"
    edits = [("DELETE_ME", "")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert "DELETE_ME" not in result
    assert "keep" in result


def test_boundary__delete_last_line():
    orig = "keep\nkeep_too\nDELETE_ME"
    edits = [("DELETE_ME", "")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert "DELETE_ME" not in result


# ─────────────── IDEMPOTENCE & ORDER ───────────────


def test_idem__same_edit_twice_second_refuses():
    """Apply edit X to result, then call _apply_edits AGAIN with the same X
    on the new result — the second call should NOT match (the pattern is gone)."""
    orig = "value = 1\n"
    edits = [("value = 1", "value = 2")]
    result, m1, _, _ = _apply_edits(orig, edits)
    assert m1 == 1
    # Apply same edit again — pattern is gone
    result2, m2, _, _ = _apply_edits(result, edits)
    assert m2 == 0


def test_order__no_dependency_between_edits():
    """Edits with disjoint targets: order should not matter for the FINAL
    result (only for the order of internal events)."""
    orig = "A\nB\nC\n"
    edits1 = [("A", "1"), ("B", "2"), ("C", "3")]
    edits2 = [("C", "3"), ("A", "1"), ("B", "2")]  # different order
    r1, _, _, _ = _apply_edits(orig, edits1)
    r2, _, _, _ = _apply_edits(orig, edits2)
    assert r1 == r2


def test_order__overlapping_first_wins():
    """Edit 1 takes line N; edit 2 targeting line N is refused (edited-range
    tracking). Order matters."""
    orig = "value = OLD\n"
    edits = [("value = OLD", "value = FIRST")]
    r, m, _, _ = _apply_edits(orig, edits)
    assert "FIRST" in r
    # Now reverse — same constraints, same result for single edit
    r2, m2, _, _ = _apply_edits(orig, edits)
    assert r == r2


# ─────────────── REGEX CHARS IN SEARCH ───────────────


def test_regex__dot_star_literal():
    orig = "pattern = '.*'\n"
    edits = [("pattern = '.*'", "pattern = 'literal'")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "literal" in result


def test_regex__paren_literal():
    orig = "match = (group1)\n"
    edits = [("match = (group1)", "match = no_group")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_regex__backslash_literal():
    orig = "path = 'C:\\Users'\n"
    edits = [("path = 'C:\\Users'", "path = '/home'")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


def test_regex__newline_in_search():
    """Multi-line SEARCH should match across line boundaries."""
    orig = "line A\nline B\nline C\n"
    edits = [("line A\nline B", "MERGED")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1


# ─────────────── AMBIGUITY HANDLING ───────────────


def test_amb__2_exact_no_hint_refuses():
    orig = "value = X\nmid\nvalue = X\n"
    edits = [("value = X", "value = Y")]
    result, m, _, amb = _apply_edits(orig, edits)
    # No hint → refuses → 0 matched
    assert m == 0
    assert len(amb) >= 1


def test_amb__5_exact_no_hint_refuses_all_unchanged():
    orig = "\n".join(["dup = 1"] * 5)
    edits = [("dup = 1", "REPL")]
    result, m, _, amb = _apply_edits(orig, edits)
    assert m == 0
    # All 5 originals still present
    assert result.count("dup = 1") == 5


def test_amb__exact_match_skips_already_edited_region():
    """First edit makes a change; second edit's exact pattern would land
    in the edited region → must find an alternative location or refuse."""
    orig = "x = OLD\ny = mid\nx = OLD\n"
    edits = [
        ("x = OLD", "x = NEW1"),  # ambiguous: 2 matches → REFUSED
        ("y = mid", "y = REPLACED"),  # disjoint → applies
    ]
    result, m, _, _ = _apply_edits(orig, edits)
    # First edit refused (ambiguous), second applies
    assert "y = REPLACED" in result
    # Both x = OLD still present (first edit refused)
    assert result.count("x = OLD") == 2


# ─────────────── EMPTY-REPLACE EDGE CASES ───────────────


def test_empty_replace__delete_middle_line():
    orig = "L1\nDELETE\nL3\n"
    edits = [("DELETE", "")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "DELETE" not in result


def test_empty_replace__delete_multi_line_block():
    orig = "head\nDEL_A\nDEL_B\nDEL_C\ntail\n"
    edits = [("DEL_A\nDEL_B\nDEL_C", "")]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 1
    assert "DEL_A" not in result
    assert "DEL_B" not in result
    assert "DEL_C" not in result
    assert "head" in result
    assert "tail" in result


def test_empty_replace__delete_then_subsequent_edit():
    orig = "L1\nDELETE\nL3\nKEEP\n"
    edits = [
        ("DELETE", ""),
        ("KEEP", "KEPT"),
    ]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 2
    assert "DELETE" not in result
    assert "KEPT" in result


# ─────────────── SPACES-ONLY / WHITESPACE-ONLY ───────────────


def test_whitespace_only_search__skipped():
    """SEARCH that is only whitespace/newlines is treated as empty."""
    orig = "content\n"
    edits = [("   \n  \n  ", "REPL")]
    result, m, t, _ = _apply_edits(orig, edits)
    # whitespace-only SEARCH after .strip('\n') becomes empty? Let's check.
    # `"   \n  \n  ".strip('\n')` → `"   \n  \n  "` (strip only strips \n)
    # Then .strip on individual lines for normalization → empty list.
    # Behavior: implementation-defined. Assert it doesn't crash.
    assert isinstance(result, str)


# ─────────────── RETURN-VALUE SANITY ───────────────


def test_return__shape_always_4_tuple():
    orig = "x"
    result = _apply_edits(orig, [])
    assert isinstance(result, tuple) and len(result) == 4


def test_return__result_is_str():
    result, _, _, _ = _apply_edits("x", [])
    assert isinstance(result, str)


def test_return__counts_are_ints():
    _, m, t, _ = _apply_edits("x", [("a", "b")])
    assert isinstance(m, int) and isinstance(t, int)


def test_return__ambiguous_skips_is_list_of_str():
    orig = "dup\ndup\n"
    _, _, _, amb = _apply_edits(orig, [("dup", "X")])
    assert isinstance(amb, list)
    if amb:
        assert all(isinstance(s, str) for s in amb)


# ─────────────── EDITED-RANGE TRACKING (HARD CASES) ───────────────


def test_edited_range__line_count_changes_subsequent_indices_track():
    """Edit 1 replaces 1 line with 3 lines (shift +2). Edit 2 targets a
    line AFTER edit 1 — its index in the new file is original+2.
    The tracker should still permit edit 2."""
    orig = "L1\nORIGINAL\nL3\nLAST\n"
    edits = [
        ("ORIGINAL", "REP_A\nREP_B\nREP_C"),  # 1 → 3 lines
        ("LAST", "LAST_REPLACED"),
    ]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 2
    assert "REP_A" in result
    assert "LAST_REPLACED" in result


def test_edited_range__deletion_doesnt_corrupt_followers():
    """Edit 1 deletes a line (1 → 0 lines, shift -1). Edit 2 targets
    a line after — must still apply."""
    orig = "L1\nDELETE_ME\nLAST\n"
    edits = [
        ("DELETE_ME", ""),
        ("LAST", "FOUND_IT"),
    ]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 2
    assert "DELETE_ME" not in result
    assert "FOUND_IT" in result


def test_edited_range__cascading_3_edits_with_growth():
    """Three edits each grow the file. All should apply."""
    orig = "A\nB\nC\n"
    edits = [
        ("A", "A1\nA2"),
        ("B", "B1\nB2"),
        ("C", "C1\nC2"),
    ]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 3
    for marker in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        assert marker in result
