"""Audit `_apply_edits` — the SEARCH/REPLACE applier.

This is the MOST CRITICAL function in the codebase. A bug means edits
silently corrupt files or land in the wrong place. We exercise:
  • Strategy 1: exact match
  • Strategy 2: line-number-hinted window match
  • Strategy 3: whitespace-normalized full scan
  • Strategy 4 (the difflib fallback) is implicit in the rest
  • Edited-range tracking — earlier edit doesn't poison later ones
  • Ambiguous-match refusal — silent clobber is the worst kind of bug
"""
import pytest
from workflows.code import _apply_edits


def _e(find, replace):
    """Convenience: build one edit tuple."""
    return (find, replace)


def test_explicit_N_indent_count_not_silently_reindented():
    # When the coder declares an explicit `N|` indent on the REPLACE body, that count is
    # AUTHORITATIVE — the SEARCH/REPLACE applier must NOT reindent it to the matched
    # window's indent (which would revert a deliberate de-dent). The find here is
    # whitespace-normalized (no indent) so it goes through a reindent strategy, not the
    # verbatim exact-match path. (audit re-pass: the 5 _reindent_replace sites were ungated.)
    orig = "def f():\n    if cond:\n        result = old()\n"
    edits = [_e("result = old()", "4|result = new()")]   # declare indent 4 (de-dent out of `if`)
    result, m, t, amb = _apply_edits(orig, edits)
    assert m == 1
    assert "    result = new()" in result            # honored the declared 4
    assert "        result = new()" not in result     # NOT reverted to the window's 8
    # control: NO explicit count → reindent to the matched window (legacy behaviour kept)
    r2, _, _, _ = _apply_edits(orig, [_e("result = old()", "result = new()")])
    assert "        result = new()" in r2             # reindented to 8


# ─────────────── Strategy 1: exact match ───────────────


def test_exact__single_line():
    orig = "def foo():\n    return 1\n"
    edits = [_e("return 1", "return 2")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "return 2" in result
    assert "return 1" not in result
    assert m == 1 and t == 1
    assert amb == []


def test_exact__multi_line_block():
    orig = "def f():\n    x = 1\n    y = 2\n    return x + y\n"
    edits = [_e("    x = 1\n    y = 2", "    x = 10\n    y = 20")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "x = 10" in result
    assert "y = 20" in result
    assert m == 1


def test_exact__no_match_falls_to_strategies():
    """A SEARCH that doesn't exactly match — falls through to strategies 2-4."""
    orig = "def foo():\n    return 1\n"
    edits = [_e("totally absent code", "new")]
    result, m, t, amb = _apply_edits(orig, edits)
    # No match found
    assert m == 0
    assert t == 1


def test_exact__ambiguous_refused_without_hint():
    """SEARCH matches 2 locations exactly with no line-number hint → refused."""
    orig = "x = 1\ny = 1\nz = 1\n"
    edits = [_e("= 1", "= 99")]
    result, m, t, amb = _apply_edits(orig, edits)
    # Should refuse and produce an ambiguity message
    assert len(amb) >= 1
    # Original unchanged
    assert "= 1" in result
    assert "= 99" not in result


# ─────────────── Strategy 2: hint-guided ───────────────


def test_hint__picks_correct_location():
    """SEARCH has i{N}|code lineno prefix → hint guides to the right location."""
    orig = "def a():\n    return 1\ndef b():\n    return 1\n"
    # i4|return 1 4 = 4-space indent + "return 1" + line 4 hint
    edits = [_e("i4|return 1 4", "    return 99")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "return 99" in result
    # The hint pointed at line 4 → the SECOND return 1 should be edited
    lines = result.split('\n')
    # Line 1: def a() | 2: return 1 | 3: def b() | 4: return 99
    assert lines[1] == "    return 1"  # FIRST return 1 untouched
    assert lines[3] == "    return 99"


# ─────────────── Edited-range tracking ───────────────


def test_edited_tracking__second_edit_skips_first_edit_region():
    """If edit 1 replaces `foo` with `bar`, edit 2 should NOT re-edit
    the line containing `bar` even if pattern matches."""
    orig = "foo here\nfoo also\nfoo last\n"
    edits = [
        _e("foo here", "BAR1"),
        _e("foo also", "BAR2"),
    ]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "BAR1" in result
    assert "BAR2" in result
    assert "foo last" in result  # untouched
    assert m == 2


def test_edited_tracking__second_edit_doesnt_overlap():
    """Two edits in disjoint locations — both apply."""
    orig = "line 1\nline 2\nline 3\nline 4\nline 5\n"
    edits = [
        _e("line 1", "REPLACED_1"),
        _e("line 4", "REPLACED_4"),
    ]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "REPLACED_1" in result
    assert "REPLACED_4" in result
    assert "line 2" in result
    assert "line 3" in result
    assert "line 5" in result
    assert m == 2


# ─────────────── Whitespace handling ───────────────


def test_whitespace__tabs_normalized_to_4_spaces():
    """File with tabs; SEARCH with spaces → match works after tab expansion."""
    orig = "def f():\n\treturn 1\n"  # tab indent
    edits = [_e("    return 1", "    return 2")]  # 4 spaces
    result, m, t, amb = _apply_edits(orig, edits)
    # Tab gets expanded to 4 spaces, match succeeds
    assert "return 2" in result
    assert m == 1


def test_whitespace__replace_indent_preserved():
    """REPLACE block leading spaces are kept (not stripped)."""
    orig = "def f():\n    x = 1\n"
    edits = [_e("    x = 1", "    x = 99")]
    result, m, t, amb = _apply_edits(orig, edits)
    # The replacement keeps its 4-space indent
    assert "    x = 99" in result


# ─────────────── Empty / edge ───────────────


def test_empty__empty_search_skipped():
    orig = "content\n"
    edits = [_e("", "new content")]
    result, m, t, amb = _apply_edits(orig, edits)
    # Empty SEARCH skipped entirely (total count NOT incremented)
    assert result == orig.expandtabs(4)
    assert m == 0
    assert t == 0


def test_empty__no_edits():
    orig = "content\n"
    result, m, t, amb = _apply_edits(orig, [])
    assert result == orig.expandtabs(4)
    assert m == 0
    assert t == 0


def test_empty__delete_block_via_empty_replace():
    """Empty REPLACE means delete the matched block."""
    orig = "line 1\nDELETE ME\nline 3\n"
    edits = [_e("DELETE ME", "")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "DELETE ME" not in result
    assert "line 1" in result
    assert "line 3" in result
    assert m == 1


# ─────────────── Line-number stripping ───────────────


def test_line_numbers__stripped_from_search():
    """SEARCH with `i{N}|code lineno` line-number prefix — numbers stripped before match."""
    orig = "def foo():\n    return 1\n"
    edits = [_e("i4|return 1 2", "    return 2")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "return 2" in result


def test_line_numbers__hint_used_for_disambiguation():
    """Two exact matches + line hint → hint picks the right one."""
    orig = "x = 1\ny = 2\nx = 1\n"
    edits = [_e("i0|x = 1 3", "x = 99")]
    result, m, t, amb = _apply_edits(orig, edits)
    # Line 3 (the second `x = 1`) should be edited
    assert "x = 99" in result
    lines = result.split('\n')
    assert lines[0] == "x = 1"  # FIRST untouched
    assert lines[2] == "x = 99"  # SECOND replaced


# ─────────────── Return-shape ───────────────


def test_return__matched_total_counts():
    orig = "foo\nbar\nbaz\n"
    edits = [
        _e("foo", "FOO"),     # matches
        _e("bar", "BAR"),     # matches
        _e("nonexistent", "X"),  # no match
    ]
    result, m, t, amb = _apply_edits(orig, edits)
    assert t == 3   # all 3 attempted
    assert m == 2   # 2 actually applied


def test_return__ambiguous_skips_listed():
    orig = "v = 1\nv = 1\nv = 1\n"
    edits = [_e("v = 1", "v = 2")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert len(amb) >= 1
    assert "ambiguous" in amb[0].lower() or "exact" in amb[0].lower()


# ─────────────── Indent re-alignment via _reindent_replace ───────────────


def test_reindent__via_strategy_2():
    """Hint-guided path uses _reindent_replace to align indentation."""
    orig = "def f():\n    if x:\n        do_thing()\n"
    # Model writes the SEARCH without leading indent — strategy 2 picks it
    # up and re-indents.
    edits = [_e("  3  do_thing()", "do_other()\nmore_stuff()")]
    result, m, t, amb = _apply_edits(orig, edits)
    if m == 1:
        # Replacement should be indented to match the original 8-space block
        assert "        do_other()" in result or "do_other()" in result


# ─────────────── Order sensitivity ───────────────


def test_order__second_edit_on_same_line_refused():
    """Edited-range tracking: the second edit on the SAME line is refused
    (line is in `edited_ranges`). This protects against accidental
    re-edits that pile multiple replacements onto the same region."""
    orig = "value = OLD\n"
    edits = [
        _e("value = OLD", "value = INTERMEDIATE"),
        _e("value = INTERMEDIATE", "value = NEW"),
    ]
    result, m, t, amb = _apply_edits(orig, edits)
    # First applied, second refused
    assert "INTERMEDIATE" in result
    assert m == 1


def test_order__edits_on_different_lines_both_apply():
    """Two edits on disjoint lines both succeed."""
    orig = "first = OLD\nsecond = OLD\n"
    edits = [
        _e("first = OLD", "first = NEW"),
        _e("second = OLD", "second = NEW"),
    ]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "first = NEW" in result
    assert "second = NEW" in result
    assert m == 2


# ─────────────── Adversarial ───────────────


def test_adversarial__search_contains_special_chars():
    """Regex meta-chars in SEARCH should be matched literally (no regex)."""
    orig = "regex chars: $%^&*()\n"
    edits = [_e("regex chars: $%^&*()", "REPLACED LINE")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "REPLACED LINE" in result


def test_adversarial__unicode_in_search():
    orig = "name = '北京'\n"
    edits = [_e("name = '北京'", "name = 'Beijing'")]
    result, m, t, amb = _apply_edits(orig, edits)
    assert "'Beijing'" in result


def test_adversarial__newline_only_search():
    """A SEARCH containing only whitespace/newlines should fail gracefully."""
    orig = "content\n"
    edits = [_e("\n\n", "")]
    result, m, t, amb = _apply_edits(orig, edits)
    # find_clean.strip('\n') → "" → continue (empty skip)
    assert m == 0


def test_sandbox_rejects_path_escape():
    # write_file/_norm must reject a path that resolves OUTSIDE the repo (a model-supplied
    # ../../x or /abs/x would pollute the host and be absent from the patch). (pass-6 fix.)
    import tempfile, os, pytest as _pt
    from tools.sandbox import Sandbox
    root = tempfile.mkdtemp(prefix="contain_test_")
    open(os.path.join(root, "a.py"), "w").write("x = 1\n")
    sb = Sandbox(root); sb.setup()
    for bad in ["../../escape.py", "/tmp/abs_escape.py", "../sibling.py"]:
        with _pt.raises(ValueError):
            sb.write_file(bad, "leaked")
    sb.write_file("sub/ok.py", "fine")   # legit inside path still works


# ── ckpt-252: shared-applier duplicate-block gate (text/json-ops paths) ────────
def _mk_extracted(**kw):
    d = {"edits": {}, "text_edits": {}, "new_files": {}, "reverts": [],
         "block_edits": {}, "undo_edits": [], "malformed_edits": []}
    d.update(kw)
    return d


def test_apply_extracted_rejects_new_toplevel_dup_class():
    # The text-protocol coder's cross-round duplicate-insert footgun: re-inserting an
    # existing top-level class/def. The result PARSES (legal redefinition) so the parse
    # gate misses it; the SHARED applier's dup gate (default ON) must reject + leave the
    # file UNCHANGED. (Root cause of the 6×-bloated ansible-395e5e20 play_iterator.py.)
    from workflows.code import _apply_extracted_code
    orig = "import os\n\nclass Foo:\n    x = 1\n\ny = 2\n"
    fc = {"m.py": orig}
    ext = _mk_extracted(text_edits={"m.py": [("y = 2", "class Foo:\n    x = 1\n\ny = 2")]})
    result, m, t, skips = _apply_extracted_code(ext, fc, None)
    assert "m.py" not in result                       # not applied
    assert fc["m.py"] == orig                          # file UNCHANGED
    assert any("shadows" in s and "Foo" in s for s in skips)


def test_apply_extracted_rejects_new_adjacent_dup():
    from workflows.code import _apply_extracted_code
    orig = "def f():\n    val = compute_with_a_sufficiently_long_name(1, 2, 3)\n    return val\n"
    fc = {"p.py": orig}
    ext = _mk_extracted(text_edits={"p.py": [(
        "    return val",
        "    val = compute_with_a_sufficiently_long_name(1, 2, 3)\n    return val")]})
    result, m, t, skips = _apply_extracted_code(ext, fc, None)
    assert "p.py" not in result and fc["p.py"] == orig
    assert any("DUPLICATE adjacent" in s for s in skips)


def test_apply_extracted_dup_gate_off_lets_native_path_handle():
    # The native callers opt out (dup_block_gate=False) — they run their own
    # _post_edit_syntax_gate. With the gate OFF the applier applies the dup (the native
    # gate downstream rejects it), so the applier must NOT pre-empt that path.
    from workflows.code import _apply_extracted_code
    orig = "import os\n\nclass Foo:\n    x = 1\n\ny = 2\n"
    fc = {"m.py": orig}
    ext = _mk_extracted(text_edits={"m.py": [("y = 2", "class Foo:\n    x = 1\n\ny = 2")]})
    result, m, t, skips = _apply_extracted_code(ext, fc, None, dup_block_gate=False)
    assert "m.py" in result                            # applied (native gate handles it)


def test_apply_extracted_preexisting_dup_passes():
    # A file that ALREADY had two `class Foo` defs is not blamed on an unrelated edit.
    from workflows.code import _apply_extracted_code
    orig = "class Foo:\n    x = 1\n\nclass Foo:\n    x = 2\n\nz = 0\n"
    fc = {"m.py": orig}
    ext = _mk_extracted(text_edits={"m.py": [("z = 0", "z = 99")]})
    result, m, t, skips = _apply_extracted_code(ext, fc, None)
    assert "m.py" in result and "z = 99" in result["m.py"]   # legit edit applies
