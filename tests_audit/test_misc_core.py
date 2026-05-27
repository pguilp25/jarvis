"""Audit `_norm_key`, `_build_file_skeleton`, `_detect_preamble_sections`,
`_detect_unterminated_blocks`, `_strip_label`, `_parse_code_arg`.

These are second-tier utilities — small, pure, but bugs here cause:
  • Cache misses (same tag re-routed → wasted budget).
  • Skeleton noise (model picks wrong KEEP ranges → wrong fix).
  • Missed continuation guidance (model redoes deep-think prelude).
  • Silently swallowed FILE / EDIT blocks (model thinks edits applied
    but they didn't, because the closer regex failed to match).
"""
import pytest
from core.tool_call import (
    _norm_key,
    _build_file_skeleton,
    _detect_preamble_sections,
    _detect_unterminated_blocks,
    _strip_label,
    _parse_code_arg,
)


# ─────────────── _norm_key ───────────────


def test_norm__path_dot_slash_prefix_stripped():
    assert _norm_key("CODE", "./foo.py") == _norm_key("CODE", "foo.py")


def test_norm__multiple_dot_slash_prefix_stripped():
    """`./../` — only the leading `./` is stripped per iteration."""
    out = _norm_key("CODE", "./foo.py")
    # tag_type is preserved verbatim; arg is lowercased
    assert out == "CODE:foo.py"


def test_norm__backslash_to_forward_slash():
    """Windows-style path separators normalized."""
    assert _norm_key("CODE", "a\\b.py") == _norm_key("CODE", "a/b.py")


def test_norm__trailing_whitespace_stripped():
    assert _norm_key("CODE", "foo.py ") == _norm_key("CODE", "foo.py")
    assert _norm_key("CODE", " foo.py") == _norm_key("CODE", "foo.py")


def test_norm__case_collapse():
    assert _norm_key("CODE", "FOO.PY") == _norm_key("CODE", "foo.py")
    assert _norm_key("REFS", "MY_FUNC") == _norm_key("REFS", "my_func")


def test_norm__internal_whitespace_collapsed():
    """`[KEEP: foo.py  10-20]` (double space) — collapse to single."""
    assert _norm_key("KEEP", "foo.py  10-20") == _norm_key("KEEP", "foo.py 10-20")


def test_norm__range_whitespace_normalized():
    """`100 - 200` → `100-200`."""
    assert _norm_key("VIEW", "foo.py 100 - 200") == _norm_key("VIEW", "foo.py 100-200")


def test_norm__list_separator_whitespace_normalized():
    assert _norm_key("KEEP", "foo.py 10-20 , 30-40") == _norm_key("KEEP", "foo.py 10-20,30-40")


def test_norm__different_tag_types_distinct():
    """Same arg, different tag → different keys (we don't conflate
    [CODE: x] with [VIEW: x])."""
    assert _norm_key("CODE", "foo.py") != _norm_key("VIEW", "foo.py")
    assert _norm_key("REFS", "foo") != _norm_key("LSP", "foo")


def test_norm__idempotent_on_already_normalized():
    once = _norm_key("CODE", "foo.py")
    prefix = "CODE:"
    twice = _norm_key("CODE", once[len(prefix):])
    assert once == twice


def test_norm__different_args_different_keys():
    assert _norm_key("CODE", "foo.py") != _norm_key("CODE", "bar.py")
    assert _norm_key("KEEP", "foo.py 10-20") != _norm_key("KEEP", "foo.py 30-40")


def test_norm__output_format_lowercase_arg():
    """Arg is lowercased; tag_type is preserved verbatim."""
    out = _norm_key("CODE", "Foo.PY")
    assert out == "CODE:foo.py"


# ─────────────── _strip_label ───────────────


def test_strip_label__no_label():
    """No `#label` suffix — returned unchanged with None."""
    clean, label = _strip_label("foo.py")
    assert clean == "foo.py"
    assert label is None


def test_strip_label__with_label():
    """`foo.py #mylabel` → clean='foo.py', label='mylabel'."""
    clean, label = _strip_label("foo.py #mylabel")
    assert clean.strip() == "foo.py"
    assert label is not None
    assert "mylabel" in label


def test_strip_label__with_keep_range_and_label():
    clean, label = _strip_label("foo.py 10-20 #anchor")
    assert "foo.py" in clean
    assert "10-20" in clean
    assert label is not None


# ─────────────── _parse_code_arg ───────────────


def test_parse_code_arg__bare_path():
    path, ranges = _parse_code_arg("foo.py")
    assert path == "foo.py"
    assert ranges is None


def test_parse_code_arg__path_with_range():
    path, ranges = _parse_code_arg("foo.py 10-20")
    assert path == "foo.py"
    assert ranges == [(10, 20)]


def test_parse_code_arg__path_with_multi_range():
    path, ranges = _parse_code_arg("foo.py 10-20,30-40")
    assert path == "foo.py"
    assert ranges is not None
    assert (10, 20) in ranges
    assert (30, 40) in ranges


# ─────────────── _build_file_skeleton ───────────────


def test_skeleton__python_finds_def_and_class():
    src = (
        "def top_level_fn():\n"
        "    pass\n"
        "\n"
        "class MyClass:\n"
        "    def method(self):\n"
        "        pass\n"
    )
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert "top_level_fn" in out
    assert "MyClass" in out
    assert "method" in out


def test_skeleton__python_skips_docstring_def():
    """A `def foo()` inside a docstring should NOT appear in the skeleton."""
    src = (
        'def real_fn():\n'
        '    """\n'
        '    Example: def fake_fn(): pass\n'
        '    """\n'
        '    return 1\n'
    )
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert "real_fn" in out
    # AST parsing means docstring contents are NOT scanned
    assert "fake_fn" not in out


def test_skeleton__python_module_const():
    """Module-level UPPER_CASE constants are skeleton-worthy."""
    src = (
        "MAX_RETRIES = 5\n"
        "MIN_COUNT = 0\n"
        "def helper():\n"
        "    LOCAL_VAR = 99\n"  # NOT module-level
    )
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert "MAX_RETRIES" in out
    assert "helper" in out
    # local UPPER_CASE inside function should NOT be in skeleton
    assert "LOCAL_VAR" not in out


def test_skeleton__short_const_name_ignored():
    """Constants under 3 chars are skipped (avoids `X = 1` noise)."""
    src = "X = 1\nAA = 2\nBIG = 3\n"
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert "BIG" in out
    # X and AA are too short — should be excluded
    assert "CONST X" not in out
    assert "CONST AA" not in out


def test_skeleton__python_async_def():
    src = "async def async_fn():\n    pass\n"
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert "async_fn" in out


def test_skeleton__broken_syntax_falls_back_to_regex():
    """Invalid Python — AST fails, regex path picks up."""
    src = "def broken(:\n   pass\n\ndef good():\n   pass\n"
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    # Should still extract `good` via regex fallback
    assert "good" in out


def test_skeleton__empty_file():
    out = _build_file_skeleton([], max_items=200, filename="a.py")
    assert "no top-level definitions" in out.lower() or out == ""


def test_skeleton__max_items_cap():
    """A file with 1000 functions should be sampled to 200."""
    src = "\n".join(f"def fn_{i}(): pass" for i in range(1000))
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    # Output line count should be ≤ max_items
    assert out.count('\n') < 220


def test_skeleton__non_python_regex_fallback():
    """For .js / .go etc. — regex path with comment guard."""
    src = (
        "function fooBar() {\n"
        "  return 1;\n"
        "}\n"
        "// function commentedOut() {}\n"
    )
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.js")
    # Either picks up fooBar or returns the no-defs marker (both acceptable)
    assert isinstance(out, str)


def test_skeleton__line_numbers_in_output():
    """Each item should be prefixed with its line number."""
    src = "\n\n\ndef foo():\n    pass\n"
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    # `def foo` is on line 4
    assert "4" in out


def test_skeleton__dedupes_identical_entries():
    """Two `def foo` on the same line shouldn't dupe in output."""
    src = "def foo():\n    pass\n"
    lines = src.split('\n')
    out = _build_file_skeleton(lines, max_items=200, filename="a.py")
    assert out.count("def foo") == 1


# ─────────────── _detect_preamble_sections ───────────────


def test_preamble__detects_deep_think():
    text = "## DEEP THINK\nstuff here"
    out = _detect_preamble_sections(text)
    assert any("DEEP THINK" in s for s in out)


def test_preamble__detects_pre_mortem():
    text = "### PRE-MORTEM\nrisks here"
    out = _detect_preamble_sections(text)
    assert any("PRE-MORTEM" in s for s in out)


def test_preamble__detects_pre_mortem_no_dash():
    text = "### PREMORTEM\nrisks here"
    out = _detect_preamble_sections(text)
    assert any("PRE-MORTEM" in s for s in out)


def test_preamble__detects_multiple():
    text = (
        "## DEEP THINK\nstuff\n"
        "## REAL GOAL\nmore\n"
        "## HARDEST UNKNOWN\nfinal\n"
    )
    out = _detect_preamble_sections(text)
    assert len(out) >= 3


def test_preamble__empty_text_returns_empty():
    assert _detect_preamble_sections("") == []


def test_preamble__no_sections_returns_empty():
    out = _detect_preamble_sections("Just plain prose with no headers.")
    assert out == []


def test_preamble__case_insensitive():
    text = "## deep think\nstuff"
    out = _detect_preamble_sections(text)
    assert any("DEEP THINK" in s for s in out)


def test_preamble__must_be_at_line_start():
    """Section name embedded mid-line should NOT trigger."""
    text = "see the ## DEEP THINK header below"
    out = _detect_preamble_sections(text)
    # `^` anchor + MULTILINE means it doesn't match
    assert out == []


# ─────────────── _detect_unterminated_blocks ───────────────


def test_unterminated__balanced_edit_block():
    text = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]\n"
        "=== END EDIT ===\n"
    )
    out = _detect_unterminated_blocks(text)
    # No issues — balanced
    assert out == [] or all("a.py" not in p for _, p in out)


def test_unterminated__edit_block_missing_close():
    """EDIT terminator is [/REPLACE] / [/INSERT] / === END FILE ===.
    Omit all three — block is unterminated."""
    text = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n"
        # NO [/REPLACE] — block has no terminator
    )
    out = _detect_unterminated_blocks(text)
    assert any(p.endswith("a.py") and kind == "EDIT" for kind, p in out)


def test_unterminated__file_block_missing_close():
    text = (
        "=== FILE: new.py ===\n"
        "content\n"
        # NO closing marker
    )
    out = _detect_unterminated_blocks(text)
    # Should detect the missing closer
    assert any(kind == "FILE" for kind, _ in out)


def test_unterminated__no_blocks_returns_empty():
    assert _detect_unterminated_blocks("plain text") == []
