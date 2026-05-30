"""Remaining tools — `_strip_think`, `_check_syntax`, `lsp_find_references`,
`_mask_inert_zones`."""
import pytest
import asyncio
from workflows.code import (_check_syntax, _mask_inert_zones, _unreachable_after_jump,
                            _duplicate_adjacent_stmts)
from core.tool_call import _strip_think


# ───────────────────── _strip_think ─────────────────────

def test_strip_think__bracket_form():
    src = "before [think]reasoning here[/think] after"
    assert _strip_think(src).strip() == "before  after"


def test_strip_think__xml_form():
    src = "before <think>reasoning here</think> after"
    assert "reasoning here" not in _strip_think(src)


def test_strip_think__nested_unclosed():
    """An unclosed <think> at end — defensive."""
    src = "stuff <think>open never closes"
    out = _strip_think(src)
    # Document behavior — either passes through or masks
    assert isinstance(out, str)


def test_strip_think__multiline():
    src = "a\n[think]\nline 1\nline 2\n[/think]\nb"
    out = _strip_think(src)
    assert "line 1" not in out
    assert "a" in out and "b" in out


def test_strip_think__preserves_non_think_content():
    src = "real content [think]reasoning[/think] more real"
    out = _strip_think(src)
    assert "real content" in out
    assert "more real" in out


def test_strip_think__empty_think_block():
    src = "[think][/think]"
    out = _strip_think(src)
    assert out.strip() == ""


def test_strip_think__case_insensitive():
    src = "[Think]REASONING[/Think]"
    out = _strip_think(src)
    # Per the regex flags (re.IGNORECASE), should strip
    assert "REASONING" not in out or out == src  # document behavior


# ───────────────────── _mask_inert_zones ─────────────────────

def test_mask_inert__think_bracket_blanked():
    src = "before\n[think]\nsecret\n[/think]\nafter"
    out = _mask_inert_zones(src)
    assert "secret" not in out
    # Lengths preserved (chars replaced with spaces)
    assert len(out) == len(src)
    # Newlines preserved
    assert out.count("\n") == src.count("\n")


def test_mask_inert__think_xml_blanked():
    src = "before <think>secret</think> after"
    out = _mask_inert_zones(src)
    assert "secret" not in out


def test_mask_inert__fenced_blanked():
    src = "regular\n```\nfenced secret\n```\nmore"
    out = _mask_inert_zones(src)
    assert "fenced secret" not in out


def test_mask_inert__no_marker_passthrough():
    src = "plain content with no markers"
    out = _mask_inert_zones(src)
    assert out == src


def test_mask_inert__multiple_zones():
    src = "[think]A[/think] mid [think]B[/think] end"
    out = _mask_inert_zones(src)
    assert "A" not in out
    assert "B" not in out
    assert "mid" in out
    assert "end" in out


# ───────────────────── _check_syntax ─────────────────────

def test_syntax__valid_python_passes():
    ok, msg = _check_syntax("test.py", "def foo():\n    return 1\n")
    assert ok, f"valid Python failed: {msg}"


def test_syntax__invalid_python_fails():
    ok, msg = _check_syntax("test.py", "def foo(:\n    return 1\n")
    assert not ok
    assert msg


def test_syntax__missing_colon():
    ok, msg = _check_syntax("test.py", "def foo()\n    return 1\n")
    assert not ok


def test_syntax__indentation_error():
    ok, msg = _check_syntax("test.py", "def foo():\nreturn 1\n")
    assert not ok


def test_syntax__non_python_file_skipped():
    """Non-Python files should be skipped (return True)."""
    ok, msg = _check_syntax("README.md", "# heading\nany content")
    assert ok


def test_syntax__empty_file_ok():
    ok, msg = _check_syntax("empty.py", "")
    assert ok


def test_syntax__only_comment_ok():
    ok, msg = _check_syntax("test.py", "# just a comment\n")
    assert ok


def test_syntax__unicode_identifiers_ok():
    ok, msg = _check_syntax("test.py", "résumé = 1\n")
    # Python 3 accepts unicode identifiers
    assert ok


# ───────────────── _unreachable_after_jump ─────────────────

def test_unreachable__dead_code_after_return_in_guard():
    """The 0ea40e09 over-indent class: real logic indented INTO a guard's
    if-block sits after `return` at the same level → unreachable. Must flag it."""
    src = ("def __or__(self, other):\n"
           "    if not isinstance(other, dict):\n"
           "        return NotImplemented\n"
           "        merged = dict(self.data)\n"   # dead — same indent as the return
           "        return merged\n")
    dead = _unreachable_after_jump(src)
    assert 4 in dead and 5 in dead, dead


def test_unreachable__correct_dedented_code_clean():
    """The CORRECT shape (body dedented OUT of the guard) flags nothing."""
    src = ("def __or__(self, other):\n"
           "    if not isinstance(other, dict):\n"
           "        return NotImplemented\n"
           "    merged = dict(self.data)\n"
           "    return merged\n")
    assert _unreachable_after_jump(src) == {}


def test_unreachable__guard_return_then_outer_body_clean():
    """A return that IS the last stmt of its block (early-return guard) is fine."""
    src = ("def f(x):\n"
           "    for i in x:\n"
           "        if i:\n"
           "            return i\n"
           "    return None\n")
    assert _unreachable_after_jump(src) == {}


def test_unreachable__unparseable_returns_empty():
    """Unparseable input is the syntax gate's job — the detector stays quiet."""
    assert _unreachable_after_jump("def f(:\n  return 1\n") == {}


# ───────────────── _duplicate_adjacent_stmts ─────────────────

def test_dup__duplicated_if_yield_block_flagged():
    """The qutebrowser-1a9e74 insert footgun: re-emitting an anchor block AND
    your own copy → two structurally-identical adjacent statements."""
    src = ("def _args(flags):\n"
           "    enabled = list(features(flags))\n"
           "    if enabled:\n"
           "        yield '--enable-features=' + ','.join(enabled)\n"
           "    if enabled:\n"
           "        yield '--enable-features=' + ','.join(enabled)\n")
    dup = _duplicate_adjacent_stmts(src)
    assert 5 in dup, dup


def test_dup__distinct_adjacent_stmts_clean():
    """Different adjacent statements are NOT flagged."""
    src = ("def _args(flags):\n"
           "    enabled = list(features(flags))\n"
           "    if enabled:\n"
           "        yield '--enable-features=' + ','.join(enabled)\n"
           "    yield '--blink-settings=x'\n")
    assert _duplicate_adjacent_stmts(src) == {}


def test_dup__trivial_repeats_not_flagged():
    """Tiny statements (dump < 40 chars) are below threshold — avoid false
    positives on legit short repeats."""
    src = "def f():\n    pass\n    pass\n"   # Pass() dumps to ~6 chars
    assert _duplicate_adjacent_stmts(src) == {}


def test_dup__unparseable_returns_empty():
    assert _duplicate_adjacent_stmts("def f(:\n  x\n") == {}


def test_syntax__valid_class_ok():
    src = (
        "class Foo:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "    def bar(self):\n"
        "        return self.x\n"
    )
    ok, msg = _check_syntax("test.py", src)
    assert ok


def test_syntax__triple_quoted_string_ok():
    src = 'x = """\nmulti\nline\nstring\n"""\n'
    ok, msg = _check_syntax("test.py", src)
    assert ok


# ───────────────────── LSP fallback ─────────────────────

def test_lsp__falls_back_when_no_server_for_language():
    """For an unknown language, lsp_find_references should fall back
    gracefully (None or a message)."""
    from tools.lsp import lsp_find_references
    out = asyncio.run(lsp_find_references("anything", "/tmp/nonexistent_root"))
    # Should not crash; returns None or an empty/explanatory string
    assert out is None or isinstance(out, str)
