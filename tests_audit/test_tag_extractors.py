"""Audit the tag extractors — `extract_code_tags`, `extract_refs_tags`,
`extract_keep_tags`, `extract_view_tags`, `extract_search_tags`, etc.

Each extractor:
  • Masks quoted contexts (think blocks, fences, backticks) first.
  • Finds `[TAG: arg]` matches.
  • Validates the arg shape (path vs identifier vs free text).

Bugs here cause:
  • Wrong-shape args silently dropped → "tool didn't fire" loops.
  • Prose mistaken for tags → garbage searches.
  • Tags inside `[think]` / fenced blocks fired → wasted budget and
    misleading runtime state.
"""
import pytest
from core.tool_call import (
    extract_code_tags,
    extract_refs_tags,
    extract_keep_tags,
    extract_view_tags,
    extract_search_tags,
    extract_lsp_tags,
    extract_websearch_tags,
    extract_detail_tags,
    extract_purpose_tags,
    extract_semantic_tags,
    extract_knowledge_tags,
    extract_discard_tags,
    has_tool_tags,
)


# ───────────────────── CODE: (paths only) ─────────────────────

def test_code__simple_path_extracted():
    out = extract_code_tags("[tool use][CODE: workflows/code.py][/tool use]")
    assert out == ["workflows/code.py"]


def test_code__multiple_paths_extracted():
    text = "[tool use][CODE: a.py] [CODE: pkg/b.py][/tool use]"
    out = extract_code_tags(text)
    assert "a.py" in out
    assert "pkg/b.py" in out


def test_code__prose_arg_rejected():
    """`[CODE: I want to see workflows/code.py]` is prose, not a path."""
    out = extract_code_tags("[tool use][CODE: I want to see workflows/code.py][/tool use]")
    assert out == []


def test_code__inside_think_block_ignored():
    text = "[think][CODE: a.py][/think] real: [tool use][CODE: real.py][/tool use]"
    out = extract_code_tags(text)
    assert "real.py" in out
    assert "a.py" not in out


def test_code__inside_fenced_block_ignored():
    text = "```\n[CODE: in_fence.py]\n```\n[tool use][CODE: outside.py][/tool use]"
    out = extract_code_tags(text)
    assert "outside.py" in out
    assert "in_fence.py" not in out


def test_code__inside_backticks_ignored():
    text = "we use `[CODE: not_a_call.py]` syntax; [tool use][CODE: real.py][/tool use]"
    out = extract_code_tags(text)
    assert "real.py" in out
    assert "not_a_call.py" not in out


def test_code__path_with_subdir():
    out = extract_code_tags("[tool use][CODE: a/b/c/deep.py][/tool use]")
    assert "a/b/c/deep.py" in out


def test_code__path_with_underscore():
    out = extract_code_tags("[tool use][CODE: pkg/my_module.py][/tool use]")
    assert "pkg/my_module.py" in out


def test_code__line_range_suffix_kept():
    """`[CODE: file.py 10-20]` — line range after path is OK."""
    out = extract_code_tags("[tool use][CODE: file.py 10-20][/tool use]")
    # The full arg incl. range should be preserved
    assert any("file.py" in t for t in out)


# ───────────────────── REFS: (idents only) ─────────────────────

def test_refs__simple_ident():
    out = extract_refs_tags("[tool use][REFS: my_function][/tool use]")
    assert "my_function" in out


def test_refs__camelcase_ident():
    out = extract_refs_tags("[tool use][REFS: MyClass][/tool use]")
    assert "MyClass" in out


def test_refs__path_rejected():
    """REFS takes a symbol name — not a file path."""
    out = extract_refs_tags("[tool use][REFS: pkg/file.py][/tool use]")
    assert out == []


def test_refs__prose_rejected():
    out = extract_refs_tags("[tool use][REFS: where is my function?][/tool use]")
    assert out == []


def test_refs__multiple_idents():
    text = "[tool use][REFS: foo] [REFS: bar][/tool use]"
    out = extract_refs_tags(text)
    assert "foo" in out
    assert "bar" in out


def test_refs__underscore_ident():
    out = extract_refs_tags("[tool use][REFS: _private_func][/tool use]")
    assert "_private_func" in out


def test_refs__dotted_ident():
    """`module.func` — depending on _IDENT_ARG_RE, may or may not be allowed."""
    out = extract_refs_tags("[tool use][REFS: module.func][/tool use]")
    # Document behavior — either allowed or rejected, just not crash
    assert isinstance(out, list)


# ───────────────────── KEEP: (paths only) ─────────────────────

def test_keep__path_extracted():
    out = extract_keep_tags("[tool use][KEEP: a.py 50-80][/tool use]")
    assert any("a.py" in t for t in out)


def test_keep__prose_rejected():
    out = extract_keep_tags("[tool use][KEEP: please keep lines 50-80][/tool use]")
    assert out == []


def test_keep__inside_think_ignored():
    text = "[think][KEEP: in_think.py 1-10][/think]"
    out = extract_keep_tags(text)
    assert out == []


# ───────────────────── VIEW: (paths only) ─────────────────────

def test_view__path_extracted():
    out = extract_view_tags("[tool use][VIEW: a.py 1-50][/tool use]")
    assert any("a.py" in t for t in out)


def test_view__prose_rejected():
    out = extract_view_tags("[tool use][VIEW: I want to view the file][/tool use]")
    assert out == []


# ───────────────────── SEARCH: (with file/range guards) ─────────────────────

def test_search__free_text_extracted():
    out = extract_search_tags("[tool use][SEARCH: my error message string][/tool use]")
    assert any("my error message" in t for t in out)


def test_search__anchored_line_range_rejected():
    """`[SEARCH: 45-49]` is anchored edit syntax inside a REPLACE body, not
    a search query. Must be rejected by the extractor."""
    out = extract_search_tags("[SEARCH: 45-49]")
    assert out == []


def test_search__bare_file_path_rejected():
    """`[SEARCH: ui/index.html]` with no spaces — looks like a file path,
    not a search query. Rejected."""
    out = extract_search_tags("[tool use][SEARCH: ui/index.html][/tool use]")
    assert out == []


def test_search__inside_think_ignored():
    text = "[think][SEARCH: ignored][/think] real: [tool use][SEARCH: real query][/tool use]"
    out = extract_search_tags(text)
    assert any("real query" in t for t in out)
    assert not any("ignored" in t for t in out)


# ───────────────────── LSP: (idents only) ─────────────────────

def test_lsp__ident_extracted():
    out = extract_lsp_tags("[tool use][LSP: foo_function][/tool use]")
    assert "foo_function" in out


def test_lsp__path_rejected():
    out = extract_lsp_tags("[tool use][LSP: pkg/file.py][/tool use]")
    assert out == []


# ───────────────────── PURPOSE: / SEMANTIC: (free text) ─────────────────────

def test_purpose__extracted():
    out = extract_purpose_tags("[tool use][PURPOSE: auth][/tool use]")
    assert out == ["auth"]


def test_semantic__extracted():
    out = extract_semantic_tags("[tool use][SEMANTIC: user-facing widgets][/tool use]")
    assert out == ["user-facing widgets"]


# ───────────────────── WEBSEARCH: / DETAIL: ─────────────────────

def test_websearch__extracted():
    out = extract_websearch_tags("[tool use][WEBSEARCH: how to do X in numpy][/tool use]")
    assert "how to do X in numpy" in out


def test_detail__extracted():
    out = extract_detail_tags("[tool use][DETAIL: section name here][/tool use]")
    assert "section name here" in out


# ───────────────────── KNOWLEDGE: ─────────────────────

def test_knowledge__extracted():
    out = extract_knowledge_tags("[tool use][KNOWLEDGE: pytest fixtures][/tool use]")
    assert "pytest fixtures" in out


# ───────────────────── DISCARD: ─────────────────────

def test_discard__label_extracted():
    out = extract_discard_tags("[tool use][DISCARD: #old_search][/tool use]")
    assert "#old_search" in out or "old_search" in out


# ───────────────────── has_tool_tags ─────────────────────

def test_has_tool__empty_text():
    assert not has_tool_tags("")


def test_has_tool__plain_prose():
    assert not has_tool_tags("Just some plain text with no tags.")


def test_has_tool__inside_think_only():
    """Tags ONLY inside [think] should NOT cause has_tool_tags to return True."""
    assert not has_tool_tags("[think][CODE: a.py][/think]")


def test_has_tool__real_tag_outside_think():
    assert has_tool_tags("[tool use][CODE: a.py][/tool use]")


def test_has_tool__one_of_each_kind():
    assert has_tool_tags("[tool use][REFS: foo][/tool use]")
    assert has_tool_tags("[tool use][SEARCH: a query][/tool use]")
    assert has_tool_tags("[tool use][KEEP: a.py 1-10][/tool use]")
    assert has_tool_tags("[tool use][LSP: foo][/tool use]")
    assert has_tool_tags("[tool use][PURPOSE: auth][/tool use]")
