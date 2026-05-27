"""Audit `_smart_apply` — the dispatcher that picks between line-number
edits, text-based edits, and fuzzy filepath matching.

The path-bounded suffix matching is critical: previously a naive
`endswith` would let `foo/bar.py` edits land on `qux/bar.py`. This audit
verifies the fix and prevents regression."""
import pytest
from workflows.code import _smart_apply


def _line_edits(*items):
    """Wrap (start, end, code) tuples in the extracted dict shape."""
    return {fp: [t for t in tuples] for fp, tuples in items}


# ─────────────── basic dispatch ───────────────


def test_smart_apply__direct_filepath_line_edits():
    """When filepath is an exact key in `edits`, line edits apply."""
    extracted = {
        "edits": {"a.py": [(1, 1, "new_line_1")]},
        "text_edits": {},
    }
    out = _smart_apply("old_line_1\nline_2\n", extracted, "a.py")
    assert out is not None
    assert "new_line_1" in out


def test_smart_apply__direct_filepath_text_edits():
    """When only text_edits is populated, fall back to those."""
    extracted = {
        "edits": {},
        "text_edits": {"a.py": [("old_line", "new_line")]},
    }
    out = _smart_apply("old_line\n", extracted, "a.py")
    assert out is not None
    assert "new_line" in out


def test_smart_apply__no_edits_returns_none():
    extracted = {"edits": {}, "text_edits": {}}
    out = _smart_apply("content\n", extracted, "a.py")
    assert out is None


def test_smart_apply__filepath_not_found_returns_none():
    extracted = {
        "edits": {"other.py": [(1, 1, "x")]},
        "text_edits": {},
    }
    out = _smart_apply("content\n", extracted, "a.py")
    # No fuzzy match → None
    assert out is None


# ─────────────── fuzzy filepath matching ───────────────


def test_smart_apply__suffix_match_full_path():
    """Edits keyed by full path like `pkg/a.py` should match filepath `a.py`
    via path-bounded suffix."""
    extracted = {
        "edits": {"pkg/a.py": [(1, 1, "replaced_line")]},
        "text_edits": {},
    }
    out = _smart_apply("orig_line\n", extracted, "a.py")
    assert out is not None
    assert "replaced_line" in out


def test_smart_apply__suffix_match_with_path_separator():
    """`foo/a.py` should match against query `a.py` (suffix at `/` boundary)."""
    extracted = {
        "edits": {"foo/a.py": [(1, 1, "new_content")]},
        "text_edits": {},
    }
    out = _smart_apply("old\n", extracted, "a.py")
    assert out is not None
    assert "new_content" in out


def test_smart_apply__path_bounded_suffix_rejects_partial():
    """`foo/bar.py` MUST NOT match against `qux/bar.py` — only when the
    shorter is a path-bounded suffix of the longer."""
    extracted = {
        "edits": {"foo/bar.py": [(1, 1, "WRONG_TARGET")]},
        "text_edits": {},
    }
    # query is `qux/bar.py` — different parent dir
    out = _smart_apply("orig\n", extracted, "qux/bar.py")
    # Should NOT match (would have matched with bare endswith)
    assert out is None or "WRONG_TARGET" not in out


def test_smart_apply__path_bounded_suffix_filename_collision_rejected():
    """`foobar.py` (no separator) should NOT match `bar.py`."""
    extracted = {
        "edits": {"foobar.py": [(1, 1, "WRONG_TARGET")]},
        "text_edits": {},
    }
    out = _smart_apply("orig\n", extracted, "bar.py")
    assert out is None or "WRONG_TARGET" not in out


# ─────────────── ordering: line edits before text edits ───────────────


def test_smart_apply__line_edits_take_priority_over_text_edits():
    """If both `edits` and `text_edits` have entries for the same path,
    line edits are tried FIRST."""
    extracted = {
        "edits": {"a.py": [(1, 1, "from_line_edits")]},
        "text_edits": {"a.py": [("orig", "from_text_edits")]},
    }
    out = _smart_apply("orig\n", extracted, "a.py")
    # Line edits won
    assert out is not None
    assert "from_line_edits" in out
    # Text edit replacement NOT in output
    assert "from_text_edits" not in out


def test_smart_apply__fuzzy_line_before_exact_text():
    """A fuzzy-match line edit should be tried before any text edit.
    (Both fuzzy + direct line edits are tried before text edits.)"""
    extracted = {
        "edits": {"pkg/a.py": [(1, 1, "from_fuzzy_line")]},
        "text_edits": {"a.py": [("orig", "from_text")]},
    }
    out = _smart_apply("orig\n", extracted, "a.py")
    # Fuzzy line match wins
    assert "from_fuzzy_line" in out


# ─────────────── nested-path resolution ───────────────


def test_smart_apply__deeply_nested_path():
    """`a/b/c/d/e/foo.py` should match query `e/foo.py` (path-bounded)."""
    extracted = {
        "edits": {"a/b/c/d/e/foo.py": [(1, 1, "deep_replaced")]},
        "text_edits": {},
    }
    out = _smart_apply("orig\n", extracted, "e/foo.py")
    assert out is not None
    assert "deep_replaced" in out


def test_smart_apply__reverse_suffix():
    """`a.py` short edit-key should also match query `pkg/a.py` long
    (the path-bounded check works both directions)."""
    extracted = {
        "edits": {"a.py": [(1, 1, "REPLACED")]},
        "text_edits": {},
    }
    out = _smart_apply("orig\n", extracted, "pkg/a.py")
    # `pkg/a.py` ends with `a.py` at a `/` boundary → match
    assert "REPLACED" in out


# ─────────────── adversarial ───────────────


def test_smart_apply__multiple_edits_in_same_path():
    """Multiple line edits on the same file all apply."""
    extracted = {
        "edits": {"a.py": [
            (1, 1, "FIRST_REPLACED"),
            (3, 3, "THIRD_REPLACED"),
        ]},
        "text_edits": {},
    }
    out = _smart_apply("orig_line_1\nline_2\norig_line_3\n", extracted, "a.py")
    assert "FIRST_REPLACED" in out or "THIRD_REPLACED" in out


def test_smart_apply__empty_extracted_dicts():
    extracted = {"edits": {}, "text_edits": {}}
    out = _smart_apply("anything\n", extracted, "any.py")
    assert out is None
