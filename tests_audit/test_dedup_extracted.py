"""Audit `_dedup_against_seen` — prevents the same edit being applied
twice when the streamed response_so_far is re-extracted at each [STOP].

This is the silent-corruption frontier:
  • Same edit hash → already applied → skip.
  • Cosmetic differences (trailing whitespace, CR/LF, path style) must
    NOT bypass dedup; otherwise the same edit re-applies and the file
    gains duplicate content.

The function mutates `extracted` IN PLACE and adds new keys to `seen_keys`.
"""
import pytest
from workflows.code import _dedup_against_seen


# ─────────────── text_edits dedup ───────────────


def test_dedup_text__first_call_keeps_all():
    extracted = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    out = _dedup_against_seen(extracted, seen)
    assert "a.py" in out["text_edits"]
    assert len(seen) == 1


def test_dedup_text__second_call_drops_dupes():
    extracted1 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    # Same edit again
    extracted2 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    # Same edit deduped — file key removed entirely since no edits left
    assert "a.py" not in out["text_edits"]


def test_dedup_text__different_replace_kept():
    extracted1 = {
        "text_edits": {"a.py": [("old", "new1")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a.py": [("old", "new2")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    # Different replacement → not a dup
    assert "a.py" in out["text_edits"]


def test_dedup_text__different_filepath_kept():
    extracted1 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"b.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "b.py" in out["text_edits"]


# ─────────────── filepath normalization ───────────────


def test_dedup__path_with_dot_slash_dedupes_as_same():
    """`a.py` and `./a.py` should hash to the same key."""
    extracted1 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"./a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "./a.py" not in out["text_edits"]


def test_dedup__path_case_insensitive():
    """`A.PY` and `a.py` dedup as same (lowercase normalization)."""
    extracted1 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"A.PY": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "A.PY" not in out["text_edits"]


def test_dedup__path_backslash_normalized():
    """Windows-style `a\\b.py` dedupes with `a/b.py`."""
    extracted1 = {
        "text_edits": {"a/b.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a\\b.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a\\b.py" not in out["text_edits"]


# ─────────────── content normalization ───────────────


def test_dedup__trailing_whitespace_normalized():
    """Trailing whitespace differences should NOT create a new key."""
    extracted1 = {
        "text_edits": {"a.py": [("old", "new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a.py": [("old   ", "new  \n")]},  # trailing ws
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a.py" not in out["text_edits"]


def test_dedup__crlf_lf_normalized():
    """CRLF and LF line endings normalize to the same key."""
    extracted1 = {
        "text_edits": {"a.py": [("old\nbody", "new\nbody")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a.py": [("old\r\nbody", "new\r\nbody")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a.py" not in out["text_edits"]


def test_dedup__blank_lines_ignored():
    """Differences in blank line count should normalize away."""
    extracted1 = {
        "text_edits": {"a.py": [("body line\n\nmore", "new\n\nbody")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a.py": [("body line\nmore", "new\nbody")]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    # Blank line dropped in _norm → same hash → dedup
    assert "a.py" not in out["text_edits"]


# ─────────────── line_edits dedup ───────────────


def test_dedup_lines__same_line_edit_deduped():
    extracted1 = {
        "text_edits": {},
        "edits": {"a.py": [(10, 12, "code")]},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {},
        "edits": {"a.py": [(10, 12, "code")]},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a.py" not in out["edits"]


def test_dedup_lines__different_line_range_kept():
    extracted1 = {
        "text_edits": {},
        "edits": {"a.py": [(10, 12, "code")]},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {},
        "edits": {"a.py": [(20, 22, "code")]},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a.py" in out["edits"]


def test_dedup_lines__different_code_kept():
    extracted1 = {
        "text_edits": {},
        "edits": {"a.py": [(10, 12, "code_v1")]},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {},
        "edits": {"a.py": [(10, 12, "code_v2")]},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "a.py" in out["edits"]


# ─────────────── new_files dedup ───────────────


def test_dedup_new_files__same_file_deduped():
    extracted1 = {
        "text_edits": {},
        "edits": {},
        "new_files": {"new.py": "content"},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {},
        "edits": {},
        "new_files": {"new.py": "content"},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "new.py" not in out["new_files"]


def test_dedup_new_files__different_content_kept():
    extracted1 = {
        "text_edits": {},
        "edits": {},
        "new_files": {"new.py": "content_v1"},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {},
        "edits": {},
        "new_files": {"new.py": "content_v2"},
    }
    out = _dedup_against_seen(extracted2, seen)
    assert "new.py" in out["new_files"]


# ─────────────── mixed ───────────────


def test_dedup__cross_kind_keys_distinct():
    """A text_edit and a line_edit with same content should NOT collide
    in dedup (different prefix in key)."""
    extracted = {
        "text_edits": {"a.py": [("body", "body")]},
        "edits": {"a.py": [(1, 1, "body")]},
        "new_files": {},
    }
    seen: set = set()
    out = _dedup_against_seen(extracted, seen)
    # Both kinds should survive
    assert "a.py" in out["text_edits"]
    assert "a.py" in out["edits"]


def test_dedup__empty_extracted_passes_through():
    extracted = {"text_edits": {}, "edits": {}, "new_files": {}}
    seen: set = set()
    out = _dedup_against_seen(extracted, seen)
    assert out["text_edits"] == {}
    assert out["edits"] == {}
    assert out["new_files"] == {}
    assert seen == set()


def test_dedup__multiple_edits_same_file_some_new():
    """A file with 2 edits, one new and one duplicate."""
    extracted1 = {
        "text_edits": {"a.py": [("v1_old", "v1_new")]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)

    extracted2 = {
        "text_edits": {"a.py": [
            ("v1_old", "v1_new"),  # duplicate
            ("v2_old", "v2_new"),  # new
        ]},
        "edits": {},
        "new_files": {},
    }
    out = _dedup_against_seen(extracted2, seen)
    # Only the new edit survives
    assert "a.py" in out["text_edits"]
    assert len(out["text_edits"]["a.py"]) == 1
    assert out["text_edits"]["a.py"][0] == ("v2_old", "v2_new")
