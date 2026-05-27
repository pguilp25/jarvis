"""THIRD-PASS FUZZ audit of `_dedup_against_seen`.

Properties:
  I1. After first call, `seen` is populated with the keys of all kept items.
  I2. Same input on second call (with `seen` from first) returns EMPTY extracted.
  I3. Item normalization is consistent (trailing whitespace, CRLF, ./prefix,
      case all dedupe correctly).
  I4. Cross-kind items don't collide (text-edit vs line-edit vs new-file).
"""
import pytest
import random
import string
from workflows.code import _dedup_against_seen


def _empty_extracted():
    return {"text_edits": {}, "edits": {}, "new_files": {}}


def _random_path(rng: random.Random) -> str:
    """Generate a random path."""
    n_parts = rng.randint(1, 4)
    parts = []
    for _ in range(n_parts):
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
        parts.append(word)
    return "/".join(parts) + ".py"


def _random_text(rng: random.Random) -> str:
    """Generate random text content."""
    return "".join(rng.choice(string.ascii_letters + " \n") for _ in range(rng.randint(5, 50)))


# ─────────────── PROPERTY: REPEATED CALL DEDUPES ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_idem__same_input_second_call_empty(seed):
    """First call populates `seen`; second call with same input → all deduped."""
    rng = random.Random(seed)
    path = _random_path(rng)
    edit = (_random_text(rng), _random_text(rng))
    extracted1 = {
        "text_edits": {path: [edit]},
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted1, seen)
    # Second call with same edit
    extracted2 = {
        "text_edits": {path: [edit]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


# ─────────────── PROPERTY: DIFFERENT REPLACE = NEW ENTRY ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_distinct__different_replace_kept(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    find = _random_text(rng)
    replace1 = _random_text(rng) + "_v1"
    replace2 = _random_text(rng) + "_v2"
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [(find, replace1)]}, "edits": {}, "new_files": {}},
        seen,
    )
    extracted2 = {
        "text_edits": {path: [(find, replace2)]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    # Different replacement → kept
    assert path in extracted2["text_edits"]


# ─────────────── PROPERTY: FILEPATH NORMALIZATION ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_path_norm__dot_slash_dedupes(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    edit = ("X", "Y")
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [edit]}, "edits": {}, "new_files": {}},
        seen,
    )
    # Re-submit with ./ prefix
    extracted2 = {
        "text_edits": {"./" + path: [edit]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


@pytest.mark.parametrize("seed", range(50))
def test_path_norm__case_dedupes(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    edit = ("X", "Y")
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [edit]}, "edits": {}, "new_files": {}},
        seen,
    )
    extracted2 = {
        "text_edits": {path.upper(): [edit]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


@pytest.mark.parametrize("seed", range(50))
def test_path_norm__backslash_dedupes(seed):
    rng = random.Random(seed)
    path = _random_path(rng)  # forward slashes
    edit = ("X", "Y")
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [edit]}, "edits": {}, "new_files": {}},
        seen,
    )
    # Same path but with backslashes
    backslash_path = path.replace("/", "\\")
    extracted2 = {
        "text_edits": {backslash_path: [edit]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


# ─────────────── PROPERTY: CONTENT NORMALIZATION ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_content_norm__trailing_ws_dedupes(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    find = _random_text(rng)
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [(find, "REPL")]}, "edits": {}, "new_files": {}},
        seen,
    )
    # Same content with trailing whitespace
    extracted2 = {
        "text_edits": {path: [(find + "   ", "REPL  \n")]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


@pytest.mark.parametrize("seed", range(50))
def test_content_norm__crlf_dedupes(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    # Multi-line content
    find_lf = "line1\nline2\nline3"
    find_crlf = "line1\r\nline2\r\nline3"
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {path: [(find_lf, "X")]}, "edits": {}, "new_files": {}},
        seen,
    )
    extracted2 = {
        "text_edits": {path: [(find_crlf, "X")]},
        "edits": {},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    assert extracted2["text_edits"] == {}


# ─────────────── PROPERTY: CROSS-KIND DON'T COLLIDE ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_cross_kind__text_edit_and_line_edit_no_collide(seed):
    """A text_edit and a line_edit with same path and same body should
    have DIFFERENT dedup keys (different `kind` prefix)."""
    rng = random.Random(seed)
    path = _random_path(rng)
    body = _random_text(rng)
    extracted = {
        "text_edits": {path: [(body, body)]},
        "edits": {path: [(1, 1, body)]},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    # Both kinds survive
    assert path in extracted["text_edits"]
    assert path in extracted["edits"]


@pytest.mark.parametrize("seed", range(50))
def test_cross_kind__text_edit_and_new_file_no_collide(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    content = _random_text(rng)
    extracted = {
        "text_edits": {path: [(content, content)]},
        "edits": {},
        "new_files": {path: content},
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    assert path in extracted["text_edits"]
    assert path in extracted["new_files"]


# ─────────────── PROPERTY: DEDUP IDEMPOTENT ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_idem__dedup_twice_no_op_second(seed):
    """Applying dedup twice = applying once + emptying."""
    rng = random.Random(seed)
    path = _random_path(rng)
    edit = (_random_text(rng), _random_text(rng))
    extracted = {"text_edits": {path: [edit]}, "edits": {}, "new_files": {}}
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    # After first call, edit kept
    assert path in extracted["text_edits"]
    # Apply again on the SAME extracted dict (it's been mutated)
    _dedup_against_seen(extracted, seen)
    # Now deduped
    assert extracted["text_edits"] == {}


# ─────────────── PROPERTY: SAME EDIT MULTIPLE FILES ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_distinct__same_edit_different_files(seed):
    """Same (find, replace) on different files — both kept."""
    rng = random.Random(seed)
    edit = ("X", "Y")
    extracted = {
        "text_edits": {
            "file_a.py": [edit],
            "file_b.py": [edit],
        },
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    assert "file_a.py" in extracted["text_edits"]
    assert "file_b.py" in extracted["text_edits"]


# ─────────────── PROPERTY: LINE-EDIT POSITION DIFFERENCE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_line_edit__different_range_kept(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    code = _random_text(rng)
    seen: set = set()
    _dedup_against_seen(
        {"text_edits": {}, "edits": {path: [(10, 12, code)]}, "new_files": {}},
        seen,
    )
    extracted2 = {
        "text_edits": {},
        "edits": {path: [(20, 22, code)]},
        "new_files": {},
    }
    _dedup_against_seen(extracted2, seen)
    # Different range → kept
    assert path in extracted2["edits"]


# ─────────────── STRESS ───────────────


def test_stress__1000_edits_all_distinct():
    """1000 distinct edits — all kept."""
    extracted = {
        "text_edits": {
            f"file_{i}.py": [(f"find_{i}", f"replace_{i}")]
            for i in range(1000)
        },
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    assert len(extracted["text_edits"]) == 1000


def test_stress__1000_edits_all_dupes():
    """1000 copies of the same edit on the same file — only one kept."""
    extracted = {
        "text_edits": {
            "same.py": [("find", "replace")] * 1000
        },
        "edits": {},
        "new_files": {},
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    # After dedup, only 1 entry survives
    assert len(extracted["text_edits"]["same.py"]) == 1


# ─────────────── EMPTY ───────────────


def test_empty__all_empty_no_op():
    """Function may add a `reverts: []` key for downstream use — assert
    the dedup-relevant keys are empty after the call."""
    extracted = {"text_edits": {}, "edits": {}, "new_files": {}}
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    assert extracted["text_edits"] == {}
    assert extracted["edits"] == {}
    assert extracted["new_files"] == {}
    assert seen == set()


# ─────────────── PRESERVES OTHER KEYS ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__other_dict_keys_preserved(seed):
    """`extracted` may have other keys like `reverts`. Dedup must not
    touch them."""
    rng = random.Random(seed)
    extracted = {
        "text_edits": {"a.py": [("x", "y")]},
        "edits": {},
        "new_files": {},
        "reverts": ["b.py"],  # unrelated key
    }
    seen: set = set()
    _dedup_against_seen(extracted, seen)
    # `reverts` untouched
    assert extracted["reverts"] == ["b.py"]
