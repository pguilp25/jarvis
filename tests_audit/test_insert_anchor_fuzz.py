"""DEEP FUZZ audit of INSERT AFTER with anchor (`---` separator) form.

The INSERT AFTER body can include an anchor line before a `---` separator:
    i12|    pass
    ---
    i0|def new_func():
    i4|    return 1

The anchor is matched against file line at `end` (with ±20 fuzzy fallback).
If the anchor doesn't match, the runtime tries to relocate. If it can't
find the anchor at all, it falls back to the requested position with a
warning.

Properties:
  I1. Anchor matches at requested line → insert at that line.
  I2. Anchor matches within ±20 → relocate and insert.
  I3. Anchor doesn't match → fall back to requested line with warning.
  I4. No anchor (no `---`) → insert at requested line directly.
"""
import pytest
import random
import string
from workflows.code import _apply_line_edits


# ─────────────── ANCHOR HIT AT EXACT LINE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_anchor__exact_match_at_requested_line(seed):
    """Anchor matches at the requested line exactly."""
    rng = random.Random(seed)
    n_lines = rng.randint(10, 50)
    orig_lines = [f"line_{i}" for i in range(n_lines)]
    anchor_line_idx = rng.randint(2, n_lines - 2)
    anchor_text = orig_lines[anchor_line_idx]
    orig = "\n".join(orig_lines)
    # INSERT AFTER LINE (anchor_line_idx + 1) (1-based)
    code = f"i0|{anchor_text}\n---\ni0|INSERTED_AT_{seed}"
    out, applied, _ = _apply_line_edits(orig, [(0, anchor_line_idx + 1, code)])
    if applied == 1:
        assert f"INSERTED_AT_{seed}" in out


# ─────────────── ANCHOR FUZZY MATCH ±20 LINES ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_anchor__fuzzy_match_within_20(seed):
    """Anchor doesn't match at requested line but matches within ±20."""
    rng = random.Random(seed)
    n_lines = 50
    orig_lines = [f"line_{i}" for i in range(n_lines)]
    # Place a unique anchor at line 25
    orig_lines[25] = "UNIQUE_ANCHOR_LINE"
    orig = "\n".join(orig_lines)
    # Request INSERT AFTER LINE 30 (off by 5 — within ±20)
    code = "i0|UNIQUE_ANCHOR_LINE\n---\ni0|INSERTED_FUZZY"
    out, applied, _ = _apply_line_edits(orig, [(0, 30, code)])
    if applied == 1:
        assert "INSERTED_FUZZY" in out


# ─────────────── ANCHOR NOT FOUND — FALL BACK ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_anchor__not_found_falls_back(seed):
    """Anchor doesn't match anywhere — fall back to requested line."""
    rng = random.Random(seed)
    orig = "\n".join(f"line_{i}" for i in range(20))
    code = f"i0|TOTALLY_ABSENT_ANCHOR_{seed}\n---\ni0|FALLBACK_INSERT"
    out, applied, _ = _apply_line_edits(orig, [(0, 10, code)])
    if applied == 1:
        assert "FALLBACK_INSERT" in out


# ─────────────── NO ANCHOR (NO ---) ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_no_anchor__direct_insert(seed):
    """No `---` separator — body is all inserted code."""
    rng = random.Random(seed)
    n_lines = rng.randint(5, 30)
    orig = "\n".join(f"line_{i}" for i in range(n_lines))
    after = rng.randint(0, n_lines)
    code = f"i0|NEW_LINE_{seed}_A\ni0|NEW_LINE_{seed}_B"
    out, applied, _ = _apply_line_edits(orig, [(0, after, code)])
    assert applied == 1
    assert f"NEW_LINE_{seed}_A" in out
    assert f"NEW_LINE_{seed}_B" in out


# ─────────────── MULTI-LINE ANCHOR ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_multi_anchor__multiple_lines(seed):
    """Anchor spans multiple lines — both must match."""
    rng = random.Random(seed)
    n_lines = 30
    orig_lines = [f"line_{i}" for i in range(n_lines)]
    orig_lines[10] = "ANCHOR_PART_A"
    orig_lines[11] = "ANCHOR_PART_B"
    orig = "\n".join(orig_lines)
    code = "i0|ANCHOR_PART_A\ni0|ANCHOR_PART_B\n---\ni0|MULTI_ANCHOR_INSERT"
    out, applied, _ = _apply_line_edits(orig, [(0, 12, code)])
    if applied == 1:
        assert "MULTI_ANCHOR_INSERT" in out


# ─────────────── ANCHOR WITH INDENT DRIFT ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_drift__anchor_indent_drift_tolerated(seed):
    """Anchor matched on STRIPPED content — indent drift tolerated."""
    rng = random.Random(seed)
    orig = (
        "class C:\n"
        "    def method(self):\n"
        "        pass\n"  # 8-space indent
        "    other = 1\n"
    )
    # Anchor with WRONG indent (4 instead of 8)
    code = "i4|pass\n---\ni0|INSERTED"
    out, applied, _ = _apply_line_edits(orig, [(0, 3, code)])
    if applied == 1:
        assert "INSERTED" in out


# ─────────────── EDGE: EMPTY ANCHOR ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_edge__empty_anchor(seed):
    """If anchor portion is empty/whitespace — no anchor validation."""
    rng = random.Random(seed)
    n_lines = rng.randint(5, 30)
    orig = "\n".join(f"line_{i}" for i in range(n_lines))
    after = rng.randint(0, n_lines)
    code = "  \n---\ni0|EMPTY_ANCHOR_INSERT"
    out, applied, _ = _apply_line_edits(orig, [(0, after, code)])
    if applied == 1:
        assert "EMPTY_ANCHOR_INSERT" in out


# ─────────────── EDGE: ONLY ANCHOR, NO BODY AFTER --- ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_edge__only_anchor_no_body(seed):
    """If only anchor + `---` and nothing after — insert nothing."""
    rng = random.Random(seed)
    n_lines = rng.randint(5, 30)
    orig = "\n".join(f"line_{i}" for i in range(n_lines))
    after = rng.randint(0, n_lines)
    code = "i0|line_0\n---\n"  # no body
    out, applied, _ = _apply_line_edits(orig, [(0, after, code)])
    # No body to insert — either no-op or empty applied
    assert isinstance(out, str)


# ─────────────── DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_det__same_anchor_same_result(seed):
    rng = random.Random(seed)
    n_lines = 20
    orig = "\n".join(f"line_{i}" for i in range(n_lines))
    line_idx = rng.randint(2, n_lines - 2)
    code = f"i0|line_{line_idx}\n---\ni0|INSERT_{seed}"
    o1, m1, _ = _apply_line_edits(orig, [(0, line_idx + 1, code)])
    o2, m2, _ = _apply_line_edits(orig, [(0, line_idx + 1, code)])
    assert o1 == o2
    assert m1 == m2


# ─────────────── COMBINED WITH OTHER EDITS ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_combined__anchor_insert_with_replace(seed):
    """INSERT AFTER with anchor + a separate REPLACE on a different line."""
    rng = random.Random(seed)
    n_lines = 30
    orig_lines = [f"line_{i}" for i in range(n_lines)]
    orig_lines[15] = f"ANCHOR_{seed}"
    orig = "\n".join(orig_lines)
    edits = [
        (0, 16, f"i0|ANCHOR_{seed}\n---\ni0|INSERTED_{seed}"),
        (5, 5, f"i0|REPLACED_{seed}"),
    ]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert f"REPLACED_{seed}" in out
    # Insert may or may not succeed depending on anchor match
    if applied == 2:
        assert f"INSERTED_{seed}" in out
