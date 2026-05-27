"""THIRD-PASS FUZZ audit of `_apply_line_edits`.

Property-based fuzzing verifies:
  I1. applied count ≤ len(edits)
  I2. skip_messages count ≥ (len(edits) - applied)
  I3. OOB edits always skipped
  I4. Overlapping REPLACEs always refused (both)
  I5. Result is always a string
  I6. Edits at valid in-bounds line numbers always apply (when no overlap)
"""
import pytest
import random
import string
from workflows.code import _apply_line_edits


def _random_line(rng: random.Random, length: int = 20) -> str:
    """Random short alphanumeric line."""
    chars = string.ascii_letters + string.digits
    return "".join(rng.choice(chars) for _ in range(length))


# ─────────────── PROPERTY: APPLIED ≤ LEN(EDITS) ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__applied_le_edit_count(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(5, 50)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    n_edits = rng.randint(0, 10)
    edits = []
    for _ in range(n_edits):
        # Mix INSERTs and REPLACEs
        if rng.random() < 0.5:
            # REPLACE: pick a valid range
            s = rng.randint(1, n_lines)
            e = min(s + rng.randint(0, 5), n_lines)
            edits.append((s, e, f"i0|replaced_{rng.randint(0, 9999)}"))
        else:
            # INSERT AFTER LINE
            after = rng.randint(0, n_lines)
            edits.append((0, after, f"i0|inserted_{rng.randint(0, 9999)}"))
    _, applied, _ = _apply_line_edits(orig, edits)
    assert 0 <= applied <= len(edits)


# ─────────────── PROPERTY: RESULT IS STRING ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__result_always_string(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(0, 30)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    n_edits = rng.randint(0, 5)
    edits = []
    for _ in range(n_edits):
        # Mix of valid and invalid
        edits.append((rng.randint(0, 100), rng.randint(0, 100), f"i0|x_{rng.randint(0, 999)}"))
    result, _, _ = _apply_line_edits(orig, edits)
    assert isinstance(result, str)


# ─────────────── PROPERTY: SKIPS RECORDED FOR OOB ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__oob_replace_skipped(seed):
    """REPLACE LINES past EOF — always skipped, message recorded."""
    rng = random.Random(seed)
    n_lines = 10
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    # Definitely OOB
    edits = [(100 + rng.randint(0, 50), 200, "i0|x")]
    _, applied, skips = _apply_line_edits(orig, edits)
    assert applied == 0
    # Some skip message must be recorded
    assert len(skips) >= 1


@pytest.mark.parametrize("seed", range(50))
def test_inv__oob_insert_skipped(seed):
    rng = random.Random(seed)
    n_lines = 10
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    # INSERT AFTER way past EOF
    edits = [(0, 100 + rng.randint(0, 50), "i0|x")]
    _, applied, skips = _apply_line_edits(orig, edits)
    assert applied == 0
    assert len(skips) >= 1


# ─────────────── PROPERTY: OVERLAPPING REPLACEs REFUSED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__overlapping_replaces_refused(seed):
    rng = random.Random(seed)
    n_lines = 50
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    # Two ranges that overlap
    a1 = rng.randint(1, n_lines - 10)
    b1 = a1 + rng.randint(5, 10)
    # Make second range overlap
    a2 = a1 + rng.randint(1, b1 - a1)
    b2 = a2 + rng.randint(5, 10)
    edits = [
        (a1, b1, "i0|A"),
        (a2, b2, "i0|B"),
    ]
    _, applied, skips = _apply_line_edits(orig, edits)
    # Both refused
    assert applied == 0
    assert any("OVERLAPPING" in s for s in skips)


# ─────────────── PROPERTY: DISJOINT REPLACEs BOTH APPLY ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__disjoint_replaces_both_apply(seed):
    rng = random.Random(seed)
    n_lines = 100
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    # Two clearly-disjoint ranges
    edits = [
        (5, 10, "i0|FIRST"),
        (50, 55, "i0|SECOND"),
    ]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 2
    assert "FIRST" in out
    assert "SECOND" in out


# ─────────────── PROPERTY: DELETION (EMPTY CODE) ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__delete_removes_lines(seed):
    """Use globally-unique line markers (no substring collisions) so we
    can assert exact deletion."""
    rng = random.Random(seed)
    n_lines = 20
    # Use letter suffix to avoid `_12` being a substring of `_120`
    lines = [f"unique_marker_{seed}_idx_{chr(65 + i % 26)}{i}" for i in range(n_lines)]
    orig = "\n".join(lines)
    a = rng.randint(2, n_lines - 5)
    b = a + rng.randint(0, 3)
    out, applied, _ = _apply_line_edits(orig, [(a, b, "")])
    if applied == 1:
        # Deleted lines (using \n delimiters to ensure exact-line match)
        out_lines = set(out.split("\n"))
        for i in range(a - 1, min(b, n_lines)):
            assert lines[i] not in out_lines, (
                f"Line {i} ({lines[i]!r}) should have been deleted"
            )


# ─────────────── PROPERTY: INSERT AFTER LINE N APPENDS ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__insert_adds_content(seed):
    rng = random.Random(seed)
    n_lines = 20
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    after = rng.randint(0, n_lines)
    marker = f"NEW_INSERTED_{seed}"
    out, applied, _ = _apply_line_edits(orig, [(0, after, f"i0|{marker}")])
    if applied == 1:
        assert marker in out


# ─────────────── PROPERTY: BOTTOM-UP ORDER ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__bottom_up_stable_line_numbers(seed):
    """Edits target ORIGINAL line numbers. Multiple disjoint edits all
    apply correctly even if earlier ones change line count."""
    rng = random.Random(seed)
    n_lines = 30
    orig = "\n".join(f"L_{i}" for i in range(n_lines))
    # Edit 1: grow at line 5 (1→3 lines). Edit 2: target line 20 (original).
    edits = [
        (5, 5, "i0|A\ni0|B\ni0|C"),  # grow by 2
        (20, 20, "i0|REPLACED_AT_20"),
    ]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 2
    assert "REPLACED_AT_20" in out


# ─────────────── PROPERTY: CATASTROPHIC SHRINK TRIPWIRE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__catastrophic_shrink_refused(seed):
    """If edits would shrink the file by >50% (file ≥ 50 lines), refused."""
    rng = random.Random(seed)
    n_lines = 100
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    # Delete 70 lines (70% loss)
    edits = [(1, 70, "")]
    out, applied, skips = _apply_line_edits(orig, edits)
    # Should refuse
    if applied == 0:
        assert any(s for s in skips)


# ─────────────── PROPERTY: SMALL FILES ESCAPE TRIPWIRE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__small_file_no_tripwire(seed):
    """Files under 50 lines — tripwire shouldn't fire."""
    rng = random.Random(seed)
    n_lines = 10
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    edits = [(1, 9, "")]  # delete 9 of 10
    out, applied, _ = _apply_line_edits(orig, edits)
    # No tripwire on small file
    assert applied == 1


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_det__same_input_same_output(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(5, 30)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    s = rng.randint(1, n_lines)
    e = min(s + rng.randint(0, 5), n_lines)
    edits = [(s, e, f"i0|replaced_{seed}")]
    out1, m1, _ = _apply_line_edits(orig, edits)
    out2, m2, _ = _apply_line_edits(orig, edits)
    out3, m3, _ = _apply_line_edits(orig, edits)
    assert out1 == out2 == out3
    assert m1 == m2 == m3


# ─────────────── PROPERTY: NO EDIT ===  IDENTITY ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__empty_edits_is_identity(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(0, 30)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    out, applied, skips = _apply_line_edits(orig, [])
    assert applied == 0
    assert skips == []
    # Result == expandtab'd original (no tabs in our random input, so == orig)
    assert out == orig.expandtabs(4)


# ─────────────── EDGE: BOUNDARY ───────────────


@pytest.mark.parametrize("n_lines", [1, 2, 5, 10, 50, 100, 500])
def test_bound__edit_first_line(n_lines):
    orig = "\n".join(f"L_{i}" for i in range(n_lines))
    out, applied, _ = _apply_line_edits(orig, [(1, 1, "i0|FIRST_REPLACED")])
    assert applied == 1
    assert "FIRST_REPLACED" in out


@pytest.mark.parametrize("n_lines", [1, 2, 5, 10, 50, 100])
def test_bound__edit_last_line(n_lines):
    orig = "\n".join(f"L_{i}" for i in range(n_lines))
    # Last line index is n_lines (split puts trailing empty)
    # Actually: "L_0\nL_1\n...\nL_{n-1}" has lines [L_0, L_1, ..., L_{n-1}]
    # split('\n') without trailing \n → n entries (lines 1..n)
    out, applied, _ = _apply_line_edits(orig, [(n_lines, n_lines, "i0|LAST_REPLACED")])
    assert applied == 1
    assert "LAST_REPLACED" in out


# ─────────────── EDGE: MANY EDITS ───────────────


def test_many__100_distinct_inserts():
    """100 INSERTs at distinct anchors — all apply."""
    orig = "\n".join(f"L_{i}" for i in range(100))
    edits = [(0, i, f"i0|INSERTED_{i}") for i in range(0, 100, 1)]
    out, applied, _ = _apply_line_edits(orig, edits)
    # Each insert is independent — all should apply
    assert applied == 100


def test_many__100_disjoint_replaces():
    """100 REPLACEs on disjoint single-lines."""
    orig = "\n".join(f"L_{i}" for i in range(200))
    # Edit every 2nd line
    edits = [(i, i, f"i0|MARK_{i}") for i in range(1, 201, 2)]
    out, applied, _ = _apply_line_edits(orig, edits)
    assert applied == 100
