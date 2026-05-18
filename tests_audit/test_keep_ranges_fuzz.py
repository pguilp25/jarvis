"""THIRD-PASS FUZZ audit of `_parse_keep_ranges` and `_filter_by_ranges`.

Verifies invariants across 100s of random inputs:
  I1. Output is sorted by start position.
  I2. No overlapping ranges in output (merge contract).
  I3. No gap-of-zero or gap-of-one pairs (merged or kept-separate per spec).
  I4. Inverted ranges (b<a) are filtered out.
  I5. Zero starts (a==0) are filtered out.
  I6. Idempotent: parsing the output formatted-as-string gives same result.
  I7. _filter_by_ranges line numbers preserved.
"""
import pytest
import random
import string
from workflows.code import _parse_keep_ranges, _filter_by_ranges


# ─────────────── PROPERTY: SORTED OUTPUT ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__sorted_output(seed):
    rng = random.Random(seed)
    n_ranges = rng.randint(0, 20)
    ranges = []
    for _ in range(n_ranges):
        a = rng.randint(1, 1000)
        b = a + rng.randint(0, 100)
        ranges.append(f"{a}-{b}")
    rng.shuffle(ranges)
    input_str = ", ".join(ranges)
    out = _parse_keep_ranges(input_str, "a.py")
    # Output sorted by start
    for i in range(len(out) - 1):
        assert out[i][0] <= out[i + 1][0]


# ─────────────── PROPERTY: NO OVERLAP IN OUTPUT ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__no_overlap_in_output(seed):
    rng = random.Random(seed)
    n_ranges = rng.randint(0, 20)
    ranges = []
    for _ in range(n_ranges):
        a = rng.randint(1, 1000)
        b = a + rng.randint(0, 50)
        ranges.append(f"{a}-{b}")
    out = _parse_keep_ranges(", ".join(ranges), "a.py")
    # No adjacent overlaps (after merge)
    for i in range(len(out) - 1):
        a1, b1 = out[i]
        a2, b2 = out[i + 1]
        # Second start must be > first end + 1 (gap ≥ 1, per merge spec)
        assert a2 > b1 + 1, f"Overlap: {out[i]} and {out[i+1]}"


# ─────────────── PROPERTY: INVERTED FILTERED OUT ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__inverted_ranges_dropped(seed):
    rng = random.Random(seed)
    # Mix valid and inverted
    ranges = []
    for _ in range(10):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        ranges.append(f"{a}-{b}")
    out = _parse_keep_ranges(", ".join(ranges), "a.py")
    # Output ranges all have start ≤ end
    for a, b in out:
        assert a <= b


# ─────────────── PROPERTY: ZERO STARTS FILTERED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__no_zero_starts(seed):
    rng = random.Random(seed)
    ranges = [f"0-{rng.randint(1, 100)}" for _ in range(5)]
    ranges += [f"{rng.randint(1, 100)}-{rng.randint(1, 200)}" for _ in range(5)]
    out = _parse_keep_ranges(", ".join(ranges), "a.py")
    for a, b in out:
        assert a > 0


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_idem__parse_format_parse(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 15)
    ranges = []
    for _ in range(n):
        a = rng.randint(1, 500)
        b = a + rng.randint(0, 50)
        ranges.append(f"{a}-{b}")
    original_str = ", ".join(ranges)
    parsed1 = _parse_keep_ranges(original_str, "a.py")
    # Format and re-parse
    if parsed1:
        formatted = ", ".join(f"{a}-{b}" for a, b in parsed1)
        parsed2 = _parse_keep_ranges(formatted, "a.py")
        assert parsed1 == parsed2


# ─────────────── PROPERTY: SEPARATORS DON'T MATTER ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__separator_variations_same_result(seed):
    rng = random.Random(seed)
    ranges = [
        (rng.randint(1, 100), rng.randint(100, 200))
        for _ in range(rng.randint(1, 8))
    ]
    range_strs = [f"{a}-{b}" for a, b in ranges]

    # Test various separators
    sep_variants = [
        ",", ", ", "  ,  ", " , ", "\n", "\t", " ",
    ]
    for sep in sep_variants:
        input_str = sep.join(range_strs)
        out = _parse_keep_ranges(input_str, "a.py")
        # All separators must produce identical output
        out_comma = _parse_keep_ranges(",".join(range_strs), "a.py")
        assert out == out_comma, f"Sep {sep!r} differs"


# ─────────────── PROPERTY: DUPLICATE INPUTS DEDUPED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__duplicate_ranges_deduped(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 100)
    b = a + rng.randint(0, 50)
    # Same range repeated 5 times
    out = _parse_keep_ranges(f"{a}-{b}, " * 5, "a.py")
    assert out == [(a, b)]


# ─────────────── PROPERTY: TRIVIAL CASES ───────────────


def test_trivial__empty():
    assert _parse_keep_ranges("", "a.py") == []


def test_trivial__whitespace():
    assert _parse_keep_ranges("   ", "a.py") == []


def test_trivial__just_text():
    assert _parse_keep_ranges("nothing here", "a.py") == []


# ─────────────── STRESS ───────────────


def test_stress__10000_ranges():
    """10000 distinct non-overlapping ranges."""
    ranges = ", ".join(f"{i * 100 + 1}-{i * 100 + 5}" for i in range(10000))
    out = _parse_keep_ranges(ranges, "a.py")
    assert len(out) == 10000


def test_stress__same_range_1000_times():
    """1000× same range → deduped to one."""
    out = _parse_keep_ranges("50-80, " * 1000, "a.py")
    assert out == [(50, 80)]


# ─────────────── PROPERTY: _filter_by_ranges LINE NUMBERS ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_filter__kept_lines_present(seed):
    rng = random.Random(seed)
    n_file_lines = rng.randint(20, 100)
    file_lines = [f"line_{i}" for i in range(1, n_file_lines + 1)]
    src = "\n".join(file_lines)
    # Pick a random valid range
    a = rng.randint(1, n_file_lines - 5)
    b = a + rng.randint(0, 5)
    out = _filter_by_ranges(src, [(a, b)], "a.py")
    # Lines in range present
    for i in range(a, b + 1):
        if i <= n_file_lines:
            assert f"line_{i}" in out


@pytest.mark.parametrize("seed", range(50))
def test_filter__outside_range_hidden(seed):
    """Lines outside any kept range should NOT appear (just hidden marker)."""
    rng = random.Random(seed)
    n_lines = rng.randint(50, 100)
    file_lines = [f"unique_line_id_{i}" for i in range(1, n_lines + 1)]
    src = "\n".join(file_lines)
    # Range in the middle, far from extremes
    mid = n_lines // 2
    a = mid - 2
    b = mid + 2
    out = _filter_by_ranges(src, [(a, b)], "a.py")
    # Line 1 should NOT appear in output (far from range)
    assert "unique_line_id_1" not in out
    # Line n_lines should NOT appear either
    assert f"unique_line_id_{n_lines}" not in out


# ─────────────── REGRESSION: KNOWN BUG CASES ───────────────


def test_regression__overlap_2_to_4_3_to_5():
    """Overlapping (2,4) and (3,5) → merge to (2,5)."""
    assert _parse_keep_ranges("2-4, 3-5", "a.py") == [(2, 5)]


def test_regression__adjacent_2_to_4_5_to_7():
    """(2,4) and (5,7): 5 == 4+1 → adjacent → merge."""
    assert _parse_keep_ranges("2-4, 5-7", "a.py") == [(2, 7)]


def test_regression__gap_2_to_4_6_to_8():
    """(2,4) and (6,8): gap of 1 (line 5 skipped) → NOT merged."""
    assert _parse_keep_ranges("2-4, 6-8", "a.py") == [(2, 4), (6, 8)]


def test_regression__chain_merge():
    """(1,10), (5,15), (12,20) — chain merge to (1,20)."""
    assert _parse_keep_ranges("1-10, 5-15, 12-20", "a.py") == [(1, 20)]
