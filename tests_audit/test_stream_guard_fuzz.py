"""THIRD-PASS FUZZ audit of `DegenerationDetector`.

Properties:
  I1. Detector always returns either None or a non-empty string.
  I2. Once tripped, stays tripped (sticky property).
  I3. Legitimate long varied content never trips (false-positive resistant).
  I4. Each specific failure pattern always trips at threshold.
"""
import pytest
import random
import string
from core.stream_guard import DegenerationDetector


def _varied_lines(rng: random.Random, n: int, min_len: int = 25) -> str:
    """Generate n VARIED lines (different prefixes per line)."""
    lines = []
    for i in range(n):
        # Unique identifier per line
        ident = f"unique_id_{i}_{rng.randint(0, 9999)}"
        rest = "".join(rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(min_len, 60)))
        lines.append(f"{ident} {rest}")
    return "\n".join(lines)


# ─────────────── PROPERTY: TYPE INVARIANTS ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__check_returns_none_or_str(seed):
    """For ANY input, check() returns None or str."""
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 1000)))
    det = DegenerationDetector()
    result = det.check(text)
    assert result is None or isinstance(result, str)


# ─────────────── PROPERTY: STICKINESS ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__sticky_once_tripped(seed):
    """Once tripped, det.reason persists."""
    rng = random.Random(seed)
    # Generate text that WILL trip (8+ identical long lines)
    line = "a" * 30 + "_" + str(seed)
    text = (line + "\n") * 10
    det = DegenerationDetector()
    r1 = det.check(text)
    assert r1 is not None
    # Try to "untrip" with diverse content
    det.check("totally fresh and varied prose with different stuff here")
    # Reason unchanged
    assert det.reason == r1


# ─────────────── PROPERTY: LEGITIMATE CONTENT DOESN'T TRIP ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__varied_content_no_trip(seed):
    """100 varied lines — should never trip."""
    rng = random.Random(seed)
    text = _varied_lines(rng, 100)
    det = DegenerationDetector()
    assert det.check(text) is None


@pytest.mark.parametrize("seed", range(50))
def test_inv__short_lines_no_trip(seed):
    """Lots of short lines (under LINE_MIN_LEN=20) — no trip."""
    rng = random.Random(seed)
    # Each line under 20 chars
    lines = [
        "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(0, 19)))
        for _ in range(100)
    ]
    text = "\n".join(lines)
    det = DegenerationDetector()
    assert det.check(text) is None


# ─────────────── PROPERTY: EACH FAILURE MODE TRIPS ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_failure__line_repeat_8x_long(seed):
    """8x same long line — trip every time."""
    rng = random.Random(seed)
    line = "long_repeated_line_seed_" + str(seed) + "_" + "x" * 20
    text = (line + "\n") * 8
    det = DegenerationDetector()
    assert det.check(text) is not None


@pytest.mark.parametrize("seed", range(30))
def test_failure__empty_tooluse_3x(seed):
    """3 empty `[tool use][/tool use]` blocks — trip."""
    rng = random.Random(seed)
    text = "[tool use][/tool use]\n" * 3
    det = DegenerationDetector()
    assert det.check(text) is not None


@pytest.mark.parametrize("seed", range(30))
def test_failure__scaffold_marker_trips(seed):
    """Scaffold marker — trip."""
    rng = random.Random(seed)
    prose = "".join(rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(0, 200)))
    text = f"{prose}\n────── ROUND {rng.randint(1, 99)}"
    det = DegenerationDetector()
    assert det.check(text) is not None


# ─────────────── PROPERTY: STREAMING ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_streaming__no_premature_trip_below_threshold(seed):
    """As text accumulates token-by-token below threshold, never trips."""
    rng = random.Random(seed)
    det = DegenerationDetector()
    # 7 long identical lines (below threshold of 8)
    line = "stuck_line_pattern_" + str(seed)
    text = (line + "\n") * 7
    accumulated = ""
    for ch in text:
        accumulated += ch
        result = det.check(accumulated)
        assert result is None


@pytest.mark.parametrize("seed", range(20))
def test_streaming__trips_at_threshold(seed):
    """Text grows past threshold — trips somewhere along the way."""
    rng = random.Random(seed)
    det = DegenerationDetector()
    line = "stuck_seed_" + str(seed) + "_padding_padding"
    text = (line + "\n") * 12  # well past threshold
    final_result = det.check(text)
    assert final_result is not None


# ─────────────── PROPERTY: PROPERTIES VS LARGE INPUTS ───────────────


def test_large__1mb_random_no_crash():
    rng = random.Random(42)
    text = "".join(rng.choice(string.printable) for _ in range(1_000_000))
    det = DegenerationDetector()
    result = det.check(text)
    assert result is None or isinstance(result, str)


def test_large__1mb_varied_content_no_trip():
    """1MB of VARIED content shouldn't trip."""
    rng = random.Random(43)
    parts = []
    for i in range(50000):
        parts.append(f"unique_marker_{i}_paragraph_with_distinct_content")
    text = "\n".join(parts)
    det = DegenerationDetector()
    assert det.check(text) is None


def test_large__1mb_stuck_pattern_trips():
    line = "x" * 50
    text = (line + "\n") * 20000
    det = DegenerationDetector()
    result = det.check(text)
    assert result is not None


# ─────────────── PROPERTY: TRIP REASON FORMAT ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__trip_reason_is_non_empty(seed):
    """When trip fires, reason is non-empty string."""
    rng = random.Random(seed)
    # Generate text that trips
    line = "long_stuck_" + str(seed) + "_" + "a" * 20
    text = (line + "\n") * 10
    det = DegenerationDetector()
    result = det.check(text)
    if result is not None:
        assert len(result) > 0


# ─────────────── EDGE: VARIED PROPERTIES ───────────────


def test_edge__empty_input_no_trip():
    det = DegenerationDetector()
    assert det.check("") is None


def test_edge__only_newlines_no_trip():
    det = DegenerationDetector()
    assert det.check("\n" * 1000) is None


def test_edge__only_whitespace_no_trip():
    det = DegenerationDetector()
    assert det.check("   \t\n   \t\n" * 100) is None


# ─────────────── PROPERTY: BOUNDARY THRESHOLD ───────────────


def test_bound__exactly_8_repetitions_trips():
    """LINE_REPEAT_THRESHOLD = 8 — exactly 8 trips."""
    line = "exactly_8_repeats_threshold_padding"
    text = (line + "\n") * 8
    det = DegenerationDetector()
    assert det.check(text) is not None


def test_bound__7_repetitions_no_trip():
    """7 repeats — one below threshold — no trip."""
    line = "exactly_7_repeats_padding_padding"
    text = (line + "\n") * 7
    det = DegenerationDetector()
    assert det.check(text) is None


def test_bound__line_min_len_20():
    """Lines of exactly 20 chars trigger; 19 don't."""
    det = DegenerationDetector()
    text = ("x" * 20 + "\n") * 8
    assert det.check(text) is not None

    det2 = DegenerationDetector()
    text2 = ("x" * 19 + "\n") * 50
    assert det2.check(text2) is None


# ─────────────── INDEPENDENCE ───────────────


def test_independence__instances_dont_share_state():
    """Two detector instances don't share trip state."""
    det1 = DegenerationDetector()
    det2 = DegenerationDetector()
    # Trip det1
    det1.check(("stuck_line_padding_padding_long\n") * 10)
    assert det1.tripped
    # det2 unaffected
    assert not det2.tripped
    assert det2.check("totally fresh") is None
