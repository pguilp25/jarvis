"""THIRD-PASS FUZZ audit of `_restore_replace_whitespace`.

Property-based fuzzing verifies:
  I1. IDEMPOTENCE: applying twice == once.
  I2. LINE-COUNT preservation OR controlled growth (mid-line packing splits).
  I3. CONTENT preservation: every non-prefix character in input survives in output.
  I4. The i{N}| prefix is REPLACED by N spaces (authoritative).
  I5. No regex-injection: special chars in content survive.
"""
import pytest
import random
import string
from workflows.code import _restore_replace_whitespace


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


@pytest.mark.parametrize("seed", range(200))
def test_idem__random_content(seed):
    """For every random content, _restore twice == once."""
    rng = random.Random(seed)
    n_lines = rng.randint(1, 20)
    parts = []
    for _ in range(n_lines):
        indent = rng.randint(0, 16)
        content = "".join(rng.choice(string.ascii_letters + " =()") for _ in range(rng.randint(1, 40)))
        parts.append(f"i{indent}|{content}")
    text = "\n".join(parts)
    once = _restore_replace_whitespace(text)
    twice = _restore_replace_whitespace(once)
    assert once == twice, f"Not idempotent: {text[:200]}"


@pytest.mark.parametrize("seed", range(100))
def test_idem__plain_text(seed):
    """Text with NO i{N}| markers should also be idempotent."""
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    once = _restore_replace_whitespace(text)
    twice = _restore_replace_whitespace(once)
    assert once == twice


# ─────────────── PROPERTY: i{N}| PREFIX EXPANDS TO N SPACES ───────────────


@pytest.mark.parametrize("n", [0, 1, 2, 4, 8, 12, 16, 20, 32, 64, 100, 256])
def test_prefix__expands_to_n_spaces(n):
    text = f"i{n}|content"
    out = _restore_replace_whitespace(text)
    assert out == " " * n + "content"


@pytest.mark.parametrize("seed", range(100))
def test_prefix__exactly_n_spaces_emitted(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 100)
    content = "x" * rng.randint(0, 50)
    text = f"i{n}|{content}"
    out = _restore_replace_whitespace(text)
    # Leading whitespace count == n
    leading_spaces = len(out) - len(out.lstrip(' '))
    assert leading_spaces == n


# ─────────────── PROPERTY: NON-PREFIX CONTENT SURVIVES ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_content__survives_after_prefix(seed):
    rng = random.Random(seed)
    indent = rng.randint(0, 16)
    # Generate content that doesn't start with whitespace or i{N}|
    safe_chars = string.ascii_letters + string.digits + "=+*-/<>()[]{}"
    content = "".join(rng.choice(safe_chars) for _ in range(rng.randint(1, 50)))
    text = f"i{indent}|{content}"
    out = _restore_replace_whitespace(text)
    # All content chars present
    for ch in content:
        assert ch in out


# ─────────────── PROPERTY: MID-LINE PACKING SPLITS ───────────────


@pytest.mark.skip(reason="v8 `iN|content` line-PACKING (multiple segments on one "
                  "physical line) was removed in the format-B/whitespace migration — "
                  "the model now emits `N:content` whitespace, never packed iN|, so "
                  "_restore_replace_whitespace no longer splits packed segments.")
@pytest.mark.parametrize("seed", range(50))
def test_pack__mid_line_splits_into_separate_lines(seed):
    """A packed line like `i0|a()i4|b()` should produce 2 output lines."""
    rng = random.Random(seed)
    n_segments = rng.randint(2, 8)
    indents = [rng.randint(0, 16) for _ in range(n_segments)]
    contents = [
        "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 20)))
        for _ in range(n_segments)
    ]
    text = "".join(f"i{i}|{c}" for i, c in zip(indents, contents))
    out = _restore_replace_whitespace(text)
    # Should split into n_segments lines
    lines = out.split('\n')
    assert len(lines) >= n_segments


# ─────────────── PROPERTY: NO PACKING WHEN LINE DOESN'T START WITH iN| ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_nopack__regular_line_unchanged(seed):
    """Lines that don't start with i{N}| should NOT be split, even if they
    contain i{N}| mid-string."""
    rng = random.Random(seed)
    prose = "".join(rng.choice(string.ascii_letters + " ") for _ in range(20))
    text = f"{prose} i4|literal_text in_middle"
    out = _restore_replace_whitespace(text)
    # Should NOT be split
    assert out == text


# ─────────────── REGRESSION: TRAILING-LINENO HANDLING ───────────────


@pytest.mark.parametrize("digits", range(1, 7))
def test_trailer__strip_after_paren_close(digits):
    """foo() <digits> — strip when 1..6 digits trail after paren-close."""
    text = f"i0|foo() {'1' * digits}"
    out = _restore_replace_whitespace(text)
    # Trailer stripped
    assert '1' * digits not in out
    assert "foo()" in out


def test_trailer__7_digit_NOT_stripped():
    """`\\d{1,6}` bound — 7 digits won't match."""
    text = "i0|foo() 1234567"
    out = _restore_replace_whitespace(text)
    assert "1234567" in out


@pytest.mark.parametrize("seed", range(50))
def test_trailer__return_value_NOT_stripped(seed):
    """`return N` for various N — N never stripped (no statement-end char before)."""
    rng = random.Random(seed)
    val = rng.randint(0, 100)
    text = f"i4|return {val}"
    out = _restore_replace_whitespace(text)
    assert str(val) in out
    assert "return" in out


@pytest.mark.parametrize("seed", range(50))
def test_trailer__assignment_value_NOT_stripped(seed):
    """`x = N` — N is the value, not stripped."""
    rng = random.Random(seed)
    val = rng.randint(0, 1000)
    text = f"i0|x = {val}"
    out = _restore_replace_whitespace(text)
    assert str(val) in out


# ─────────────── EDGE: PURE TRAILER LINES ───────────────


@pytest.mark.parametrize("digits", range(1, 7))
def test_pure__digits_only_becomes_blank(digits):
    """`i0| N` (just digits, blank-line trailer) → becomes blank."""
    text = f"i0| {'1' * digits}"
    out = _restore_replace_whitespace(text)
    assert out == ""  # truly empty


# ─────────────── EDGE: UNICODE PRESERVATION ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_unicode__preserved(seed):
    rng = random.Random(seed)
    unicode_chars = ["北京", "العربية", "🎉", "🚀", "ümlaut", "café"]
    text = f"i4|name = '{rng.choice(unicode_chars)}'"
    out = _restore_replace_whitespace(text)
    # Original unicode preserved
    for u in unicode_chars:
        if u in text:
            assert u in out


# ─────────────── EDGE: LARGE INPUTS ───────────────


def test_edge__1000_lines_with_prefix():
    """1000 lines all with i{N}| prefix."""
    text = "\n".join(f"i{i % 16}|line_{i}" for i in range(1000))
    out = _restore_replace_whitespace(text)
    assert out.count('\n') == 999  # 1000 lines = 999 newlines


def test_edge__long_indent():
    """i1000|content — 1000 spaces of indent."""
    text = "i1000|content"
    out = _restore_replace_whitespace(text)
    assert out == " " * 1000 + "content"


def test_edge__very_long_content():
    """Single line with 10K-char content."""
    content = "x" * 10000
    text = f"i0|{content}"
    out = _restore_replace_whitespace(text)
    assert out == content


# ─────────────── EDGE: SPECIAL CHARS ───────────────


def test_special__regex_chars_in_content_preserved():
    """Regex metachars must pass through literally."""
    text = "i0|pattern = '.*'"
    out = _restore_replace_whitespace(text)
    assert "'.*'" in out


def test_special__backslash_in_content():
    text = "i0|path = 'C:\\\\Users'"
    out = _restore_replace_whitespace(text)
    assert "C:\\\\Users" in out


def test_special__pipe_in_content():
    """Content containing `|` (not as part of i{N}| prefix)."""
    text = "i0|x = a | b"
    out = _restore_replace_whitespace(text)
    assert "a | b" in out


# ─────────────── DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_det__same_input_same_output(seed):
    rng = random.Random(seed)
    indent = rng.randint(0, 16)
    content = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 40)))
    text = f"i{indent}|{content}"
    out1 = _restore_replace_whitespace(text)
    out2 = _restore_replace_whitespace(text)
    out3 = _restore_replace_whitespace(text)
    assert out1 == out2 == out3


# ─────────────── BOUNDARY ───────────────


def test_bound__empty_string():
    assert _restore_replace_whitespace("") == ""


def test_bound__single_newline():
    assert _restore_replace_whitespace("\n") == "\n"


def test_bound__only_prefix_no_content():
    """i4| with no content."""
    text = "i4|"
    out = _restore_replace_whitespace(text)
    assert out == "    "  # just 4 spaces


def test_bound__empty_content_lines():
    """Multiple lines, each just a prefix."""
    text = "i0|\ni4|\ni8|"
    out = _restore_replace_whitespace(text)
    lines = out.split('\n')
    assert lines[0] == ""
    assert lines[1] == "    "
    assert lines[2] == "        "
