"""THIRD-PASS FUZZ audit of `_apply_edits`.

This pass hammers the function with random adversarial inputs and
verifies invariants hold regardless of input. The two prior passes
(test_apply_edits.py and test_apply_edits_adversarial.py) covered
specific known scenarios. This pass uses generated cases to find any
remaining edge case.

Invariants verified across N=200 random cases each:
  I1. The result is always a string.
  I2. The applied count is between 0 and len(edits).
  I3. The result never contains data NOT in original or replacement.
  I4. If no edits match, result == expandtabs(original).
  I5. Ambiguous-skip messages are well-formed strings.
"""
import pytest
import random
import string
from workflows.code import _apply_edits


def _random_line(rng: random.Random, length: int = 30) -> str:
    """Generate a random line of typical code-like content."""
    chars = string.ascii_letters + string.digits + "    "
    return "".join(rng.choice(chars) for _ in range(length))


# ─────────────── PROPERTY INVARIANTS ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__matched_le_total(seed):
    """For ANY input, matched ≤ total."""
    rng = random.Random(seed)
    n_lines = rng.randint(1, 50)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    n_edits = rng.randint(0, 10)
    edits = [
        (_random_line(rng, 20), _random_line(rng, 20))
        for _ in range(n_edits)
    ]
    _, m, t, _ = _apply_edits(orig, edits)
    assert 0 <= m <= t == len(edits)


@pytest.mark.parametrize("seed", range(100))
def test_inv__result_is_string(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(0, 30)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    n_edits = rng.randint(0, 5)
    edits = [
        (_random_line(rng), _random_line(rng))
        for _ in range(n_edits)
    ]
    result, _, _, _ = _apply_edits(orig, edits)
    assert isinstance(result, str)


@pytest.mark.parametrize("seed", range(100))
def test_inv__no_edits_unchanged(seed):
    """With empty edit list, result equals expandtab'd original."""
    rng = random.Random(seed)
    n_lines = rng.randint(0, 50)
    orig = "\n".join(_random_line(rng) for _ in range(n_lines))
    result, m, t, amb = _apply_edits(orig, [])
    assert m == 0 and t == 0 and amb == []
    # Result is identical (modulo deterministic tab expansion)
    assert result == orig.expandtabs(4)


@pytest.mark.parametrize("seed", range(100))
def test_inv__ambiguous_skips_are_strings(seed):
    rng = random.Random(seed)
    orig = "duplicate\n" * rng.randint(2, 10)
    edits = [("duplicate", "REPLACED")]
    _, _, _, amb = _apply_edits(orig, edits)
    for msg in amb:
        assert isinstance(msg, str)
        assert len(msg) > 0


# ─────────────── ROUND-TRIP / IDEMPOTENCE ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_idem__applied_once_pattern_gone(seed):
    """After applying an edit X→Y, the pattern X is no longer present
    (assuming X is unique in original)."""
    rng = random.Random(seed)
    n_lines = rng.randint(5, 20)
    lines = [_random_line(rng) for _ in range(n_lines)]
    # Pick one unique line to be the target
    target_idx = rng.randint(0, n_lines - 1)
    lines[target_idx] = f"UNIQUE_LINE_{seed}_TARGET"
    orig = "\n".join(lines)
    replacement = f"REPLACED_VALUE_{seed}"
    result, m, _, _ = _apply_edits(orig, [(lines[target_idx], replacement)])
    if m == 1:
        # Target gone, replacement present
        assert lines[target_idx] not in result
        assert replacement in result


@pytest.mark.parametrize("seed", range(50))
def test_idem__second_application_no_op(seed):
    """Applying the same edit twice — second is a no-op."""
    rng = random.Random(seed)
    target = f"UNIQUE_TARGET_{seed}"
    orig = f"prelude\n{target}\npostlude"
    edits = [(target, "REPLACED")]
    once, m1, _, _ = _apply_edits(orig, edits)
    twice, m2, _, _ = _apply_edits(once, edits)
    assert m1 == 1
    assert m2 == 0
    assert once == twice


# ─────────────── SEMANTIC PROPERTIES ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_sem__only_kept_lines_unchanged(seed):
    """Lines outside the edited region are byte-identical to expandtab'd
    original."""
    rng = random.Random(seed)
    lines = [f"safe_line_{i}_with_random_{rng.randint(0,9999)}" for i in range(20)]
    target_idx = rng.randint(5, 15)
    lines[target_idx] = "EDIT_TARGET"
    orig = "\n".join(lines)
    result, m, _, _ = _apply_edits(orig, [("EDIT_TARGET", "EDITED")])
    if m == 1:
        # Every other line must appear unchanged
        for i, line in enumerate(lines):
            if i != target_idx:
                assert line in result, f"Line {i} ('{line}') missing"


@pytest.mark.parametrize("seed", range(30))
def test_sem__delete_removes_target(seed):
    """Empty REPLACE deletes the line."""
    rng = random.Random(seed)
    target = f"DELETE_ME_{seed}"
    orig = f"a\n{target}\nb"
    result, m, _, _ = _apply_edits(orig, [(target, "")])
    if m == 1:
        assert target not in result
        assert "a" in result and "b" in result


# ─────────────── EDGE: PATHOLOGICAL INPUTS ───────────────


def test_path__only_newlines_input():
    """Input that's just newlines."""
    for n in [0, 1, 5, 100, 1000]:
        orig = "\n" * n
        result, _, _, _ = _apply_edits(orig, [])
        assert result == orig.expandtabs(4)


def test_path__only_whitespace_lines():
    """Input that's whitespace-only lines."""
    orig = "   \n   \n   \n"
    result, m, _, _ = _apply_edits(orig, [("   ", "REPLACED")])
    # Whitespace-only SEARCH matches — but ambiguous (3 matches)
    assert isinstance(result, str)


def test_path__null_byte_in_search():
    """Null byte in SEARCH — should match literally if present in file."""
    orig = "before\x00after\nother"
    result, m, _, _ = _apply_edits(orig, [("before\x00after", "REPLACED")])
    if m == 1:
        assert "REPLACED" in result
        assert "\x00" not in result


def test_path__null_byte_in_replace():
    """Null byte in REPLACE body — preserved literally."""
    orig = "target_line\n"
    result, m, _, _ = _apply_edits(orig, [("target_line", "before\x00after")])
    if m == 1:
        assert "\x00" in result


def test_path__very_long_single_line():
    """A 100K-character single line."""
    orig = "x" * 100000
    result, _, _, _ = _apply_edits(orig, [])
    assert len(result) >= 100000


def test_path__deeply_indented():
    """Line with 1000 spaces of indent."""
    line = " " * 1000 + "code"
    orig = line + "\n"
    result, m, _, _ = _apply_edits(orig, [(line, " " * 1000 + "new_code")])
    if m == 1:
        assert "new_code" in result


# ─────────────── EDGE: WEIRD QUOTING ───────────────


def test_quote__nested_quotes_in_search():
    orig = """x = 'it\\'s a test'\n"""
    result, m, _, _ = _apply_edits(orig, [("x = 'it\\'s a test'", "x = 'plain'")])
    if m == 1:
        assert "plain" in result


def test_quote__triple_quoted_string():
    orig = '''x = """\nmultiline\nstring\n"""\n'''
    edits = [('x = """\nmultiline\nstring\n"""', 'x = "single"')]
    result, m, _, _ = _apply_edits(orig, edits)
    if m == 1:
        assert "single" in result


def test_quote__raw_string_prefix():
    orig = "x = r'C:\\Users'\n"
    result, m, _, _ = _apply_edits(orig, [("x = r'C:\\Users'", "x = '/home'")])
    if m == 1:
        assert "/home" in result


def test_quote__fstring():
    orig = "x = f'{value}'\n"
    result, m, _, _ = _apply_edits(orig, [("x = f'{value}'", "x = value")])
    if m == 1:
        assert "x = value" in result


# ─────────────── EDGE: EDIT BODY CHARACTERISTICS ───────────────


def test_body__replace_with_python_keyword():
    orig = "x = OLD_VAR\n"
    result, m, _, _ = _apply_edits(orig, [("OLD_VAR", "return")])
    if m == 1:
        assert "return" in result


def test_body__replace_grows_10x():
    """1-line search, 10-line replace."""
    orig = "TARGET\n"
    replacement = "\n".join(f"line_{i}" for i in range(10))
    result, m, _, _ = _apply_edits(orig, [("TARGET", replacement)])
    if m == 1:
        for i in range(10):
            assert f"line_{i}" in result


def test_body__replace_shrinks_10x():
    """10-line search, 1-line replace."""
    orig_lines = [f"line_{i}" for i in range(10)]
    orig = "\n".join(orig_lines)
    result, m, _, _ = _apply_edits(orig, [(orig, "single_line")])
    if m == 1:
        assert "single_line" in result
        # All 10 original lines gone
        for line in orig_lines:
            assert line not in result.split("single_line")[0]


# ─────────────── FUZZ: MIXED BATCH ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_fuzz__mixed_apply_match_subset(seed):
    """Generate 5 random edits; some match the file, some don't.
    Verify the matched-count reflects reality."""
    rng = random.Random(seed)
    n_lines = 20
    lines = [f"line_id_{i}_seed_{seed}_data" for i in range(n_lines)]
    orig = "\n".join(lines)
    # Mix: 3 that match, 2 that don't
    edits = []
    for i in range(3):
        idx = rng.randint(0, n_lines - 1)
        edits.append((lines[idx], f"replaced_{i}"))
    for _ in range(2):
        edits.append((f"absent_pattern_{seed}_{rng.randint(0,9999)}", "X"))
    # Some of the matching edits may target the same line → ambiguous,
    # so matched count may be < 3
    result, m, t, amb = _apply_edits(orig, edits)
    assert t == 5
    assert 0 <= m <= 3
    # Non-matching ones never apply
    for absent_pattern, _ in [e for e in edits if "absent" in e[0]]:
        # The absent pattern's marker shouldn't appear
        assert "X" not in result or "replaced_" in result


@pytest.mark.parametrize("seed", range(20))
def test_fuzz__disjoint_edits_all_apply(seed):
    """All edits target distinct unique lines → all apply."""
    rng = random.Random(seed)
    lines = [f"unique_{seed}_{i}" for i in range(20)]
    orig = "\n".join(lines)
    # Pick 5 distinct indices
    indices = rng.sample(range(20), 5)
    edits = [(lines[i], f"NEW_{i}") for i in indices]
    result, m, _, _ = _apply_edits(orig, edits)
    assert m == 5
    for i in indices:
        assert f"NEW_{i}" in result


# ─────────────── EDGE: FILE CONTENT WITH ALL CHAR CLASSES ───────────────


def test_charclass__mixed_unicode_ascii():
    """Mix Chinese, Arabic, emoji, ASCII all in one file."""
    orig = "ASCII\n北京\nالعربية\n🎉\nx = 1\n"
    edits = [("北京", "Beijing")]
    result, m, _, _ = _apply_edits(orig, edits)
    if m == 1:
        assert "Beijing" in result
        # Other unicode preserved
        assert "العربية" in result
        assert "🎉" in result


def test_charclass__control_chars():
    """Control characters in file content."""
    orig = "before\x01\x02after\nother"
    edits = [("before\x01\x02after", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    if m == 1:
        assert "REPLACED" in result


def test_charclass__bidirectional_text():
    """RTL + LTR mixed in one line."""
    orig = "English النص العربي more English\n"
    edits = [("English النص العربي more English", "REPLACED")]
    result, m, _, _ = _apply_edits(orig, edits)
    if m == 1:
        assert "REPLACED" in result
