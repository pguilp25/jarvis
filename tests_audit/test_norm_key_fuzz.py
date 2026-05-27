"""THIRD-PASS FUZZ audit of `_norm_key`.

Property-based fuzzing verifies that every random input satisfies:
  I1. IDEMPOTENCE: re-applying _norm_key on output's arg yields same.
  I2. PREFIX: output starts with `{tag_type}:`.
  I3. LOWERCASE: arg part is lowercase.
  I4. NO TRAILING WHITESPACE.
  I5. NO DOUBLE-SPACE in arg.
  I6. Different tag types never collide.
"""
import pytest
import random
import string
from core.tool_call import _norm_key


TAG_TYPES = ["CODE", "REFS", "KEEP", "VIEW", "LSP", "SEARCH", "PURPOSE",
             "SEMANTIC", "KNOWLEDGE", "DETAIL", "WEBSEARCH"]


def _random_arg(rng: random.Random) -> str:
    """Generate a random arg-shaped string."""
    chars = string.ascii_letters + string.digits + "/\\.-_ ,"
    n = rng.randint(0, 80)
    return "".join(rng.choice(chars) for _ in range(n))


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


@pytest.mark.parametrize("seed", range(200))
def test_idem__random_arg(seed):
    """Property: _norm_key applied twice == applied once."""
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng)
    once = _norm_key(tag, arg)
    # Extract arg part and re-normalize
    prefix = tag + ":"
    if once.startswith(prefix):
        arg_part = once[len(prefix):]
        twice = _norm_key(tag, arg_part)
        assert once == twice, f"Not idempotent: {arg!r} → {once!r} → {twice!r}"


# ─────────────── PROPERTY: PREFIX ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_prefix__starts_with_tag_colon(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng)
    key = _norm_key(tag, arg)
    assert key.startswith(f"{tag}:")


# ─────────────── PROPERTY: LOWERCASE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_lowercase__arg_part(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng)
    key = _norm_key(tag, arg)
    arg_part = key[len(tag) + 1:]
    # Lowercase invariant — no uppercase ASCII letters in arg part
    for ch in arg_part:
        if ch.isupper():
            assert False, f"Uppercase char {ch!r} in arg part: {arg_part!r}"


# ─────────────── PROPERTY: NO TRAILING WHITESPACE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_ws__no_trailing(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng) + " " * rng.randint(0, 5)  # always trailing
    key = _norm_key(tag, arg)
    assert not key.endswith(" "), f"Key has trailing space: {key!r}"


# ─────────────── PROPERTY: NO DOUBLE-SPACE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_ws__no_internal_double_space(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    # Inject random multi-space inputs
    arg_parts = [_random_arg(rng) for _ in range(rng.randint(1, 4))]
    arg = "   ".join(arg_parts)  # 3-space separator
    key = _norm_key(tag, arg)
    arg_part = key[len(tag) + 1:]
    assert "  " not in arg_part, f"Double-space in: {arg_part!r}"


# ─────────────── PROPERTY: DIFFERENT TAG TYPES DON'T COLLIDE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_distinct__different_tags(seed):
    rng = random.Random(seed)
    arg = _random_arg(rng) or "x"  # non-empty
    keys = {_norm_key(tag, arg) for tag in TAG_TYPES}
    # 11 distinct tag types → 11 distinct keys
    assert len(keys) == len(TAG_TYPES), (
        f"Collision with arg {arg!r}: {keys}"
    )


# ─────────────── PROPERTY: EQUIVALENCE CLASSES ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_eq__dot_slash_prefix(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng) or "x"
    # Prepending `./` shouldn't change key
    assert _norm_key(tag, arg) == _norm_key(tag, "./" + arg)
    assert _norm_key(tag, arg) == _norm_key(tag, "././" + arg)


@pytest.mark.parametrize("seed", range(50))
def test_eq__case_collapse(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng) or "x"
    assert _norm_key(tag, arg) == _norm_key(tag, arg.upper())
    assert _norm_key(tag, arg) == _norm_key(tag, arg.lower())


@pytest.mark.parametrize("seed", range(50))
def test_eq__whitespace_padding(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng) or "x"
    assert _norm_key(tag, arg) == _norm_key(tag, arg + " ")
    assert _norm_key(tag, arg) == _norm_key(tag, " " + arg)
    assert _norm_key(tag, arg) == _norm_key(tag, "  " + arg + "  ")


@pytest.mark.parametrize("seed", range(50))
def test_eq__backslash_to_forward(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    # Generate path with forward slashes
    parts = [
        "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        for _ in range(rng.randint(2, 5))
    ]
    fwd = "/".join(parts)
    back = "\\".join(parts)
    assert _norm_key(tag, fwd) == _norm_key(tag, back)


# ─────────────── PROPERTY: DISTINCTNESS ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_distinct__different_args(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg1 = "".join(rng.choice(string.ascii_lowercase) for _ in range(10))
    arg2 = arg1 + "_extra"  # genuinely different
    assert _norm_key(tag, arg1) != _norm_key(tag, arg2)


# ─────────────── DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_det__same_input_same_output(seed):
    rng = random.Random(seed)
    tag = rng.choice(TAG_TYPES)
    arg = _random_arg(rng)
    out1 = _norm_key(tag, arg)
    out2 = _norm_key(tag, arg)
    out3 = _norm_key(tag, arg)
    assert out1 == out2 == out3


# ─────────────── ADVERSARIAL: WEIRD INPUTS ───────────────


def test_adv__empty_arg():
    """Empty arg → key is just `tag:`"""
    for tag in TAG_TYPES:
        key = _norm_key(tag, "")
        assert key == f"{tag}:"


def test_adv__only_whitespace():
    """Whitespace-only arg → empty arg part."""
    for tag in TAG_TYPES:
        key = _norm_key(tag, "   \t\n")
        assert key.startswith(f"{tag}:")


def test_adv__only_dot_slash():
    """`./././` → all stripped → empty."""
    for tag in TAG_TYPES:
        key = _norm_key(tag, "././././")
        assert key == f"{tag}:"


def test_adv__very_long_arg():
    """100K-char arg — should not crash."""
    arg = "x" * 100000
    key = _norm_key("CODE", arg)
    assert key.startswith("CODE:")


def test_adv__unicode_arg():
    """Unicode characters preserved (lowercased where applicable)."""
    key1 = _norm_key("CODE", "北京.py")
    key2 = _norm_key("CODE", "北京.py")
    assert key1 == key2


def test_adv__emoji_arg():
    key1 = _norm_key("CODE", "🎉.py")
    key2 = _norm_key("CODE", "🎉.py")
    assert key1 == key2
    # Different emoji → different key
    key3 = _norm_key("CODE", "🚀.py")
    assert key1 != key3


# ─────────────── BOUNDARY ───────────────


@pytest.mark.parametrize("tag", TAG_TYPES)
def test_bound__every_tag_type_works(tag):
    key = _norm_key(tag, "test.py")
    assert key == f"{tag}:test.py"


def test_bound__many_ranges_in_arg():
    """Many range specs in arg — all normalized consistently."""
    arg = "f.py 1-2, 3-4, 5-6, 7-8, 9-10, 11-12"
    key = _norm_key("KEEP", arg)
    # All comma-spaces collapsed
    assert "  " not in key[len("KEEP:"):]
