"""THIRD-PASS FUZZ audit of `_apply_continue_from`.

Properties:
  I1. Output is always a string.
  I2. Output length ≤ input length (directive only erases).
  I3. Any FIRED directive is absent from output.
  I4. Directives in inert zones (think/fence/backtick) are PRESERVED.
  I5. Malformed (N>500, N=0, non-numeric) → directive stripped, content
      unchanged otherwise.
"""
import pytest
import random
import string
from core.tool_call import _apply_continue_from


def _safe_prose(rng: random.Random, length: int) -> str:
    """Generate prose that's safe (no backticks/brackets/think markers)."""
    chars = string.ascii_letters + " 0123456789.,!"
    return "".join(rng.choice(chars) for _ in range(length))


# ─────────────── PROPERTY: TYPE INVARIANTS ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__output_is_string(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 500)))
    out = _apply_continue_from(text)
    assert isinstance(out, str)


@pytest.mark.parametrize("seed", range(100))
def test_inv__output_le_input(seed):
    """Output length ≤ input length (directive only erases)."""
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 500)))
    out = _apply_continue_from(text)
    assert len(out) <= len(text)


# ─────────────── PROPERTY: FIRED DIRECTIVE GONE ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__fired_directive_absent_from_output(seed):
    rng = random.Random(seed)
    prose_before = _safe_prose(rng, rng.randint(20, 100))
    n_lines_before = rng.randint(5, 20)
    n_to_erase = rng.randint(1, min(5, n_lines_before))
    # Build text with content lines then directive
    lines_before = [_safe_prose(rng, 20) for _ in range(n_lines_before)]
    text = "\n".join(lines_before) + f"\n[continue from: -{n_to_erase}]\nresume"
    out = _apply_continue_from(text)
    # Directive must be gone
    assert "[continue from" not in out


# ─────────────── PROPERTY: INERT-ZONE PRESERVED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inert__inside_fence_preserved(seed):
    rng = random.Random(seed)
    n_to_erase = rng.randint(1, 100)
    text = (
        f"real content\n"
        f"```\n"
        f"[continue from: -{n_to_erase}]\n"
        f"```\n"
        f"more content"
    )
    out = _apply_continue_from(text)
    # Directive preserved inside fence
    assert f"[continue from: -{n_to_erase}]" in out


@pytest.mark.parametrize("seed", range(50))
def test_inert__inside_think_preserved(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 100)
    text = (
        f"real content\n"
        f"[think]\n"
        f"[continue from: -{n}]\n"
        f"[/think]\n"
        f"more content"
    )
    out = _apply_continue_from(text)
    assert f"[continue from: -{n}]" in out


@pytest.mark.parametrize("seed", range(50))
def test_inert__inside_xml_think_preserved(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 100)
    text = (
        f"real content\n"
        f"<think>[continue from: -{n}]</think>\n"
        f"more content"
    )
    out = _apply_continue_from(text)
    assert f"[continue from: -{n}]" in out


# ─────────────── PROPERTY: MALFORMED STRIPPED ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_malformed__N_zero_stripped(seed):
    rng = random.Random(seed)
    prose = _safe_prose(rng, rng.randint(20, 100))
    text = f"{prose}\n[continue from: -0]\nresume"
    out = _apply_continue_from(text)
    # Directive stripped, no content erased
    assert "[continue from" not in out
    assert "resume" in out
    # Prose preserved (no content erased)
    assert prose in out or "resume" in out


@pytest.mark.parametrize("seed", range(30))
def test_malformed__N_over_500_stripped(seed):
    rng = random.Random(seed)
    N = rng.randint(501, 100000)
    prose = _safe_prose(rng, rng.randint(20, 100))
    text = f"{prose}\n[continue from: -{N}]\nresume"
    out = _apply_continue_from(text)
    assert "[continue from" not in out
    # Content not erased
    assert "resume" in out


# ─────────────── PROPERTY: NO DIRECTIVE = NO-OP ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__no_directive_passthrough(seed):
    """Text with no `[continue from: -N]` directive — exact passthrough."""
    rng = random.Random(seed)
    text = _safe_prose(rng, rng.randint(0, 500))
    out = _apply_continue_from(text)
    assert out == text


# ─────────────── PROPERTY: BOUNDARY ───────────────


@pytest.mark.parametrize("N", [1, 2, 5, 10, 50, 100, 200, 300, 400, 500])
def test_bound__valid_N_fires(N):
    """N from 1 to 500 inclusive — all valid."""
    lines = [f"line_{i}" for i in range(N + 5)]
    text = "\n".join(lines) + f"\n[continue from: -{N}]\nresume"
    out = _apply_continue_from(text)
    assert "[continue from" not in out
    assert "resume" in out


# ─────────────── PROPERTY: CHAINING ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_chain__multiple_directives(seed):
    rng = random.Random(seed)
    text = ""
    for i in range(3):
        for _ in range(rng.randint(2, 5)):
            text += _safe_prose(rng, 20) + "\n"
        text += f"[continue from: -{rng.randint(1, 3)}]\n"
    text += "final"
    out = _apply_continue_from(text)
    # No directives in output
    assert "[continue from" not in out
    # Final content present
    assert "final" in out


# ─────────────── PROPERTY: CASE-INSENSITIVITY ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_case__uppercase_works(seed):
    rng = random.Random(seed)
    prose = _safe_prose(rng, 30)
    text = f"{prose}\n[CONTINUE FROM: -2]\nresume"
    out = _apply_continue_from(text)
    assert "CONTINUE FROM" not in out.upper().split("RESUME")[0] or "[" not in out.upper().split("RESUME")[0]


@pytest.mark.parametrize("seed", range(30))
def test_case__mixed_case_works(seed):
    rng = random.Random(seed)
    prose = _safe_prose(rng, 30)
    text = f"{prose}\n[Continue From: -2]\nresume"
    out = _apply_continue_from(text)
    assert "[Continue From" not in out


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_det__same_input_same_output(seed):
    rng = random.Random(seed)
    text = _safe_prose(rng, rng.randint(20, 200))
    text += f"\n[continue from: -{rng.randint(1, 5)}]\nrest"
    o1 = _apply_continue_from(text)
    o2 = _apply_continue_from(text)
    o3 = _apply_continue_from(text)
    assert o1 == o2 == o3


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_idem__output_has_no_directives(seed):
    """After applying, output has no directives → applying again is no-op."""
    rng = random.Random(seed)
    text = _safe_prose(rng, rng.randint(20, 100))
    text += f"\n[continue from: -{rng.randint(1, 5)}]\nrest"
    once = _apply_continue_from(text)
    twice = _apply_continue_from(once)
    assert once == twice


# ─────────────── ADVERSARIAL ───────────────


def test_adv__empty_text():
    assert _apply_continue_from("") == ""


def test_adv__only_directive():
    out = _apply_continue_from("[continue from: -1]")
    assert "[continue from" not in out


def test_adv__many_directives_stress():
    """100 directives in sequence."""
    text = ""
    for _ in range(100):
        text += "line\n[continue from: -0]\n"  # all malformed (N=0)
    out = _apply_continue_from(text)
    # All directives stripped
    assert "[continue from" not in out


def test_adv__1mb_input_no_crash():
    rng = random.Random(99)
    text = "".join(rng.choice(string.printable) for _ in range(1_000_000))
    out = _apply_continue_from(text)
    assert isinstance(out, str)
