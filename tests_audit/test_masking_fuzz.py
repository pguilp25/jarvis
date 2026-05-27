"""THIRD-PASS FUZZ audit of `_mask_for_signals` / `_mask_quoted_tags`.

Property-based fuzzing: generate random inputs with embedded signals
and verify invariants:

  I1. LENGTH preserved: len(masked) == len(text) ALWAYS.
  I2. NEWLINE positions preserved: every \\n in text is at same index in masked.
  I3. CHAR-class preserved: ONLY `[` characters can change to \\x00.
  I4. IDEMPOTENCE: masking twice == masking once.
  I5. NON-BRACKET chars unchanged.
  I6. Signals inside ` ``` ... ``` ` ALWAYS masked.
  I7. Signals inside `<think>...</think>` ALWAYS masked.
  I8. Signals inside `[think]...[/think]` ALWAYS masked.
  I9. Signals inside `` `...` `` (inline backticks) ALWAYS masked.
"""
import pytest
import random
from core.tool_call import _mask_for_signals, _mask_quoted_tags


# ─────────────── PROPERTY-BASED FUZZ TESTING ───────────────


def _random_text(rng: random.Random, length: int) -> str:
    """Generate random text with mixed content."""
    chars = "abcdefghijklmnopqrstuvwxyz [](){}<>`\n"
    return "".join(rng.choice(chars) for _ in range(length))


@pytest.mark.parametrize("seed", range(100))
def test_inv__length_preserved(seed):
    """For ANY input, len(masked) == len(input)."""
    rng = random.Random(seed)
    text = _random_text(rng, rng.randint(0, 500))
    masked = _mask_for_signals(text)
    assert len(masked) == len(text)


@pytest.mark.parametrize("seed", range(100))
def test_inv__newlines_preserved(seed):
    rng = random.Random(seed)
    text = _random_text(rng, rng.randint(0, 500))
    masked = _mask_for_signals(text)
    # Every \n in text must be at same index in masked
    for i, ch in enumerate(text):
        if ch == '\n':
            assert masked[i] == '\n'


@pytest.mark.parametrize("seed", range(100))
def test_inv__only_bracket_can_change(seed):
    """For every position i, masked[i] either equals text[i] or text[i]=='['."""
    rng = random.Random(seed)
    text = _random_text(rng, rng.randint(0, 500))
    masked = _mask_for_signals(text)
    for i in range(len(text)):
        if masked[i] != text[i]:
            assert text[i] == '['
            assert masked[i] == '\x00'


@pytest.mark.parametrize("seed", range(100))
def test_inv__idempotent(seed):
    rng = random.Random(seed)
    text = _random_text(rng, rng.randint(0, 500))
    once = _mask_for_signals(text)
    twice = _mask_for_signals(once)
    assert once == twice


@pytest.mark.parametrize("seed", range(50))
def test_inv__quoted_tags_idempotent(seed):
    rng = random.Random(seed)
    text = _random_text(rng, rng.randint(0, 500))
    once = _mask_quoted_tags(text)
    twice = _mask_quoted_tags(once)
    assert once == twice


# ─────────────── INJECTED SIGNAL FUZZ ───────────────


SIGNAL_FORMS = [
    "[STOP][CONFIRM_STOP]",
    "[DONE][CONFIRM_DONE]",
    "[FORCE DONE][CONFIRM_FORCE_DONE]",
    "[CONTINUE][CONFIRM_CONTINUE]",
    "[PLAN DONE][CONFIRM_PLAN_DONE]",
]


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("signal", SIGNAL_FORMS)
def test_inv__signal_inside_fence_always_masked(seed, signal):
    """ANY random prose around a fence with a signal inside — signal masked."""
    rng = random.Random(seed)
    prose_before = _random_text(rng, rng.randint(0, 100))
    prose_after = _random_text(rng, rng.randint(0, 100))
    text = f"{prose_before}\n```\nx {signal} y\n```\n{prose_after}"
    masked = _mask_for_signals(text)
    # Find signal's first bracket position
    sig_pos = text.find(signal)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("signal", SIGNAL_FORMS)
def test_inv__signal_inside_think_xml_always_masked(seed, signal):
    rng = random.Random(seed)
    prose_before = _random_text(rng, rng.randint(0, 100))
    prose_after = _random_text(rng, rng.randint(0, 100))
    text = f"{prose_before}<think>{signal}</think>{prose_after}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(signal)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("signal", SIGNAL_FORMS)
def test_inv__signal_inside_think_bracket_always_masked(seed, signal):
    rng = random.Random(seed)
    prose_before = _random_text(rng, rng.randint(0, 100))
    prose_after = _random_text(rng, rng.randint(0, 100))
    text = f"{prose_before}[think]{signal}[/think]{prose_after}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(signal)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("signal", SIGNAL_FORMS)
def test_inv__signal_inside_inline_backtick_always_masked(seed, signal):
    """Generated prose uses SAFE chars (no backticks/brackets) so we
    control the backtick-pairing in the test. Inline backticks must
    not span newlines, so prose stays on one line."""
    rng = random.Random(seed)
    safe = "abcdefghijklmnopqrstuvwxyz0123456789 "
    prose_before = "".join(rng.choice(safe) for _ in range(rng.randint(0, 50)))
    text = f"{prose_before} `{signal}` more"
    masked = _mask_for_signals(text)
    sig_pos = text.find(signal)
    assert masked[sig_pos] != '['


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("signal", SIGNAL_FORMS)
def test_inv__signal_OUTSIDE_quoted_zones_visible(seed, signal):
    """A signal OUTSIDE any quoted zone is VISIBLE."""
    rng = random.Random(seed)
    # Build prose without any backticks, fences, think markers
    safe_chars = "abcdefghijklmnopqrstuvwxyz \n"
    prose_before = "".join(rng.choice(safe_chars) for _ in range(rng.randint(0, 100)))
    prose_after = "".join(rng.choice(safe_chars) for _ in range(rng.randint(0, 100)))
    text = f"{prose_before}\n{signal}\n{prose_after}"
    masked = _mask_for_signals(text)
    sig_pos = text.find(signal)
    assert masked[sig_pos] == '[', (
        f"Signal {signal!r} should be visible but was masked at pos {sig_pos}"
    )


# ─────────────── NESTING FUZZ ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__multiple_signal_kinds_in_quoted_all_masked(seed):
    """All 5 signals inside a fenced block — all masked."""
    rng = random.Random(seed)
    all_signals = " ".join(rng.sample(SIGNAL_FORMS, k=5))
    text = f"```\n{all_signals}\n```"
    masked = _mask_for_signals(text)
    for sig in SIGNAL_FORMS:
        sig_pos = text.find(sig)
        if sig_pos >= 0:
            assert masked[sig_pos] != '[', f"{sig} not masked"


@pytest.mark.parametrize("seed", range(30))
def test_inv__alternating_quoted_unquoted_signals(seed):
    """Mix of quoted and unquoted signals — quoted hidden, unquoted visible."""
    rng = random.Random(seed)
    quoted = rng.choice(SIGNAL_FORMS)
    unquoted = rng.choice(SIGNAL_FORMS)
    text = f"```\n{quoted}\n```\nreal: {unquoted}"
    masked = _mask_for_signals(text)
    q_pos = text.find(quoted)
    u_pos = text.find(unquoted, q_pos + 1) if quoted == unquoted else text.find(unquoted)
    if u_pos >= 0 and q_pos >= 0 and q_pos != u_pos:
        assert masked[q_pos] != '['
        assert masked[u_pos] == '['


# ─────────────── ADVERSARIAL: SPECIFIC ATTACK VECTORS ───────────────


def test_adv__signal_split_across_streamed_chunks():
    """A signal split: model streams `[STOP]` THEN `[CONFIRM_STOP]` arrives
    separately. If we get only the first half, no fire."""
    # Simulate: text contains only [STOP] (no CONFIRM yet)
    text = "partial response then [STOP]"
    masked = _mask_for_signals(text)
    # The bracket is visible but the signal regex won't match
    from core.tool_call import STOP_TAG
    assert STOP_TAG.search(masked) is None


def test_adv__signal_inside_unclosed_think_during_stream():
    """`<think>...streaming... [STOP][CONFIRM_STOP]` — unclosed at EOT
    means the model hasn't finished thinking. Signal must be masked."""
    text = "<think>I'm reasoning about [STOP][CONFIRM_STOP] which means stop"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    assert masked[sig_pos] != '['


def test_adv__double_open_think_one_close():
    """Two `<think>` opens with only one `</think>` close — second think is
    unclosed, signal inside masked."""
    text = "<think>first thought</think> ok then <think>second [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    assert masked[sig_pos] != '['


def test_adv__signal_immediately_after_close_visible():
    """`</think>[STOP][CONFIRM_STOP]` — signal AFTER close → visible."""
    text = "<think>internal</think>[STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    assert masked[sig_pos] == '['


def test_adv__signal_pretending_to_be_in_fence():
    """`'''[STOP][CONFIRM_STOP]'''` — triple-SINGLE-quote (Python docstring)
    NOT the same as triple-BACKTICK (markdown fence). Signal NOT masked."""
    text = "'''[STOP][CONFIRM_STOP]'''"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    # Triple single quotes are NOT a fence — signal visible
    assert masked[sig_pos] == '['


def test_adv__backtick_inside_backtick_doesnt_break():
    """`` `` `` — double backticks (markdown inline code with backtick). """
    text = "Code: ``literally`a`backtick`` then [STOP][CONFIRM_STOP]"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    # The signal is outside the backtick zones
    assert masked[sig_pos] == '['


def test_adv__case_insensitive_think_tag():
    """`<THINK>[STOP][CONFIRM_STOP]</THINK>` — uppercase."""
    text = "<THINK>[STOP][CONFIRM_STOP]</THINK>"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    assert masked[sig_pos] != '['


def test_adv__mixed_case_bracket_think():
    text = "[ThInK][STOP][CONFIRM_STOP][/tHiNk]"
    masked = _mask_for_signals(text)
    sig_pos = text.find("[STOP]")
    assert masked[sig_pos] != '['


def test_adv__massive_random_input_no_crash():
    """1MB of random data — should not crash."""
    rng = random.Random(12345)
    text = _random_text(rng, 1_000_000)
    masked = _mask_for_signals(text)
    assert len(masked) == len(text)


# ─────────────── BOUNDARY EDGE CASES ───────────────


def test_boundary__exactly_one_char_input():
    """Single character — must not crash."""
    for ch in "[](){}<>`abc \n":
        masked = _mask_for_signals(ch)
        assert len(masked) == 1


def test_boundary__empty_string():
    assert _mask_for_signals("") == ""
    assert _mask_quoted_tags("") == ""


def test_boundary__only_open_bracket():
    text = "["
    masked = _mask_for_signals(text)
    # No tool-use enforcement, no quoted context → visible
    assert masked == "["


def test_boundary__only_open_then_close_quoted():
    """`\\[`  — escaped bracket → masked."""
    text = "\\["
    masked = _mask_for_signals(text)
    assert masked[1] != '['  # masked


# ─────────────── DETERMINISM ───────────────


def test_det__same_input_same_output():
    """Multiple calls on same input → identical results."""
    text = "<think>[STOP][CONFIRM_STOP]</think> [DONE][CONFIRM_DONE]"
    out1 = _mask_for_signals(text)
    out2 = _mask_for_signals(text)
    out3 = _mask_for_signals(text)
    assert out1 == out2 == out3


def test_det__different_instances_same_function():
    """The function has no global state — independent calls don't interfere."""
    text_a = "[think]A[/think]"
    text_b = "[think]B[/think]"
    out_a = _mask_for_signals(text_a)
    out_b = _mask_for_signals(text_b)
    out_a2 = _mask_for_signals(text_a)
    assert out_a == out_a2
    # B doesn't leak into A's result
    assert "B" not in out_a
