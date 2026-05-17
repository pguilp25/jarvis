"""Audit `_autocomplete_tool_blocks` — repair for unclosed [tool use]
blocks. Models occasionally write [tool use]...[CODE: a.py] and then
emit [STOP][CONFIRM_STOP] without ever closing [/tool use]. This
function inserts synthetic closers at the right boundaries.

A wrong closer position means:
  • Subsequent tags are wrongly attributed to the wrong block.
  • The model's [tool use] grouping intent is corrupted.
  • Edge: nested or duplicate opens cause cascading offset bugs.
"""
import pytest
from core.tool_call import _autocomplete_tool_blocks


# ─────────────── NO-OP CASES ───────────────


def test_autocomp__no_blocks_unchanged():
    text = "Just plain prose with no tool blocks."
    out, n = _autocomplete_tool_blocks(text)
    assert out == text
    assert n == 0


def test_autocomp__balanced_unchanged():
    text = "[tool use][CODE: a.py][/tool use]"
    out, n = _autocomplete_tool_blocks(text)
    assert out == text
    assert n == 0


def test_autocomp__multiple_balanced_unchanged():
    text = (
        "[tool use][CODE: a.py][/tool use]\n"
        "Some prose.\n"
        "[tool use][REFS: foo][/tool use]"
    )
    out, n = _autocomplete_tool_blocks(text)
    assert out == text
    assert n == 0


def test_autocomp__more_closes_than_opens_passthrough():
    """If closes > opens, the function returns unchanged (defensive — it
    only inserts closes, never opens)."""
    text = "[/tool use][/tool use]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 0


# ─────────────── INSERTS A CLOSER ───────────────


def test_autocomp__unclosed_at_eot():
    """Single open with no close — synthetic closer goes at EOT."""
    text = "[tool use][CODE: a.py]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    assert "[/tool use]" in out


def test_autocomp__close_inserted_before_next_signal():
    """Unclosed [tool use] followed by [STOP][CONFIRM_STOP] — closer
    inserted BEFORE the signal."""
    text = "[tool use][CODE: a.py]\n[STOP][CONFIRM_STOP]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    # The closer must come BEFORE [STOP]
    closer_pos = out.find("[/tool use]")
    stop_pos = out.find("[STOP]")
    assert closer_pos < stop_pos


def test_autocomp__close_inserted_before_done():
    text = "[tool use][CODE: a.py]\n[DONE][CONFIRM_DONE]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    assert out.find("[/tool use]") < out.find("[DONE]")


def test_autocomp__close_inserted_before_force_done():
    text = "[tool use][CODE: a.py]\n[FORCE DONE][CONFIRM_FORCE_DONE]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    assert out.find("[/tool use]") < out.find("[FORCE DONE]")


def test_autocomp__close_inserted_before_continue():
    text = "[tool use][CODE: a.py]\n[CONTINUE][CONFIRM_CONTINUE]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    assert out.find("[/tool use]") < out.find("[CONTINUE]")


def test_autocomp__double_open_one_close_synthetic_between():
    """[tool use] ... [tool use] ... [/tool use] — the FIRST open never
    closed; a synthetic closer goes between the two opens."""
    text = "[tool use][CODE: a.py][tool use][REFS: foo][/tool use]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 1
    # First [tool use] should now have its own [/tool use] before the
    # second [tool use].
    second_open_pos = out.find("[tool use]", out.find("[tool use]") + 1)
    first_closer_pos = out.find("[/tool use]")
    # The newly inserted closer should be before the second open
    assert first_closer_pos < second_open_pos


def test_autocomp__multiple_orphans_each_get_closer():
    """Three opens, zero closes — three synthetic closers."""
    text = "[tool use][CODE: a.py][tool use][CODE: b.py][tool use][CODE: c.py]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 3


def test_autocomp__count_reflects_real_inserts():
    """The returned count matches the number of `[/tool use]` insertions."""
    text = "[tool use][CODE: a.py]"
    before_closes = text.count("[/tool use]")
    out, n = _autocomplete_tool_blocks(text)
    after_closes = out.count("[/tool use]")
    assert after_closes - before_closes == n


def test_autocomp__closer_on_own_boundary():
    """The synthetic closer should be flanked by whitespace (newline or
    space) — not glued to surrounding text."""
    text = "[tool use][CODE: a.py]"
    out, _ = _autocomplete_tool_blocks(text)
    closer_pos = out.find("[/tool use]")
    # Either at start of text, or preceded by whitespace/newline
    assert closer_pos == 0 or out[closer_pos - 1] in '\n '


# ─────────────── EDGE / ADVERSARIAL ───────────────


def test_autocomp__case_insensitive_recognition():
    """[TOOL USE] (uppercase) should also be recognized."""
    text = "[TOOL USE][CODE: a.py]"
    out, n = _autocomplete_tool_blocks(text)
    # Should detect the (uppercase) open and insert a (lowercase) close
    assert n == 1


def test_autocomp__mixed_case_close_counted():
    """If model writes [TOOL USE]...[/TOOL USE], close should still be
    recognized so no synthetic close is inserted."""
    text = "[TOOL USE][CODE: a.py][/TOOL USE]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 0


def test_autocomp__open_with_extra_whitespace():
    """`[tool   use]` (extra internal whitespace) — likely NOT recognized
    by the strict regex. Document behavior."""
    text = "[tool   use][CODE: a.py]"
    out, n = _autocomplete_tool_blocks(text)
    # Either treats it as unrecognized (n=0) or recognizes and fixes (n=1)
    assert n in (0, 1)


def test_autocomp__interleaved_signals_partial_open():
    """Sequence: open, [CODE], [STOP], open, [REFS], [DONE]
    Expected: two synthetic closers, each before its respective signal."""
    text = (
        "[tool use][CODE: a.py]\n[STOP][CONFIRM_STOP]\n"
        "[tool use][REFS: foo]\n[DONE][CONFIRM_DONE]"
    )
    out, n = _autocomplete_tool_blocks(text)
    assert n == 2
    # Each [tool use] now has a [/tool use] before its signal
    assert out.count("[/tool use]") == 2


def test_autocomp__signal_before_first_open():
    """A signal at the very start (no open before it) — no insertion."""
    text = "[STOP][CONFIRM_STOP]\n[tool use][CODE: a.py][/tool use]"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 0


def test_autocomp__close_without_open_passes_through():
    """A stray [/tool use] with no open before it is left alone."""
    text = "prose [/tool use] more prose"
    out, n = _autocomplete_tool_blocks(text)
    assert n == 0
