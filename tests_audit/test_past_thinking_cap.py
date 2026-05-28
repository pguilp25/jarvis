"""The text tool loop must bound past-thinking growth (stability audit #4).

_past_thinking_keep_from picks the first round to keep so the NEWEST rounds fit a
token budget — preventing a long loop from growing the prompt into a silent
HTTP-400 context overflow. Tested with a deterministic word-count token fn.
"""
from core.tool_call import _past_thinking_keep_from


def _count(text):
    return len(text.split())


def test_everything_fits_keeps_all():
    rounds = ["a b c", "d e", "f"]
    assert _past_thinking_keep_from(rounds, cap_tokens=10_000, count_fn=_count) == 0


def test_single_round_never_dropped():
    assert _past_thinking_keep_from(["huge " * 1000], cap_tokens=1, count_fn=_count) == 0


def test_empty_is_zero():
    assert _past_thinking_keep_from([], cap_tokens=10, count_fn=_count) == 0


def test_over_budget_keeps_newest_drops_oldest():
    # 5 rounds, each ~10 tokens; cap fits ~2 newest rounds (+marker overhead)
    rounds = [" ".join(["w"] * 10) for _ in range(5)]
    keep = _past_thinking_keep_from(rounds, cap_tokens=25, count_fn=_count)
    assert keep > 0                      # something was elided
    assert keep < len(rounds)            # but not everything
    # the kept tail must fit the budget
    kept = rounds[keep:]
    assert _count("\n".join(kept)) <= 25 + 50 * len(kept)


def test_always_keeps_at_least_last_round():
    # even if the last round alone exceeds the cap, it is retained
    rounds = ["small", " ".join(["big"] * 1000)]
    keep = _past_thinking_keep_from(rounds, cap_tokens=5, count_fn=_count)
    assert keep == len(rounds) - 1       # only the last round kept
