"""Salvage a plan that a reasoning model wrote inside the thinking channel.

When a planner/merger reasons its way to the correct plan entirely inside
<think> (or [think]) and emits a thin/empty visible body, _strip_think would
zero it and the run would discard a correct plan. _salvage_plan_from_think
recovers it. (Root-caused on pylint-4551, where glm-5.1's correct plan lived in
<think> and was thrown away for a weaker structured draft.)
"""
from core.tool_call import _salvage_plan_from_think, _strip_think


def test_salvage_lifts_from_plan_marker_in_native_think():
    txt = ("<think>let me weigh the options... the fix belongs in inspector.py\n"
           "=== PLAN ===\n## TASK SHAPE: FIX\n### STEP 1: edit inspector.py\n"
           "=== END PLAN ===</think>")
    out = _salvage_plan_from_think(txt)
    assert out.startswith("=== PLAN ===")
    assert "STEP 1" in out


def test_salvage_lifts_from_bracket_think():
    txt = "[think]reasoning...\n## GOAL: fix the bug\n### STEP 1: do X[/think]"
    out = _salvage_plan_from_think(txt)
    assert out.startswith("## GOAL")


def test_salvage_returns_full_reasoning_when_no_marker():
    txt = "<think>the bug is in cbook.py Grouper, weakref pickle issue, fix __getstate__</think>"
    out = _salvage_plan_from_think(txt)
    assert "cbook.py" in out and "Grouper" in out


def test_salvage_empty_when_no_think():
    assert _salvage_plan_from_think("just visible text, no think block") == ""


def test_strip_then_salvage_complementary():
    # the exact failure shape: rich think, ~empty visible
    txt = ("<think>=== PLAN ===\n### STEP 1: edit inspector.py and utils.py\n"
           "=== END PLAN ===</think>\n[PLAN DONE][CONFIRM_PLAN_DONE]")
    stripped = _strip_think(txt).strip()
    # after stripping think + signals, the visible body is essentially empty
    assert "STEP 1" not in stripped
    # but salvage recovers the real plan
    salvaged = _salvage_plan_from_think(txt)
    assert "STEP 1" in salvaged and "inspector.py" in salvaged


def test_strip_handles_malformed_close_bracket():
    # owl-alpha typo'd the close as `[/think>` on a26; the [think] reasoning must
    # still be stripped, not leak into the plan body the coder reads.
    txt = "[think]planner meta noise, tools unavailable[/think>\n## GOAL: fix it\n### STEP 1: edit urls.py"
    out = _strip_think(txt).strip()
    assert "planner meta noise" not in out
    assert out.startswith("## GOAL")
    assert "STEP 1" in out


def test_strip_handles_close_with_inner_whitespace():
    txt = "[think] reasoning [/ think ]\nvisible plan"
    out = _strip_think(txt).strip()
    assert "reasoning" not in out
    assert out == "visible plan"


def test_salvage_recovers_from_malformed_close():
    # consistency invariant: if strip would zero a plan-INSIDE [think] with a
    # malformed close, salvage must still pull it back (same close forms).
    txt = "[think]reasoning...\n## GOAL: fix the bug\n### STEP 1: do X[/think>"
    assert "STEP 1" not in _strip_think(txt)           # strip zeroed it
    out = _salvage_plan_from_think(txt)                # salvage recovers it
    assert out.startswith("## GOAL") and "STEP 1" in out


def test_salvage_caps_overlong_think_dump():
    # a 70K-char think dump must NOT come back as a giant "plan" (pylint-4551)
    from core.tool_call import _salvage_plan_from_think, _SALVAGE_MAX_CHARS
    huge = "<think>" + ("blah reasoning words " * 5000) + "</think>"
    out = _salvage_plan_from_think(huge)
    assert len(out) <= _SALVAGE_MAX_CHARS + 100   # capped (+ small note)
    assert "tail kept" in out
