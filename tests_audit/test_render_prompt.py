"""Drift guard for the render harness (ckpt-197) — and for the shared prompt-assembly functions
it exercises. The harness renders the FULL artifact the coder receives by calling the SAME functions
the live coder uses (build_implement_native_prompt + finalize_coder_system). If those drift from
what the model actually gets, these fail — which is the whole point (CLAUDE.md: verify the rendered
artifact, never let audited ≠ live)."""
import os

from behavioral_audit.render_prompt import assemble, render_growing_view
from workflows.code import build_implement_native_prompt, IMPLEMENT_NATIVE_PROMPT
from core.native_tools import finalize_coder_system, _INDENT_FORMAT_BLOCK, CODER_TOOLS


def test_assembled_system_is_live_prompt_plus_indent_block():
    art = assemble("STEP 1: do X.", "(none)", {"a/b.py": "x = 1\n"})
    # the system the model sees = the LIVE coder prompt + the always-on indent block
    assert IMPLEMENT_NATIVE_PROMPT in art["system"]
    assert _INDENT_FORMAT_BLOCK in art["system"]
    assert art["tools"] is CODER_TOOLS and len(art["tools"]) >= 10


def test_user_turn_carries_step_and_injected_file():
    art = assemble("STEP 1: rename foo.", "IFACE LINE", {"a/b.py": "def foo():\n    return 1\n"})
    assert "STEP 1: rename foo." in art["user"]
    assert "IFACE LINE" in art["user"]
    assert "a/b.py" in art["user"] and "def foo" in art["user"]   # injected in full
    assert art["injected"] == ["a/b.py"] and art["overflow"] == []


def test_small_files_inject_huge_file_overflows():
    big = "x = 1\n" * 60000   # ~360k chars > the 160k inject budget
    art = assemble("STEP", "(none)", {"a/small.py": "y = 1\n", "a/huge.py": big})
    assert "a/small.py" in art["injected"]
    assert "a/huge.py" in art["overflow"]                          # too big to preload
    assert "read on demand" in art["user"]


def test_error_feedback_appears_in_user_turn():
    art = assemble("STEP", "(none)", {"a/b.py": "x = 1\n"},
                   error_feedback="Your last edit was REJECTED.")
    assert "Your last edit was REJECTED." in art["user"] or "REJECTED" in art["system"]


def test_finalize_is_idempotent_in_structure():
    # finalize must always append the indent block exactly once for a fresh system
    s = finalize_coder_system("BASE")
    assert s.count(_INDENT_FORMAT_BLOCK) == 1 and s.startswith("BASE")


def test_growing_view_dedups_and_labels_holes(capsys):
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(900))   # 1800 lines
    render_growing_view(big, [(3, 8), (101, 106), (3, 6)], path="big.py")  # last is a re-read
    out = capsys.readouterr().out
    assert "GROWING VIEW" in out
    assert "[(3, 8), (101, 106)]" in out          # the covered re-read added nothing (dedup)
    assert "not read" in out and "contains" in out  # the gap is a labelled hole
