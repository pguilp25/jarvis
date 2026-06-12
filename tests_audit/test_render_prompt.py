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
    # the system the model sees = the LIVE json-ops coder prompt + the always-on indent block +
    # the JSON-OPS protocol override. The IMPLEMENT prompt is present but with batch() neutralised
    # for json-ops, so assert its distinctive opening rather than the raw constant verbatim.
    assert IMPLEMENT_NATIVE_PROMPT[:80] in art["system"]          # live coder prompt, opening intact
    assert "JSON-OPS has NO batch tool" in art["system"]           # batch() neutralised for json-ops
    assert _INDENT_FORMAT_BLOCK in art["system"]
    assert art["tools"] == []   # json-ops PRIMARY coder is TEXT mode — no tools array sent


def test_all_system_no_user_turn_and_fenced():
    # ALL-SYSTEM (user 2026-06-11): the coder sends ONE system message and NO user turn; the
    # per-step task is wrapped in the [USER REQUEST] … [END OF USER REQUEST] fence.
    art = assemble("STEP 1: rename foo.", "IFACE LINE", {"a/b.py": "def foo():\n    return 1\n"})
    assert art["user"] == ""                                       # no user turn
    assert "[USER REQUEST]" in art["system"] and "[END OF USER REQUEST]" in art["system"]
    assert "STEP 1: rename foo." in art["system"]
    assert "IFACE LINE" in art["system"]
    assert "a/b.py" in art["system"] and "def foo" in art["system"]   # injected in full
    assert art["injected"] == ["a/b.py"] and art["overflow"] == []


def test_batch_neutralisation_does_not_corrupt_file_content():
    # BLOCKER caught in adversarial review (2026-06-11): the json-ops batch() neutralisation must
    # rewrite ONLY the IMPLEMENT prompt instructions, NEVER injected file content (the file view now
    # lives in the system message). A project file mentioning batch(calls)/with batch() must render
    # verbatim, or the coder copies a corrupted `old` and reject-thrashes.
    src = "async def go():\n    async with batch() as b:\n        b.add(batch(calls))\n"
    art = assemble("STEP: edit it", "(none)", {"a/q.py": src})
    assert "async with batch() as b" in art["system"]   # file content NOT rewritten
    assert "b.add(batch(calls))" in art["system"]        # file content NOT rewritten
    assert "JSON-OPS has NO batch tool" in art["system"]  # but the IMPLEMENT prompt's own CTA IS neutralised


def test_small_files_inject_huge_file_overflows():
    big = "x = 1\n" * 60000   # ~360k chars > the 160k inject budget
    art = assemble("STEP", "(none)", {"a/small.py": "y = 1\n", "a/huge.py": big})
    assert "a/small.py" in art["injected"]
    assert "a/huge.py" in art["overflow"]                          # too big to preload
    assert "TOO LARGE TO PRELOAD" in art["system"]


def test_big_target_file_routes_to_growing_view_not_full_inject():
    # ckpt-198: the a26 root cause. A file OVER the view cap (even the step's TARGET, even if it
    # fits the char budget) must NOT be dumped in full — it goes read-on-demand so its first read
    # opens the def-index and reads reveal ranges into the ONE growing view. (ckpt-196's view could
    # only fire on non-injected files; the big target was still dumped whole → it never fired.)
    big = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(700))   # ~2100 lines, <160k
    art = assemble("STEP: edit big.py", "(none)", {"pkg/big.py": big})
    assert art["injected"] == []                                   # NOT dumped in full
    assert art["overflow"] == ["pkg/big.py"]                       # routed to the growing view
    assert "TOO LARGE TO PRELOAD" in art["system"]
    assert "return x + 699" not in art["system"]                   # big-file BODY not inlined


def test_error_feedback_appears_in_system():
    art = assemble("STEP", "(none)", {"a/b.py": "x = 1\n"},
                   error_feedback="Your last edit was REJECTED.")
    assert "REJECTED" in art["system"]


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
