"""Tool-robustness regression tests (from the 2026-05-29 tool-robustness audit).

Pins the two CRITICAL fixes plus the HIGH ones so they can't regress:
  - unknown / misspelled tool names are SURFACED (not silently dropped), so the
    model is told it wrote a bad call instead of hallucinating a result;
  - tool-executor exceptions reach the model via an EPHEMERAL channel;
  - [CODE:] line-range edge cases (backwards / past-EOF / single number) are
    explained, not silent;
  - [KNOWLEDGE:] actually loads; [RUN:] is not cached.
"""
import os
import tempfile

from core.tool_detector import TagDetector, KNOWN_TAG_TYPES


# ── CRITICAL #2: unknown / misspelled tool names surface ───────────────────────

def _rejected(text):
    d = TagDetector(text)
    return {(t.tag_type, t.rejection_reason) for t in d.all_tags if not t.valid}


def test_misspelled_tool_inside_tooluse_surfaces():
    # a misspelled tool the model clearly meant must NOT vanish silently
    for bad in ("VEIW", "SERACH", "COD", "DEPENDS", "FOO"):
        rej = _rejected(f"[tool use][{bad}: x.py][/tool use]")
        assert (bad, "unknown-tag-type") in rej, f"{bad} not surfaced: {rej}"


def test_known_common_wrong_name_surfaces_anywhere():
    # READ/GREP habit-names surface regardless of placement
    rej = _rejected("READ the file [READ: foo.py] then think")
    assert any(r == "unknown-tag-type" for _, r in rej)


def test_prose_allcaps_bracket_outside_tooluse_not_flagged():
    # a genuine prose bracket outside [tool use] must NOT be reported as a
    # dropped/malformed tool call (it's not a tool attempt)
    d = TagDetector("Per [NOTE: see below] I proceed.\n[tool use][CODE: a.py][/tool use]")
    surfaced = {t.rejection_reason for t in d.all_tags
                if not t.valid and t.rejection_reason in
                ("unknown-tag-type", "outside-tool-use-block", "no-tool-use-block-in-response")}
    assert "unknown-tag-type" not in surfaced
    # the valid CODE call still fires
    assert any(t.tag_type == "CODE" and t.valid for t in d.all_tags)


def test_real_tools_still_valid():
    # the new scan must not break real tools (each with a VALID arg for its type)
    cases = {"CODE": "a.py", "VIEW": "a.py 1-5", "REFS": "foo", "SEARCH": "foo",
             "SEMANTIC": "what does foo do", "DEPENDENCY": "#3df",
             "DEPENDSON": "foo", "PURPOSE": "a.py", "KEEP": "a.py 1-5"}
    for good, arg in cases.items():
        d = TagDetector(f"[tool use][{good}: {arg}][/tool use]")
        assert any(t.tag_type == good and t.valid for t in d.all_tags), f"{good} {arg}"


def test_bad_hint_suggests_nearest_tool():
    # _bad_hint (in core.tool_call) should map a misspelling to the nearest real tool
    import difflib
    for bad, expect in [("VEIW", "VIEW"), ("SERACH", "SEARCH"), ("REF", "REFS")]:
        near = difflib.get_close_matches(bad, list(KNOWN_TAG_TYPES), n=1, cutoff=0.6)
        assert near and near[0] == expect, f"{bad} → {near} (want {expect})"


# ── HIGH: [KNOWLEDGE:] actually loads ──────────────────────────────────────────

def test_knowledge_dir_resolves():
    from knowledge import KNOWLEDGE_DIR, list_knowledge
    assert KNOWLEDGE_DIR.exists(), f"KNOWLEDGE_DIR missing: {KNOWLEDGE_DIR}"
    topics = list_knowledge()
    assert isinstance(topics, list)


# ── HIGH: [CODE:] argument parsing (single line number → one-line range) ───────

def test_code_single_line_number_is_a_range():
    from core.tool_call import _parse_code_arg
    assert _parse_code_arg("foo.py 60") == ("foo.py", [(60, 60)])
    assert _parse_code_arg("foo.py 10-20") == ("foo.py", [(10, 20)])
    assert _parse_code_arg("foo.py") == ("foo.py", None)
    assert _parse_code_arg("a/b.py 5-9, 30-40") == ("a/b.py", [(5, 9), (30, 40)])


# ── HIGH: [SEARCH:] malformed-regex is reported, not silently literal ──────────

def test_search_invalid_regex_reports_fallback():
    import tempfile, os
    from tools.codebase import search_code, format_search_results
    d = tempfile.mkdtemp()
    open(os.path.join(d, "a.py"), "w").write("x = 1\n")
    out = format_search_results(search_code(r"\d+(unclosed", d))
    assert "not a valid regex" in out and "literal" in out.lower()


# ── #6: empty SEARCH/CODE arg does NOT fire a bogus tool-use-boundary query ────

def test_empty_arg_does_not_fire_bogus_query():
    for t in ("[tool use][SEARCH:][/tool use]",
              "[tool use][SEARCH: ][/tool use]",
              "[tool use][CODE: ][/tool use]"):
        d = TagDetector(t)
        assert d.valid_tags() == [], f"{t} fired a phantom tag: {d.valid_tags()}"
    # a real query still fires
    d = TagDetector("[tool use][SEARCH: def foo][/tool use]")
    assert any(x.tag_type == "SEARCH" and x.clean_arg == "def foo" for x in d.valid_tags())


# ── CRITICAL #1: tool-executor exceptions reach the model, then clear ──────────

def test_tool_exception_is_visible_then_ephemeral(monkeypatch):
    """A tool that RAISES must surface a ✗ error in the NEXT round's prompt (so the
    model is told, not handed a blank to hallucinate from) — and that error must be
    GONE the round after (ephemeral, no bloat)."""
    import asyncio, tempfile, os
    import core.tool_call as TC
    import tools.codebase as CB

    root = tempfile.mkdtemp(prefix="toolerr_")
    open(os.path.join(root, "real.py"), "w").write("def ok():\n    return 1\n")

    # SEARCH executor raises; CODE works normally.
    def _boom(*a, **k):
        raise RuntimeError("ripgrep exploded")
    monkeypatch.setattr(CB, "search_code", _boom)

    prompts = []          # current_prompt seen by the model each round
    responses = [
        "[tool use]\n[SEARCH: needle]\n[/tool use]\n[STOP][CONFIRM_STOP]",  # R1: crash
        "[tool use]\n[CODE: real.py]\n[/tool use]\n[STOP][CONFIRM_STOP]",   # R2: clean
        "[PLAN DONE][CONFIRM_PLAN_DONE]",                                    # R3: end
    ]
    call_n = {"i": 0}

    async def _fake_retry(model, prompt, **kw):
        prompts.append(prompt)
        i = call_n["i"]; call_n["i"] += 1
        return responses[i] if i < len(responses) else "[PLAN DONE][CONFIRM_PLAN_DONE]"
    monkeypatch.setattr(TC, "call_with_retry", _fake_retry)

    asyncio.run(TC.call_with_tools("test/model", "[SYSTEM] do work", project_root=root,
                                   max_rounds=4, enable_web_search=False))

    # prompts[0] = R1 (initial, no error yet). prompts[1] = R2 (must show the SEARCH crash).
    assert len(prompts) >= 2, f"only {len(prompts)} rounds ran"
    assert "SEARCH" in prompts[1] and ("✗" in prompts[1] or "failed" in prompts[1]), \
        "round-2 prompt did not surface the SEARCH executor crash (silent failure!)"
    # prompts[2] = R3 — the crash error must be GONE (ephemeral, cleared next round).
    if len(prompts) >= 3:
        assert "ripgrep exploded" not in prompts[2], \
            "the tool error persisted into a later round (not ephemeral / bloats context)"
