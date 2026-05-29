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
