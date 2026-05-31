"""Tracked regression tests for the NATIVE tool-calling coder (core/native_tools).

The native path drives gpt-oss-120b via structured function calls instead of
JARVIS's text protocol. These tests pin the contract every tool must honour:
full-parity tool surface, the n| (LINENO:INDENT|code) read/edit format, the
indent-count expansion, and — critically for stability — that every failure
mode returns a clear string instead of crashing the coder.

Async tools are driven via asyncio.run (the repo has no pytest-asyncio), matching
the pattern in tests_audit/test_remaining.py.
"""
import asyncio
import os
import shutil
import tempfile
import textwrap

import pytest

from core.native_tools import _dispatch, CODER_TOOLS
from tools.sandbox import Sandbox

_HAS_RG = shutil.which("rg") is not None

SRC = textwrap.dedent('''\
    """A tiny module for testing the native coder tools."""


    def greet(name):
        """Return a greeting for name."""
        return "hello " + name


    class Counter:
        """Counts things."""

        def __init__(self):
            self.n = 0

        def bump(self):
            self.n = self.n + 1
            return self.n
''')


def _mk_ctx():
    root = tempfile.mkdtemp(prefix="nativetool_")
    rel = "mod.py"
    with open(os.path.join(root, rel), "w") as f:
        f.write(SRC)
    sb = Sandbox(root)
    sb.setup()
    sb.load_file(rel)
    ctx = {"file_contents": {rel: SRC}, "sandbox": sb, "project_root": root,
           "viewed_versions": {}, "purpose_map": "", "detailed_map": "",
           "files_changed": set()}
    return ctx, rel, root


def _cleanup(root):
    shutil.rmtree(root, ignore_errors=True)


def _disp(name, args, ctx):
    return asyncio.run(_dispatch(name, args, ctx))


# ── tool surface ─────────────────────────────────────────────────────────────
def test_schemas_cover_full_toolset():
    names = {t["function"]["name"] for t in CODER_TOOLS}
    assert names == {"read_file", "find_refs", "find_callers", "search_text",
                     "file_purpose", "semantic_search", "depends_on",
                     "edit_file", "create_file", "replace_lines", "finish"}


def test_every_schema_is_wellformed():
    for t in CODER_TOOLS:
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        # every required field must be declared in properties
        for req in params.get("required", []):
            assert req in params["properties"], (fn["name"], req)


# ── read_file: n| format + ranges + skeleton ────────────────────────────────
def test_read_file_prefix_format():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel}, ctx)
        assert ":0|" in out          # LINENO:INDENT| with 0 indent at module level
        assert "greet" in out
    finally:
        _cleanup(root)


def test_read_file_shows_indent_count():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel}, ctx)
        # body lines are indented 4/8 spaces → INDENT count appears
        assert ":4|" in out or ":8|" in out
    finally:
        _cleanup(root)


def test_read_file_range():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel, "start_line": 4, "end_line": 6}, ctx)
        assert "greet" in out
    finally:
        _cleanup(root)


def test_read_file_missing_path_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
    finally:
        _cleanup(root)


def test_read_file_nonexistent_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": "does/not/exist.py"}, ctx)
        assert isinstance(out, str)   # must return, not raise
    finally:
        _cleanup(root)


# ── replace_lines: apply, indent-count expansion, rejects ────────────────────
def test_replace_lines_applies_and_expands_indent():
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)   # read-before-edit contract
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '4|return "hi " + name'}, ctx)
        assert out.startswith("✓ Applied")
        assert ctx["file_contents"][rel].split("\n")[5] == '    return "hi " + name'
        assert rel in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_raw_indent_passthrough():
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '    return "yo " + name'}, ctx)
        assert out.startswith("✓ Applied")
        assert ctx["file_contents"][rel].split("\n")[5] == '    return "yo " + name'
    finally:
        _cleanup(root)


def test_replace_lines_with_seeded_viewed_version_applies():
    # The workflow seeds viewed_versions from the file block it injects into the
    # prompt, so the coder's FIRST replace_lines lands without a wasted read
    # round (rough-edge #1). Simulate that seeding here.
    ctx, rel, root = _mk_ctx()
    try:
        ctx["viewed_versions"][rel] = ctx["file_contents"][rel]
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '4|return "hi " + name'}, ctx)
        assert out.startswith("✓ Applied")
        assert rel in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_on_nonexistent_file_rejected_no_junk():
    # The pylint-4551 nonsense: a placeholder step targeted a non-existent file
    # ("main"), the coder issued replace_lines, and the applier FABRICATED a junk
    # file from the REPLACE bodies (leaking 0|/4| prefixes) and marked it success.
    # Now it must REJECT and create NO file.
    ctx, rel, root = _mk_ctx()
    try:
        # exact failing shape: an empty placeholder entry (as phase_implement set
        # file_contents["main"]="" + seeded viewed_versions with it)
        ctx["file_contents"]["ghost.py"] = ""
        ctx["viewed_versions"]["ghost.py"] = ""
        out = _disp("replace_lines",
                    {"path": "ghost.py", "start_line": 1, "end_line": 1,
                     "new_content": "0|# Entry point\nif __name__ == '__main__':"}, ctx)
        assert isinstance(out, str) and out.startswith("✗"), out
        assert "ghost.py" not in ctx["files_changed"]
        # the applier must NOT have fabricated the file on the sandbox
        assert (ctx["sandbox"].load_file("ghost.py") or "") == ""
    finally:
        _cleanup(root)


def test_create_file_is_the_path_for_new_files():
    # the correct way to make a new file — create_file, not replace_lines
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("create_file",
                    {"path": "brand_new.py", "content": "x = 1\n"}, ctx)
        assert out.startswith("✓ Created")
        assert "brand_new.py" in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_without_read_is_rejected_not_crash():
    # The applier requires a prior read (viewed_versions gate). Editing blind
    # must reject cleanly with guidance, never crash or silently corrupt.
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '4|return "hi " + name'}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        assert rel not in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_bad_range_rejects_cleanly():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 9999, "end_line": 99999,
                     "new_content": "0|x = 1"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
    finally:
        _cleanup(root)


def test_replace_lines_missing_args_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("replace_lines", {"path": rel}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
    finally:
        _cleanup(root)


def test_replace_lines_noninteger_lines_no_crash():
    # a non-numeric start_line must reject cleanly, NEVER raise (would kill the run)
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": "L6", "end_line": "L6",
                     "new_content": "x = 1"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        assert "integer" in out.lower()
    finally:
        _cleanup(root)


def test_read_file_noninteger_lines_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel, "start_line": "abc", "end_line": "def"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        assert "integer" in out.lower()
    finally:
        _cleanup(root)


def test_read_file_inverted_range_is_not_silent():
    # start_line > end_line currently yields a header with an EMPTY body and no
    # signal — the model can hallucinate. It must say the range is invalid.
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel, "start_line": 5, "end_line": 2}, ctx)
        assert isinstance(out, str)
        low = out.lower()
        assert "invalid" in low or "start_line must be ≤ end_line" in low or \
               "no lines" in low, f"silent empty range: {out!r}"
    finally:
        _cleanup(root)


def test_read_file_out_of_range_is_not_silent():
    # start_line beyond EOF yields a header with an empty body and no signal.
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel, "start_line": 9000, "end_line": 9001}, ctx)
        assert isinstance(out, str)
        low = out.lower()
        assert "out of range" in low or "beyond" in low or "only" in low or \
               "has" in low and "line" in low, f"silent OOB range: {out!r}"
    finally:
        _cleanup(root)


def test_read_file_negative_line_reports_range_not_missing_file():
    # Negative line numbers must not be reported as "FILE NOT FOUND" (wrong cause).
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel, "start_line": -5, "end_line": -1}, ctx)
        assert isinstance(out, str)
        assert "not found" not in out.lower(), f"negative line misreported as missing file: {out!r}"
    finally:
        _cleanup(root)


def test_replace_lines_inverted_range_message_not_duplicated():
    # The invalid-range message must appear ONCE, not twice (malformed_edits +
    # skips both carry it → "invalid range ... | invalid range ...").
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 5, "end_line": 2,
                     "new_content": "0|x = 1"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        assert out.lower().count("invalid range") <= 1, f"duplicated reason: {out!r}"
    finally:
        _cleanup(root)


def test_replace_lines_negative_range_says_why():
    # A negative range currently rejects with the vague "no change produced
    # (range may be invalid)" — it should name the range problem and how to fix.
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": -1, "end_line": 2,
                     "new_content": "0|x = 1"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        low = out.lower()
        assert "range" in low and ("1 ≤" in out or ">= 1" in low or "positive" in low
                                   or "out of bounds" in low), \
            f"vague negative-range reject: {out!r}"
    finally:
        _cleanup(root)


def test_semantic_search_failure_has_marker_and_fallback_hint():
    # When embeddings are unavailable / 0 hits, the result is
    # "(semantic search unavailable: ...)" — no ✗ marker, no "try search_text".
    # The model can't tell it failed and isn't told the alternative.
    import core.native_tools as nt

    async def _boom(*a, **k):
        return "(semantic search unavailable: Embed API HTTP 403)"
    ctx, rel, root = _mk_ctx()
    orig = None
    try:
        import tools.embeddings as emb
        orig = emb.semantic_retrieve
        emb.semantic_retrieve = _boom
        out = _disp("semantic_search", {"query": "counting"}, ctx)
        assert isinstance(out, str)
        assert out.startswith("✗"), f"failure not marked: {out!r}"
        assert "search_text" in out or "find_refs" in out, f"no fallback hint: {out!r}"
    finally:
        if orig is not None:
            emb.semantic_retrieve = orig
        _cleanup(root)


def test_unknown_tool_suggests_correct_name_for_common_aliases():
    # READ→read_file, GREP/SEARCH→search_text, CODE→read_file: a weak model that
    # emits an alias should be told the RIGHT tool, not just the full list.
    ctx, rel, root = _mk_ctx()
    try:
        for alias, want in (("READ", "read_file"), ("GREP", "search_text"),
                            ("SEARCH", "search_text"), ("CODE", "read_file"),
                            ("VIEW", "read_file")):
            out = _disp(alias, {}, ctx)
            assert isinstance(out, str) and out.startswith("✗")
            # the current code only lists all tools; ideally it names `want` as
            # the likely intended tool.
            assert want in out, (alias, out)
    finally:
        _cleanup(root)


def test_replace_lines_noop_is_rejected():
    # a byte-identical replace must NOT report success or pollute files_changed
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        # line 6 currently is `    return "hello " + name` (4 spaces) — re-write it identical
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '4|return "hello " + name'}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
        assert "no-op" in out.lower()
        assert rel not in ctx["files_changed"]
    finally:
        _cleanup(root)


# ── create_file: greenfield / new-module support ─────────────────────────────
def test_create_file_makes_new_file():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("create_file",
                    {"path": "new_mod.py", "content": "def f():\n    return 42\n"}, ctx)
        assert out.startswith("✓ Created")
        assert "new_mod.py" in ctx["files_changed"]
        produced = ctx["file_contents"]["new_mod.py"]
        assert "def f():" in produced and "return 42" in produced
        # in-memory and on-sandbox content agree
        assert ctx["sandbox"].load_file("new_mod.py") == produced
    finally:
        _cleanup(root)


def test_create_file_refuses_to_clobber_existing():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("create_file", {"path": rel, "content": "x = 1\n"}, ctx)
        assert out.startswith("✗") and "already exists" in out
        # original untouched
        assert "greet" in ctx["file_contents"][rel]
    finally:
        _cleanup(root)


def test_create_file_missing_path_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("create_file", {"content": "x=1"}, ctx)
        assert isinstance(out, str) and out.startswith("✗")
    finally:
        _cleanup(root)


def test_created_file_is_then_editable():
    # after create_file, the file is "viewed" so replace_lines can edit it
    ctx, rel, root = _mk_ctx()
    try:
        _disp("create_file", {"path": "n.py", "content": "a = 1\nb = 2\n"}, ctx)
        out = _disp("replace_lines",
                    {"path": "n.py", "start_line": 1, "end_line": 1,
                     "new_content": "0|a = 99"}, ctx)
        assert out.startswith("✓ Applied")
        assert ctx["file_contents"]["n.py"].split("\n")[0] == "a = 99"
    finally:
        _cleanup(root)


# ── lookup tools: each returns a string, never raises ────────────────────────
@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
def test_find_refs():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("find_refs", {"symbol": "greet"}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


@pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
def test_search_text():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("search_text", {"pattern": "class Counter"}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


def test_file_purpose():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("file_purpose", {"path": rel}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


def test_semantic_search():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("semantic_search", {"query": "counting things"}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


def test_depends_on():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("depends_on", {"symbol": "Counter"}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


def test_find_callers_no_map_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("find_callers", {"tag": "nope"}, ctx)
        assert isinstance(out, str) and len(out) > 0
    finally:
        _cleanup(root)


def test_lookup_tools_empty_arg_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        for name in ("find_refs", "find_callers", "search_text",
                     "file_purpose", "semantic_search", "depends_on"):
            out = _disp(name, {}, ctx)
            assert isinstance(out, str) and out.startswith("✗"), (name, out)
    finally:
        _cleanup(root)


# ── control tools ────────────────────────────────────────────────────────────
def test_finish_signals_done():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("finish", {"summary": "done"}, ctx)
        assert isinstance(out, tuple) and out[0] == "__FINISH__"
    finally:
        _cleanup(root)


def test_unknown_tool_message():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("bogus_tool", {}, ctx)
        assert isinstance(out, str) and out.startswith("✗ Unknown")
    finally:
        _cleanup(root)


# ── call_nvidia_tools response-shape guard (_extract_tool_message) ───────────
def test_extract_message_valid():
    import json as _j
    from clients.nvidia import _extract_tool_message
    raw = _j.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    msg = _extract_tool_message(raw, "m")
    assert msg["content"] == "hi"


def test_extract_message_malformed_shapes_raise_clearly():
    import json as _j
    import pytest as _p
    from clients.nvidia import _extract_tool_message
    # non-JSON body
    with _p.raises(RuntimeError, match="non-JSON"):
        _extract_tool_message("<html>502 bad gateway</html>", "m")
    # non-object JSON
    with _p.raises(RuntimeError, match="non-object"):
        _extract_tool_message("[1,2,3]", "m")
    # error-shaped 200 with a rate-limit message → RuntimeError carrying '429'
    # so the retry layer still classifies it transient
    with _p.raises(RuntimeError, match="429"):
        _extract_tool_message(_j.dumps({"error": {"message": "rate limited", "code": 429}}), "m")
    # missing choices
    with _p.raises(RuntimeError, match="no choices"):
        _extract_tool_message(_j.dumps({"id": "x"}), "m")
    # empty choices
    with _p.raises(RuntimeError, match="no choices"):
        _extract_tool_message(_j.dumps({"choices": []}), "m")
    # choice without a message
    with _p.raises(RuntimeError, match="no message"):
        _extract_tool_message(_j.dumps({"choices": [{"finish_reason": "stop"}]}), "m")


# ── native coder PROMPT ⇄ schema lockstep (overhaul: two coder prompts) ────────
def test_native_prompt_tools_match_schema():
    """The native coder prompt (IMPLEMENT_NATIVE_PROMPT) must name EXACTLY the
    functions in CODER_TOOLS — no stale tool (e.g. the cut symbol_detail), no
    missing tool (e.g. depends_on). Drift here silently mis-teaches the primary
    coder."""
    import re
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    named = set(re.findall(r'•\s*(\w+)\(', IMPLEMENT_NATIVE_PROMPT))
    schema = {t["function"]["name"] for t in CODER_TOOLS}
    assert named == schema, f"native prompt vs schema drift: only-in-prompt={named-schema}, only-in-schema={schema-named}"


def test_native_prompt_has_no_text_edit_format():
    """The native coder uses function calls — it must NOT carry the text
    [edit]/=== EDIT: protocol (that's the TEXT coder's IMPLEMENT_PROMPT)."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    for tok in ("=== EDIT:", "[edit:", "[REPLACE LINES", "[STOP][CONFIRM_STOP]"):
        assert tok not in IMPLEMENT_NATIVE_PROMPT, f"native prompt leaks text-protocol token {tok!r}"


def test_coder_prompts_ground_on_spec_literals():
    """ckpt 64/66: the #1 SWE-bench-Pro fail was the coder inventing a
    plausible-but-wrong value ('editor' vs the spec's 'Editor'). Both coder
    prompts must tell it to match the REQUIREMENTS/INTERFACE literals exactly and
    not invent them. STRICT PROTOCOL: they must NOT instruct reading the held-out
    failing test (it isn't on disk, and peeking would invalidate the score)."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT, IMPLEMENT_PROMPT
    for nm, t in (("native", IMPLEMENT_NATIVE_PROMPT), ("text", IMPLEMENT_PROMPT)):
        low = t.lower()
        assert "don't invent" in low, f"{nm} coder prompt dropped the don't-invent-the-value rule"
        assert "'Editor'" in t and "'editor'" in t, \
            f"{nm} coder prompt dropped the Editor≠editor exact-literal example"
        assert "requirements" in low and "interface" in low, \
            f"{nm} coder prompt must ground literals on the requirements/interface spec"
        # strict protocol: do not nudge the coder to read the failing test
        assert "read the failing test" not in low and "match its asserted" not in low, \
            f"{nm} coder prompt still points at the held-out test (protocol violation)"


def test_native_prompt_has_indent_by_scope_reasoning():
    """ckpt 69: gpt-oss intermittently emits a new method's `def` one level too
    deep (8| instead of 4|) → nested → AttributeError. The native coder prompt must
    require reasoning about indent-by-SCOPE (match the sibling in the target scope,
    not the line above) BEFORE the edit."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    low = IMPLEMENT_NATIVE_PROMPT.lower()
    assert "indent by scope" in low, "native prompt missing the indent-by-scope step"
    assert "sibling" in low, "native prompt must tell the coder to match a sibling's indent"


def test_native_prompt_has_anti_over_elaboration_rule():
    """ckpt 71: the coder over-elaborates (returns a wrapped type when a plain dict
    is asked, appends an extra command flag, combines 'X or Y' alternatives). The
    prompt must say the spec is a CEILING — implement only the stated behaviour."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    low = IMPLEMENT_NATIVE_PROMPT.lower()
    assert "ceiling" in low, "native prompt missing the spec-is-a-ceiling rule"
    assert "pick exactly one" in low, "native prompt must say pick ONE of offered alternatives"


def test_native_prompt_has_thinking_toolkit_reflexes():
    """ckpt 73: the coder CoT encodes Claude's tacit faculties as triggered reflexes —
    a live state-model ('be the interpreter') and calibrated uncertainty ('calibrate'
    exact tokens), plus parallel-consistency and type-snap. Pin them."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    low = IMPLEMENT_NATIVE_PROMPT.lower()
    for reflex in ("be the interpreter", "assume-and-check", "siblings move together", "type-snap", "hard contract"):
        assert reflex in low, f"native prompt missing reflex: {reflex!r}"
    # the calibration substitute must NOT rely on the model feeling its uncertainty
    assert "you can't feel the difference between knowing and guessing" in low
    assert "collide" in low, "missing the collide-with-the-example principle"


# ── syntax + unreachable gates (parity with the text coder's parse gate) ───────
def test_replace_lines_syntax_gate_rejects_unparseable():
    """A native edit that makes a previously-parseable .py file fail to compile
    must be REJECTED, not written + reported '✓ Applied' (that's how an
    IndentationError reached a final patch before this gate). File stays unchanged."""
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        # replace the greet def (lines 4-6) with a colon-less def → SyntaxError
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 4, "end_line": 6,
                     "new_content": "def greet(name)\n    return name"}, ctx)
        assert out.startswith("✗"), out
        assert "parse" in out.lower()
        assert ctx["file_contents"][rel] == SRC          # unchanged
        assert rel not in ctx["files_changed"]
        assert ctx["sandbox"].load_file(rel) == SRC      # disk unchanged
    finally:
        _cleanup(root)


def test_replace_lines_unreachable_gate_rejects_dead_code():
    """The 0ea40e09 failure: valid Python where the edit buried real logic as
    dead code after a return (success path silently returns None). Must reject."""
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        # replace greet's body (line 6) with a return followed by an unreachable stmt
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '    return "hi"\n    print("dead")'}, ctx)
        assert out.startswith("✗"), out
        assert "unreachable" in out.lower()
        assert ctx["file_contents"][rel] == SRC          # unchanged
        assert rel not in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_syntax_gate_allows_valid_edit():
    """The gates must NOT block a clean edit — a valid one still applies."""
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": '    return "hey " + name'}, ctx)
        assert out.startswith("✓ Applied"), out
        assert rel in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_replace_lines_dup_gate_rejects_duplicate_block():
    """ckpt 65: re-emitting an anchor block AND a copy → duplicate adjacent
    statements (the 1a9e74bf bug). The gate must reject, not write."""
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)
        # greet's body (line 6) → a yield then an identical yield (dup adjacent)
        out = _disp("replace_lines",
                    {"path": rel, "start_line": 6, "end_line": 6,
                     "new_content": ('    self.cache.store(name, "hello", overwrite=True)\n'
                                     '    self.cache.store(name, "hello", overwrite=True)')}, ctx)
        assert out.startswith("✗"), out
        assert "duplicate" in out.lower()
        assert ctx["file_contents"][rel] == SRC
        assert rel not in ctx["files_changed"]
    finally:
        _cleanup(root)


def test_create_file_syntax_gate_rejects_unparseable():
    """A new .py module that doesn't parse would ImportError on first import —
    reject before writing."""
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("create_file",
                    {"path": "broken.py", "content": "def f(\n    return 1"}, ctx)
        assert out.startswith("✗"), out
        assert "broken.py" not in ctx["files_changed"]
        assert ctx["sandbox"].load_file("broken.py") in (None, "")
    finally:
        _cleanup(root)


# ── edit_file: content-anchored JSON hunks (ckpt 79) ───────────────────────────
# The primary edit tool — replaces fragile line-range replace_lines with the text
# coder's content-matched SEARCH/REPLACE, expressed as native JSON {old,new} hunks
# so gpt-oss produces it reliably. Context lines anchor by content, not line #.

def test_edit_file_in_toolset_and_primary():
    names = [t["function"]["name"] for t in CODER_TOOLS]
    assert "edit_file" in names
    # edit_file is listed BEFORE replace_lines (the model prefers earlier tools)
    assert names.index("edit_file") < names.index("replace_lines")
    schema = next(t for t in CODER_TOOLS if t["function"]["name"] == "edit_file")
    props = schema["function"]["parameters"]["properties"]
    assert set(props) == {"path", "hunks"}
    assert props["hunks"]["type"] == "array"


def test_edit_file_changes_by_content_not_line_number():
    ctx, rel, root = _mk_ctx()
    try:
        # no start_line — unique `old` located by content (4-space indent matches SRC)
        out = _disp("edit_file", {"path": rel, "hunks": [
            {"old": ['    return "hello " + name'],
             "new": ['    return "hi " + name']}]}, ctx)
        assert out.startswith("✓"), out
        assert 'return "hi " + name' in ctx["file_contents"][rel]
        # persisted to the sandbox too
        assert 'return "hi " + name' in ctx["sandbox"].load_file(rel)
    finally:
        _cleanup(root)


def test_edit_file_insert_keeps_anchor():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("edit_file", {"path": rel, "hunks": [
            {"old": ['    def bump(self):'],
             "new": ['    def reset(self):',
                     '        self.n = 0',
                     '',
                     '    def bump(self):']}]}, ctx)
        assert out.startswith("✓"), out
        src = ctx["file_contents"][rel]
        assert "def reset(self):" in src and "def bump(self):" in src  # anchor preserved
    finally:
        _cleanup(root)


def test_edit_file_delete_with_empty_new():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("edit_file", {"path": rel, "hunks": [
            {"start_line": 5, "old": ['    """Return a greeting for name."""'],
             "new": []}]}, ctx)
        assert out.startswith("✓"), out
        assert "Return a greeting for name" not in ctx["file_contents"][rel]
    finally:
        _cleanup(root)


def test_edit_file_rejects_unfound_old_unchanged():
    ctx, rel, root = _mk_ctx()
    before = ctx["file_contents"][rel]
    try:
        out = _disp("edit_file", {"path": rel, "hunks": [
            {"old": ['        return "this line is not in the file at all"'],
             "new": ['        return "x"']}]}, ctx)
        assert out.startswith("✗"), out
        assert ctx["file_contents"][rel] == before     # untouched
    finally:
        _cleanup(root)


def test_edit_file_empty_hunks_rejected():
    ctx, rel, root = _mk_ctx()
    try:
        assert _disp("edit_file", {"path": rel, "hunks": []}, ctx).startswith("✗")
        assert _disp("edit_file", {"path": rel}, ctx).startswith("✗")
        # a hunk with empty `old` can't anchor
        out = _disp("edit_file", {"path": rel, "hunks": [{"old": [], "new": ["x"]}]}, ctx)
        assert out.startswith("✗") and "old" in out.lower()
    finally:
        _cleanup(root)


def test_edit_alias_routes_to_edit_file():
    # a model reaching for a wrong verb is steered to edit_file, not replace_lines
    ctx, rel, root = _mk_ctx()
    try:
        for wrong in ("edit", "replace", "str_replace", "patch"):
            out = _disp(wrong, {}, ctx)
            assert "edit_file" in out, f"{wrong} → {out}"
    finally:
        _cleanup(root)


def test_edit_file_start_line_disambiguates_repeated_old():
    # `return False` appears twice; start_line picks WHICH one (the ckpt-79 ambiguity fix)
    src = ('def a(x):\n    if x:\n        return False\n    return True\n\n'
           'def b(y):\n    if y:\n        return False\n    return True\n')
    def fresh():
        return {"file_contents": {"m.py": src}, "sandbox": None,
                "viewed_versions": {}, "project_root": ".", "files_changed": set()}
    # ambiguous without start_line → rejected with guidance, file untouched
    ctx = fresh()
    out = _disp("edit_file", {"path": "m.py",
                "hunks": [{"old": ['        return False'], "new": ['        return None']}]}, ctx)
    assert out.startswith("✗") and "appears 2 times" in out
    assert ctx["file_contents"]["m.py"] == src
    # start_line=8 edits ONLY the second occurrence
    ctx = fresh()
    out = _disp("edit_file", {"path": "m.py",
                "hunks": [{"start_line": 8, "old": ['        return False'],
                           "new": ['        return None']}]}, ctx)
    assert out.startswith("✓"), out
    lines = ctx["file_contents"]["m.py"].split("\n")
    assert lines[2].strip() == "return False"   # first occurrence untouched
    assert lines[7].strip() == "return None"     # second occurrence changed


def test_edit_file_sorts_out_of_order_hunks():
    # The numbered [edit] applier needs file order; the model may send hunks in
    # any order. _do_edit must sort them (not reject 'out of order' → retry loop).
    src = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None,
           "viewed_versions": {}, "project_root": ".", "files_changed": set()}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 5, "old": ["e = 5"], "new": ["e = 50"]},   # later line FIRST
        {"start_line": 1, "old": ["a = 1"], "new": ["a = 10"]},   # earlier line SECOND
    ]}, ctx)
    assert out.startswith("✓"), out
    lines = ctx["file_contents"]["m.py"].split("\n")
    assert lines[0] == "a = 10" and lines[4] == "e = 50"
