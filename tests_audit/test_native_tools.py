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
                     "edit_file", "create_file", "replace_lines", "run_code", "finish"}


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


# ── read_file: REAL-whitespace format (ckpt 88, aligned with edit_file copy) ──
def test_read_file_prefix_ws_format():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel}, ctx)
        assert "greet" in out
        # prefix_ws format `LINENO:INDENT|<real spaces>code`: shows BOTH the indent
        # NUMBER (authoritative for edits) AND the real spaces (so the coder sees it).
        assert ":8|" in out                       # the indent number is present
        assert ":8|        self.n = 0" in out      # number + 8 real spaces + code
        assert ":0|def greet(name):" in out        # col-0 def
    finally:
        _cleanup(root)


def test_read_file_shows_real_indentation():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("read_file", {"path": rel}, ctx)
        # a method body line carries its ACTUAL 8 leading spaces (after the `8|`)
        assert "        self.n = 0" in out
        # a module-level def is at column 0
        assert "def greet(name):" in out
    finally:
        _cleanup(root)


def test_diff_rows_are_not_editable_input_strict():
    # STRICT design (pass-4): a raw diff row is NOT editable input — the applier does NOT
    # silently strip its gutter (that shape is ambiguous with YAML/config). Pasting a diff
    # `-` row as `old` does NOT match; the reject teaches the canonical INDENT|code form.
    from core.native_tools import _expand_indent_lines as ex, _do_edit
    from core.edit_diff import render_diff
    old = "class C:\n    def f(self):\n        return 1\n"
    new = "class C:\n    def f(self):\n        return 2\n"
    diff = render_diff(old, new, "m.py")
    minus = next(l for l in diff.splitlines() if l.split(":", 1)[-1].startswith("- "))
    assert ex([minus]) == [minus]                          # left literal, NOT transformed
    ctx = {"file_contents": {"m.py": old}, "files_changed": set()}
    r = _do_edit({"path": "m.py", "hunks": [{"old": [minus], "new": ["8|return 9"]}]}, ctx)
    assert not r.startswith("✓")
    assert "INDENT|code" in r and "diff" in r.lower()      # targeted, helpful reject
    # the canonical forms DO apply
    ctx2 = {"file_contents": {"m.py": old}, "files_changed": set()}
    r2 = _do_edit({"path": "m.py", "hunks": [{"old": ["8|return 1"], "new": ["8|return 2"]}]}, ctx2)
    assert r2.startswith("✓") and "return 2" in ctx2["file_contents"]["m.py"]


def test_route_parser_accepts_colonless_go_to_step():
    # The reviewer prompt's prose shows the bare `[GO TO STEP]` / `[GO TO PLAN]` form;
    # the parser must accept it (a colon-required parser silently dropped the rejection →
    # a lost FAIL ships a bug). (audit re-pass MED fix.)
    from core.review_verify import parse_route
    assert parse_route("[GO TO STEP]").kind == "step"
    assert parse_route("[GO TO STEP 3]").kind == "step"
    assert parse_route("[GO TO PLAN]").kind == "plan"
    assert parse_route("[GO TO STEP 3: fix it]").kind == "step"   # colon form still works
    assert parse_route("[APPROVED]").kind == "approved"
    # word-boundary: PLANNER/STEPPED must NOT match the keyword as a prefix (pass-3 fix)
    assert parse_route("[GO TO PLANNER] handle it").kind == "none"
    assert parse_route("[GO TO STEPPED] carefully").kind == "none"


def test_expand_indent_never_corrupts_real_content():
    # STRICT design: only the two UNAMBIGUOUS canonical forms are transformed; everything
    # else is literal. A YAML/dict mapping, a diff row, a dict-slice, a bitmask — all survive
    # verbatim (no guessed transform can corrupt real content). (audit pass-4: strict.)
    from core.native_tools import _expand_indent_lines as ex
    for literal in ["443:  description", "200:  OK", "8080:  proxy backend",
                    " 3:+     return 2", "12:-     return 1", "    data[5: 10]",
                    "    flags = 0o755 | 0o644", "5: 'value',"]:
        assert ex([literal]) == [literal], f"corrupted: {literal!r} -> {ex([literal])!r}"
    # canonical forms still expand
    assert ex(["8|return x"]) == ["        return x"]
    assert ex(["3:4|    def foo"]) == ["    def foo"]


def test_appears_annotation_strip_is_safe():
    # The `|appears N (#tag)` blast-radius annotation is stripped when copied, but a real
    # code line that merely CONTAINS "|appears <n>" (no `(#hex)`) must NOT be truncated.
    from core.native_tools import _expand_indent_lines as ex
    assert ex(["5:4|    def baz(self): |appears 1 (#085)"]) == ["    def baz(self):"]
    real = '    raise ValueError("symbol |appears 3 times")'
    assert ex(["4|" + real.lstrip()]) == [real]      # NOT truncated


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
    """ckpt 71 / ckpt 116 (RIGHT-SIZE): the coder must neither over-elaborate
    (gold-plate: wrapped type, extra flag, combining 'X or Y') NOR under-build
    (collapse required logic to a naive shortcut). The prompt must carry the
    right-size rule: cut extras, keep required logic, pick ONE of alternatives."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    low = IMPLEMENT_NATIVE_PROMPT.lower()
    assert "right-size" in low or "gold-plat" in low, "native prompt missing the right-size / no-gold-plating rule"
    assert "pick exactly one" in low, "native prompt must say pick ONE of offered alternatives"
    # the ckpt-116 nuance: minimal must NOT mean the naivest shortcut
    assert "correct implementation" in low or "never required logic" in low, \
        "right-size rule must say minimal = simplest CORRECT impl, not the naive shortcut"


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
    # ckpt 119: edit_file gained grounding-CoT fields (goal/traced/check), enforced
    # only when JARVIS_EDIT_COT is set (optional in the base schema).
    assert set(props) == {"path", "hunks", "goal", "traced", "check"}
    assert props["hunks"]["type"] == "array"
    assert {"goal", "traced", "check"} <= set(props)


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


def test_edit_file_old_not_found_message_is_stale_aware():
    # `old` matches nowhere: if we already edited the file, the `old` is from a view
    # taken BEFORE that edit — NOT the wrong file. Trust-the-view (ckpt 111): the
    # guidance must point the coder at the in-context DIFF / a targeted range, and
    # must NOT tell it to re-read the whole file (that re-read is what blew f631's
    # context window). It must still be distinguishable from the wrong-file case.
    src = "x = 1\ny = 2\n"
    base = lambda changed: {"file_contents": {"m.py": src}, "sandbox": None,
                            "viewed_versions": {}, "project_root": ".",
                            "files_changed": ({"m.py"} if changed else set())}
    # already edited → stale-view guidance pointing at the diff, NOT a full re-read
    r_edited = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 1, "old": ["this line is not present"], "new": ["z = 9"]}]},
        base(True))
    assert r_edited.startswith("✗")
    assert "EDITED" in r_edited and "WRONG FILE" not in r_edited
    assert "diff" in r_edited.lower()                       # points at the in-context diff
    assert "stale" in r_edited.lower()                      # still flags the stale copy
    assert "read_file m.py with" in r_edited                # only a targeted range, if needed
    # never edited → WRONG-FILE guidance + copy-exactly / range; never "re-read whole"
    r_fresh = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 1, "old": ["this line is not present"], "new": ["z = 9"]}]},
        base(False))
    assert r_fresh.startswith("✗") and "WRONG FILE" in r_fresh
    assert "range" in r_fresh.lower()                       # offers a targeted range
    assert "re-read" not in r_fresh.lower()                 # never tells it to re-read the file


def test_view_at_invariant_holds_across_a_full_sequence():
    """STAYS-THAT-WAY guarantee: after EVERY op, the trust-the-view invariant holds —
    (a) if view_at[path] is set, a full read is short-circuited (never re-dumps), and
    (b) file_contents[path] equals the sandbox (the view the coder trusts is real);
    and every edit/create/replace leaves the touched file in view_at (the coder is
    never told to re-read what it just changed)."""
    ctx, rel, root = _mk_ctx()
    try:
        def check():
            for p in list(ctx.get("view_at", {})):
                r = _disp("read_file", {"path": p}, ctx)
                assert r.startswith("ℹ"), f"{p} in view_at but a full read was not short-circuited:\n{r}"
                if ctx.get("sandbox") is not None:
                    assert ctx["file_contents"][p] == ctx["sandbox"].load_file(p), \
                        f"{p}: trusted view diverged from the sandbox"
        # 1) full read → file enters view_at
        out = _disp("read_file", {"path": rel}, ctx)
        assert not out.startswith("ℹ") and rel in ctx["view_at"]; check()
        # 2) range read of the SAME file does NOT downgrade its full-view status
        _disp("read_file", {"path": rel, "start_line": 1, "end_line": 2}, ctx)
        assert rel in ctx["view_at"]; check()
        # 3) edit_file → still in view_at, content tracks the sandbox
        e = _disp("edit_file", {"path": rel, "hunks": [
            {"start_line": 6, "old": ['    return "hello " + name'],
             "new": ['    return "hi " + name']}]}, ctx)
        assert e.startswith("✓"), e
        assert rel in ctx["view_at"]; check()
        # 4) replace_lines on the same file → still consistent
        rl = _disp("replace_lines", {"path": rel, "start_line": 1, "end_line": 1,
                                      "new_content": '"""A tiny module (edited)."""'}, ctx)
        assert rl.startswith("✓"), rl
        assert rel in ctx["view_at"]; check()
        # 5) create_file → the new file is in view_at (you just wrote it; no read needed)
        c = _disp("create_file", {"path": "brand_new.py", "content": "VALUE = 42\n"}, ctx)
        assert c.startswith("✓"), c
        assert "brand_new.py" in ctx["view_at"]; check()
    finally:
        _cleanup(root)


def test_range_read_of_unseen_file_does_not_mark_it_fully_in_context():
    """A RANGE read shows only a slice, so it must NOT enter view_at — otherwise a
    later FULL read the coder genuinely needs would be wrongly short-circuited."""
    ctx, rel, root = _mk_ctx()
    try:
        r = _disp("read_file", {"path": rel, "start_line": 1, "end_line": 2}, ctx)
        assert not r.startswith("ℹ")
        assert rel not in ctx.get("view_at", {})        # partial view ≠ full view
        full = _disp("read_file", {"path": rel}, ctx)   # the needed full read is allowed
        assert not full.startswith("ℹ") and "class Counter" in full
        assert rel in ctx["view_at"]                    # now it's fully in context
    finally:
        _cleanup(root)


def test_unassigned_enum_members_flags_forgotten_case():
    from core.native_tools import _unassigned_enum_members
    # `unknown` is DEFINED but never assigned anywhere → flagged (forgotten case).
    src = (
        "import enum\n"
        "class VersionChange(enum.Enum):\n"
        "    unknown = 0\n"
        "    equal = 1\n"
        "    major = 2\n"
        "def f(a, b):\n"
        "    if a == b:\n"
        "        return VersionChange.equal\n"
        "    return VersionChange.major\n"
    )
    dead = _unassigned_enum_members(src)
    assert "VersionChange.unknown" in dead
    assert "VersionChange.equal" not in dead and "VersionChange.major" not in dead


def test_unassigned_enum_members_clean_when_all_used():
    # f631 shape: every member is referenced somewhere → no false positive.
    from core.native_tools import _unassigned_enum_members
    src = (
        "import enum\n"
        "class VersionChange(enum.Enum):\n"
        "    unknown = 0\n"
        "    equal = 1\n"
        "def g(x):\n"
        "    if x is None:\n"
        "        return VersionChange.unknown\n"
        "    return VersionChange.equal\n"
    )
    assert _unassigned_enum_members(src) == []
    # non-enum / unparseable input never raises
    assert _unassigned_enum_members("def h(:\n bad") == []
    assert _unassigned_enum_members("x = 1\n") == []


def test_read_short_circuit_escalates_on_repeat():
    # A full re-read of an in-context file is declined with ℹ; repeating it must
    # escalate (no silent spin to the round budget).
    ctx = {"file_contents": {"m.py": "a = 1\n"}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(),
           "view_at": {"m.py": "step 1 (loaded at the start)"}}
    r1 = _disp("read_file", {"path": "m.py"}, ctx)
    assert r1.startswith("ℹ") and "⚠" not in r1                # first decline: gentle
    r2 = _disp("read_file", {"path": "m.py"}, ctx)
    assert r2.startswith("ℹ") and "⚠" in r2 and "STOP" in r2   # second: escalated
    assert not r2.startswith("✗")                              # still inert to the fail-counter


def test_edit_file_pure_insert_with_empty_old():
    # Adding new code: model leaves old empty, gives start_line + new. Must insert
    # AFTER start_line (not reject 'old is empty' → the f327 5x reject).
    src = "class C:\n    def a(self):\n        return 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 3, "old": [], "new": ["", "    def b(self):", "        return 2"]}]}, ctx)
    assert out.startswith("✓"), out
    body = ctx["file_contents"]["m.py"]
    assert "def a(self)" in body and "def b(self)" in body and body.count("return 1") == 1
    # but empty old with NO start_line is still a clear reject
    ctx2 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
            "project_root": ".", "files_changed": set()}
    assert _disp("edit_file", {"path": "m.py", "hunks": [{"old": [], "new": ["x"]}]},
                 ctx2).startswith("✗")


# ── comprehension GPS (ckpt 90): harness computes the blast-radius the weak ──
# model can't hold, and hands back the exact remaining edit (not a dead-end error).

def test_dangling_ref_reject_points_at_the_use_site():
    # Remove a helper's DEFINITION but leave a call to it → the reject must name
    # WHERE it's still used + that the def was removed (f327's 5x NameError).
    src = ("def _is_fqcn(s):\n    return True\n\n"
           "def is_valid(name):\n    return _is_fqcn(name)\n")
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    # delete the def of _is_fqcn (lines 1-2), leaving the call on line 5
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 1, "old": ["def _is_fqcn(s):", "    return True"], "new": []}]}, ctx)
    assert out.startswith("✗"), out
    assert "_is_fqcn" in out and "REMOVED its definition" in out
    assert "still USED" in out and "_is_fqcn(name)" in out   # points at the dangling call
    assert ctx["file_contents"]["m.py"] == src               # not applied


def test_orphaned_block_reject_names_the_header():
    # Delete a try-block's body leaving `try:` empty → reject must say the BODY
    # was deleted and the header kept (f327's empty-block SyntaxError), not the
    # generic indent hint.
    src = "def f():\n    try:\n        risky()\n    except Exception:\n        pass\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 3, "old": ["        risky()"], "new": []}]}, ctx)
    assert out.startswith("✗"), out
    assert "DELETED the body" in out and "try" in out
    assert ctx["file_contents"]["m.py"] == src


# ── trust-the-view: short-circuit redundant re-reads + stamp diffs with WHEN ──
# (ckpt 112) f631 re-read a 900-line file 5× → blew the 131072-token context
# window mid-step → the method-creation step aborted and the def was lost.

def test_read_file_short_circuits_a_full_reread():
    # A full read records the view; a SECOND full read is refused (not re-dumped),
    # but a targeted RANGE read is still allowed.
    ctx, rel, root = _mk_ctx()
    try:
        first = _disp("read_file", {"path": rel}, ctx)
        assert "greet" in first                              # real content returned
        assert rel in ctx.get("view_at", {})                # view recorded
        second = _disp("read_file", {"path": rel}, ctx)
        assert second.startswith("ℹ") and "ALREADY in your context" in second
        assert "def greet" not in second                    # did NOT re-dump the file
        rng = _disp("read_file", {"path": rel, "start_line": 1, "end_line": 3}, ctx)
        assert not rng.startswith("ℹ")                       # range read still works
    finally:
        _cleanup(root)


def test_read_file_short_circuit_after_edit_points_to_diff():
    # After an edit, the file is in context (diff + unchanged remainder); a full
    # re-read is refused and points at the edit's diff — never a re-read.
    src = "a = 1\nb = 2\nc = 3\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "round": 4, "step_num": 2}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 2, "old": ["b = 2"], "new": ["b = 20"]}]}, ctx)
    assert out.startswith("✓") and "step 2, round 4" in out  # diff stamped with WHEN
    assert "m.py" in ctx.get("view_at", {})
    r = _disp("read_file", {"path": "m.py"}, ctx)
    assert r.startswith("ℹ") and "ALREADY in your context" in r
    assert "diff after your edit" in r and "step 2, round 4" in r


def test_seeded_view_at_short_circuits_injected_target():
    # The caller seeds view_at for files injected into the prompt → a redundant
    # full read of an injected target is short-circuited.
    ctx = {"file_contents": {"m.py": "a = 1\n"}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(),
           "view_at": {"m.py": "step 1 (loaded at the start)"}}
    r = _disp("read_file", {"path": "m.py"}, ctx)
    assert r.startswith("ℹ") and "step 1 (loaded at the start)" in r


def test_edit_diff_stamped_with_round_when_no_step():
    src = "a = 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "round": 7}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 1, "old": ["a = 1"], "new": ["a = 2"]}]}, ctx)
    assert out.startswith("✓") and "round 7" in out and "step" not in out.split("\n")[0]


def test_edit_cot_gate_rejects_ungrounded_when_flag_on():
    """ckpt 119 (JARVIS_EDIT_COT): edit tools REQUIRE grounded goal/traced/check.
    Ungrounded / guessed / unquoted edits are REJECTED; a grounded edit (traced
    quotes a real line) applies. Flag OFF = no enforcement (regression-safe)."""
    import os, importlib
    src = "class Bar:\n    def a(self):\n        return 1\n"
    mkctx = lambda: {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
                     "project_root": ".", "files_changed": set()}
    hunk = [{"start_line": 3, "old": ["        return 1"], "new": ["        return 2"]}]
    os.environ["JARVIS_EDIT_COT"] = "1"
    try:
        import core.native_tools as nt; importlib.reload(nt)
        run = lambda a: asyncio.run(nt._dispatch("edit_file", a, mkctx()))
        # missing grounding → reject
        assert run({"path": "m.py", "hunks": hunk}).startswith("✗")
        # hedge in traced → reject
        assert "GUESS" in run({"path": "m.py", "goal": "make a() return 2 per spec",
            "traced": "it probably returns 1", "check": "calling a() returns 2 not 1", "hunks": hunk})
        # traced doesn't quote a real line → reject
        assert run({"path": "m.py", "goal": "make a() return 2 per spec",
            "traced": "the method returns the number one", "check": "a() gives 2",
            "hunks": hunk}).startswith("✗")
        # grounded (traced quotes `return 1`) → apply
        ok = run({"path": "m.py", "goal": "make a() return 2 as the spec requires",
            "traced": "a() runs `return 1`", "check": "a() returns 2 not 1", "hunks": hunk})
        assert ok.startswith("✓"), ok
    finally:
        os.environ.pop("JARVIS_EDIT_COT", None)
        import core.native_tools as nt; importlib.reload(nt)
    # flag OFF: ungrounded edit applies (no enforcement)
    assert asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "hunks": hunk}, mkctx())).startswith("✓")
