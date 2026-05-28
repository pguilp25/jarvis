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
                     "file_purpose", "semantic_search", "symbol_detail",
                     "create_file", "replace_lines", "finish"}


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


def test_symbol_detail():
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("symbol_detail", {"symbol": "Counter"}, ctx)
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
                     "file_purpose", "semantic_search", "symbol_detail"):
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
