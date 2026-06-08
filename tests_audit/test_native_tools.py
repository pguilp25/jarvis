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
    # ckpt-138: collapsed to ONE edit tool (edit_file old→new). replace_lines removed.
    assert names == {"read_file", "list_dir", "keep", "batch", "find_refs", "find_callers",
                     "search_text", "file_purpose", "semantic_search", "depends_on",
                     "edit_file", "create_file", "run_code", "finish"}


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
        # prefix_ws format `LINENO ⇥INDENT|<real spaces>code` (ckpt-143 naturalized):
        # line# bare-left, `⇥INDENT` marks the indent (authoritative for edits), then
        # `|` + the real spaces (so the coder sees the nesting) + code.
        assert " ⇥8|" in out                       # the indent number is present, ⇥-marked
        assert " ⇥8|        self.n = 0" in out      # ⇥number + 8 real spaces + code
        assert " ⇥0|def greet(name):" in out        # col-0 def
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


def test_structural_indent_slip_guard():
    # ckpt-224 / Cluster D (f631 FAIL): a def/class/decorator declared with a NUMBER that disagrees
    # with the TYPED leading spaces is a fat-fingered number — trust the typed spaces so a method
    # doesn't get EJECTED to module scope (where it parses but is semantically dead).
    from core.native_tools import _expand_indent_lines as ex
    assert ex(["0|    def __bool__(self):"]) == ["    def __bool__(self):"]   # slip → stays nested
    assert ex(["0|    @property"]) == ["    @property"]                       # decorator slip
    assert ex(["0|        class Inner:"]) == ["        class Inner:"]          # class slip
    # a REAL module-level def is typed with ZERO leading spaces → number wins, stays at col 0
    assert ex(["0|def top_level():"]) == ["def top_level():"]
    # a NON-structural intentional dedent stays number-authoritative (typed spaces ignored)
    assert ex(["4|        x = 1"]) == ["    x = 1"]
    # normal method form (no typed spaces) is unaffected
    assert ex(["4|def bar(self):"]) == ["    def bar(self):"]


def test_repair_tool_use_wrapper_recovers_dropped_open():
    # ckpt-224 / Cluster A: a model that drops the leading `[` of `[tool use]` (owl-alpha merger)
    # must still have its block recognised — esp. in the MIXED case where a well-formed block
    # would otherwise turn enforce-masking ON and drop the malformed block's tags.
    from core.tool_call import _repair_tool_use_wrapper as rp
    fixed = rp("intro tool use]\n[VIEW: a.py 1-50]\n[/tool use]\n[STOP][CONFIRM_STOP]")
    assert fixed.count("[tool use]") == 1 and fixed.count("[/tool use]") == 1
    # well-formed input is untouched; a prose mention of "[tool use]" is not corrupted
    wf = "[tool use]\n[VIEW: a.py 1-9]\n[/tool use]"
    assert rp(wf) == wf
    assert "[tool use]" in rp("see the [tool use] wrapper docs")  # prose stays well-formed, no dup


def test_strip_leaked_channel_tokens():
    # ckpt-224 / Cluster L: leaked harmony/LongCat control tokens are dropped, real <think> kept.
    from core.tool_call import _strip_leaked_channel_tokens as s
    assert s("a <channel|> b </longcat_think> c") == "a  b  c"
    assert s("x <|channel|>y<|message|>z") == "x yz"
    assert s("keep <think>real</think>") == "keep <think>real</think>"


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


def test_create_file_blocks_import_shims_and_new_toplevel_pkg():
    # ckpt-167: after a run_code ModuleNotFoundError on a missing dep/framework module
    # (web, infogami, yaml…), the weak coder tries to manufacture a stub package — that
    # junk pollutes the patch and breaks `git apply` in the real env (regressed a PASSING
    # instance, 4a5d2a7d ✓→broken). create_file must refuse import-shims while STILL
    # allowing genuine new files inside existing packages.
    ctx, rel, root = _mk_ctx()
    try:
        sbdir = ctx["sandbox"].sandbox_dir
        # (2) DYNAMIC — a module run_code just failed to import → refuse the shim
        ctx["_failed_imports"] = {"web"}
        out = _disp("create_file", {"path": "web/__init__.py", "content": "x=1\n"}, ctx)
        assert out.startswith("✗") and "web" in out and "shim" in out, out
        ctx.pop("_failed_imports", None)
        # (4) NEW top-level package not in the repo → refuse (no failed-import needed)
        out2 = _disp("create_file", {"path": "infogami/core/x.py", "content": "x=1\n"}, ctx)
        assert out2.startswith("✗") and "top-level" in out2, out2
        # (3) root module shadowing a dep → refuse
        out3 = _disp("create_file", {"path": "yaml.py", "content": "x=1\n"}, ctx)
        assert out3.startswith("✗") and "yaml" in out3, out3
        # LEGIT — a genuinely new file INSIDE an existing package must still work
        os.makedirs(os.path.join(sbdir, "pkg"))
        out4 = _disp("create_file",
                     {"path": "pkg/newmod.py", "content": "def f():\n    return 1\n"}, ctx)
        assert out4.startswith("✓ Created"), out4
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


def test_list_dir_tree_with_line_counts():
    # list_dir lists folders (file counts) + files (LINE counts); no path = project root,
    # a folder path expands it; escapes/missing/file paths are clean ✗, never a crash.
    import os as _os
    root = tempfile.mkdtemp(prefix="listdir_")
    try:
        _os.makedirs(_os.path.join(root, "pkg"))
        with open(_os.path.join(root, "top.py"), "w") as f:
            f.write("x = 1\ny = 2\nz = 3\n")            # 3 lines
        with open(_os.path.join(root, "pkg", "mod.py"), "w") as f:
            f.write("def a():\n    return 1\n")
        ctx = {"file_contents": {}, "sandbox": None, "project_root": root, "files_changed": set()}
        top = _disp("list_dir", {}, ctx)                 # project root
        assert "=== TREE:" in top
        assert "pkg/" in top and "files)" in top         # a folder with a file count
        assert "top.py" in top and "3 lines" in top      # a file with its line count
        sub = _disp("list_dir", {"path": "pkg"}, ctx)    # expand one level
        assert "pkg/mod.py" in sub and "2 lines" in sub
        assert _disp("list_dir", {"path": "../etc"}, ctx).startswith("✗")     # escape refused
        assert _disp("list_dir", {"path": "nope"}, ctx).startswith("✗")       # missing folder
        assert _disp("list_dir", {"path": "top.py"}, ctx).startswith("✗")     # a file, not a folder
        assert isinstance(_disp("list_dir", {"path": 5}, ctx), str)           # wrong type → no crash
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
    # Tool bullets start a line as `  - name(` or `  • name(` — anchor on the
    # line-start bullet so trace examples like `open(` mid-sentence aren't matched.
    named = set(re.findall(r'(?m)^\s*[-•]\s*(\w+)\(', IMPLEMENT_NATIVE_PROMPT))
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
    """ckpt 69 + ckpt-135: gpt-oss emits a `def` at the wrong indent → nested/ejected →
    IndentationError/AttributeError. The native coder prompt must require reasoning about
    a def's SCOPE number, match a SIBLING in the target scope, and (ckpt-135) warn BOTH
    directions — under-indent (4→0 ejection) AND over-indent (0→4 nesting, the ba3abfb6
    regression) — not just the col-0 case."""
    from core.prompts_v8 import IMPLEMENT_NATIVE_PROMPT
    low = IMPLEMENT_NATIVE_PROMPT.lower()
    assert "scope" in low, "native prompt missing the scope-number indent step"
    assert "sibling" in low, "native prompt must tell the coder to match a sibling's indent"
    assert "over-indent" in low and "under-indent" in low, \
        "indent reflex must warn BOTH directions (over- AND under-indent), not just col-0"


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
    assert "replace_lines" not in names      # ckpt-138: collapsed to one edit tool
    schema = next(t for t in CODER_TOOLS if t["function"]["name"] == "edit_file")
    props = schema["function"]["parameters"]["properties"]
    # ckpt-138: dead-simple search→replace — path/old/new. ckpt-151: + `edits`
    # array for batching multiple old/new pairs into ONE atomic call. Only `path`
    # is hard-required (old/new used solo, OR edits used for a batch).
    assert set(props) == {"path", "old", "new", "edits"}
    assert props["old"]["type"] == "array" and props["new"]["type"] == "array"
    assert props["edits"]["type"] == "array"
    assert schema["function"]["parameters"]["required"] == ["path"]


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
    # must EXPLICITLY forbid a re-read (ckpt-154 don't-re-read steering) and never
    # instruct re-dumping the whole file
    assert "do not re-read" in r_fresh.lower()
    assert "whole file" not in r_fresh.lower() or "never re-dump the whole file" in r_fresh.lower()


def test_view_at_invariant_holds_across_a_full_sequence():
    """STAYS-THAT-WAY guarantee: after EVERY op, the view invariant holds —
    (a) a full read of an in-view file is SERVED, never a ✗ refusal (ckpt-150: ℹ if
    unchanged, ✓ UPDATED with a diff if it changed since first seen — the coder can
    always SEE the current code), and (b) file_contents[path] equals the sandbox (the
    view the coder trusts is real); and every edit/create/replace leaves the touched
    file in view_at (the coder is never told to re-read what it just changed)."""
    ctx, rel, root = _mk_ctx()
    try:
        def check():
            for p in list(ctx.get("view_at", {})):
                r = _disp("read_file", {"path": p}, ctx)
                assert r.startswith("ℹ") or r.startswith("✓"), \
                    f"{p} in view_at but a full read was not served:\n{r[:200]}"
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


def test_range_read_small_file_serves_whole_and_marks_viewed():
    """ckpt-178 range-nibbling fix: a range read of a SMALL (≤3000) file serves the WHOLE
    file once (it's cheap to hold) and marks it viewed → every later read short-circuits,
    so the coder can't nibble it in dozens of overlapping ranges (the f631/a26 timeout)."""
    ctx, rel, root = _mk_ctx()
    try:
        r = _disp("read_file", {"path": rel, "start_line": 1, "end_line": 2}, ctx)
        assert "FULL file" in r and "class Counter" in r     # served WHOLE, not just lines 1-2
        assert "asked for lines 1-2" in r                    # tells coder why it got the whole file
        assert rel in ctx["view_at"]                         # marked viewed → later reads short-circuit
        # ckpt-210: a NARROW re-read (≤200 lines) is now SERVED, not refused. The whole-file
        # view can be TRIMMED out of a long step's history while view_at persists in ctx, so the
        # old refusal trapped the coder in a re-read spiral (a26 step 7 → 1800s timeout). Wide
        # re-reads still short-circuit (no f631 re-dump bloat) — see the dedicated test below.
        again = _disp("read_file", {"path": rel, "start_line": 5, "end_line": 6}, ctx)
        assert not again.startswith("ℹ")                          # served, not refused
        assert "RANGE" in again and "lines 5-6" in again          # the requested slice, freshly rendered
    finally:
        _cleanup(root)


def test_range_read_large_file_accumulates_into_one_growing_view():
    """ckpt-196 (replaces the ckpt-178 slice+short-circuit contract): a >cap file is navigated by
    ONE growing view. A range read REVEALS that range as real code; a re-read fully covered by what
    was already revealed changes nothing and is NOT refused (no 'already read' error); a fresh
    region EXPANDS the same view (both ranges now shown), with the un-read gap labelled as a hole."""
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="bigrange_")
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(2000))   # 4000 lines (>cap)
    with open(_os.path.join(root, "big.py"), "w") as _f:
        _f.write(big)
    ctx = {"file_contents": {"big.py": big}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "step_num": 1}
    try:
        r = _disp("read_file", {"path": "big.py", "start_line": 100, "end_line": 200}, ctx)
        assert "GROWING VIEW" in r and "return 50" in r          # revealed range shown as real code
        assert "big.py" not in ctx.get("view_at", {})            # partial view ≠ full view held
        assert ctx["_served_ranges"]["big.py"] == [(100, 200)]
        # re-read fully covered by what's revealed → NOT refused, just the same growing view
        covered = _disp("read_file", {"path": "big.py", "start_line": 120, "end_line": 180}, ctx)
        assert not covered.lstrip().startswith("✗") and "already read" not in covered
        assert "GROWING VIEW" in covered and "return 50" in covered
        assert ctx["_served_ranges"]["big.py"] == [(100, 200)]   # nothing new revealed
        # a fresh region EXPANDS the one view — both ranges now present + the gap labelled
        fresh = _disp("read_file", {"path": "big.py", "start_line": 900, "end_line": 1000}, ctx)
        assert "return 50" in fresh and "return 450" in fresh    # both revealed ranges shown
        assert "not read" in fresh                               # the gap between them is a labelled hole
        assert ctx["_served_ranges"]["big.py"] == [(100, 200), (900, 1000)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_batch_runs_multiple_lookups_one_round():
    # ckpt-181: batch runs several read-only lookups in one round; refuses edits; never crashes.
    ctx, rel, root = _mk_ctx()
    try:
        out = _disp("batch", {"calls": [
            {"tool": "read_file", "args": {"path": rel}},
            {"tool": "search_text", "args": {"pattern": "class Counter"}},
            {"tool": "find_refs", "args": {"symbol": "greet"}},
        ]}, ctx)
        assert "op[1] read_file" in out and "op[2] search_text" in out and "op[3] find_refs" in out
        assert "greet" in out                                    # the read_file result is in there
        assert rel in ctx.get("view_at", {})                     # the batched read marked it viewed
        # by default (flag off) an edit sub-call is refused — batch is read-only
        bad = _disp("batch", {"calls": [{"tool": "edit_file", "args": {"path": rel}}]}, ctx)
        assert "not allowed here" in bad
        # ckpt-181b: an ALL-failing batch (here: all non-batchable — the realistic repeat-mistake
        # of batching edits) returns ✗ so the loop's reject-counter/fallover engages.
        allfail = _disp("batch", {"calls": [{"tool": "edit_file", "args": {}},
                                            {"tool": "run_code", "args": {"command": "ls"}}]}, ctx)
        assert allfail.startswith("✗") and "all 2 op" in allfail
        # a batch with ≥1 real result does NOT start with ✗ (not a reject)
        partial = _disp("batch", {"calls": [{"tool": "read_file", "args": {"path": rel}},
                                            {"tool": "edit_file", "args": {}}]}, ctx)
        assert not partial.startswith("✗")
        # malformed / wrong-type calls don't crash
        assert isinstance(_disp("batch", {"calls": "nope"}, ctx), str)
        assert isinstance(_disp("batch", {"calls": [5, {"tool": "list_dir"}]}, ctx), str)
    finally:
        _cleanup(root)


def test_batch_only_mode_runs_edit_then_finish():
    # ckpt-182: under JARVIS_BATCH_ONLY, batch carries ALL ops (incl. edit + finish). An
    # edit+finish batch applies the edit and returns the __FINISH__ tuple (loop ends via the
    # normal finish path). Default (flag off) keeps batch read-only (other tests cover that).
    os.environ["JARVIS_BATCH_ONLY"] = "1"
    try:
        from core.native_tools import _do_batch
        src = "def f():\n    return 1\n"
        ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
               "project_root": ".", "files_changed": set()}
        r = asyncio.run(_do_batch({"calls": [
            {"tool": "edit_file", "args": {"path": "m.py", "old": ["4|return 1"], "new": ["4|return 2"]}},
            {"tool": "finish", "args": {"summary": "changed it"}}]}, ctx))
        assert isinstance(r, tuple) and r[0] == "__FINISH__" and r[1] == "changed it"
        assert ctx["file_contents"]["m.py"] == "def f():\n    return 2\n"   # the batched edit applied
        assert "m.py" in ctx["files_changed"]
    finally:
        os.environ.pop("JARVIS_BATCH_ONLY", None)


def test_keep_errors_if_file_not_viewed():
    # ckpt-179: keep must reject a file/range the coder hasn't actually viewed.
    ctx = {"file_contents": {"m.py": "a=1\nb=2\nc=3\n"}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}   # nothing viewed (view_at absent)
    r = _disp("keep", {"path": "m.py", "ranges": [[1, 2]]}, ctx)
    assert r.startswith("✗") and "haven't viewed" in r


def test_keep_after_view_succeeds_and_validates_range():
    # After reading a ≤1k file (whole-viewed), keep a real range → ✓; an out-of-file range → ✗.
    ctx, rel, root = _mk_ctx()
    try:
        _disp("read_file", {"path": rel}, ctx)                 # now fully viewed
        ok = _disp("keep", {"path": rel, "ranges": [[1, 3]]}, ctx)
        assert ok.startswith("✓") and "keeping lines 1-3" in ok
        assert ctx["_kept"][rel] == [(1, 3)]
        bad = _disp("keep", {"path": rel, "ranges": [[1, 99999]]}, ctx)
        assert bad.startswith("✗") and "outside" in bad
        junk = _disp("keep", {"path": rel, "ranges": "notalist"}, ctx)  # wrong type → clean ✗, no crash
        assert junk.startswith("✗")
    finally:
        _cleanup(root)


def test_def_index_lists_defs_no_bodies():
    from core.native_tools import _def_index
    src = "import os\nclass Foo:\n    def bar(self):\n        return 1\ndef baz():\n    return 2\n"
    idx = _def_index(src, "m.py")
    assert "class Foo" in idx and "def bar" in idx and "def baz" in idx
    assert "return 1" not in idx and "import os" not in idx     # names only, no bodies/other lines


def test_expand_indent_strips_stray_tab_marker():
    # ckpt-178: a stray ⇥ (U+21E5, the view's indent glyph) copied into code must be removed —
    # it's never valid Python and caused an endless SyntaxError reject-loop (c580 via mistral).
    from core.native_tools import _expand_indent_lines as ex
    assert ex(["0|⇥def f():"]) == ["def f():"]          # ⇥ after the INDENT| gutter, in the code
    assert ex(["4|⇥    return 1"]) == ["    return 1"]   # indent from the number, ⇥ gone
    assert "⇥" not in "".join(ex(["⇥x = 1"]))           # literal path also stripped


def test_edit_with_stray_tab_marker_applies_not_loops():
    # The ⇥ must be stripped so the edit LANDS (parses) instead of looping on SyntaxError.
    src = "def f():\n    return 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"old": ["8|    return 1"], "new": ["8|⇥    return 2"]}]}, ctx)
    assert out.startswith("✓"), out
    assert "⇥" not in ctx["file_contents"]["m.py"] and "return 2" in ctx["file_contents"]["m.py"]


def test_edit_rejects_duplicate_toplevel_def():
    # ckpt-178: adding a SECOND top-level def of an existing name (the c580 dup-def, far apart)
    # is rejected — the gate previously only caught ADJACENT duplicates.
    src = "def widen(h):\n    return h\n\n\ndef other():\n    return 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    # insert a SECOND `def widen` after other() — a far-apart duplicate
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"old": ["6|    return 1"], "new": ["0|    return 1", "0|", "0|", "0|def widen(h):", "4|    return h.upper()"]}]}, ctx)
    assert out.startswith("✗") and "widen" in out and "shadows" in out
    assert ctx["file_contents"]["m.py"] == src              # rejected → file unchanged


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


def test_reread_unchanged_shortcircuits_no_body():
    # NEW CONTRACT (view-redesign): the coder ALREADY holds an unchanged file's view, so a
    # re-read does NOT re-dump it — it returns one line ("you already have this, up to date"),
    # no body, nothing capped. Repeating escalates the nudge. Never a ✗ (inert to fail-counter).
    src = "a = 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {"m.py": src},
           "project_root": ".", "files_changed": set(),
           "view_at": {"m.py": "step 1 (loaded at the start)"}}
    r1 = _disp("read_file", {"path": "m.py"}, ctx)
    assert r1.startswith("ℹ") and "ALREADY have" in r1 and "UP TO DATE" in r1
    assert "a = 1" not in r1                                   # no re-dump of the body
    r2 = _disp("read_file", {"path": "m.py"}, ctx)
    assert r2.startswith("ℹ") and "2×" in r2                   # repeat: escalated nudge
    assert not r2.startswith("✗")                              # inert to the fail-counter


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

def test_salvage_parses_leaked_tool_call_from_reasoning():
    # ckpt-148: gpt-oss on a cheap provider intermittently returns finish_reason=stop with
    # the tool call left as TEXT in the harmony reasoning channel. We parse it ourselves
    # ("do the native ourselves") instead of depending on the provider / paying for a
    # stricter one. Covers the real leak shapes seen in the f327 run.
    from core.native_tools import _salvage_inline_tool_call as sal
    # bare args object after intent words (f327 round-9 actual leak) -> read_file
    r = sal('Let us view around line 40-60.{"path":"x.py","start_line":40,"end_line":70}', 0)
    assert r and r["function"]["name"] == "read_file" and r["id"] == "salvage_0"
    # path+pattern -> search_text (pattern wins over path)
    assert sal('{"path":"m.py","pattern":"_is_fqcn"}', 1)["function"]["name"] == "search_text"
    # {name, arguments} wrapper
    assert sal('{"name":"read_file","arguments":{"path":"a.py"}}', 2)["function"]["name"] == "read_file"
    # edit with a brace INSIDE a string literal — string-aware matcher must not break
    e = sal('Now edit. {"old":["if x: {"],"new":["if x: pass"],"path":"a.py"}', 3)
    assert e and e["function"]["name"] == "edit_file"
    # plain reasoning with NO call -> None (no false salvage)
    assert sal('We should think about whether the function returns True here.', 4) is None
    # only real CODER_TOOLS tools are salvaged
    assert sal('{"name":"rm_rf","arguments":{"x":1}}', 5) is None


def test_edit_anchors_on_copied_view_line_number():
    # ckpt-144: copying the view line VERBATIM for `old` (keeping the `LINENO ⇥INDENT|`
    # prefix) anchors the edit on BOTH the line number AND the content. `return x`
    # repeats (lines 3 and 6); without a line anchor that's ambiguous, but the copied
    # view line for line 6 must edit line 6 and leave line 3 untouched.
    src = "def a():\n    x = 1\n    return x\n\ndef b():\n    return x\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    out = _disp("edit_file", {"path": "m.py",
                              "old": ["6 ⇥4|    return x"], "new": ["4|return x + 1"]}, ctx)
    assert out.startswith("✓"), out
    res = ctx["file_contents"]["m.py"].split("\n")
    assert res[2] == "    return x"          # line 3 (first occurrence) UNCHANGED
    assert res[5] == "    return x + 1"      # line 6 (the anchored one) changed
    # the bare write-form (no lineno) on a repeated line is still ambiguous → rejects
    ctx2 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
            "project_root": ".", "files_changed": set()}
    out2 = _disp("edit_file", {"path": "m.py",
                               "old": ["4|return x"], "new": ["4|return x + 1"]}, ctx2)
    assert out2.startswith("✗") and "appears 2 times" in out2


def test_edit_file_batch_applies_atomically_with_one_diff():
    # ckpt-151: `edits` = a batch of changes applied TOGETHER (one consolidated diff).
    # A multi-site refactor (delete a def AND fix its only call site) in ONE batch must
    # NOT trip the dangling-ref gate — the intermediate "deleted but still called" state
    # never exists, so the whole edit section lands clean.
    src = "def _h(x):\n    return x\n\ndef use(n):\n    return _h(n)\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "view_at": {}}
    out = _disp("edit_file", {"path": "m.py", "edits": [
        {"old": ["def _h(x):", "    return x"], "new": []},          # delete the helper
        {"old": ["    return _h(n)"], "new": ["    return n + 1"]},  # fix its call site
    ]}, ctx)
    assert out.startswith("✓"), out                        # applied — no dangling-ref reject
    res = ctx["file_contents"]["m.py"]
    assert "def _h" not in res and "_h(" not in res         # both edits landed
    assert "return n + 1" in res
    assert "Applied 2 edit(s)" in out and "CUMULATIVE change" in out


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

def test_reread_serves_current_content_not_refusal():
    # NEW CONTRACT: the FIRST full read serves the content (incrusted in the tree); a SECOND
    # full read short-circuits ("already have it") instead of re-dumping; a RANGE read of an
    # unseen slice still works (never refused).
    ctx, rel, root = _mk_ctx()
    try:
        first = _disp("read_file", {"path": rel}, ctx)
        assert "greet" in first                              # real content returned
        assert "=== VIEW:" in first and rel in first         # incrusted VIEW header names THIS file
        assert rel in ctx.get("view_at", {})                # view recorded
        second = _disp("read_file", {"path": rel}, ctx)
        assert second.startswith("ℹ") and "ALREADY have" in second   # short-circuit, no re-dump
        # ckpt-210: a NARROW range re-read of an already-viewed ≤cap file is now SERVED (the
        # whole-file view can be trimmed out of a long step's history → refusing it traps the
        # coder in a re-read spiral; a26 timeout). Only a WIDE re-read still short-circuits.
        rng = _disp("read_file", {"path": rel, "start_line": 1, "end_line": 3}, ctx)
        assert not rng.startswith("ℹ") and "RANGE" in rng and "lines 1-3" in rng
    finally:
        _cleanup(root)


def test_wide_reread_of_viewed_small_file_still_short_circuits():
    # ckpt-210: the narrow-re-read-serves fix must NOT reopen the f631 re-dump bloat. A WIDE
    # (>200-line) re-read of an already-viewed ≤cap file still short-circuits with a refusal
    # that points the coder at a narrow sub-range; only narrow (≤200-line) re-reads are served.
    import tempfile, os as _os
    root = _os.path.abspath(tempfile.mkdtemp(prefix="widereread_"))
    src = "".join(f"x{i} = {i}\n" for i in range(400))   # 400 lines (≤cap)
    with open(_os.path.join(root, "m.py"), "w") as _f:
        _f.write(src)
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "step_num": 1}
    try:
        _disp("read_file", {"path": "m.py"}, ctx)              # whole-file serve marks it viewed
        assert "m.py" in ctx.get("view_at", {})
        wide = _disp("read_file", {"path": "m.py", "start_line": 10, "end_line": 350}, ctx)   # 341 lines
        assert wide.startswith("ℹ") and "WHOLE file" in wide   # wide re-read still refused (no re-dump)
        narrow = _disp("read_file", {"path": "m.py", "start_line": 10, "end_line": 60}, ctx)  # 51 lines
        assert not narrow.startswith("ℹ") and "RANGE" in narrow   # narrow re-read served
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reread_after_edit_shows_diff_and_current_content():
    # NEW CONTRACT: after an edit, the edit RESULT already handed the coder the cumulative
    # diff = the file's live state, so a full re-read short-circuits ("already have it; your
    # edits are applied") rather than re-dumping. The edit's own diff is vs the STEP-START
    # baseline (cumulative), with live line numbers.
    src = "a = 1\nb = 2\nc = 3\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "round": 4, "step_num": 2,
           "_first_seen": {"m.py": src}}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 2, "old": ["b = 2"], "new": ["b = 20"]}]}, ctx)
    assert out.startswith("✓")
    assert "CUMULATIVE change" in out and "START of this step" in out   # diff framed vs step-start
    assert "2:+ b = 20" in out                               # the diff shows the change
    assert "m.py" in ctx.get("view_at", {})
    r = _disp("read_file", {"path": "m.py"}, ctx)
    assert r.startswith("ℹ") and "ALREADY have" in r         # short-circuit, not a re-dump
    assert "applied" in r                                    # told its edits are applied


def test_reread_injected_target_shortcircuits():
    # NEW CONTRACT: a file injected into the prompt (view_at + viewed_versions seeded at step
    # start) is ALREADY in the coder's context, so a full re-read short-circuits — it doesn't
    # re-dump the block that's already sitting in the user turn.
    ctx = {"file_contents": {"m.py": "a = 1\n"}, "sandbox": None,
           "viewed_versions": {"m.py": "a = 1\n"}, "project_root": ".", "files_changed": set(),
           "view_at": {"m.py": "step 1 (loaded at the start)"}}
    r = _disp("read_file", {"path": "m.py"}, ctx)
    assert r.startswith("ℹ") and "ALREADY have" in r and "a = 1" not in r


def test_large_file_first_read_is_def_index_not_full_dump():
    # CONTRACT: a file over _FULL_VIEW_CAP (1000 lines) is NEVER dumped in full — on the FIRST
    # whole-file read (no ranges revealed yet) it comes back as a DEF/CLASS INDEX (names + line
    # numbers, no bodies) + a "read a range" redirect. Reads then FILL it IN (accumulating view).
    big = "".join(f"def f{i}():\n    x = 1\n    y = 2\n    return {i}\n" for i in range(300))  # 1200 lines, 300 defs
    ctx = {"file_contents": {"big.py": big}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set()}
    r = _disp("read_file", {"path": "big.py"}, ctx)
    assert "TOO LARGE" in r and "line range" in r            # the >1000 view
    assert "def f0" in r and "def f299" in r                 # the def INDEX (names + lines, ≤500 cap)
    assert "return 0" not in r and "x = 1" not in r          # but NO bodies (just the index)
    assert len(r) < len(big)                                 # bounded
    assert "big.py" not in ctx.get("view_at", {})            # NOT a full view → range reads required


def test_moderate_file_first_read_serves_full_not_skeleton():
    # The skeleton trap fix, under the new contract: a MODERATE file (≤ _FULL_VIEW_CAP=1000
    # lines) read WITHOUT a range serves the REAL, line-numbered content — NOT a skeleton — so
    # the coder sees real indentation (guessing it wrong → IndentationError was the a26c325b /
    # 395e5e20 budget-exhaust on the ckpt-165 night run). Tested on the FIRST read (a re-read of
    # an unchanged file now short-circuits — see test_reread_unchanged_shortcircuits_no_body).
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="midfile_")
    mod = "".join(f"def f{i}():\n    return {i}\n" for i in range(350))   # 700 lines (≤3000)
    with open(_os.path.join(root, "mid.py"), "w") as _f:
        _f.write(mod)
    ctx = {"file_contents": {"mid.py": mod}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set()}
    try:
        r = _disp("read_file", {"path": "mid.py"}, ctx)
        assert "=== VIEW:" in r and "FULL file" in r             # full-content path, clearly labelled
        assert "TOO LARGE" not in r and "STRUCTURE only" not in r  # NOT the structure path
        assert "return 349" in r                                # real body lines present, not just defs
        assert "    return 0" in r or "⇥4|    return 0" in r     # real indentation visible
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_big_file_edit_invalidates_served_ranges_and_reread_serves_fresh():
    # ckpt-194 → SUPERSEDED by ckpt-205. ckpt-194 CLEARED _served_ranges on edit (to avoid stale
    # ranges) but that discarded the coder's accumulated view → re-read storm. ckpt-205 instead
    # SHIFTS the ranges by the edit's line delta and keeps the file in growing-view mode, so the
    # view SURVIVES the edit (no re-read needed) and a re-read is never the stale "already read"
    # refusal nor a no-body "you already have it".
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="bigreread_")
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(700))   # 1400 lines (>cap=1000)
    with open(_os.path.join(root, "big.py"), "w") as _f:
        _f.write(big)
    ctx = {"file_contents": {"big.py": big}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "round": 3, "step_num": 1,
           "_first_seen": {"big.py": big}}
    try:
        # 1) a range read of a >cap file reveals it + enters growing-view mode
        rng = _disp("read_file", {"path": "big.py", "start_line": 3, "end_line": 8}, ctx)
        assert "def f1" in rng
        assert ctx["_served_ranges"]["big.py"] == [(3, 8)]
        # 2) edit a line inside that range (delta 0 → range unchanged)
        out = _disp("edit_file", {"path": "big.py", "hunks": [
            {"start_line": 4, "old": ["    return 1"], "new": ["    return 111"]}]}, ctx)
        assert out.startswith("✓"), out
        # ckpt-205: the edit SHIFTS/keeps the revealed range (does NOT clear it)
        assert ctx["_served_ranges"]["big.py"] == [(3, 8)]
        # 3) a RANGE re-read of the same span is not the stale "you already read a-b" refusal
        rng2 = _disp("read_file", {"path": "big.py", "start_line": 3, "end_line": 8}, ctx)
        assert "you already read" not in rng2 and not rng2.lstrip().startswith("ℹ")
        # 4) a WHOLE-file re-read serves the GROWING VIEW (revealed region + its edit), never a
        # no-body "ALREADY have" — the coder keeps its accumulated view across the edit
        whole = _disp("read_file", {"path": "big.py"}, ctx)
        assert "GROWING VIEW" in whole and "return 111" in whole
        assert "ALREADY have" not in whole
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_accumulated_view_renders_holes_between_revealed_ranges():
    # ckpt-196: the growing view shows revealed ranges as real code, with the un-read GAP between
    # them rendered as a labelled hole that lists the defs inside it — clear even with holes.
    from core.native_tools import _accumulated_view
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(700))   # 1400 lines
    ctx = {"file_contents": {"big.py": big}, "_served_ranges": {"big.py": [(3, 4), (101, 102)]}}
    v = _accumulated_view(ctx, "big.py", big)
    assert "GROWING VIEW" in v
    assert "return 1" in v and "return 50" in v          # both revealed ranges shown as real code
    assert "not read" in v                               # the gap between them is a labelled hole
    assert "contains" in v and "5:def f2" in v           # the hole lists the defs inside it


def test_merge_ranges_combines_overlapping_and_adjacent():
    from core.native_tools import _merge_ranges
    assert _merge_ranges([(10, 20), (15, 25)]) == [(10, 25)]      # overlap
    assert _merge_ranges([(10, 20), (21, 30)]) == [(10, 30)]      # adjacent (touching) → one block
    assert _merge_ranges([(30, 40), (10, 20)]) == [(10, 20), (30, 40)]  # sorted, disjoint
    assert _merge_ranges([(5, 5), ("bad", None), (5, 5)]) == [(5, 5)]   # dedup + skip malformed


def test_supersede_prior_file_views_leaves_one_live_view():
    # ckpt-196: a read of a >cap file returns the ONE growing view; older views (and prior KEPT
    # blocks) of THAT file collapse to a pointer, the newest stays, OTHER files are untouched.
    from core.native_tools import _supersede_prior_file_views
    msgs = [
        {"role": "tool", "content": "=== VIEW: a/big.py — 2000 lines — GROWING VIEW (5) ===\n1 ⇥0|x"},
        {"role": "tool", "content": "=== VIEW: a/other.py — 50 lines (FULL file) ===\nstuff"},
        {"role": "tool", "content": "⟪KEPT only lines 10-20 of a/big.py (you called keep)…⟫\nkept"},
        {"role": "tool", "content": "=== VIEW: a/big.py — 2000 lines — GROWING VIEW (40) ===\nNEWEST"},
    ]
    n = _supersede_prior_file_views(msgs, "a/big.py")
    assert n == 2                                              # the old view + the KEPT block
    assert "⟪earlier view of a/big.py" in msgs[0]["content"]
    assert "⟪earlier view of a/big.py" in msgs[2]["content"]
    assert "stuff" in msgs[1]["content"] and "⟪earlier" not in msgs[1]["content"]  # other file untouched
    assert "NEWEST" in msgs[3]["content"]                      # the newest (last) view is left intact


def test_keep_sets_served_ranges_to_kept():
    # ckpt-196: keep is the trim for the growing view — it sets the revealed ranges to exactly
    # what's kept, so a later read shows only the kept ranges (+ new) and the rest is a hole.
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="keepacc_")
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(700))   # 1400 lines (>cap)
    with open(_os.path.join(root, "big.py"), "w") as _f:
        _f.write(big)
    ctx = {"file_contents": {"big.py": big}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "step_num": 1}
    try:
        _disp("read_file", {"path": "big.py", "start_line": 3, "end_line": 8}, ctx)
        _disp("read_file", {"path": "big.py", "start_line": 100, "end_line": 110}, ctx)
        assert ctx["_served_ranges"]["big.py"] == [(3, 8), (100, 110)]
        r = _disp("keep", {"path": "big.py", "ranges": [[100, 110]]}, ctx)
        assert r.startswith("✓")
        assert ctx["_served_ranges"]["big.py"] == [(100, 110)]   # revealed trimmed to kept
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shorthand_insert_carries_start_line():
    # ckpt-199: the natural single insert `edit_file(path, old=[], new=[...], start_line=N)` must
    # work — it was dropping the top-level start_line when synthesizing the hunk, so it hit the
    # "old is empty" reject while the identical nested `edits=[{...,start_line}]` inserted cleanly
    # (recurring empty-old insert failure, e.g. f327). Verifies the shorthand now matches nested.
    src = "a = 1\nb = 2\nc = 3\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "files_changed": set(), "round": 1, "_first_seen": {"m.py": src}}
    out = _disp("edit_file", {"path": "m.py", "old": [], "new": ["x = 99"], "start_line": 2}, ctx)
    assert out.startswith("✓"), out
    assert ctx["file_contents"]["m.py"] == "a = 1\nb = 2\nx = 99\nc = 3\n"
    # and the nested form still works identically (the reference behavior)
    ctx2 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
            "files_changed": set(), "round": 1, "_first_seen": {"m.py": src}}
    out2 = _disp("edit_file", {"path": "m.py", "edits": [
        {"old": [], "new": ["x = 99"], "start_line": 2}]}, ctx2)
    assert out2.startswith("✓") and ctx2["file_contents"]["m.py"] == ctx["file_contents"]["m.py"]


def test_keep_small_file_does_not_dead_end_dropped_region():
    # bughunt #1/#4 (ckpt-200): keep evicts the file body; a SMALL fully-held file used to keep its
    # view_at, so a later read of a DROPPED region was refused ("you already hold the whole file")
    # → the coder could never anchor an edit there → stale-anchor loop → budget exhaust. keep now
    # drops the fully-held claim, so a dropped region RE-SERVES.
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="keepdead_")
    src = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"   # 20-line small file
    with open(_os.path.join(root, "s.py"), "w") as _f:
        _f.write(src)
    ctx = {"file_contents": {"s.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "step_num": 1}
    try:
        _disp("read_file", {"path": "s.py"}, ctx)                  # full read → view_at set
        assert "s.py" in ctx.get("view_at", {})
        k = _disp("keep", {"path": "s.py", "ranges": [[1, 2]]}, ctx)
        assert k.startswith("✓")
        assert "s.py" not in ctx.get("view_at", {})                # fully-held claim dropped
        r = _disp("read_file", {"path": "s.py", "start_line": 10, "end_line": 12}, ctx)
        assert not (r.lstrip().startswith("ℹ") and "already" in r)  # NOT refused
        assert "line10" in r                                       # re-serves the dropped region
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_msg_is_view_of_is_header_precise():
    # bughunt #15/#16 (ckpt-200): the supersede/keep-eviction loops used a loose `path in content`
    # test, so a path merely MENTIONED in another file's view (an import line) collapsed THAT file's
    # view by mistake. _msg_is_view_of is header-precise.
    from core.native_tools import _msg_is_view_of
    v = "=== VIEW: a/urls.py — 2000 lines — GROWING VIEW (5) ===\n1 ⇥0|x"
    assert _msg_is_view_of(v, "a/urls.py")
    assert not _msg_is_view_of(v, "a/url.py")                  # not a prefix false-match
    other = "=== VIEW: a/other.py — 10 lines (FULL file) ===\n1 ⇥0|import a.urls  # a/urls.py"
    assert _msg_is_view_of(other, "a/other.py")
    assert not _msg_is_view_of(other, "a/urls.py")            # the loose substring bug — must NOT match
    kept = "⟪KEPT only lines 1-2 of a/urls.py (you called keep)…⟫\nkept"
    assert _msg_is_view_of(kept, "a/urls.py") and not _msg_is_view_of(kept, "a/other.py")


def test_edit_diff_stamped_with_round_when_no_step():
    src = "a = 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "round": 7}
    out = _disp("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 1, "old": ["a = 1"], "new": ["a = 2"]}]}, ctx)
    # the WHEN-stamp (right after the em-dash) must be round-only, not the "step S, round R" form
    assert out.startswith("✓") and "— round 7." in out and ", round 7" not in out


def test_edit_cot_verification_removed_even_when_flag_on():
    """ckpt-133: the EDIT-COT VERIFICATION is removed. The grounding SLOTS
    (goal/traced/check) and the prompt that invites them are KEPT — the coder may
    still reason in them — but the harness NO LONGER REJECTS an edit for missing or
    ungrounded fields. (The verbatim-`traced`-quote teeth tripped weak models into
    8×-reject loops on hard steps, cost an instance a timeout, and forced a rigid
    reasoning template that gamed the format instead of helping.) So even with the
    flag ON, an edit with NO grounding fields APPLIES — `old` already carries the
    real, content-verified line, so no real grounding is lost."""
    import os, importlib
    src = "class Bar:\n    def a(self):\n        return 1\n"
    mkctx = lambda: {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
                     "project_root": ".", "files_changed": set()}
    hunk = [{"start_line": 3, "old": ["        return 1"], "new": ["        return 2"]}]
    os.environ["JARVIS_EDIT_COT"] = "1"
    try:
        import core.native_tools as nt; importlib.reload(nt)
        run = lambda a: asyncio.run(nt._dispatch("edit_file", a, mkctx()))
        # flag ON but NO grounding fields → APPLIES (verification gone, no reject)
        assert run({"path": "m.py", "hunks": hunk}).startswith("✓")
        # a hedged / unquoted `traced` no longer rejects either
        assert run({"path": "m.py", "goal": "make a() return 2 per spec",
            "traced": "it probably returns 1", "check": "calling a() returns 2",
            "hunks": hunk}).startswith("✓")
        # a grounded edit still applies (slots are harmless when filled)
        ok = run({"path": "m.py", "goal": "make a() return 2 as the spec requires",
            "traced": "a() runs `return 1`", "check": "a() returns 2 not 1", "hunks": hunk})
        assert ok.startswith("✓"), ok
    finally:
        os.environ.pop("JARVIS_EDIT_COT", None)
        import core.native_tools as nt; importlib.reload(nt)
    # flag OFF: ungrounded edit applies (unchanged)
    assert asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "hunks": hunk}, mkctx())).startswith("✓")


def test_edit_success_messages_dont_tell_coder_to_paste_a_diff():
    # ckpt-130: the strict applier no longer accepts a diff row as editable input, so the
    # post-edit success messages must NOT tell the coder to copy its next `old` from the
    # diff — they must point to the canonical INDENT|code form. (pass-5 leftover fix.)
    import os
    os.environ.pop("JARVIS_EDIT_COT", None)
    from core.native_tools import _do_edit, _do_replace
    ctx = {"file_contents": {"m.py": "def f():\n    return 1\n"}, "files_changed": set()}
    r = _do_edit({"path": "m.py", "hunks": [{"old": ["8|return 1"], "new": ["8|return 2"]}]}, ctx)
    assert r.startswith("✓") and "INDENT|code" in r
    assert "from this diff" not in r.lower()
    ctx2 = {"file_contents": {"m.py": "def f():\n    return 1\n"}, "files_changed": set(),
            "viewed_versions": {"m.py": "def f():\n    return 1\n"}}
    r2 = _do_replace({"path": "m.py", "start_line": 2, "end_line": 2, "new_content": "8|return 2"}, ctx2)
    assert "copy your next edit's `old` line(s) from this diff" not in r2


def test_old_not_found_reject_shows_actual_lines_for_recovery():
    """ckpt-134: the #1 reject-loop cause was a skeleton view of a big file → the coder
    builds `old` for a body it never saw → permanent no-match loop. The no-match reject
    must now SHOW the real current lines (LINENO:INDENT|code) at the intended site so the
    coder can copy a valid `old` instead of re-sending an imagined one."""
    import os
    os.environ.pop("JARVIS_EDIT_COT", None)
    import core.native_tools as nt
    src = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    # start_line points at a real region but `old` is imagined → must reject WITH the real lines
    r = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "hunks": [
        {"start_line": 3, "old": ["8|y = 999  # imagined"], "new": ["8|y = 5"]}]}, ctx))
    assert r.startswith("✗")
    assert "ACTUAL current lines" in r
    assert "3 ⇥4|    y = 2" in r       # real line 3 in canonical view form: LINENO ⇥INDENT|<real spaces>code
    # ckpt-147: byte-identical to a normal read-view line (no extra prefix), so copying it
    # VERBATIM as `old` parses cleanly through _VIEW_LINE_RE instead of failing to match.
    from core.native_tools import _expand_indent_lines as _ex
    _hint_line = next(l.strip() for l in r.splitlines() if "⇥4|" in l and "y = 2" in l)
    assert _ex([_hint_line]) == ["    y = 2"]   # round-trips: a verbatim copy resolves
    # no start_line: fuzzy-locate still surfaces the nearest real line
    r2 = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "hunks": [
        {"old": ["    return x + y - 1"], "new": ["    return x"]}]}, ctx))
    assert r2.startswith("✗") and "ACTUAL current lines" in r2 and "return x + y" in r2


def test_block_reject_tells_coder_to_replace_whole_block():
    """ckpt-137/138: a whole-block reject (stranded return = unreachable / duplicated
    anchor) must tell the coder to put the ENTIRE block (the resolved line span) in
    old→new, not patch a fragment. (ckpt-138 collapsed to one edit tool, so the message
    no longer points at replace_lines — it points at making `old` the whole span.)"""
    import os
    os.environ.pop("JARVIS_EDIT_COT", None)
    import core.native_tools as nt
    src = "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a\n"   # clean baseline, no dead code
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    # rewrite the 4-line body so the result leaves code unreachable after a return
    r = asyncio.run(nt._dispatch("edit_file", {"path": "m.py",
        "old": ["4|a = 1", "4|b = 2", "4|c = 3", "4|return a"],
        "new": ["4|return 1", "4|dead1 = 1", "4|dead2 = 2"]}, ctx))
    assert r.startswith("✗")
    assert "whole CONTIGUOUS span" in r and "2 to 5" in r, \
        "block reject must tell the coder to put the whole span (lines 2-5) in old/new"


def test_indent_parse_fail_retry_trusts_typed_spaces():
    """ckpt-155: the `0|    x` dual-channel slip (number 0, but real spaces typed) makes the
    number-based expansion produce indent 0 → IndentationError. The harness retries trusting
    the typed spaces and the edit lands. An intentional dedent (smaller number + stale spaces)
    PARSES, so it never reaches the retry — number stays authoritative."""
    import core.native_tools as nt
    # SLIP: insert a method body with `0|        return 2` (8 spaces typed, number 0).
    src = "class C:\n    def a(self):\n        return 1\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "old": [
        "2 ⇥4|    def a(self):", "3 ⇥8|        return 1"],
        "new": ["4|def a(self):", "8|return 1", "4|def b(self):", "0|        return 2"]}, ctx))
    assert r.startswith("✓"), r
    assert "typed spaces" in r                      # surfaced the auto-fix
    res = ctx["file_contents"]["m.py"]
    assert "        return 2" in res                # body landed at indent 8, not 0
    import ast; ast.parse(res)                      # and the file parses

    # DEDENT: `8|            y = 1` (number 8, 12 spaces) — number-based parses, number wins.
    src2 = "def f():\n    if x:\n            y = 1\n"
    ctx2 = {"file_contents": {"m.py": src2}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r2 = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "old": [
        "2 ⇥4|    if x:", "3 ⇥12|            y = 1"],
        "new": ["4|if x:", "8|            y = 1"]}, ctx2))
    assert r2.startswith("✓"), r2
    assert "typed spaces" not in r2                 # no retry: number-based already parsed
    assert "\n        y = 1" in ctx2["file_contents"]["m.py"]   # dedented to 8 (the number)


def test_overlapping_hunks_merge_context_reject_changed_conflict():
    """ckpt-158: hunks anchor on the pre-edit file, so two hunks sharing only UNCHANGED
    context lines are harmless — MERGE them (dedupe the shared line) and apply as one.
    REJECT only when two hunks both CHANGE the same shared line. Far-apart hunks (any
    order) apply; cross-hunk order is auto-sorted."""
    import core.native_tools as nt
    src = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
    # CONTEXT overlap: A changes b (keeps c), B changes d (keeps c) → share line 3 (c) as
    # context in both → MERGE & apply.
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "edits": [
        {"old": ["2 ⇥0|b = 2", "3 ⇥0|c = 3"], "new": ["0|b = 20", "0|c = 3"]},
        {"old": ["3 ⇥0|c = 3", "4 ⇥0|d = 4"], "new": ["0|c = 3", "0|d = 40"]}]}, ctx))
    assert r.startswith("✓"), r
    res = ctx["file_contents"]["m.py"]
    assert "b = 20" in res and "d = 40" in res and res.count("c = 3") == 1, res

    # CHANGED-line conflict: both hunks change line 3 (c) differently → reject.
    ctx2 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r2 = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "edits": [
        {"old": ["2 ⇥0|b = 2", "3 ⇥0|c = 3"], "new": ["0|b = 20", "0|c = 30"]},
        {"old": ["3 ⇥0|c = 3", "4 ⇥0|d = 4"], "new": ["0|c = 99", "0|d = 4"]}]}, ctx2))
    assert r2.startswith("✗") and "conflict" in r2.lower(), r2
    assert "out of order" not in r2

    # far-apart hunks given in REVERSED order apply cleanly (sort handles order)
    ctx3 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r3 = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "edits": [
        {"old": ["5 ⇥0|e = 5"], "new": ["0|e = 50"]},
        {"old": ["1 ⇥0|a = 1"], "new": ["0|a = 10"]}]}, ctx3))
    assert r3.startswith("✓"), r3
    assert "a = 10" in ctx3["file_contents"]["m.py"] and "e = 50" in ctx3["file_contents"]["m.py"]


def test_indent_retry_covers_col0_statement_and_no_leak_on_reject():
    """ckpt-157 (bigger-audit follow-up): (a) the retry must fire for a col-0 `return`
    slip — ast.parse ACCEPTS a module-level return but compile() rejects it, so the gate
    is compile-based; (b) a genuine syntax error must leave file_contents UNCHANGED (the
    reject says 'file UNCHANGED — view current', which must be TRUE)."""
    import core.native_tools as nt, ast
    # (a) col-0 return slip WITH a content change → retry trusts typed spaces → lands at 4
    src = "def greet(name):\n    msg = 'hi'\n    return msg + name\n"
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r = asyncio.run(nt._dispatch("edit_file", {"path": "m.py", "old": [
        "2 ⇥4|    msg = 'hi'", "3 ⇥4|    return msg + name"],
        "new": ["4|msg = 'hi'", "0|    return msg.upper() + name"]}, ctx))
    assert r.startswith("✓"), r
    assert "typed spaces" in r
    out = ctx["file_contents"]["m.py"]
    assert "    return msg.upper() + name" in out
    ast.parse(out)
    # (b) genuine syntax error → clean reject, file_contents identical to src (no leak)
    ctx2 = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {}, "files_changed": set()}
    r2 = asyncio.run(nt._dispatch("edit_file", {"path": "m.py",
        "old": ["3 ⇥4|    return msg + name"], "new": ["4|return ("]}, ctx2))
    assert r2.startswith("✗"), r2
    assert ctx2["file_contents"]["m.py"] == src, "LEAK: rejected content left in file_contents"


def test_growing_view_survives_edits_no_reread_storm():
    # ROOT FIX (ckpt-205): the 79-reads/10-edits storm cause was ckpt-194 POPPING the file's
    # revealed ranges on every edit — the coder lost its accumulated view and re-read every region
    # before each subsequent edit. Now the ranges SHIFT (not cleared) and the file stays in
    # growing-view mode across edits, so a later read of a NEW region still carries the earlier one.
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="growpersist_")
    big = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(700))   # ~2100 lines (>cap)
    with open(_os.path.join(root, "u.py"), "w") as _f:
        _f.write(big)
    ctx = {"file_contents": {"u.py": big}, "sandbox": None, "viewed_versions": {},
           "project_root": root, "files_changed": set(), "step_num": 1, "view_at": {},
           "_first_seen": {"u.py": big}}
    try:
        _disp("read_file", {"path": "u.py", "start_line": 1300, "end_line": 1320}, ctx)
        assert ctx["_served_ranges"]["u.py"] == [(1300, 1320)]
        assert "u.py" in ctx.get("_accum", set())              # growing-view mode
        # edit a line in the revealed region (same line count → delta 0)
        out = _disp("edit_file", {"path": "u.py", "start_line": 1301,
                                  "old": ["1301 ⇥4|    return x + 433"], "new": ["4|    return x + 999"]}, ctx)
        assert out.startswith("✓"), out
        assert ctx["_served_ranges"].get("u.py") == [(1300, 1320)]   # SHIFTED/kept, NOT popped
        # read a NEW region — the view must STILL carry the first region (the edit), no re-read needed
        r2 = _disp("read_file", {"path": "u.py", "start_line": 1480, "end_line": 1500}, ctx)
        assert ctx["_served_ranges"]["u.py"] == [(1300, 1320), (1480, 1500)]
        assert "GROWING VIEW" in r2                              # growing view still engaged post-edit
        assert "return x + 999" in r2                            # the earlier region + its edit retained
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resync_served_ranges_shifts_by_line_delta():
    from core.native_tools import _resync_served_ranges_after_edit
    before = "\n".join(f"L{i}" for i in range(1, 301))
    al = [f"L{i}" for i in range(1, 301)]
    al[104:104] = ["NEW1", "NEW2", "NEW3"]              # insert after line 104 → L=105, delta +3
    after = "\n".join(al)
    ctx = {"_served_ranges": {"f": [(10, 20), (100, 110), (200, 210)]}, "_accum": {"f"}}
    _resync_served_ranges_after_edit(ctx, "f", before, after)
    assert ctx["_served_ranges"]["f"] == [(10, 20), (100, 113), (203, 213)]   # above / spans / below
    # a NON-growing-view file (not in _accum) still gets the old pop behavior
    ctx2 = {"_served_ranges": {"g": [(1, 5)]}, "_accum": set()}
    _resync_served_ranges_after_edit(ctx2, "g", "a\n", "a\nb\n")
    assert "g" not in ctx2["_served_ranges"]


def test_multihunk_stale_start_lines_apply_not_out_of_order():
    # ckpt-206: the coder often gives STALE start_lines (it edits from a view whose numbers
    # shifted). When each hunk's `old` is UNIQUELY locatable, the edit must APPLY — not get the
    # whole batch rejected "edit lines out of order" (a26: 5 of 7 rejects were this → retry pileup).
    src = "def a():\n    return 1\n\ndef b():\n    return 2\n"   # return 1 @ line 2, return 2 @ line 5
    ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
           "files_changed": set(), "round": 1, "_first_seen": {"m.py": src}}
    out = _disp("edit_file", {"path": "m.py", "edits": [
        {"start_line": 2, "old": ["    return 2"], "new": ["4|    return 22"]},  # content is at line 5
        {"start_line": 5, "old": ["    return 1"], "new": ["4|    return 11"]},  # content is at line 2
    ]}, ctx)
    assert out.startswith("✓"), out                          # applied, NOT "out of order"
    assert "out of order" not in out
    assert ctx["file_contents"]["m.py"] == "def a():\n    return 11\n\ndef b():\n    return 22\n"


def test_read_file_accepts_line_start_end_aliases():
    # ckpt-209: gpt-oss repeatedly sent line_start/line_end (and start/end) instead of the schema's
    # start_line/end_line, so read_file DROPPED the range → whole-file/def-index → the coder re-read
    # the SAME region 11× without ever getting its lines (a26's read-spin). Aliases must be honored.
    import tempfile, os as _os
    root = tempfile.mkdtemp(prefix="alias_")
    big = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(700))   # ~2100 lines (>cap)
    with open(_os.path.join(root, "u.py"), "w") as _f:
        _f.write(big)
    for k_s, k_e in [("line_start", "line_end"), ("lineStart", "lineEnd"), ("start", "end")]:
        ctx = {"file_contents": {"u.py": big}, "sandbox": None, "viewed_versions": {},
               "project_root": root, "files_changed": set(), "step_num": 1, "view_at": {},
               "_first_seen": {"u.py": big}}
        r = _disp("read_file", {"path": "u.py", k_s: 1300, k_e: 1320}, ctx)
        assert ctx["_served_ranges"]["u.py"] == [(1300, 1320)], f"{k_s}/{k_e} dropped: {ctx.get('_served_ranges')}"
        assert "return x + 433" in r                          # the wanted region, not a def-index
    shutil.rmtree(root, ignore_errors=True)
    # the half-range error still fires for a genuine single bound (no alias present)
    from core.native_tools import _range_arg
    assert _range_arg({"start_line": 5}) == 5 and _range_arg({"path": "x"}) is None


# ───────────────────── Batch A fixes (ckpt-213) ─────────────────────

def test_read_file_carrying_edits_reroutes_to_edit(monkeypatch=None):
    # #17: a read_file call that carries an edit-only `edits` payload is a mis-named edit_file —
    # re-route on the uniquely-identifying key instead of dropping it and returning a no-op re-read.
    ctx, rel, root = _mk_ctx()
    try:
        # read_file{path, edits:[…]} must APPLY the edit (✓), not return an ℹ re-read no-op.
        r = _disp("read_file", {"path": rel, "edits": [
            {"old": ['    return "hello " + name'], "new": ['    return "hi " + name']}]}, ctx)
        assert r.startswith("✓"), r
        assert 'hi ' in ctx["file_contents"][rel]
    finally:
        _cleanup(root)


def test_search_text_path_scopes_results():
    # #10: search_text honors an optional `path` scope (ripgrep -g) instead of silently searching
    # repo-wide. A scoped search to one file must NOT return hits from other files.
    # #1 (ckpt-221 REGRESSION guard): use a DEEP nested path — a ripgrep -g glob containing '/' is
    # anchored to the absolute search root, so a bare `lib/.../urls.py` glob never matches → false
    # "no matches". The fix prefixes `**/`. The old test passed by luck on a ONE-level path; this
    # 4-level path is the shape that actually broke in production (a26 lib/ansible/module_utils/).
    import tempfile, os as _os
    root = _os.path.abspath(tempfile.mkdtemp(prefix="searchscope_"))
    _deep = _os.path.join(root, "lib", "ansible", "module_utils")
    _os.makedirs(_deep, exist_ok=True)
    with open(_os.path.join(_deep, "target.py"), "w") as f:
        f.write("def open_url():\n    pass\n")
    with open(_os.path.join(root, "other.py"), "w") as f:
        f.write("# open_url called here\nopen_url()\n")
    ctx = {"project_root": root, "file_contents": {}, "sandbox": None, "files_changed": set()}
    try:
        scoped = _disp("search_text", {"pattern": "open_url",
                                       "path": "lib/ansible/module_utils/target.py"}, ctx)
        assert "target.py" in scoped, scoped     # the DEEP-path scope MUST still find the symbol
        assert "other.py" not in scoped          # the scope excluded the other file
        wide = _disp("search_text", {"pattern": "open_url"}, ctx)
        assert "other.py" in wide                # repo-wide still sees both
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_docstring_insert_warning_flags_code_in_docstring():
    # #6: a real statement wedged inside a triple-quoted docstring is inert; warn (the parse gate
    # can't catch it). A genuine docstring-PROSE edit or code OUTSIDE the docstring is NOT flagged.
    from core.native_tools import _docstring_insert_warning
    before = ('class R:\n    def __init__(self):\n        """Doc.\n\n        >>> R()\n'
              '        """\n        self.a = 1\n')
    after_bug = ('class R:\n    def __init__(self):\n        """Doc.\n\n        >>> R()\n'
                 '        self.b = 2\n        """\n        self.a = 1\n')   # self.b INSIDE docstring
    after_ok = ('class R:\n    def __init__(self):\n        """Doc.\n\n        >>> R()\n'
                '        """\n        self.a = 1\n        self.b = 2\n')     # self.b after docstring
    assert "INSIDE a docstring" in _docstring_insert_warning(before, after_bug)
    assert _docstring_insert_warning(before, after_ok) == ""
    assert _docstring_insert_warning(before, before) == ""                  # no change → no warn


def test_growing_view_marks_requested_range_and_collapses_header():
    # #3: a ranged read of a >cap file leads with the requested range (named + ◀ REQUESTED marker)
    # and COLLAPSES a large already-read leading header instead of re-printing it on every read.
    from core.native_tools import _accumulated_view
    big = "".join(f"line{i} = {i}\n" for i in range(1, 2001))   # 2000 lines (>cap)
    # header 1-60 read earlier, focus is a deep range 1500-1520
    ctx = {"file_contents": {"big.py": big}, "_served_ranges": {"big.py": [(1, 60), (1500, 1520)]}}
    v = _accumulated_view(ctx, "big.py", big, focus=(1500, 1520))
    assert "you requested lines 1500-1520" in v          # named at top
    assert "◀ REQUESTED" in v                            # the focus line is marked
    assert "line1500 = 1500  ◀ REQUESTED" in v
    assert "lines 1-60 read earlier — collapsed" in v    # the leading header is collapsed, not re-printed
    assert "line30 = 30" not in v                        # collapsed header body is NOT rendered
    # but if the focus IS the header, it is rendered (not collapsed)
    v2 = _accumulated_view(ctx, "big.py", big, focus=(1, 60))
    assert "line30 = 30" in v2 and "◀ REQUESTED" in v2
