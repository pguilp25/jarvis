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


# ── MEDIUM: self-explaining failure messages (what / why / how) ────────────────

def test_view_not_found_tells_how_to_recover():
    """VIEW on a missing file must say WHY (not found) and HOW (search/refs)."""
    import asyncio, tempfile, os
    import core.tool_call as TC

    root = tempfile.mkdtemp(prefix="viewnf_")
    open(os.path.join(root, "real.py"), "w").write("x = 1\n")

    blob = asyncio.run(TC._run_view(["nope.py 1-5"], root, {}))
    assert "file not found" in blob.lower()
    assert "SEARCH" in blob or "REFS" in blob, f"no recovery hint: {blob}"


def test_websearch_failure_says_do_not_retry():
    import asyncio
    import core.tool_call as TC
    import tools.search as S

    async def _boom(*a, **k):
        raise RuntimeError("no network")
    orig = S.web_search
    S.web_search = _boom
    try:
        out = asyncio.run(TC._run_web_searches(["anything"]))
    finally:
        S.web_search = orig
    assert "do NOT retry" in out and ("SEARCH" in out or "CODE" in out)


def test_semantic_unavailable_says_do_not_retry():
    """The SEMANTIC key-missing / build-failure message must steer to alternatives."""
    import inspect
    import tools.embeddings as E
    src = inspect.getsource(E)
    assert "Do NOT retry" in src and "[SEARCH:" in src


# ── 2nd-pass audit: parser silent-drops (missing/spaced colon, ] in arg) ───────

def test_missing_colon_tool_is_surfaced_not_dropped():
    """`[SEARCH foo]` (no colon) must be flagged, not silently vanish."""
    for bad in ("[tool use]\n[SEARCH foo]\n[/tool use]",
                "[tool use]\n[CODE foo.py]\n[/tool use]"):
        d = TagDetector(bad)
        assert d.valid_tags() == [], f"fired despite missing colon: {bad}"
        assert any(t.rejection_reason == "missing-colon" for t in d.all_tags), \
            f"missing colon not surfaced: {bad}"


def test_space_before_colon_is_accepted():
    d = TagDetector("[tool use]\n[SEARCH : my query]\n[/tool use]")
    assert any(t.tag_type == "SEARCH" and t.clean_arg == "my query"
               for t in d.valid_tags())


def test_bracket_in_search_arg_fires_one_balanced_query():
    """A regex char-class `]` inside a SEARCH arg must NOT truncate the query
    nor produce a duplicate truncated tag — exactly one full query fires."""
    d = TagDetector('[tool use]\n[SEARCH: re.match(r"[0-9]+")]\n[/tool use]')
    vt = [t for t in d.valid_tags() if t.tag_type == "SEARCH"]
    assert len(vt) == 1, f"expected 1 query, got {[t.clean_arg for t in vt]}"
    assert vt[0].clean_arg == 're.match(r"[0-9]+")'


def test_prose_missing_colon_outside_tooluse_not_flagged():
    d = TagDetector("I will [NOTE check the cache] then continue.")
    assert not any(t.rejection_reason == "missing-colon" for t in d.all_tags)


# ── 2nd-pass audit: KEEP/VIEW failure paths persist (not blank to the model) ───

def test_view_empty_file_persists_message():
    import asyncio, tempfile, os
    import core.tool_call as TC
    root = tempfile.mkdtemp(prefix="viewempty_")
    os.makedirs(os.path.join(root, ".jarvis_sandbox"), exist_ok=True)
    open(os.path.join(root, "empty.py"), "w").write("")
    pl = {}
    out = asyncio.run(TC._run_view(["empty.py 1-5"], root, pl))
    assert "EMPTY" in out
    # persisted under a key so the next round sees it (not a silent blank)
    assert any("EMPTY" in v for v in pl.values()), "empty-file VIEW not persisted"


def test_view_executor_crash_does_not_escape_loop(monkeypatch):
    """A raise inside _run_view must be caught and surfaced, never crash the loop."""
    import asyncio, tempfile, os
    import core.tool_call as TC
    root = tempfile.mkdtemp(prefix="viewboom_")
    open(os.path.join(root, "real.py"), "w").write("x=1\n")

    async def _boom(*a, **k):
        raise RuntimeError("view exploded")
    monkeypatch.setattr(TC, "_run_view", _boom)

    prompts = []
    responses = ["[tool use]\n[VIEW: real.py 1-3]\n[/tool use]\n[STOP][CONFIRM_STOP]",
                 "[PLAN DONE][CONFIRM_PLAN_DONE]"]
    n = {"i": 0}
    async def _fake_retry(model, prompt, **kw):
        prompts.append(prompt)
        i = n["i"]; n["i"] += 1
        return responses[i] if i < len(responses) else "[PLAN DONE][CONFIRM_PLAN_DONE]"
    monkeypatch.setattr(TC, "call_with_retry", _fake_retry)

    # Must not raise.
    asyncio.run(TC.call_with_tools("test/model", "[SYSTEM] do work", project_root=root,
                                   max_rounds=4, enable_web_search=False))
    assert len(prompts) >= 2, "loop crashed on the VIEW exception"
    assert "VIEW" in prompts[1] and ("✗" in prompts[1] or "failed" in prompts[1])


# ── 2nd-pass audit: native read_file / create_file arg validation ──────────────

def test_native_read_one_bound_is_rejected():
    import asyncio
    from core.native_tools import _do_read
    ctx = {"project_root": "/tmp", "file_contents": {}}
    out = asyncio.run(_do_read({"path": "x.py", "start_line": 50}, ctx))
    assert out.startswith("✗") and "BOTH" in out


def test_native_create_empty_content_is_rejected():
    import asyncio, inspect
    from core.native_tools import _do_create
    ctx = {"file_contents": {}, "sandbox": None}
    res = _do_create({"path": "new.py", "content": "   "}, ctx)
    out = asyncio.run(res) if inspect.iscoroutine(res) else res
    assert out.startswith("✗") and "content" in out.lower()


# ── 2nd-pass audit: edit_block CRLF + keep/del-same-line ───────────────────────

def test_edit_block_crlf_stays_uniform():
    from core.edit_block import apply_edit_block
    src = "def f():\r\n    x = 1\r\n    return x\r\n"
    new, _info, _w = apply_edit_block(src, "[edit]\n2:-     x = 1\n2:+     x = 10\n[/edit]")
    assert new is not None and "\n" not in new.replace("\r\n", ""), "mixed line endings"


def test_edit_block_keep_and_delete_same_line_clear_message():
    from core.edit_block import apply_edit_block
    new, info, _ = apply_edit_block("a=1\nx=1\nb=2\n",
                                    "[edit]\n2: x=1\n2:- x=1\n2:+ x=10\n[/edit]")
    assert new is None and ("KEPT and DELETED" in info or "listed twice" in info)
    assert "out of order" not in info


# ── 2nd-pass audit: embeddings poisoned-cache guard ────────────────────────────

def test_embeddings_poisoned_build_detected():
    from tools.embeddings import _build_wholly_failed
    assert _build_wholly_failed([{"vec": [0] * 8}, {"vec": [0] * 8}])
    assert not _build_wholly_failed([{"vec": [0] * 8}, {"vec": [1, 2, 3, 4, 5]}])
    assert not _build_wholly_failed([])


def test_embeddings_cosine_is_finite():
    from tools.embeddings import cosine_similarity
    import math
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert math.isfinite(cosine_similarity([1, 2, 3], [4, 5, 6]))


# ── 3rd-pass audit: unbalanced `[` in an arg must not swallow a following tag ───

def test_unbalanced_bracket_does_not_swallow_following_tag():
    """A stray unbalanced `[` in a SEARCH arg must not consume a following tag —
    the next tag (here a missing-colon REFS) must still be surfaced."""
    d = TagDetector("[tool use][SEARCH: arr[i\n[REFS sym][/tool use]")
    assert any(t.tag_type == "REFS" and t.rejection_reason == "missing-colon"
               for t in d.all_tags), "following REFS tag was swallowed"
    # and a well-formed tag after a stray `[` is unaffected
    d2 = TagDetector("[tool use][SEARCH: foo[bar\n[CODE: a.py][/tool use]")
    assert any(t.tag_type == "CODE" and t.valid and t.clean_arg == "a.py"
               for t in d2.valid_tags())


# ── 3rd-pass audit: no-op edits are reported as NO CHANGE, not ✓ APPLIED ───────

def test_noop_edit_reported_as_no_change_all_three_callbacks():
    """The coder / self-check / reviewer mid-stream apply callbacks must report a
    byte-identical (no-op) edit as 'NO CHANGE', never '✓ MODIFIED/FIX APPLIED'."""
    import inspect
    import workflows.code as C
    src = inspect.getsource(C)
    # the guard string + its three uses (one per callback) must be present
    assert src.count("⚠ NO CHANGE") >= 3, "a no-op apply callback still lacks the guard"
    assert src.count("old == content") >= 3


# ── MEDIUM (deferred): path-traversal containment on CODE/VIEW/KEEP ────────────

def test_path_traversal_is_refused_reads_stay_in_project():
    import asyncio, tempfile, os
    import core.tool_call as TC
    root = tempfile.mkdtemp(prefix="travtest_")
    os.makedirs(os.path.join(root, ".jarvis_sandbox"), exist_ok=True)
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    open(os.path.join(root, "pkg", "mod.py"), "w").write("def f():\n    return 1\n")
    sd = tempfile.mkdtemp(prefix="travsecret_")
    open(os.path.join(sd, "passwd"), "w").write("ROOT_SECRET=hunter2\n")
    rel = os.path.relpath(os.path.join(sd, "passwd"), root)   # ../travsecret_*/passwd

    # CODE: relative traversal + absolute both refused, no leak
    for arg in (rel, os.path.join(sd, "passwd")):
        out = asyncio.run(TC._run_code_reads([arg], root))
        assert "OUTSIDE" in out and "ROOT_SECRET" not in out, f"leaked via {arg}"
    # VIEW + KEEP refused
    assert "ROOT_SECRET" not in asyncio.run(TC._run_view([f"{rel} 1-2"], root, {}))
    assert "ROOT_SECRET" not in asyncio.run(TC._run_keep([f"{rel} 1-2"], root, {}, None))
    # legit in-project reads still work
    assert "def f" in asyncio.run(TC._run_code_reads(["pkg/mod.py"], root))
    assert "def f" in asyncio.run(TC._run_view(["pkg/mod.py 1-2"], root, {}))


def test_path_escapes_root_helper():
    from core.tool_call import _path_escapes_root
    assert _path_escapes_root("../etc/passwd", "/proj")
    assert _path_escapes_root("a/../../etc/passwd", "/proj")
    assert _path_escapes_root("/etc/passwd", "/proj")
    assert not _path_escapes_root("pkg/mod.py", "/proj")
    assert not _path_escapes_root("a/b/../c.py", "/proj")   # stays inside


# ── LS tree tool (ckpt 78): planner navigates the REAL filesystem ──────────────
# Replaces the regex-scraped-from-prose file list that let a planner "copy" a
# path that doesn't exist (e.g. galaxy/collection/dataclasses.py when the code
# is in galaxy/dependency_resolution/dataclasses.py).

def test_ls_is_a_known_tag_not_a_wrong_tool():
    from core.tool_detector import KNOWN_TAG_TYPES, COMMON_WRONG_TOOLS
    assert "LS" in KNOWN_TAG_TYPES
    assert "LS" not in COMMON_WRONG_TOOLS   # promoted to a real tool


def test_ls_tag_validates_paths_rejects_prose():
    d = TagDetector("[tool use][LS: lib/ansible/galaxy/dependency_resolution][/tool use]")
    assert d.valid_args("LS") == ["lib/ansible/galaxy/dependency_resolution"]
    d2 = TagDetector("[tool use][LS: is there a galaxy folder anywhere here][/tool use]")
    assert d2.valid_args("LS") == []        # sentence-shaped → rejected


def _mk_repo():
    import tempfile, os
    root = tempfile.mkdtemp()
    # two same-named files in different folders — the exact ambiguity that bit us
    for p in ("galaxy/collection/api.py",
              "galaxy/dependency_resolution/dataclasses.py",
              "utils/log.py", "README.md"):
        full = os.path.join(root, p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").write("x = 1\n")
    # noise that must be hidden
    os.makedirs(os.path.join(root, ".git/objects"), exist_ok=True)
    open(os.path.join(root, ".git/objects/abc"), "w").write("blob")
    os.makedirs(os.path.join(root, "__pycache__"), exist_ok=True)
    open(os.path.join(root, "galaxy/collection/api.cpython-311.pyc"), "w").write("z")
    return root


def test_tree_top_level_collapsed_with_counts_and_no_noise():
    from core.exploration_tools import build_repo_tree
    out = build_repo_tree(_mk_repo())
    assert "galaxy/" in out and "utils/" in out and "README.md" in out
    assert "file" in out                       # folders carry counts
    assert ".git" not in out and "__pycache__" not in out   # noise hidden
    assert "api.py" not in out                 # collapsed — not expanded yet


def test_ls_expands_to_real_children_and_disambiguates():
    from core.exploration_tools import list_dir_entries
    root = _mk_repo()
    out = list_dir_entries(root, "galaxy")
    # the real sub-folders are revealed; the planner can now SEE which is which
    assert "galaxy/collection/" in out
    assert "galaxy/dependency_resolution/" in out
    deeper = list_dir_entries(root, "galaxy/dependency_resolution")
    assert "galaxy/dependency_resolution/dataclasses.py" in deeper
    assert "api.cpython-311.pyc" not in list_dir_entries(root, "galaxy/collection")


def test_ls_refuses_traversal_and_guides_on_missing_or_file():
    from core.exploration_tools import list_dir_entries
    root = _mk_repo()
    assert "escapes the project root" in list_dir_entries(root, "../../../etc")
    assert "no such folder" in list_dir_entries(root, "galaxy/nope")
    out_file = list_dir_entries(root, "README.md")
    assert "FILE, not a folder" in out_file and "[CODE:" in out_file


# ── [LS:] shows what each .py file DEFINES (ckpt 83) ───────────────────────────
# Root cause of the f327e65d 30-round search loop: the planner couldn't tell which
# file defines a symbol, so it put is_valid_collection_name in dataclasses.py when
# that method lives on AnsibleCollectionRef in _collection_finder.py. Fix folds the
# symbols INTO the existing [LS:] tool — not a separate map — so navigating reveals
# what's in each file.

def test_ls_annotates_py_files_with_their_symbols():
    import os, tempfile
    from core.exploration_tools import list_dir_entries
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "galaxy/dependency_resolution"), exist_ok=True)
    os.makedirs(os.path.join(root, "utils/collection_loader"), exist_ok=True)
    with open(os.path.join(root, "galaxy/dependency_resolution/dataclasses.py"), "w") as f:
        f.write("def _is_fqcn(s):\n    return True\n\nclass _ComputedReqKindsMixin:\n    pass\n")
    with open(os.path.join(root, "utils/collection_loader/_collection_finder.py"), "w") as f:
        f.write("class AnsibleCollectionRef:\n    def is_valid_collection_name(self, n):\n        return True\n")
    # the helper file shows its defs/class — and NOT AnsibleCollectionRef
    out_d = list_dir_entries(root, "galaxy/dependency_resolution")
    assert "_is_fqcn" in out_d and "_ComputedReqKindsMixin" in out_d
    assert "AnsibleCollectionRef" not in out_d
    # the finder file shows the class AND its METHOD by name → planner can't
    # misplace is_valid_collection_name (the f327e65d bug). Methods in braces,
    # NOT parens (parens would read as base classes).
    out_f = list_dir_entries(root, "utils/collection_loader")
    assert "class AnsibleCollectionRef" in out_f
    assert "is_valid_collection_name" in out_f          # the method is visible here
    assert "{is_valid_collection_name" in out_f          # braces, not parens
    assert "is_valid_collection_name" not in out_d        # and NOT in the helper file


def test_ls_symbol_annotation_is_best_effort_on_bad_py():
    import os, tempfile
    from core.exploration_tools import list_dir_entries, _py_top_symbols
    root = tempfile.mkdtemp()
    with open(os.path.join(root, "broken.py"), "w") as f:
        f.write("def (((:\n  not python\n")
    # unparseable file → no annotation, but listing must not crash
    out = list_dir_entries(root, "")
    assert "broken.py" in out
    assert _py_top_symbols(os.path.join(root, "broken.py")) == ""


# ── [TRACE:] tool (ckpt 99): enforced line-grounded flow-trace → discriminating test ─
def test_trace_is_a_known_tag():
    from core.tool_detector import TagDetector, KNOWN_TAG_TYPES
    assert "TRACE" in KNOWN_TAG_TYPES
    d = TagDetector("[tool use][TRACE: is_valid_collection_name keyword handling][/tool use]")
    assert d.valid_args("TRACE") == ["is_valid_collection_name keyword handling"]


def test_trace_template_has_the_format_slots():
    from core.exploration_tools import build_trace_template
    t = build_trace_template("netrc must not override Bearer")
    for slot in ("GOAL", "FLOW", "@ <file>:<lineno>", "EDGE", "TEST",
                 "SETUP", "ACTION", "EXPECT", "RUNNABLE", "CATCHES the bug"):
        assert slot in t, slot


def test_verify_trace_lines_grounds_to_real_lines():
    import os, tempfile
    from core.exploration_tools import verify_trace_lines
    root = tempfile.mkdtemp()
    with open(os.path.join(root, "m.py"), "w") as f:
        f.write("def f():\n    x = 1\n    return x\n")
    # real line at its real number → clean
    assert verify_trace_lines("does a thing @ m.py:2 | x = 1", root) == ""
    # imagined line → flagged
    bad = verify_trace_lines("does a thing @ m.py:2 | not_a_real_line()", root)
    assert "not grounded" in bad and "m.py:2" in bad
    # nonexistent file → flagged
    assert "not found" in verify_trace_lines("x @ nope.py:1 | y = 2", root)
