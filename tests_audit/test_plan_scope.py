"""Tests for the deterministic plan-scope helpers (core.plan_scope)."""
from core.plan_scope import (
    union_file_scopes, majority_files, format_candidate_block,
    imported_modules, modules_to_files,
    referenced_files_outside_scope, completeness_lint, format_plan_gaps,
    rank_relevant_tests, imported_symbols, missing_symbols,
    coverage_steps,
)


# ── contract detection: a test imports a symbol that doesn't exist yet ─────────
# (the pylint-4551 root cause — utils.get_annotation/infer_node/get_annotation_label
#  imported by the test, absent in utils.py → ImportError → 0 tests collected)
def test_imported_symbols_and_missing_detects_contract():
    test_src = (
        "import pytest\n"
        "from pylint.pyreverse.utils import get_annotation, get_annotation_label, infer_node\n"
        "from pylint.pyreverse.diadefs import DiadefsHandler\n"
        "from unittest import mock\n"
    )
    isyms = imported_symbols(test_src)
    assert isyms["pylint.pyreverse.utils"] == {
        "get_annotation", "get_annotation_label", "infer_node"}
    utils_before = "import os\ndef is_exception(n):\n    return True\n"
    assert sorted(missing_symbols(isyms["pylint.pyreverse.utils"], utils_before)) == [
        "get_annotation", "get_annotation_label", "infer_node"]
    utils_after = utils_before + "def get_annotation(n):\n    pass\n" \
        "def get_annotation_label(a):\n    pass\ndef infer_node(n):\n    pass\n"
    assert missing_symbols(isyms["pylint.pyreverse.utils"], utils_after) == []


def test_missing_symbols_no_false_positive():
    assert missing_symbols({"Foo"}, "class Foo:\n    pass\n") == []
    assert missing_symbols({"BAZ"}, "BAZ = 1\n") == []
    assert missing_symbols({"foo"}, "from x import bar as foo\n") == []
    assert missing_symbols({"x"}, "x, y = 1, 2\n") == []
    assert missing_symbols({"nope"}, "def other():\n    pass\n") == ["nope"]


def test_imported_symbols_handles_paren_and_alias():
    syms = imported_symbols("from a.b import (one, two as t)\n")
    assert "one" in syms["a.b"] and "two" in syms["a.b"]


# ── test ranking (the pylint utils.py regression) ──────────────────────────────
def test_rank_surfaces_gold_test_past_noise_stems():
    """Regression: pylint scope includes __init__/main; those generic stems used
    to (as whole-path substrings) rank nearly every test 0, burying the real gold
    test (unittest_pyreverse_writer.py) past the cap so its import of
    pyreverse.utils was never seen. Ranking by basename against MEANINGFUL stems
    must surface it within the cap."""
    scope = [
        "pylint/pyreverse/__init__.py", "pylint/pyreverse/main.py",
        "pylint/pyreverse/inspector.py", "pylint/pyreverse/writer.py",
        "pylint/pyreverse/diagrams.py",
    ]
    # 20 noise tests whose paths contain 'main'/'init' (so the OLD substring rule
    # ranked them 0) sort alphabetically ahead of the gold test.
    noise = [f"tests/functional/a_main_case_{i:02d}_test.py" for i in range(20)]
    gold = "tests/unittest_pyreverse_writer.py"
    ranked = rank_relevant_tests(noise + [gold], scope, cap=16)
    assert gold in ranked, "gold test buried past the cap by generic-stem noise"
    assert ranked[0] == gold  # writer/inspector/diagrams match → ranked first


def test_rank_ignores_non_test_and_non_py():
    out = rank_relevant_tests(
        ["test_a.py", "src/widget.py", "tests/data.txt", "test_b.py"], ["widget.py"])
    assert out == ["test_a.py", "test_b.py"]


def test_rank_respects_cap():
    files = [f"test_{i:03d}.py" for i in range(50)]
    assert len(rank_relevant_tests(files, [], cap=16)) == 16


# ── union / votes ─────────────────────────────────────────────────────────────
def test_union_counts_distinct_draft_votes():
    per_draft = [
        ["a.py", "b.py"],
        ["b.py", "c.py"],
        ["b.py"],
    ]
    union, votes = union_file_scopes(per_draft)
    assert votes == {"a.py": 1, "b.py": 3, "c.py": 1}
    # sorted by votes desc then name
    assert union[0] == "b.py"
    assert set(union) == {"a.py", "b.py", "c.py"}


def test_union_dedups_within_a_draft():
    union, votes = union_file_scopes([["x.py", "x.py", "x.py"]])
    assert votes["x.py"] == 1


def test_union_handles_empty():
    union, votes = union_file_scopes([])
    assert union == [] and votes == {}


def test_majority_requires_threshold_drafts():
    _, votes = union_file_scopes([["a.py", "b.py"], ["b.py", "c.py"], ["b.py"]])
    # 3 drafts, threshold 2 → only b.py (3 votes)
    assert majority_files(votes, n_drafts=3, threshold=2) == ["b.py"]
    # with only 1 draft there's no agreement signal
    assert majority_files({"a.py": 1}, n_drafts=1, threshold=2) == []


def test_candidate_block_lists_each_with_votes():
    union, votes = union_file_scopes([["a.py", "b.py"], ["b.py"]])
    block = format_candidate_block(union, votes, n_drafts=2)
    assert "CANDIDATE FILES" in block
    assert "a.py  (named by 1/2 drafts)" in block
    assert "b.py  (named by 2/2 drafts)" in block
    assert format_candidate_block([], {}, 0) == ""


# ── test-derived scope ────────────────────────────────────────────────────────
def test_import_parsing():
    src = (
        "import os\n"
        "from pylint.pyreverse.utils import get_annotation, infer_node\n"
        "from pylint.pyreverse import inspector as insp\n"
        "import matplotlib.cbook\n"
    )
    mods = imported_modules(src)
    assert "pylint.pyreverse.utils" in mods
    assert "pylint.pyreverse" in mods
    assert "matplotlib.cbook" in mods
    assert "os" in mods


def test_modules_to_files_suffix_match():
    project = ["pylint/pyreverse/utils.py", "pylint/pyreverse/inspector.py", "x/y.py"]
    files = modules_to_files(
        {"pylint.pyreverse.utils", "pylint.pyreverse.inspector", "nonexistent.mod"},
        project)
    assert files == ["pylint/pyreverse/inspector.py", "pylint/pyreverse/utils.py"]


def test_pylint_scenario_test_would_have_flagged_utils():
    # the real pylint-4551 test imports get_annotation/infer_node from utils.py,
    # a file the plan never named — this is exactly what #2 recovers.
    test_src = "from pylint.pyreverse.utils import get_annotation, infer_node\n"
    project = ["pylint/pyreverse/utils.py", "pylint/pyreverse/diagrams.py"]
    req = modules_to_files(imported_modules(test_src), project)
    assert "pylint/pyreverse/utils.py" in req


# ── dependency backstop ───────────────────────────────────────────────────────
def test_referenced_files_outside_scope():
    out = ("foo defined in pkg/a.py; used in pkg/b.py:12, pkg/c.py:88; "
           "and /usr/lib/site.py")
    project = ["pkg/a.py", "pkg/b.py", "pkg/c.py"]
    scope = ["pkg/a.py"]
    missing = referenced_files_outside_scope(out, scope, project)
    assert missing == ["pkg/b.py", "pkg/c.py"]   # callers not in scope
    # /usr/lib/site.py isn't a project file → ignored


# ── completeness lint ─────────────────────────────────────────────────────────
def test_lint_flags_required_file_not_in_scope():
    gaps = completeness_lint(
        plan_text="### STEP 1: edit diagrams.py\nFILES: diagrams.py",
        scope_files=["diagrams.py"],
        required_files=["utils.py", "diagrams.py"])
    assert len(gaps) == 1 and "utils.py" in gaps[0]


def test_lint_flags_requirements_without_steps():
    gaps = completeness_lint(
        plan_text="## REQUIREMENTS\n- do X\n- do Y\n(no steps)",
        scope_files=[], required_files=[])
    assert any("no `### STEP`" in g for g in gaps)


def test_lint_clean_when_covered():
    gaps = completeness_lint(
        plan_text="## REQUIREMENTS\n### STEP 1: x\nFILES: a.py",
        scope_files=["a.py"], required_files=["a.py"])
    assert gaps == []


def test_format_plan_gaps():
    assert format_plan_gaps([]) == ""
    note = format_plan_gaps(["a.py: missing"])
    # wording updated in ckpt-166 (imperative "REQUIRED SCOPE THE PLAN OMITTED")
    assert "REQUIRED SCOPE" in note and "a.py: missing" in note


# ── coverage_steps: consensus file MENTIONED but never STEPPED (395e5e20) ──────
# The merger can NAME a ≥2-draft file in prose (so it's "in scope" and the
# _required backstop skips it) yet emit NO `### STEP` for it — the per-step coder
# then never edits it. coverage_steps synthesizes a real STEP that the step parser
# picks up so the file actually gets covered.
from workflows.code import _extract_impl_steps   # the real step parser


def test_coverage_steps_adds_step_for_unstepped_consensus_file():
    plan = "## STEPS\n### STEP 1: do play_iterator\nFILES: play_iterator.py\n"
    proj = ["play_iterator.py", "strategy/__init__.py", "strategy/linear.py"]
    consensus = ["play_iterator.py", "strategy/__init__.py", "strategy/linear.py"]
    text, added = coverage_steps(plan, consensus, {"play_iterator.py"}, proj,
                                 {"strategy/__init__.py": 3, "strategy/linear.py": 2})
    assert set(added) == {"strategy/__init__.py", "strategy/linear.py"}
    # the synthesized steps must PARSE via the real parser and yield those files
    parsed = _extract_impl_steps(plan + text)
    parsed_files = {f for s in parsed for f in s.get("files") or []}
    assert "strategy/__init__.py" in parsed_files and "strategy/linear.py" in parsed_files
    # numbering continues from the existing max (STEP 1 → 2, 3)
    assert "### STEP 2:" in text and "### STEP 3:" in text


def test_coverage_steps_noop_when_all_consensus_stepped():
    plan = "### STEP 1: a\nFILES: a.py\n### STEP 2: b\nFILES: b.py\n"
    text, added = coverage_steps(plan, ["a.py", "b.py"], {"a.py", "b.py"},
                                 ["a.py", "b.py"], {})
    assert text == "" and added == []


def test_coverage_steps_excludes_tests_and_nonproject():
    plan = "### STEP 1: a\nFILES: a.py\n"
    # test file + a file not in the project pool must NOT be forced
    text, added = coverage_steps(plan, ["tests/test_x.py", "ghost.py", "real.py"],
                                 {"a.py"}, ["a.py", "real.py"],
                                 {"tests/test_x.py": 3, "ghost.py": 2, "real.py": 2})
    assert added == ["real.py"]


def test_coverage_steps_caps_additions():
    plan = "### STEP 1: a\nFILES: a.py\n"
    proj = [f"f{i}.py" for i in range(10)]
    text, added = coverage_steps(plan, proj, set(), proj, {f: 2 for f in proj}, max_add=4)
    assert len(added) == 4


def test_coverage_steps_handles_empty_inputs():
    assert coverage_steps("", [], set(), [], {}) == ("", [])
    assert coverage_steps(None, None, None, None, None) == ("", [])


def test_extract_impl_steps_handles_config_and_stub_paths():
    # bughunt #D (ckpt-204): _PATH_RE dropped/mis-extracted config & stub files the planner is told
    # to step (setup.cfg→setup.c, docs/index.rst→docs/index.rs, .pyi/.ini/.in were lost).
    import workflows.code as wc
    plan = ("## STEPS\n\n"
            "### STEP 1: update setup config\nFILES: setup.cfg, src/pkg/mod.py\nDo the thing.\n\n"
            "### STEP 2: stubs and docs\nFILES: stubs/foo.pyi, docs/index.rst, tox.ini, MANIFEST.in\nMore.\n")
    allf = [f for s in wc._extract_impl_steps(plan) for f in s.get("files", [])]
    for want in ("setup.cfg", "src/pkg/mod.py", "stubs/foo.pyi", "docs/index.rst", "tox.ini", "MANIFEST.in"):
        assert want in allf, f"{want} missing from {allf}"
    assert "setup.c" not in allf and "docs/index.rs" not in allf   # no greedy mis-extraction
