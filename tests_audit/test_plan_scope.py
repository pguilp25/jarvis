"""Tests for the deterministic plan-scope helpers (core.plan_scope)."""
from core.plan_scope import (
    union_file_scopes, majority_files, format_candidate_block,
    imported_modules, modules_to_files,
    referenced_files_outside_scope, completeness_lint, format_plan_gaps,
)


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
    assert "PLAN GAPS" in note and "a.py: missing" in note
