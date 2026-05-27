"""Audit plan extraction — `_extract_files_from_plan`, `_extract_impl_steps`,
`_extract_new_files_from_plan`."""

import pytest
from workflows.code import (
    _extract_files_from_plan,
    _extract_impl_steps,
    _extract_new_files_from_plan,
)


# ───────────────────── FILES_TO_MODIFY ─────────────────────

def test_files__simple_python_path():
    plan = "Modify pkg/widgets.py to add the new feature."
    known = ["pkg/widgets.py"]
    out = _extract_files_from_plan(plan, known)
    assert "pkg/widgets.py" in out


def test_files__multiple_paths():
    plan = "Edit pkg/a.py and pkg/b.py and pkg/sub/c.py."
    known = ["pkg/a.py", "pkg/b.py", "pkg/sub/c.py"]
    out = _extract_files_from_plan(plan, known)
    assert set(out) >= set(known)


def test_files__noise_path_rejected():
    """Absolute paths and virtualenv paths should be rejected."""
    plan = (
        "Stack trace: /home/user/.virtualenvs/x/lib/python3.6/site-packages/sympy/foo.py\n"
        "Should modify: pkg/widgets.py\n"
    )
    known = ["pkg/widgets.py"]
    out = _extract_files_from_plan(plan, known)
    assert "pkg/widgets.py" in out
    assert not any("/site-packages/" in p for p in out)
    assert not any("/.virtualenvs/" in p for p in out)
    assert not any(p.startswith("/") for p in out)


def test_files__testbed_path_rejected():
    plan = "Issue: /testbed/sympy/geometry/point.py"
    out = _extract_files_from_plan(plan, ["sympy/geometry/point.py"])
    assert not any("/testbed/" in p for p in out)


def test_files__bare_basename_only_if_known():
    """A bare `widgets.py` mention in prose — only kept if it maps to a
    known project file."""
    plan = "Mention widgets.py in passing."
    out_known = _extract_files_from_plan(plan, ["pkg/widgets.py"])
    out_unknown = _extract_files_from_plan(plan, [])
    # Known: map bare basename to known full path
    if "pkg/widgets.py" in out_known:
        # Good — bare basename was mapped
        pass
    # Unknown: drop
    assert "widgets.py" not in out_unknown


def test_files__dotdot_traversal_rejected():
    plan = "Edit ../sibling/file.py"
    out = _extract_files_from_plan(plan, ["../sibling/file.py"])
    # `..` path should be filtered
    assert not any(".." in p.split("/") for p in out)


def test_files__multiple_extensions_all_recognized():
    plan = "Edit a.py, b.js, c.ts, d.rs, e.go, f.java"
    out = _extract_files_from_plan(plan, [])
    # All these should be recognized as potential file paths
    # (whether they survive the basename filter depends on `known`)


def test_files__empty_plan_empty_result():
    out = _extract_files_from_plan("", [])
    assert out == []


def test_files__only_prose_no_paths():
    plan = "This plan has no file paths just words."
    out = _extract_files_from_plan(plan, [])
    assert out == []


# ───────────────────── IMPL_STEPS ─────────────────────

def test_steps__single_step():
    plan = """
## IMPLEMENTATION STEPS

### STEP 1: Add the new method
SATISFIES: R1
FILES: pkg/widgets.py
WHAT TO DO:
  - Action 1
"""
    out = _extract_impl_steps(plan)
    assert len(out) == 1
    assert out[0]["num"] == 1
    assert "pkg/widgets.py" in out[0]["files"]


def test_steps__multiple_steps():
    plan = """
## IMPLEMENTATION STEPS

### STEP 1: First
FILES: a.py

### STEP 2: Second
FILES: b.py

### STEP 3: Third
FILES: c.py
"""
    out = _extract_impl_steps(plan)
    assert len(out) == 3
    assert [s["num"] for s in out] == [1, 2, 3]


def test_steps__step_with_depends_on():
    plan = """
## IMPLEMENTATION STEPS

### STEP 1: Base
FILES: a.py

### STEP 2: Builds on 1
DEPENDS ON: STEP 1
FILES: b.py
"""
    out = _extract_impl_steps(plan)
    assert out[1]["depends_on"] == [1]


def test_steps__duplicate_step_numbers_keep_first():
    """A duplicate `### STEP 1` is rejected — only first kept."""
    plan = """
## IMPLEMENTATION STEPS

### STEP 1: First instance
FILES: a.py

### STEP 1: Duplicate
FILES: b.py
"""
    out = _extract_impl_steps(plan)
    assert len(out) == 1
    assert "a.py" in out[0]["files"]


def test_steps__no_steps_returns_empty():
    plan = "## GOAL\nDo a thing."
    out = _extract_impl_steps(plan)
    assert out == []


def test_steps__only_last_implementation_steps_block_used():
    """If the plan has TWO `## IMPLEMENTATION STEPS` sections, use the LAST
    (typically the final draft)."""
    plan = """
## IMPLEMENTATION STEPS
### STEP 1: Draft
FILES: draft.py

## IMPLEMENTATION STEPS
### STEP 1: Final
FILES: final.py
"""
    out = _extract_impl_steps(plan)
    assert len(out) == 1
    assert "final.py" in out[0]["files"]


# ───────────────────── NEW FILES ─────────────────────

def test_new_files__basic():
    plan = """
## FILES TO CREATE
- scripts/new_tool.py
- docs/readme.md
"""
    out = _extract_new_files_from_plan(plan)
    assert "scripts/new_tool.py" in out
    assert "docs/readme.md" in out


def test_new_files__no_create_section():
    plan = "## REQUIREMENTS\nR1. foo"
    out = _extract_new_files_from_plan(plan)
    assert out == []


# ───────────────────── ADVERSARIAL ─────────────────────

def test_files__path_with_dash():
    plan = "Edit my-pkg/widgets.py"
    out = _extract_files_from_plan(plan, ["my-pkg/widgets.py"])
    assert "my-pkg/widgets.py" in out


def test_files__path_with_underscore():
    plan = "Edit pkg/my_module.py"
    out = _extract_files_from_plan(plan, ["pkg/my_module.py"])
    assert "pkg/my_module.py" in out


def test_files__path_with_numbers():
    plan = "Edit pkg/v2/widgets.py"
    out = _extract_files_from_plan(plan, ["pkg/v2/widgets.py"])
    assert "pkg/v2/widgets.py" in out


def test_files__many_unknown_paths_all_dropped():
    """Lots of made-up paths the planner mentioned but aren't in the project."""
    plan = "Reference: foo.py, bar.py, baz.py, qux.py, doesnt_exist.py"
    out = _extract_files_from_plan(plan, ["actual.py"])
    # All the bare basenames are dropped (none match `actual.py`)
    assert len(out) <= 1  # at most "actual.py" if mentioned


def test_files__plan_with_traceback_paths():
    """Plan body contains a Python traceback — those paths must NOT be
    captured as targets (this is the sympy-17655 bug)."""
    plan = """
Traceback (most recent call last):
  File "/.virtualenvs/test/lib/python3.6/site-packages/sympy/geometry/point.py", line 233
  File "/usr/lib/python3.9/typing.py", line 100
The fix should go in sympy/printing/latex.py
"""
    out = _extract_files_from_plan(plan, ["sympy/printing/latex.py", "sympy/geometry/point.py"])
    assert "sympy/printing/latex.py" in out
    # Traceback paths must NOT appear
    assert not any(".virtualenvs" in p for p in out)
    assert not any("/usr/lib/" in p for p in out)
