"""Audit `_guess_filename` — heuristic that picks a default filename for
new-file content when the planner didn't specify one.

This runs on every "create new file" path that lacks a name; bugs cause:
  • A React file written as `main.py` → import errors.
  • A standalone HTML demo saved as `main.py` → broken artifact.
  • Java code saved as `main.py` → unrunnable.
"""
import pytest
from workflows.code import _guess_filename


# ─────────────── detection from content ───────────────


def test_guess__html_doctype():
    assert _guess_filename("a task", "<!DOCTYPE html>\n<html>") == "index.html"


def test_guess__html_lowercase_doctype():
    assert _guess_filename("a task", "<!doctype html>") == "index.html"


def test_guess__bare_html_tag():
    assert _guess_filename("a task", "<html>\n<body>") == "index.html"


def test_guess__react_import():
    assert _guess_filename("any", "import React from 'react'") == "App.jsx"


def test_guess__react_from_react():
    assert _guess_filename("any", "from react import x") == "App.jsx"


def test_guess__go_main():
    assert _guess_filename("any", "package main\n\nfunc main() {}") == "main.go"


def test_guess__rust_main():
    assert _guess_filename("any", "fn main() { println!(\"hi\"); }") == "main.rs"


def test_guess__java_main():
    assert _guess_filename(
        "any", "public class Main { public static void main(String[] args) {} }"
    ) == "Main.java"


def test_guess__cpp_with_include():
    """C/C++ — `#include` detected."""
    out = _guess_filename("any", "#include <iostream>\nint main() {}")
    # Should be .c or .cpp
    assert out in {"main.c", "main.cpp"}


def test_guess__cpp_iostream_picks_cpp():
    """`#include <iostream>` → C++ specifically."""
    assert _guess_filename("any", "#include <iostream>\nint main() {}") == "main.cpp"


def test_guess__cpp_cstdio_picks_cpp():
    """`#include <cstdio>` is C++-style (vs `<stdio.h>` which is C)."""
    assert _guess_filename("any", "#include <cstdio>\nint main() {}") == "main.cpp"


def test_guess__c_plain_include_picks_c():
    """Regression: `#include <stdio.h>` should be C, not C++ (bug fixed:
    `iostream or 'cstdio' in content_start` was always-truthy)."""
    assert _guess_filename("any", "#include <stdio.h>\nint main() { return 0; }") == "main.c"


def test_guess__c_string_h_picks_c():
    assert _guess_filename("any", "#include <string.h>\nint main() {}") == "main.c"


def test_guess__lean_theorem():
    assert _guess_filename("any", "theorem foo : 1 = 1 := rfl") == "proof.lean"


def test_guess__lean_mathlib_import():
    assert _guess_filename("any", "import Mathlib") == "proof.lean"


def test_guess__json_object():
    """JSON file detected from leading `{`."""
    assert _guess_filename("any", '{"key": "value"}') == "data.json"


# ─────────────── detection from task ───────────────


def test_guess__task_html():
    assert _guess_filename("Build an HTML webpage", "") == "index.html"


def test_guess__task_css():
    assert _guess_filename("Write some CSS for the stylesheet", "") == "style.css"


def test_guess__task_react():
    assert _guess_filename("Create a React component", "") == "App.jsx"


def test_guess__task_lean():
    assert _guess_filename("Write a formal proof in Lean", "") == "proof.lean"


def test_guess__task_rust():
    assert _guess_filename("Set up a cargo project in Rust", "") == "main.rs"


def test_guess__task_browser_game():
    assert _guess_filename("Make a browser game", "") == "index.html"


# ─────────────── default fallback ───────────────


def test_guess__default_python():
    """No content/task hints → default to Python."""
    assert _guess_filename("a generic task", "x = 1") == "main.py"


def test_guess__empty_inputs():
    """Empty task and content — still returns a default."""
    assert _guess_filename("", "") == "main.py"


# ─────────────── precedence: content beats task ───────────────


def test_guess__content_html_overrides_python_task():
    """Content detection runs first: HTML doctype overrides 'Python' in task."""
    out = _guess_filename("Write a Python script", "<!DOCTYPE html>")
    assert out == "index.html"


def test_guess__content_react_overrides_html_task():
    out = _guess_filename("Make an HTML page", "import React from 'react'")
    assert out == "App.jsx"


def test_guess__content_go_overrides_rust_task():
    out = _guess_filename("Write Rust code", "package main\nfunc main() {}")
    assert out == "main.go"


# ─────────────── adversarial ───────────────


def test_guess__only_first_200_chars_examined():
    """Content beyond 200 chars should NOT influence detection."""
    padding = "# python comment\n" * 50
    big_content = padding + "<!DOCTYPE html>"  # html starts past char 200
    out = _guess_filename("any", big_content)
    # The html marker is past the 200-char window → not detected from content
    # Falls through to task — no task hints → default
    assert out == "main.py" or out == "index.html"


def test_guess__case_insensitive_task_match():
    """Task keyword matching should be case-insensitive."""
    out1 = _guess_filename("make HTML page", "")
    out2 = _guess_filename("make html page", "")
    assert out1 == out2


def test_guess__react_takes_jsx_extension():
    """React detection picks JSX over JS."""
    out = _guess_filename("any react component task", "")
    assert out == "App.jsx"
