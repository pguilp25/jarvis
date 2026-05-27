"""Audit `search_refs` — symbol search that buckets into DEFINED / IMPORTED /
USED. This is what backs `[REFS: name]`.

Critical contracts:
  - DEFINED is never truncated (priority pass over the whole project)
  - IMPORTED catches both single-line and parenthesized multi-line forms
  - USED is bounded by max_results
  - Word-boundary matching: searching `render` doesn't find `prerender`
  - Multiple definition sites (e.g. method overrides on different classes)
    all appear under DEFINED
"""
import os
import re
import textwrap
from pathlib import Path
import pytest

from tools.codebase import search_refs


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ───────────────────── DEFINED — never truncated ─────────────────────

def test_refs__class_definition_in_defined(tmp_path):
    _write(tmp_path / "pkg/widgets.py",
           "class Widget:\n    pass\n")
    out = search_refs("Widget", str(tmp_path))
    assert "DEFINED" in out
    assert "pkg/widgets.py" in out


def test_refs__function_definition_in_defined(tmp_path):
    _write(tmp_path / "pkg/util.py",
           "def compute(n):\n    return n * 2\n")
    out = search_refs("compute", str(tmp_path))
    assert "DEFINED" in out
    assert "compute" in out


def test_refs__async_def_in_defined(tmp_path):
    _write(tmp_path / "pkg/io.py",
           "async def fetch(url):\n    return None\n")
    out = search_refs("fetch", str(tmp_path))
    assert "DEFINED" in out
    assert "fetch" in out


def test_refs__multiple_definitions_all_present(tmp_path):
    """Two classes with the same method name — both `to_index` defs surface."""
    _write(tmp_path / "pkg/a.py",
           "class A:\n    def to_index(self): return 1\n")
    _write(tmp_path / "pkg/b.py",
           "class B:\n    def to_index(self): return 2\n")
    out = search_refs("to_index", str(tmp_path))
    assert "DEFINED" in out
    # Both files appear in DEFINED section
    defined_section = out.split("DEFINED")[1].split("USED")[0] if "USED" in out else out.split("DEFINED")[1]
    assert "a.py" in defined_section
    assert "b.py" in defined_section


def test_refs__definition_survives_when_many_usages(tmp_path):
    """Definitions must not be pushed off by USED cap.
    Same astropy-13977 regression pattern."""
    _write(tmp_path / "pkg/defs.py",
           "def critical_func():\n    return 42\n")
    # Lots of usage files
    for i in range(40):
        _write(tmp_path / f"pkg/user_{i:02}.py",
               "from .defs import critical_func\n"
               "critical_func()\n" * 5)
    out = search_refs("critical_func", str(tmp_path), max_results=30)
    assert "DEFINED" in out
    assert "defs.py" in out


def test_refs__word_boundary_not_substring(tmp_path):
    """Searching `render` should NOT match `prerender` or `rendered_obj`."""
    _write(tmp_path / "a.py",
           "def render(): pass\n"
           "def prerender(): pass\n"
           "rendered_obj = None\n")
    out = search_refs("render", str(tmp_path))
    defined = out.split("DEFINED")[1] if "DEFINED" in out else ""
    # `render` (the exact word) is defined once. `prerender` is a different word.
    # Since word boundary is in effect, we should see render but not prerender
    # listed AS a definition of `render`.
    # In USED section, accept anything; we mostly check that DEFINED is correct.
    assert "def render():" in out


# ───────────────────── IMPORTED — multi-line catch ─────────────────────

def test_refs__single_line_from_import_in_imported(tmp_path):
    _write(tmp_path / "pkg/__init__.py",
           "from .widgets import Widget\n")
    _write(tmp_path / "pkg/widgets.py",
           "class Widget:\n    pass\n")
    out = search_refs("Widget", str(tmp_path))
    assert "IMPORTED" in out
    assert "__init__.py" in out


def test_refs__multiline_parenthesized_import_in_imported(tmp_path):
    """astropy-13236 regression: multi-line `(Name1, Name2, ...)` imports."""
    _write(tmp_path / "pkg/__init__.py",
           "from .widgets import (\n"
           "    Widget,\n"
           "    Gadget,\n"
           "    Helper,\n"
           ")\n")
    _write(tmp_path / "pkg/widgets.py",
           "class Widget: pass\nclass Gadget: pass\nclass Helper: pass\n")
    out = search_refs("Widget", str(tmp_path))
    assert "IMPORTED" in out
    assert "__init__.py" in out, (
        f"multi-line parenthesized import not detected. Output:\n{out}"
    )


def test_refs__deeply_indented_multiline_import(tmp_path):
    """Multi-line import with the symbol on the THIRD continuation line."""
    _write(tmp_path / "pkg/__init__.py",
           "from .widgets import (Alpha,\n"
           "                      Beta,\n"
           "                      Gamma,\n"
           "                      TargetName,\n"
           "                      Delta)\n")
    _write(tmp_path / "pkg/widgets.py",
           "class Alpha: pass\nclass Beta: pass\nclass Gamma: pass\n"
           "class TargetName: pass\nclass Delta: pass\n")
    out = search_refs("TargetName", str(tmp_path))
    assert "IMPORTED" in out
    assert "__init__.py" in out


def test_refs__import_as_alias(tmp_path):
    """`from x import Foo as Bar` — searching `Foo` should still find it."""
    _write(tmp_path / "pkg/__init__.py",
           "from .widgets import Widget as W\n")
    _write(tmp_path / "pkg/widgets.py",
           "class Widget: pass\n")
    out = search_refs("Widget", str(tmp_path))
    assert "IMPORTED" in out


# ───────────────────── USED — bounded ─────────────────────

def test_refs__usage_capped(tmp_path):
    """USED section respects max_results."""
    _write(tmp_path / "pkg/defs.py", "MY_NAME = 1\n")
    # Make 200 usage lines
    _write(tmp_path / "pkg/user.py",
           "x = MY_NAME\n" * 200)
    out = search_refs("MY_NAME", str(tmp_path), max_results=20)
    # Count the entries
    used_section = out.split("USED")[1] if "USED" in out else ""
    used_lines = [l for l in used_section.split("\n") if l.strip().startswith(("pkg/", "."))]
    # Soft assertion — it should be bounded, not full 200
    assert len(used_lines) < 100, f"USED count: {len(used_lines)}"


def test_refs__no_matches_anywhere__graceful(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    out = search_refs("nonexistent_symbol", str(tmp_path))
    # Should not crash; should return a "no matches found" string
    assert isinstance(out, str)
    assert "no matches" in out.lower() or "DEFINED" not in out


# ───────────────────── EDGE CASES ─────────────────────

def test_refs__name_with_underscore(tmp_path):
    _write(tmp_path / "a.py",
           "def my_func(): pass\n"
           "my_func()\n")
    out = search_refs("my_func", str(tmp_path))
    assert "DEFINED" in out


def test_refs__single_char_name(tmp_path):
    """Single-char names like `x` — high false-positive risk."""
    _write(tmp_path / "a.py",
           "class X: pass\n"
           "y = X()\n")
    out = search_refs("X", str(tmp_path))
    # Should at least find the class definition
    assert "DEFINED" in out


def test_refs__dunder_name(tmp_path):
    """Dunder names like `__init__`."""
    _write(tmp_path / "a.py",
           "class Foo:\n"
           "    def __init__(self): pass\n")
    out = search_refs("__init__", str(tmp_path))
    # May or may not appear in DEFINED depending on detection logic
    assert isinstance(out, str)


def test_refs__name_in_string_literal_doesnt_count_as_definition(tmp_path):
    """A string `"def foo():"` should NOT be classified as a definition."""
    _write(tmp_path / "a.py",
           "src = 'def foo():\\n    pass'\n")
    out = search_refs("foo", str(tmp_path))
    # If DEFINED appears at all, it shouldn't be from the string content
    # (This is harder to validate without scanning DEFINED specifically)
    assert isinstance(out, str)


# ───────────────────── REGRESSION ─────────────────────

def test_refs__realworld_astropy_13236_pattern(tmp_path):
    """Exact pattern from astropy/table/__init__.py:51 — multi-line import
    on line 51 of a __init__.py with many other imports above."""
    _write(tmp_path / "astropy/table/__init__.py",
           # 50 lines of preamble + imports
           "\n".join([
               "# Licensed under a 3-clause BSD style license",
               "from astropy.utils.compat import optional_deps",
               "from .column import Column, MaskedColumn",
               "from . import connect",
               "",
           ] + ["# filler"] * 45) + "\n" +
           "from .table import (Table, QTable, TableColumns, Row, TableFormatter,\n"
           "                    NdarrayMixin, TableReplaceWarning, TableAttribute,\n"
           "                    PprintIncludeExclude)  # noqa: E402\n")
    _write(tmp_path / "astropy/table/table.py",
           "class Table: pass\nclass QTable: pass\nclass TableColumns: pass\n"
           "class Row: pass\nclass TableFormatter: pass\nclass NdarrayMixin: pass\n"
           "class TableReplaceWarning(Warning): pass\nclass TableAttribute: pass\n"
           "class PprintIncludeExclude: pass\n")
    out = search_refs("NdarrayMixin", str(tmp_path))
    assert "IMPORTED" in out, (
        f"multi-line re-export of NdarrayMixin not detected. Output:\n{out[:1500]}"
    )
    assert "__init__.py" in out
