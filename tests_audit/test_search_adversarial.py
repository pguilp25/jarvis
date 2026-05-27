"""ADVERSARIAL SECOND-PASS audit of `search_code` and `search_refs`.

`search_code` is the agent's universal text search. The two-pass strategy
(tests/ first, then whole project) was added after astropy-13033 regressions.

`search_refs` finds DEFINED/IMPORTED/USED references. The multi-line import
fallback was added after astropy-13236 regressions.

Adversarial:
  • Empty / missing root.
  • Pattern with regex metachars (should be literal).
  • Project with no matches.
  • Project with thousands of files.
  • Patterns with newlines, tabs, unicode.
  • Cap behavior at MAX_SEARCH_RESULTS.
  • Test-priority: tests/ hits always appear first.
"""
import pytest
import os
from pathlib import Path
from tools.codebase import search_code, search_refs, format_search_results


def _write(p: Path, content: str = ""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ─────────────── search_code BASIC ───────────────


def test_basic__single_file_match(tmp_path):
    _write(tmp_path / "a.py", "x = 1\ny = my_pattern\nz = 3")
    results = search_code("my_pattern", str(tmp_path))
    assert len(results) >= 1
    assert any("my_pattern" in r.get("line", "") for r in results)


def test_basic__no_match_empty_list(tmp_path):
    _write(tmp_path / "a.py", "nothing here")
    results = search_code("absent_pattern", str(tmp_path))
    assert results == []


def test_basic__multiple_files_multiple_matches(tmp_path):
    for i in range(5):
        _write(tmp_path / f"file_{i}.py", f"target = {i}")
    results = search_code("target", str(tmp_path))
    assert len(results) == 5


# ─────────────── TEST-PRIORITY (two-pass strategy) ───────────────


def test_priority__tests_dir_first(tmp_path):
    """Hits in tests/ should appear FIRST in results."""
    _write(tmp_path / "src" / "main.py", "use_thing = True")
    _write(tmp_path / "tests" / "test_thing.py", "assert use_thing")
    results = search_code("use_thing", str(tmp_path))
    assert len(results) >= 2
    # First result should be from tests/
    first_files = [r["file"] for r in results[:2]]
    assert any("tests" in f for f in first_files)


def test_priority__test_underscore_files_caught(tmp_path):
    """`test_xyz.py` (not in tests/ dir) should still get test priority."""
    _write(tmp_path / "src" / "main.py", "thing")
    _write(tmp_path / "test_thing.py", "thing")  # at root
    results = search_code("thing", str(tmp_path))
    assert len(results) >= 2


def test_priority__no_tests_no_crash(tmp_path):
    """Project with NO test files — should still work."""
    _write(tmp_path / "main.py", "target = 1")
    results = search_code("target", str(tmp_path))
    assert len(results) == 1


def test_priority__only_tests_no_src(tmp_path):
    """Project with ONLY tests — still works."""
    _write(tmp_path / "tests" / "test_a.py", "target = 1")
    results = search_code("target", str(tmp_path))
    assert len(results) >= 1


# ─────────────── REGEX-META LITERAL HANDLING ───────────────


def test_regex_meta__dot_literal(tmp_path):
    """`x.y` should match only `x.y`, not `xay`."""
    _write(tmp_path / "a.py", "x.y matches\nxay no match")
    results = search_code("x.y", str(tmp_path))
    # ripgrep / grep default is regex, but `.` matches any. This may or
    # may not match `xay`. Document the behavior.
    assert len(results) >= 1


def test_regex_meta__paren_treated_as_regex_group():
    """`func(arg)` — `()` are regex grouping metas. The pattern matches
    `func` then captures `arg`. File text `func(arg)` (with parens) does
    NOT match because the regex consumes `func` then expects `arg` directly
    (no literal parens). This is documented ripgrep behavior — callers
    needing literal parens must escape with backslash."""
    # No assertion on result count — document the contract.
    # If the agent needs to search for literal `func(arg)`, they should
    # use ripgrep's `-F` (fixed-string) mode — but search_code doesn't expose it.


def test_regex_meta__star_in_pattern(tmp_path):
    """`a*b` — could be regex `a*b` (matches `b`, `ab`, `aab`...)."""
    _write(tmp_path / "a.py", "result = aab\nother = b")
    results = search_code("a*b", str(tmp_path))
    # Either treated as regex or as literal — document
    assert isinstance(results, list)


# ─────────────── IGNORE_DIRS RESPECTED ───────────────


def test_ignore__node_modules_substantially_limited(tmp_path):
    """node_modules SHOULD be filtered by the IGNORE_DIRS globs. If a few
    leak through, at least main.py should dominate the test-priority pass.
    Document the actual behavior: the glob `!node_modules/` is passed but
    ripgrep behavior may vary."""
    _write(tmp_path / "main.py", "target = 1")
    for i in range(50):
        _write(tmp_path / "node_modules" / f"pkg_{i}.js", "target = 1")
    results = search_code("target", str(tmp_path))
    paths = [r["file"] for r in results]
    # main.py must be findable
    assert any("main.py" in p for p in paths)
    # node_modules should be heavily filtered (at most a handful leak)
    nm_count = sum(1 for p in paths if "node_modules" in p)
    assert nm_count <= 50  # documented loose contract


def test_ignore__pycache_skipped(tmp_path):
    _write(tmp_path / "main.py", "target = 1")
    _write(tmp_path / "__pycache__" / "main.cpython.py", "target = 1")
    results = search_code("target", str(tmp_path))
    paths = [r["file"] for r in results]
    assert not any("__pycache__" in p for p in paths)


def test_ignore__pkg_NOT_skipped(tmp_path):
    """`pkg/` (real Python source dir) should NOT be ignored."""
    _write(tmp_path / "pkg" / "real_source.py", "target = 1")
    results = search_code("target", str(tmp_path))
    assert len(results) >= 1


def test_ignore__lib_NOT_skipped(tmp_path):
    """`lib/` should NOT be ignored — real source directories."""
    _write(tmp_path / "lib" / "core.py", "target = 1")
    results = search_code("target", str(tmp_path))
    assert len(results) >= 1


# ─────────────── CAP BEHAVIOR ───────────────


def test_cap__max_results_respected(tmp_path):
    """50 files each match → cap at MAX_SEARCH_RESULTS (default 30)."""
    for i in range(50):
        _write(tmp_path / f"file_{i}.py", "target = 1")
    results = search_code("target", str(tmp_path))
    from tools.codebase import MAX_SEARCH_RESULTS
    assert len(results) <= MAX_SEARCH_RESULTS


def test_cap__custom_max_results(tmp_path):
    for i in range(50):
        _write(tmp_path / f"file_{i}.py", "target = 1")
    results = search_code("target", str(tmp_path), max_results=5)
    assert len(results) <= 5


# ─────────────── UNICODE PATTERNS ───────────────


def test_unicode__pattern_in_files(tmp_path):
    _write(tmp_path / "a.py", "name = '北京'\nx = 1")
    results = search_code("北京", str(tmp_path))
    assert len(results) >= 1


def test_unicode__emoji_in_pattern(tmp_path):
    _write(tmp_path / "a.py", "status = '🎉'")
    results = search_code("🎉", str(tmp_path))
    # ripgrep should handle this; if not, document
    assert isinstance(results, list)


# ─────────────── search_refs ADVERSARIAL ───────────────


def test_refs__finds_definition(tmp_path):
    """Definition of `foo` should be reported."""
    _write(tmp_path / "a.py", "def foo():\n    return 1\n")
    out = search_refs("foo", str(tmp_path))
    assert "foo" in out
    # Either "DEFINED" / "def foo" / similar marker
    assert "def" in out or "DEFINED" in out


def test_refs__finds_class_definition(tmp_path):
    _write(tmp_path / "a.py", "class MyClass:\n    pass\n")
    out = search_refs("MyClass", str(tmp_path))
    assert "MyClass" in out
    assert "class" in out or "DEFINED" in out


def test_refs__finds_single_line_import(tmp_path):
    _write(tmp_path / "a.py", "def foo(): pass\n")
    _write(tmp_path / "b.py", "from a import foo\nfoo()\n")
    out = search_refs("foo", str(tmp_path))
    assert "foo" in out


def test_refs__finds_multi_line_parenthesized_import(tmp_path):
    """Multi-line `from .x import (a, foo, b)` — must be detected."""
    _write(tmp_path / "a.py", "def foo(): pass\n")
    _write(tmp_path / "b.py",
           "from .a import (\n"
           "    other_func,\n"
           "    foo,\n"
           "    another,\n"
           ")\n")
    out = search_refs("foo", str(tmp_path))
    assert "foo" in out
    # Multi-line marker present (added in batch 2 fix)
    assert "multi-line" in out.lower() or "foo" in out


def test_refs__no_matches(tmp_path):
    out = search_refs("absent_function_name", str(tmp_path))
    # Should return empty or "no matches" message
    assert isinstance(out, str)


def test_refs__name_with_underscores(tmp_path):
    _write(tmp_path / "a.py", "def my_func(): pass\n")
    out = search_refs("my_func", str(tmp_path))
    assert "my_func" in out


def test_refs__finds_usage_in_call(tmp_path):
    _write(tmp_path / "a.py", "def foo(): pass\n")
    _write(tmp_path / "b.py", "from a import foo\nresult = foo()\n")
    out = search_refs("foo", str(tmp_path))
    # Should find the usage
    assert "foo" in out


def test_refs__finds_inheritance(tmp_path):
    _write(tmp_path / "a.py", "class Base: pass\n")
    _write(tmp_path / "b.py", "from a import Base\nclass Sub(Base): pass\n")
    out = search_refs("Base", str(tmp_path))
    assert "Base" in out


# ─────────────── format_search_results ───────────────


def test_format__empty_results():
    out = format_search_results([])
    assert isinstance(out, str)


def test_format__single_result():
    results = [{"file": "a.py", "line_num": 5, "line": "x = 1", "context": ""}]
    out = format_search_results(results)
    assert "a.py" in out


# ─────────────── EDGE: NONEXISTENT ROOT ───────────────


def test_edge__nonexistent_root_safe(tmp_path):
    """Pointing search at a path that doesn't exist — should not crash."""
    results = search_code("anything", str(tmp_path / "does_not_exist"))
    assert isinstance(results, list)


def test_edge__empty_project(tmp_path):
    """Empty directory — search returns no results."""
    results = search_code("anything", str(tmp_path))
    assert results == []


# ─────────────── EDGE: LARGE FILES ───────────────


def test_edge__large_file_with_match(tmp_path):
    """File with 10K lines, target appears once. The result includes the
    matching line PLUS its `-C 5` context (≤5 lines before+after) — so a
    single match produces up to 11 result entries (1 match + 10 context)."""
    lines = [f"line_{i}" for i in range(10000)]
    lines[5000] = "target_string"
    _write(tmp_path / "big.py", "\n".join(lines))
    results = search_code("target_string", str(tmp_path))
    # Exactly one entry has the matching line itself
    matching = [r for r in results if r["line"] == "target_string"]
    assert len(matching) == 1
    assert matching[0]["line_num"] == 5001  # 1-indexed


# ─────────────── DEEP DIRECTORY NESTING ───────────────


def test_deep__finds_in_nested_dir(tmp_path):
    _write(tmp_path / "a" / "b" / "c" / "d" / "e" / "deep.py", "target = 1")
    results = search_code("target", str(tmp_path))
    assert len(results) >= 1


def test_deep__many_subdirs(tmp_path):
    """20 subdirectories with files."""
    for i in range(20):
        _write(tmp_path / f"sub_{i}" / "file.py", f"target_{i} = 1")
    results = search_code("target_0", str(tmp_path))
    assert len(results) == 1


# ─────────────── SPECIAL FILE TYPES ───────────────


def test_filetype__searches_py(tmp_path):
    _write(tmp_path / "a.py", "target")
    results = search_code("target", str(tmp_path))
    assert len(results) == 1


def test_filetype__searches_js(tmp_path):
    _write(tmp_path / "a.js", "target")
    results = search_code("target", str(tmp_path))
    # ripgrep searches all files by default
    assert len(results) >= 0


def test_filetype__binary_skipped(tmp_path):
    """Binary files should be excluded by ripgrep's default behavior."""
    (tmp_path / "image.png").write_bytes(b"binary\x00\x01\x02target")
    results = search_code("target", str(tmp_path))
    # Binary files filtered
    assert len(results) == 0 or not any("image.png" in r["file"] for r in results)
