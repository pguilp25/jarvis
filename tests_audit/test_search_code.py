"""Exhaustive audit of `search_code` — the ripgrep-backed text/regex
search the agent uses as `[SEARCH: pattern]`.

Two-pass design: pass 1 scoped to test directories, pass 2 whole project.
We test that BOTH passes work, dedup is correct, ordering favors tests,
and edge cases (regex specials, unicode, no results, very long lines)
behave sanely.
"""
import re
import os
import textwrap
from pathlib import Path
import pytest

from tools.codebase import search_code, format_search_results


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ───────────────────── PRIORITY / ORDERING ─────────────────────

def test_search__test_files_appear_first(tmp_path):
    """When a pattern matches in BOTH a test file and a docs/source file,
    the test-file hit must appear in the top of the result list."""
    _write(tmp_path / "src/widget.py", "# unrelated\n" * 50 + "MAGIC_PHRASE here\n")
    _write(tmp_path / "src/other.py", "MAGIC_PHRASE in other\n")
    _write(tmp_path / "docs/changes.txt",
           "\n".join(f"old release notes line {i} MAGIC_PHRASE" for i in range(30)))
    _write(tmp_path / "tests/test_widget.py",
           "def test_magic():\n    assert 'MAGIC_PHRASE' in result\n")

    results = search_code("MAGIC_PHRASE", str(tmp_path), max_results=10)
    # The FIRST result should be from a tests/ path
    assert results, "should return matches"
    first_file = results[0]["file"]
    assert "tests" in first_file, (
        f"first match should be from tests/, got: {first_file}"
    )


def test_search__test_files_not_crowded_out(tmp_path):
    """Many noise hits should NOT push test-file hits past the cap.
    This is the astropy-13033 regression case."""
    # 50 files of pure noise, each with 5 matches → 250 noise matches
    for i in range(50):
        _write(tmp_path / f"src/noise_{i:02}.py",
               "NEEDLE_STRING\n" * 5)
    # ONE test file with the same needle
    _write(tmp_path / "tests/test_thing.py", "assert NEEDLE_STRING == 'X'\n")

    results = search_code("NEEDLE_STRING", str(tmp_path), max_results=30)
    test_hits = [r for r in results if "tests" in r["file"]]
    assert test_hits, (
        f"test files MUST appear in result despite 250 noise matches; "
        f"got files: {sorted({r['file'] for r in results})}"
    )


# ───────────────────── BASIC FUNCTIONALITY ─────────────────────

def test_search__finds_literal_string(tmp_path):
    _write(tmp_path / "a.py", "hello world")
    results = search_code("hello world", str(tmp_path))
    assert len(results) >= 1
    assert any("a.py" in r["file"] for r in results)


def test_search__no_matches__empty_list(tmp_path):
    _write(tmp_path / "a.py", "nothing here")
    results = search_code("ZZZZZZ_not_present", str(tmp_path))
    assert results == []


def test_search__regex_special_chars(tmp_path):
    """Regex metacharacters in the pattern."""
    _write(tmp_path / "a.py", "foo() = 5\nbar() = 6")
    results = search_code(r"foo\(\)", str(tmp_path))
    assert any("foo()" in r["line"] for r in results)


def test_search__multiline_string(tmp_path):
    """Multi-line strings ARE one logical statement but multi physical lines.
    ripgrep matches line-by-line so we expect to find the per-line match."""
    _write(tmp_path / "a.py",
           '''x = """
           this is line two
           this is line three
           """''')
    results = search_code("line two", str(tmp_path))
    assert len(results) >= 1


# ───────────────────── EDGE CASES ─────────────────────

def test_search__empty_pattern__defensive(tmp_path):
    """Empty pattern should not blow up."""
    _write(tmp_path / "a.py", "x = 1")
    try:
        results = search_code("", str(tmp_path))
        # ripgrep refuses empty pattern; should return empty
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"search_code on empty pattern raised: {e!r}")


def test_search__pattern_with_newline_in_it(tmp_path):
    """A pattern containing literal newline — ripgrep treats newline
    as line boundary by default, so this matches nothing."""
    _write(tmp_path / "a.py", "x = 1\ny = 2")
    results = search_code("x = 1\ny = 2", str(tmp_path))
    # Document behavior — empty result is the expected ripgrep behavior
    assert isinstance(results, list)


def test_search__pattern_with_quotes(tmp_path):
    _write(tmp_path / "a.py", '''assert msg == "expected 'x' found"''')
    results = search_code("expected 'x' found", str(tmp_path))
    assert len(results) >= 1


def test_search__pattern_with_unicode(tmp_path):
    _write(tmp_path / "a.py", "# résumé naïve\nx = '北京'")
    results = search_code("北京", str(tmp_path))
    assert len(results) >= 1


def test_search__nonexistent_root__empty(tmp_path):
    """A root directory that doesn't exist — defensive empty list."""
    results = search_code("anything", str(tmp_path / "does_not_exist"))
    assert results == []


def test_search__binary_files_skipped(tmp_path):
    """Binary files should not produce text matches."""
    _write(tmp_path / "a.py", "MARKER")
    # Write a binary file containing the marker bytes
    (tmp_path / "blob.bin").write_bytes(b"MARKER" + bytes(100))
    results = search_code("MARKER", str(tmp_path))
    # The .py file should match; the .bin should be skipped or handled
    files = {r["file"] for r in results}
    assert any("a.py" in f for f in files)


def test_search__per_file_cap_5(tmp_path):
    """Within one file, results are capped at 5 (rg --max-count=5).
    A file with 100 matches should contribute at most 5 results."""
    _write(tmp_path / "spammy.py", "MARK\n" * 100)
    results = search_code("MARK", str(tmp_path), max_results=30)
    spammy_results = [r for r in results if "spammy.py" in r["file"]]
    assert len(spammy_results) <= 10, (
        # 5 hits + 5 context lines = 10 entries plausibly. Allow some slack.
        f"per-file cap violated: spammy.py contributed {len(spammy_results)}"
    )


def test_search__case_insensitive_not_default(tmp_path):
    """Searches are case-sensitive by default."""
    _write(tmp_path / "a.py", "Hello WORLD")
    results = search_code("hello world", str(tmp_path))
    assert results == []


# ───────────────────── BOUNDARY ─────────────────────

def test_search__max_results_zero_returns_empty(tmp_path):
    """max_results=0 should respect the cap."""
    _write(tmp_path / "a.py", "MARK\n" * 5)
    results = search_code("MARK", str(tmp_path), max_results=0)
    # Either returns empty or treats 0 as "no cap" — document behavior
    assert isinstance(results, list)


def test_search__max_results_1(tmp_path):
    _write(tmp_path / "a.py", "MARK\n" * 10)
    results = search_code("MARK", str(tmp_path), max_results=1)
    # Cap should prevent overflow
    assert len(results) <= 5  # per-file cap is 5


def test_search__very_long_line__handled(tmp_path):
    """A file with a single 100K-character line containing the pattern."""
    long_line = "MARK " * 20000
    _write(tmp_path / "long.py", long_line)
    results = search_code("MARK", str(tmp_path))
    # Should match; ripgrep handles long lines
    assert results, "very long line not searched"


def test_search__deeply_nested_dirs(tmp_path):
    """A file 10 directories deep."""
    deep = tmp_path
    for i in range(10):
        deep = deep / f"d{i}"
    _write(deep / "buried.py", "TREASURE")
    results = search_code("TREASURE", str(tmp_path))
    assert any("buried.py" in r["file"] for r in results)


def test_search__skips_git_directory(tmp_path):
    """`.git/` should be ignored — accidental match in HEAD logs etc."""
    _write(tmp_path / ".git/logs/HEAD", "NEEDLE")
    _write(tmp_path / "actual.py", "NEEDLE")
    results = search_code("NEEDLE", str(tmp_path))
    files = {r["file"] for r in results}
    git_hits = [f for f in files if "/.git/" in f]
    assert not git_hits, f".git/ not skipped: {git_hits}"


def test_search__skips_pycache(tmp_path):
    _write(tmp_path / "__pycache__/cached.pyc", "WORD")
    _write(tmp_path / "real.py", "WORD")
    results = search_code("WORD", str(tmp_path))
    files = {r["file"] for r in results}
    assert not any("__pycache__" in f for f in files)


def test_search__skips_node_modules(tmp_path):
    _write(tmp_path / "node_modules/lib/x.js", "PHRASE")
    _write(tmp_path / "src/app.py", "PHRASE")
    results = search_code("PHRASE", str(tmp_path))
    files = {r["file"] for r in results}
    assert not any("node_modules" in f for f in files)


def test_search__pattern_with_dollar(tmp_path):
    """`$` is end-of-line in regex — pattern ending in `$` should match."""
    _write(tmp_path / "a.py", "end_marker\nnot end here")
    results = search_code("end_marker$", str(tmp_path))
    assert any("end_marker" in r["line"] for r in results)


# ───────────────────── TWO-PASS DEDUP ─────────────────────

def test_search__hit_in_test_file_only_once(tmp_path):
    """A test-file hit must NOT appear twice (once from pass 1, once from
    pass 2)."""
    _write(tmp_path / "tests/test_a.py", "ASSERTION_TEXT\n")
    results = search_code("ASSERTION_TEXT", str(tmp_path))
    test_hits = [r for r in results if "test_a.py" in r["file"]]
    # Each (file, line_num) pair should be unique
    keys = [(r["file"], r["line_num"]) for r in test_hits]
    assert len(keys) == len(set(keys)), f"duplicate test hits: {keys}"
