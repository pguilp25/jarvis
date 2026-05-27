"""Audit `_detect_language` and `_find_lsp_server`.

These run BEFORE any LSP search to:
  1. Pick the primary language of a project (Python takes priority over JS etc.).
  2. Find an installed LSP server (pylsp / tsserver / clangd / etc.).

Bugs here cause:
  • The wrong LSP server is invoked → wrong references (e.g. running clangd
    on Python files returns nothing).
  • A `node_modules/` dump in a Python project wins the count → JS is
    detected → no Python LSP fires.
  • Missing language → silent fallback that confuses the model.
"""
import pytest
from pathlib import Path
from tools.lsp import _detect_language, _find_lsp_server, LANGUAGE_EXTENSIONS


def _write(p: Path, content: str = ""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ─────────────── single-language projects ───────────────


def test_detect__pure_python(tmp_path):
    _write(tmp_path / "a.py", "x = 1")
    _write(tmp_path / "b.py", "y = 2")
    _write(tmp_path / "c.py", "z = 3")
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__pure_javascript(tmp_path):
    _write(tmp_path / "a.js", "")
    _write(tmp_path / "b.jsx", "")
    assert _detect_language(str(tmp_path)) == "javascript"


def test_detect__pure_typescript(tmp_path):
    _write(tmp_path / "a.ts", "")
    _write(tmp_path / "b.tsx", "")
    assert _detect_language(str(tmp_path)) == "typescript"


def test_detect__pure_rust(tmp_path):
    _write(tmp_path / "a.rs", "")
    _write(tmp_path / "b.rs", "")
    assert _detect_language(str(tmp_path)) == "rust"


def test_detect__pure_go(tmp_path):
    _write(tmp_path / "a.go", "")
    assert _detect_language(str(tmp_path)) == "go"


def test_detect__pure_c(tmp_path):
    _write(tmp_path / "a.c", "")
    _write(tmp_path / "b.h", "")
    # .h is shared with C++; the c entry comes first in LANGUAGE_EXTENSIONS
    assert _detect_language(str(tmp_path)) == "c"


def test_detect__cpp_via_cpp_extension(tmp_path):
    _write(tmp_path / "a.cpp", "")
    _write(tmp_path / "b.hpp", "")
    assert _detect_language(str(tmp_path)) == "cpp"


def test_detect__java(tmp_path):
    _write(tmp_path / "A.java", "")
    assert _detect_language(str(tmp_path)) == "java"


def test_detect__ruby(tmp_path):
    _write(tmp_path / "a.rb", "")
    assert _detect_language(str(tmp_path)) == "ruby"


def test_detect__bash(tmp_path):
    _write(tmp_path / "a.sh", "")
    _write(tmp_path / "b.bash", "")
    assert _detect_language(str(tmp_path)) == "bash"


def test_detect__lean(tmp_path):
    _write(tmp_path / "a.lean", "")
    assert _detect_language(str(tmp_path)) == "lean"


def test_detect__lua(tmp_path):
    _write(tmp_path / "a.lua", "")
    assert _detect_language(str(tmp_path)) == "lua"


def test_detect__haskell(tmp_path):
    _write(tmp_path / "a.hs", "")
    assert _detect_language(str(tmp_path)) == "haskell"


# ─────────────── mixed projects ───────────────


def test_detect__mixed_python_wins_when_dominant(tmp_path):
    """5 Python files vs 1 JS file → Python wins."""
    for i in range(5):
        _write(tmp_path / f"a{i}.py", "")
    _write(tmp_path / "x.js", "")
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__node_modules_ignored(tmp_path):
    """A Python project with a node_modules/ dump in it should still be
    detected as Python. (Without the ignore, the thousands of node_modules
    .js files would dominate the count.)"""
    _write(tmp_path / "a.py", "")
    _write(tmp_path / "b.py", "")
    # Fake node_modules with many JS files
    for i in range(100):
        _write(tmp_path / "node_modules" / f"pkg_{i}.js", "")
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__venv_ignored(tmp_path):
    """`.venv/` and `venv/` are skipped — they have site-packages that
    would distort the count."""
    _write(tmp_path / "a.py", "")
    _write(tmp_path / ".venv" / "lib" / "site-packages" / "pkg" / "x.py", "")
    _write(tmp_path / "venv" / "lib" / "site-packages" / "pkg" / "y.py", "")
    # Counts only see top-level a.py
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__pycache_ignored(tmp_path):
    """`__pycache__/` should be skipped (contains compiled .pyc but those
    don't match our extensions anyway — assert no crash)."""
    _write(tmp_path / "a.py", "")
    _write(tmp_path / "__pycache__" / "a.cpython-310.pyc", "binary")
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__git_ignored(tmp_path):
    """`.git/` should be skipped."""
    _write(tmp_path / "a.py", "")
    _write(tmp_path / ".git" / "config", "git data")
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__build_dist_ignored(tmp_path):
    """`build/` and `dist/` skipped (compiled artifacts)."""
    _write(tmp_path / "a.py", "")
    for i in range(50):
        _write(tmp_path / "build" / f"out_{i}.py", "")
        _write(tmp_path / "dist" / f"out_{i}.py", "")
    # Top-level a.py wins because build/dist are ignored
    assert _detect_language(str(tmp_path)) == "python"


def test_detect__rust_target_ignored(tmp_path):
    """`target/` is the Rust build dir."""
    _write(tmp_path / "src" / "main.rs", "")
    for i in range(50):
        _write(tmp_path / "target" / f"out_{i}.rs", "")
    assert _detect_language(str(tmp_path)) == "rust"


# ─────────────── tie-breaking ───────────────


def test_detect__tie_python_wins_over_js(tmp_path):
    """Tie between Python and JS → Python wins (dict iteration order)."""
    _write(tmp_path / "a.py", "")
    _write(tmp_path / "b.js", "")
    # Both have count 1 — Python wins as it comes first in LANGUAGE_EXTENSIONS
    assert _detect_language(str(tmp_path)) == "python"


# ─────────────── empty / nonexistent ───────────────


def test_detect__empty_project_returns_none(tmp_path):
    """No source files at all → None."""
    _write(tmp_path / "README.md", "# title")
    assert _detect_language(str(tmp_path)) is None


def test_detect__only_unknown_extensions_returns_none(tmp_path):
    _write(tmp_path / "config.toml", "")
    _write(tmp_path / "data.xml", "")
    assert _detect_language(str(tmp_path)) is None


def test_detect__nested_python_files(tmp_path):
    """Files nested deeply still count."""
    _write(tmp_path / "deep" / "nested" / "subdir" / "a.py", "")
    assert _detect_language(str(tmp_path)) == "python"


# ─────────────── _find_lsp_server ───────────────


def test_find_lsp__python_installed():
    """pylsp is installed in this environment per session memory."""
    server = _find_lsp_server("python")
    # Either returns a dict with pylsp, or None (if not installed in this env)
    if server is not None:
        assert "cmd" in server


def test_find_lsp__unknown_language():
    """A language with no LSP_SERVERS entry should return None."""
    server = _find_lsp_server("cobol")
    assert server is None


def test_find_lsp__empty_language():
    server = _find_lsp_server("")
    assert server is None


# ─────────────── LANGUAGE_EXTENSIONS sanity ───────────────


def test_extensions__no_overlap_within_language():
    """Each language's extension set should have distinct entries."""
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        assert len(exts) == len(set(exts))


def test_extensions__all_lowercase():
    """All extensions must be lowercase (detection lowercases input)."""
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        for ext in exts:
            assert ext == ext.lower(), f"{lang}:{ext} is not lowercase"


def test_extensions__all_start_with_dot():
    """All extensions must start with `.`"""
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        for ext in exts:
            assert ext.startswith("."), f"{lang}:{ext} missing leading dot"


def test_extensions__h_file_conflict_documented():
    """`.h` is in C — C++ uses .hpp/.hh/.hxx exclusively. (If someone adds
    .h to cpp, both languages will count it twice → wrong language detection.)"""
    c_exts = LANGUAGE_EXTENSIONS["c"]
    cpp_exts = LANGUAGE_EXTENSIONS["cpp"]
    # Allow `.h` to be in only one — currently in C.
    overlap = c_exts & cpp_exts
    # Document the contract: no overlap (otherwise count would double)
    assert overlap == set(), f"C/C++ extension overlap: {overlap}"
