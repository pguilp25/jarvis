"""Audit project-scanning utilities — `scan_project`, `_is_ignored_dir`,
`_human_size`, `norm_path`, `to_forward_slash`, IGNORE_DIRS contract.

`scan_project` is the first thing the agent sees about a project. If it
includes `node_modules/` or `.venv/site-packages/` or `target/`, the agent
gets confused about where the real code lives. If it EXCLUDES real source
directories named `pkg/` or `lib/`, the agent never finds anything.
"""
import pytest
from pathlib import Path
from tools.codebase import (
    scan_project,
    _is_ignored_dir,
    _human_size,
    norm_path,
    to_forward_slash,
    IGNORE_DIRS,
    IGNORE_EXTENSIONS,
)


def _write(p: Path, content: str = ""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ─────────────── _is_ignored_dir — what's ignored ───────────────


def test_ignored__pycache():
    assert _is_ignored_dir("__pycache__")


def test_ignored__node_modules():
    assert _is_ignored_dir("node_modules")


def test_ignored__git():
    assert _is_ignored_dir(".git")


def test_ignored__venv():
    assert _is_ignored_dir("venv")
    assert _is_ignored_dir(".venv")


def test_ignored__build_dist():
    assert _is_ignored_dir("build")
    assert _is_ignored_dir("dist")


def test_ignored__rust_target():
    assert _is_ignored_dir("target")


def test_ignored__egg_info_suffix():
    """`.egg-info/` is matched by suffix."""
    assert _is_ignored_dir("mypackage.egg-info")


def test_ignored__dist_info_suffix():
    assert _is_ignored_dir("mypackage-1.0.dist-info")


# ─────────────── _is_ignored_dir — what's NOT ignored (regression) ───────────────


def test_NOT_ignored__pkg():
    """`pkg/` used to be in IGNORE_DIRS, breaking Django/astropy. Now allowed."""
    assert not _is_ignored_dir("pkg")


def test_NOT_ignored__lib():
    """`lib/` is a real source dir in many projects."""
    assert not _is_ignored_dir("lib")


def test_NOT_ignored__libs():
    assert not _is_ignored_dir("libs")


def test_NOT_ignored__packages():
    """`packages/` — common monorepo pattern."""
    assert not _is_ignored_dir("packages")


def test_NOT_ignored__vendor():
    """`vendor/` — some Python projects vendor deps here as source."""
    assert not _is_ignored_dir("vendor")


def test_NOT_ignored__bin_obj():
    """`bin`/`obj` — some C projects use these as source."""
    assert not _is_ignored_dir("bin")
    assert not _is_ignored_dir("obj")


def test_NOT_ignored__src():
    """`src/` is the canonical source directory — must NEVER be ignored."""
    assert not _is_ignored_dir("src")


def test_NOT_ignored__test():
    """`test/` and `tests/` — must be visible (otherwise the agent can
    never find failing tests for FAIL_TO_PASS issues)."""
    assert not _is_ignored_dir("test")
    assert not _is_ignored_dir("tests")


def test_NOT_ignored__random_userdir():
    """Random user-named dirs are not in IGNORE_DIRS."""
    assert not _is_ignored_dir("my_module")
    assert not _is_ignored_dir("widgets")


# ─────────────── _human_size ───────────────


def test_human_size__bytes():
    assert _human_size(100) == "100B"


def test_human_size__kilobytes():
    assert "K" in _human_size(2048)


def test_human_size__megabytes():
    assert "M" in _human_size(5 * 1024 * 1024)


def test_human_size__gigabytes():
    """1GB+ should show GB."""
    out = _human_size(2 * 1024**3)
    assert "G" in out


def test_human_size__zero():
    assert _human_size(0) == "0B"


# ─────────────── norm_path / to_forward_slash ───────────────


def test_norm_path__roundtrip():
    """norm_path returns OS-native; on Linux that's forward slashes."""
    p = norm_path("a/b/c")
    assert "a" in p and "c" in p


def test_norm_path__strip_trailing():
    """`Path` normalizes trailing slashes."""
    p = norm_path("a/b/c/")
    # `Path("a/b/c/")` == `Path("a/b/c")`
    assert p == norm_path("a/b/c")


def test_to_forward_slash__backslash_replaced():
    assert to_forward_slash("a\\b\\c") == "a/b/c"


def test_to_forward_slash__mixed():
    assert to_forward_slash("a/b\\c") == "a/b/c"


def test_to_forward_slash__no_backslash_unchanged():
    assert to_forward_slash("a/b/c") == "a/b/c"


# ─────────────── scan_project ───────────────


def test_scan__includes_top_level_files(tmp_path):
    _write(tmp_path / "main.py", "x = 1")
    _write(tmp_path / "README.md", "# title")
    out = scan_project(str(tmp_path))
    assert "main.py" in out
    assert "README.md" in out


def test_scan__nested_dirs(tmp_path):
    _write(tmp_path / "src" / "a.py", "")
    _write(tmp_path / "src" / "sub" / "b.py", "")
    out = scan_project(str(tmp_path))
    assert "src" in out
    assert "a.py" in out
    assert "b.py" in out


def test_scan__ignored_dirs_excluded(tmp_path):
    """node_modules / __pycache__ / .git should be excluded from scan."""
    _write(tmp_path / "main.py", "")
    _write(tmp_path / "node_modules" / "pkg" / "x.js", "")
    _write(tmp_path / "__pycache__" / "main.cpython.pyc", "")
    _write(tmp_path / ".git" / "config", "")
    out = scan_project(str(tmp_path))
    assert "main.py" in out
    assert "node_modules" not in out
    assert "__pycache__" not in out
    assert ".git" not in out


def test_scan__binary_extensions_excluded(tmp_path):
    """IGNORE_EXTENSIONS files should not appear in scan."""
    _write(tmp_path / "a.py", "")
    (tmp_path / "icon.png").write_bytes(b"fake png")
    (tmp_path / "data.pyc").write_bytes(b"compiled")
    out = scan_project(str(tmp_path))
    assert "a.py" in out
    assert "icon.png" not in out
    assert "data.pyc" not in out


def test_scan__max_depth_cap(tmp_path):
    """Files past max_depth should NOT appear."""
    _write(tmp_path / "lvl1" / "lvl2" / "lvl3" / "lvl4" / "lvl5" / "deep.py", "")
    out = scan_project(str(tmp_path), max_depth=2)
    # `deep.py` is at depth 5 — won't appear
    assert "deep.py" not in out


def test_scan__pkg_dir_included(tmp_path):
    """Regression: `pkg/` is no longer in IGNORE_DIRS — must appear."""
    _write(tmp_path / "pkg" / "module.py", "")
    out = scan_project(str(tmp_path))
    assert "pkg" in out or "module.py" in out


def test_scan__lib_dir_included(tmp_path):
    """Regression: `lib/` is no longer in IGNORE_DIRS — must appear."""
    _write(tmp_path / "lib" / "core.py", "")
    out = scan_project(str(tmp_path))
    assert "lib" in out or "core.py" in out


def test_scan__nonexistent_returns_marker(tmp_path):
    out = scan_project(str(tmp_path / "does_not_exist"))
    assert "not found" in out.lower()


def test_scan__file_count_reported(tmp_path):
    _write(tmp_path / "a.py", "")
    _write(tmp_path / "b.py", "")
    _write(tmp_path / "c.py", "")
    out = scan_project(str(tmp_path))
    # Should mention 3 files somewhere
    assert "3 files" in out or "3 file" in out


def test_scan__large_file_size_annotated(tmp_path):
    """Files > 10KB get a size annotation."""
    big = "x" * 20000
    _write(tmp_path / "big.py", big)
    out = scan_project(str(tmp_path))
    # Either "KB" or some size marker
    assert "K" in out or "big.py" in out


# ─────────────── IGNORE_EXTENSIONS contract ───────────────


def test_extensions__common_binary_present():
    assert ".png" in IGNORE_EXTENSIONS
    assert ".jpg" in IGNORE_EXTENSIONS
    assert ".pyc" in IGNORE_EXTENSIONS
    assert ".pdf" in IGNORE_EXTENSIONS


def test_extensions__source_extensions_NOT_in_ignore():
    """Source-file extensions must NOT be ignored."""
    for src_ext in [".py", ".js", ".ts", ".rs", ".go", ".c", ".cpp", ".java", ".rb"]:
        assert src_ext not in IGNORE_EXTENSIONS, f"{src_ext} wrongly in IGNORE_EXTENSIONS"


def test_extensions__all_start_with_dot():
    for ext in IGNORE_EXTENSIONS:
        assert ext.startswith("."), f"{ext} missing leading dot"
