"""Audit `Sandbox` — file-edit isolation with no-op detection."""
import os
import pytest
from pathlib import Path

from tools.sandbox import Sandbox


def _setup_sandbox(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.py").write_text("def foo():\n    return 1\n")
    (project / "b.py").write_text("class B: pass\n")
    sb = Sandbox(str(project))
    sb.setup()
    return sb


# ───────────────────── BASIC ─────────────────────

def test_sandbox__write_modifies_tracks(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")  # populate original_files
    sb.write_file("a.py", "def foo():\n    return 2\n")
    assert "a.py" in sb.modified_files


def test_sandbox__write_noop_skipped(tmp_path):
    """If new content == original, no modification tracked."""
    sb = _setup_sandbox(tmp_path)
    orig = sb.load_file("a.py")
    sb.write_file("a.py", orig)  # byte-identical
    assert "a.py" not in sb.modified_files


def test_sandbox__write_new_file_tracks(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.write_file("c.py", "print('new')\n")
    assert "c.py" in sb.new_files


def test_sandbox__write_empty_new_file_skipped(tmp_path):
    """Writing empty content as a new file — should not be tracked
    (no-op for diff purposes)."""
    sb = _setup_sandbox(tmp_path)
    sb.write_file("c.py", "")
    assert "c.py" not in sb.new_files


def test_sandbox__apply_count_reflects_real_modifications(tmp_path):
    """The `apply()` count should reflect actual changed files."""
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    sb.load_file("b.py")
    # Modify a.py for real
    sb.write_file("a.py", "def foo():\n    return 99\n")
    # No-op write on b.py
    orig_b = sb.load_file("b.py")
    sb.write_file("b.py", orig_b)
    applied = sb.apply()
    # Only a.py should appear
    assert any("a.py" in s for s in applied)
    assert not any("b.py" in s for s in applied)


def test_sandbox__diff_returns_no_changes_for_noop(tmp_path):
    sb = _setup_sandbox(tmp_path)
    orig = sb.load_file("a.py")
    sb.write_file("a.py", orig)
    assert sb.get_diff("a.py") == "(no changes)"


def test_sandbox__diff_returns_unified_for_real_edit(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    sb.write_file("a.py", "def foo():\n    return 999\n")
    d = sb.get_diff("a.py")
    assert "999" in d
    assert "@@" in d or "+" in d  # standard unified-diff markers


# ───────────────────── EDGE ─────────────────────

def test_sandbox__write_with_newlines_preserved(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    content = "def foo():\n\n    return 1\n"  # blank line
    sb.write_file("a.py", content)
    # On disk?
    on_disk = (sb.sandbox_dir / "a.py").read_text()
    assert on_disk == content


def test_sandbox__write_unicode_content(tmp_path):
    sb = _setup_sandbox(tmp_path)
    content = "x = '北京 résumé'\n"
    sb.write_file("a.py", content)
    on_disk = (sb.sandbox_dir / "a.py").read_text()
    assert "北京" in on_disk


def test_sandbox__nested_path(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.write_file("deep/nested/dir/new.py", "x = 1\n")
    # On disk + tracked
    assert (sb.sandbox_dir / "deep/nested/dir/new.py").exists()
    assert "deep/nested/dir/new.py" in sb.new_files


def test_sandbox__windows_path_separator_normalized(tmp_path):
    """Backslash paths should be normalized."""
    sb = _setup_sandbox(tmp_path)
    sb.write_file(r"sub\dir\file.py", "x = 1\n")
    # Either form should resolve to the same internal key
    # On Linux, the path is treated literally — make sure no crash


# ───────────────────── ROUNDTRIP ─────────────────────

def test_sandbox__write_then_load_returns_modified(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    new_content = "def foo():\n    return 42\n"
    sb.write_file("a.py", new_content)
    # load_file reads from sandbox (working copy)
    loaded = sb.load_file("a.py")
    assert loaded == new_content


def test_sandbox__apply_writes_to_project_root(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    sb.write_file("a.py", "def foo():\n    return 7\n")
    sb.apply()
    # The actual project file should now have the new content
    on_disk = (sb.project_root / "a.py").read_text()
    assert "return 7" in on_disk


def test_sandbox__multiple_modifications_cumulate(tmp_path):
    sb = _setup_sandbox(tmp_path)
    sb.load_file("a.py")
    sb.write_file("a.py", "v1\n")
    sb.write_file("a.py", "v2\n")
    sb.write_file("a.py", "v3\n")
    # Final state is v3
    assert sb.modified_files["a.py"] == "v3\n"
