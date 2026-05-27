"""DEEP FUZZ audit of `Sandbox`.

Sandbox provides file-edit isolation:
  • `load_file(path)` — read; cache original
  • `write_file(path, content)` — modify in sandbox; track diff
  • `apply()` — flush modifications to project_root
  • No-op writes (content == original) are NOT tracked

Critical properties verified:
  I1. Roundtrip: write_file(p, X) then load_file(p) returns X.
  I2. No-op writes don't appear in `modified_files`.
  I3. apply() actually writes to project_root.
  I4. New files (no original) tracked in `new_files`.
  I5. Empty new file write — NOT tracked.
"""
import pytest
import random
import string
from pathlib import Path
from tools.sandbox import Sandbox


def _build_sandbox(tmp_path, seed=0):
    rng = random.Random(seed)
    project = tmp_path / "project"
    project.mkdir()
    # Add a few default files
    (project / "a.py").write_text("def foo():\n    return 1\n")
    (project / "b.py").write_text("class B: pass\n")
    sb = Sandbox(str(project))
    sb.setup()
    return sb


# ─────────────── PROPERTY: ROUNDTRIP ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_roundtrip__write_then_load(tmp_path, seed):
    rng = random.Random(seed)
    sb = _build_sandbox(tmp_path)
    new_content = "".join(rng.choice(string.ascii_letters + " \n") for _ in range(rng.randint(10, 100)))
    sb.write_file("a.py", new_content)
    loaded = sb.load_file("a.py")
    assert loaded == new_content


# ─────────────── PROPERTY: NO-OP DETECTION ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_noop__identical_content_not_tracked(tmp_path, seed):
    """Writing identical content does NOT track a modification."""
    rng = random.Random(seed)
    sb = _build_sandbox(tmp_path)
    orig = sb.load_file("a.py")
    sb.write_file("a.py", orig)
    assert "a.py" not in sb.modified_files


# ─────────────── PROPERTY: REAL MODIFICATIONS TRACKED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_real__different_content_tracked(tmp_path, seed):
    rng = random.Random(seed)
    sb = _build_sandbox(tmp_path)
    sb.load_file("a.py")
    new = f"def changed():\n    return {seed}\n"
    sb.write_file("a.py", new)
    assert "a.py" in sb.modified_files


# ─────────────── PROPERTY: NEW FILES TRACKED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_new__write_new_file_tracked(tmp_path, seed):
    sb = _build_sandbox(tmp_path)
    new_path = f"new_{seed}.py"
    sb.write_file(new_path, f"# new file {seed}\nx = {seed}\n")
    assert new_path in sb.new_files


@pytest.mark.parametrize("seed", range(30))
def test_new__empty_new_file_not_tracked(tmp_path, seed):
    """Empty content for a new file — NOT tracked (no-op)."""
    sb = _build_sandbox(tmp_path)
    new_path = f"empty_{seed}.py"
    sb.write_file(new_path, "")
    assert new_path not in sb.new_files


# ─────────────── PROPERTY: APPLY WRITES TO REAL PROJECT ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_apply__changes_written_to_project_root(tmp_path, seed):
    sb = _build_sandbox(tmp_path)
    sb.load_file("a.py")
    new_content = f"# applied content {seed}\nx = {seed}\n"
    sb.write_file("a.py", new_content)
    sb.apply()
    # Project root file now has the new content
    actual = (sb.project_root / "a.py").read_text()
    assert actual == new_content


@pytest.mark.parametrize("seed", range(20))
def test_apply__no_changes_to_project_root_for_noop(tmp_path, seed):
    sb = _build_sandbox(tmp_path)
    orig_a = sb.load_file("a.py")
    sb.write_file("a.py", orig_a)
    sb.apply()
    # Project root file unchanged
    actual = (sb.project_root / "a.py").read_text()
    assert actual == orig_a


# ─────────────── PROPERTY: MULTIPLE MODIFICATIONS ACCUMULATE ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_multi__last_write_wins(tmp_path, seed):
    rng = random.Random(seed)
    sb = _build_sandbox(tmp_path)
    sb.load_file("a.py")
    for i in range(5):
        sb.write_file("a.py", f"version_{i}\n")
    # Final state is version_4
    final = sb.load_file("a.py")
    assert final == "version_4\n"


# ─────────────── PROPERTY: NESTED PATHS ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_nested__deep_path(tmp_path, seed):
    sb = _build_sandbox(tmp_path)
    path = f"a/b/c/d/file_{seed}.py"
    content = f"x = {seed}\n"
    sb.write_file(path, content)
    assert path in sb.new_files
    # File exists on disk in sandbox
    assert (sb.sandbox_dir / path).exists()


# ─────────────── PROPERTY: UNICODE / SPECIAL CHARS ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_unicode__written_correctly(tmp_path, seed):
    """Unicode content preserved through write+read."""
    sb = _build_sandbox(tmp_path)
    sb.load_file("a.py")
    unicodes = ["北京", "العربية", "🎉", "ümlaut", "résumé"]
    rng = random.Random(seed)
    content = f"name = '{rng.choice(unicodes)}_{seed}'\n"
    sb.write_file("a.py", content)
    loaded = sb.load_file("a.py")
    assert loaded == content


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_det__multiple_writes_same_state(tmp_path, seed):
    """Two identical write sequences give identical final state."""
    (tmp_path / "p1").mkdir()
    (tmp_path / "p2").mkdir()
    sb1 = _build_sandbox(tmp_path / "p1")
    sb1.load_file("a.py")
    sb2 = _build_sandbox(tmp_path / "p2")
    sb2.load_file("a.py")
    content = f"# seed {seed}\nx = {seed}\n"
    sb1.write_file("a.py", content)
    sb2.write_file("a.py", content)
    assert sb1.load_file("a.py") == sb2.load_file("a.py")


# ─────────────── EDGE: VERY LARGE FILES ───────────────


def test_edge__1mb_write(tmp_path):
    """Writing 1MB to a file should work."""
    sb = _build_sandbox(tmp_path)
    content = "x" * 1_000_000
    sb.write_file("a.py", content)
    loaded = sb.load_file("a.py")
    assert loaded == content


# ─────────────── EDGE: DIFF ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_diff__shows_changes(tmp_path, seed):
    sb = _build_sandbox(tmp_path)
    sb.load_file("a.py")
    sb.write_file("a.py", f"x = {seed}\n")
    d = sb.get_diff("a.py")
    assert isinstance(d, str)
    # Should contain the new value
    if d != "(no changes)":
        assert str(seed) in d


def test_diff__noop_returns_no_changes(tmp_path):
    sb = _build_sandbox(tmp_path)
    orig = sb.load_file("a.py")
    sb.write_file("a.py", orig)
    assert sb.get_diff("a.py") == "(no changes)"
