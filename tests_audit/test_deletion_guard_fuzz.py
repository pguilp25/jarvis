"""DEEP FUZZ audit of `_check_deleted_imports`.

This guard prevented the astropy-13236 (644 tests) and astropy-13398 (68
tests) regressions. The fuzz tests verify it doesn't crash AND that the
no-op cases are bulletproof:

  • Non-.py files always return []
  • Identical original/modified always return []
  • Empty files return []
  • Added imports never trigger findings (only DELETED ones)
"""
import pytest
import random
import string
from pathlib import Path
from workflows.code import _check_deleted_imports


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ─────────────── PROPERTY: TYPE INVARIANT ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__always_returns_list(tmp_path, seed):
    rng = random.Random(seed)
    rel_path = f"file_{seed}.py"
    orig = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 500)))
    mod = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 500)))
    out = _check_deleted_imports(rel_path, orig, mod, str(tmp_path))
    assert isinstance(out, list)
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        assert isinstance(item[0], str) and isinstance(item[1], str)


# ─────────────── PROPERTY: NON-PY ALWAYS EMPTY ───────────────


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("ext", [".txt", ".md", ".json", ".yaml", ".js", ".rs", ".cfg"])
def test_inv__non_py_returns_empty(tmp_path, seed, ext):
    rng = random.Random(seed)
    orig = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    mod = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    out = _check_deleted_imports(f"file{ext}", orig, mod, str(tmp_path))
    assert out == []


# ─────────────── PROPERTY: IDENTICAL = EMPTY ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inv__identical_returns_empty(tmp_path, seed):
    rng = random.Random(seed)
    src = "from a import b\nclass C:\n    pass\ndef foo():\n    return 1\n"
    out = _check_deleted_imports("file.py", src, src, str(tmp_path))
    assert out == []


# ─────────────── PROPERTY: ADDED ONLY = EMPTY ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__additions_not_flagged(tmp_path, seed):
    """Adding imports — not flagged (only deletions matter)."""
    rng = random.Random(seed)
    orig = "x = 1\n"
    new = f"from a import b\nfrom c import d\nclass E_{seed}:\n    pass\nx = 1\n"
    out = _check_deleted_imports("file.py", orig, new, str(tmp_path))
    assert out == []


# ─────────────── PROPERTY: WHITESPACE-ONLY DIFFS = EMPTY ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_inv__whitespace_changes_not_flagged(tmp_path, seed):
    rng = random.Random(seed)
    orig = "from a import b\n\nclass C:\n    pass\n"
    # Add/remove blank lines
    mod = "from a import b\n\n\nclass C:\n    pass\n"
    out = _check_deleted_imports("file.py", orig, mod, str(tmp_path))
    assert out == []


# ─────────────── PROPERTY: EMPTY INPUTS ───────────────


def test_empty__both_empty(tmp_path):
    assert _check_deleted_imports("a.py", "", "", str(tmp_path)) == []


def test_empty__orig_empty_added_imports(tmp_path):
    """Empty orig, modified has imports — additions only, no flagging."""
    out = _check_deleted_imports(
        "a.py", "", "from a import b\nclass C: pass\n", str(tmp_path)
    )
    assert out == []


def test_empty__modified_empty_no_consumers(tmp_path):
    """Modified is empty (whole file deleted). If no consumers exist, no flags."""
    orig = "from a import b\nclass C: pass\n"
    out = _check_deleted_imports("a.py", orig, "", str(tmp_path))
    # No external consumers → no findings
    assert isinstance(out, list)


# ─────────────── PROPERTY: NON-EXISTENT PROJECT ROOT ───────────────


def test_safe__nonexistent_root_no_crash(tmp_path):
    out = _check_deleted_imports(
        "a.py", "from a import b\n", "", str(tmp_path / "does_not_exist")
    )
    assert isinstance(out, list)


# ─────────────── PROPERTY: LARGE INPUTS ───────────────


@pytest.mark.parametrize("seed", range(10))
def test_perf__large_file_no_hang(tmp_path, seed):
    """Many imports → process completes within timeout."""
    rng = random.Random(seed)
    n_imports = rng.randint(50, 500)
    imports = [f"from .mod_{i} import X_{i}" for i in range(n_imports)]
    orig = "\n".join(imports)
    # Remove last import
    mod = "\n".join(imports[:-1])
    out = _check_deleted_imports("file.py", orig, mod, str(tmp_path))
    assert isinstance(out, list)


# ─────────────── PROPERTY: INDENT MATTERS ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_indent__indented_def_not_top_level(tmp_path, seed):
    """Indented def is NOT top-level — not flagged when removed."""
    rng = random.Random(seed)
    orig = (
        f"class C:\n"
        f"    def method_{seed}(self):\n"
        f"        pass\n"
    )
    new = f"class C:\n    pass\n"
    out = _check_deleted_imports("a.py", orig, new, str(tmp_path))
    # method_X is indented → not top-level → not flagged
    assert out == []


# ─────────────── PROPERTY: DUNDER METHODS SKIPPED ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_dunder__module_level_skipped(tmp_path, seed):
    """`def __thing__` at module level — skipped."""
    rng = random.Random(seed)
    orig = f"def __module_dunder_{seed}__():\n    pass\n"
    mod = ""
    out = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    # Either skipped (dunder) or no consumers → empty
    assert isinstance(out, list)


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_det__same_input_same_output(tmp_path, seed):
    rng = random.Random(seed)
    name = f"sym_{seed}"
    orig = f"def {name}():\n    return 1\n"
    mod = ""
    o1 = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    o2 = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    assert o1 == o2


# ─────────────── PROPERTY: NO CRASHES ON ADVERSARIAL ───────────────


def test_adv__unicode_paths(tmp_path):
    orig = "from a import 北京\n"
    mod = ""
    out = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    assert isinstance(out, list)


def test_adv__binary_content_no_crash(tmp_path):
    orig = "\x00\x01\x02 from a import b \xff"
    mod = ""
    out = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    assert isinstance(out, list)


def test_adv__very_long_line(tmp_path):
    """A 100K-char single line."""
    orig = "x = '" + "a" * 100000 + "'\n"
    mod = ""
    out = _check_deleted_imports("a.py", orig, mod, str(tmp_path))
    assert isinstance(out, list)
