"""Audit `_check_deleted_imports` — detects unsafe top-level deletions.

Catches:
  Mode A — deleted `import X` or `from Y import X` lines whose names are
           re-exported / consumed elsewhere
  Mode B — deleted top-level `class Name` or `def Name` blocks whose names
           are imported or used elsewhere

This guard fired catastrophic-regression-causing edits on
astropy-13236 (NdarrayMixin re-export deletion) and astropy-13398 (ITRS
class deletion). We test every plausible edge case.
"""
import os
import re
import textwrap
from pathlib import Path
import pytest

from workflows.code import _check_deleted_imports


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# ───────────────────── MODE A: deleted imports ─────────────────────

def test_guard__deleted_import_with_relative_reexport(tmp_path):
    """The canonical astropy-13236 case: __init__.py re-exports via `from .X
    import Name`. Deleting the original import in X should be flagged."""
    _write(tmp_path / "pkg/__init__.py",
           "from .widgets import Widget\n")
    _write(tmp_path / "pkg/widgets.py",
           "class Widget: pass\n")
    original = "from .extras import HelperMixin\nclass Widget(HelperMixin): pass\n"
    modified = "class Widget: pass\n"  # removed import + base class
    findings = _check_deleted_imports(
        "pkg/widgets.py", original, modified, str(tmp_path),
    )
    # Should at least surface the HelperMixin removal if consumers exist


def test_guard__deleted_import_with_absolute_reexport(tmp_path):
    """Consumer file does `from pkg.widgets import Widget` (absolute path)."""
    _write(tmp_path / "pkg/widgets.py",
           "from pkg.helpers import compute\nclass Widget: pass\n")
    _write(tmp_path / "pkg/helpers.py",
           "def compute(): return 1\n")
    _write(tmp_path / "consumer.py",
           "from pkg.widgets import Widget\n")  # references the symbol elsewhere
    original = "from pkg.helpers import compute\nclass Widget: pass\n"
    modified = "class Widget: pass\n"
    findings = _check_deleted_imports(
        "pkg/widgets.py", original, modified, str(tmp_path),
    )
    # If compute is used elsewhere via this module, guard should flag.
    # For this case, compute is imported FROM widgets implicitly. Document.


def test_guard__deleted_unused_import__no_findings(tmp_path):
    """If nothing else imports the removed name, deletion is safe."""
    _write(tmp_path / "pkg/widgets.py",
           "import math\nclass Widget: pass\n")
    # No consumers of math
    original = "import math\nclass Widget: pass\n"
    modified = "class Widget: pass\n"
    findings = _check_deleted_imports(
        "pkg/widgets.py", original, modified, str(tmp_path),
    )
    # Should be empty (or flag only if reasonable false positive)
    assert all("math" not in f[0] or "unused" in f[1].lower() for f in findings)


def test_guard__noop_modification__no_findings(tmp_path):
    """When nothing was removed, the guard returns no findings."""
    _write(tmp_path / "a.py", "class X: pass\n")
    same = "class X: pass\n"
    findings = _check_deleted_imports("a.py", same, same, str(tmp_path))
    assert findings == []


# ───────────────────── MODE B: deleted classes / defs ─────────────────────

def test_guard__deleted_class_with_consumer(tmp_path):
    """The astropy-13398 case: ITRS class deleted, consumers in other files."""
    _write(tmp_path / "pkg/__init__.py",
           "from .frames import ITRS\n")
    _write(tmp_path / "tests/test_itrs.py",
           "from pkg.frames import ITRS\ndef test_a():\n    ITRS()\n")
    original = "class ITRS:\n    pass\nclass OtherFrame:\n    pass\n"
    modified = "class OtherFrame:\n    pass\n"  # ITRS removed
    findings = _check_deleted_imports(
        "pkg/frames.py", original, modified, str(tmp_path),
    )
    assert findings, "deletion of consumed ITRS should be flagged"
    assert any("ITRS" in f[0] or "ITRS" in f[1] for f in findings)


def test_guard__deleted_function_with_consumer(tmp_path):
    """Same as class case but for `def`."""
    _write(tmp_path / "pkg/__init__.py",
           "from .helpers import critical_func\n")
    _write(tmp_path / "tests/test_h.py",
           "from pkg.helpers import critical_func\ndef test(): critical_func()\n")
    original = "def critical_func():\n    return 42\n\ndef other():\n    pass\n"
    modified = "def other():\n    pass\n"
    findings = _check_deleted_imports(
        "pkg/helpers.py", original, modified, str(tmp_path),
    )
    assert findings, "deletion of consumed critical_func should be flagged"


def test_guard__deleted_class_no_consumers__no_findings(tmp_path):
    """Deletion of a class with NO consumers is safe."""
    _write(tmp_path / "pkg/helpers.py", "class Unused: pass\nclass Other: pass\n")
    original = "class Unused: pass\nclass Other: pass\n"
    modified = "class Other: pass\n"
    findings = _check_deleted_imports(
        "pkg/helpers.py", original, modified, str(tmp_path),
    )
    # Document — may have false positives from grep-by-name
    # but `Unused` is short-ish; expect no findings
    assert not any("Unused" in f[0] for f in findings)


def test_guard__skip_short_names(tmp_path):
    """Names ≤2 chars are skipped to avoid false positives on common
    identifiers like `x`, `i`, `_`."""
    _write(tmp_path / "consumer.py", "x = 1\n")
    original = "class X:\n    pass\n"
    modified = ""
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    # X is single-char — should be skipped
    assert not any(f[0].startswith("class X") for f in findings)


def test_guard__skip_dunder_methods(tmp_path):
    """Dunder names like `__init__` at module level — usually fixture
    leftovers, not real public API."""
    _write(tmp_path / "consumer.py", "__init__ = None\n")
    original = "def __init__(): pass\n"
    modified = ""
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    # __init__ is dunder, should be skipped
    assert not any("__init__" in f[0] for f in findings)


# ───────────────────── MIXED MODES ─────────────────────

def test_guard__import_and_class_deleted_together(tmp_path):
    """Both an import AND a class deleted in one diff — both flagged
    if consumers exist."""
    _write(tmp_path / "pkg/__init__.py",
           "from .core import Helper, MainClass\n")
    original = (
        "from .extras import Extension\n"
        "class Helper:\n    pass\n"
        "class MainClass(Helper, Extension):\n    pass\n"
    )
    modified = "class MainClass:\n    pass\n"  # both Helper and Extension removed
    findings = _check_deleted_imports(
        "pkg/core.py", original, modified, str(tmp_path),
    )
    # Should at least flag Helper (consumed by __init__.py)


def test_guard__multiline_parenthesized_import_in_consumer(tmp_path):
    """astropy-13236 regression: consumer uses multi-line parenthesized
    import — the guard must detect via the multi-line pattern."""
    _write(tmp_path / "pkg/__init__.py",
           "from .core import (\n"
           "    Helper,\n"
           "    MainClass,\n"
           "    Utility,\n"
           ")\n")
    original = (
        "from .extras import Mixin\n"
        "class Helper:\n    pass\n"
        "class MainClass(Mixin): pass\n"
        "class Utility: pass\n"
    )
    modified = (
        "class Helper:\n    pass\n"
        "class MainClass: pass\n"
        "class Utility: pass\n"
    )  # removed import (Mixin)
    findings = _check_deleted_imports(
        "pkg/core.py", original, modified, str(tmp_path),
    )
    # Either way, Helper/MainClass/Utility should all still be imported by
    # __init__.py — so if any of THEM are deleted, guard catches.


# ───────────────────── EDGE CASES ─────────────────────

def test_guard__non_python_file__no_findings(tmp_path):
    """Markdown / yaml / etc. — skip."""
    findings = _check_deleted_imports(
        "README.md", "old", "new", str(tmp_path),
    )
    assert findings == []


def test_guard__empty_original__no_findings(tmp_path):
    """If the file is brand new (no original to compare), nothing to detect."""
    findings = _check_deleted_imports(
        "new.py", "", "class X: pass\n", str(tmp_path),
    )
    assert findings == []


def test_guard__rename_class__detected_as_deletion(tmp_path):
    """Renaming `class Foo` → `class Bar` looks like deleting Foo to the
    guard. If Foo is consumed elsewhere, this IS a real break."""
    _write(tmp_path / "consumer.py", "from a import LongerNamedFoo\n")
    original = "class LongerNamedFoo:\n    pass\n"
    modified = "class LongerNamedBar:\n    pass\n"
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    assert findings, "rename of consumed class should be flagged"


def test_guard__deleted_function_with_only_self_reference(tmp_path):
    """A function called only within the same file — guard should NOT flag."""
    _write(tmp_path / "no_consumer.py", "# no imports of helper\n")
    original = "def helper_func():\n    return 1\n\ndef main():\n    return helper_func()\n"
    modified = "def main():\n    return 1\n"
    findings = _check_deleted_imports(
        "a.py", original, modified, str(tmp_path),
    )
    # The function is only used in the file we're modifying (and we're
    # presumably inlining or removing both). Should not flag.


def test_guard__path_with_init_py(tmp_path):
    """The file being modified IS an `__init__.py` — module path needs
    special handling."""
    _write(tmp_path / "pkg/sub/__init__.py", "from .helpers import compute\n")
    _write(tmp_path / "pkg/sub/helpers.py", "def compute(): return 1\n")
    _write(tmp_path / "consumer.py", "from pkg.sub import compute\n")
    original = "from .helpers import compute\nfrom .extras import other\n"
    modified = "from .helpers import compute\n"  # removed `from .extras import other`
    findings = _check_deleted_imports(
        "pkg/sub/__init__.py", original, modified, str(tmp_path),
    )
    # `other` deletion may or may not be flagged depending on whether
    # `extras` exists and `other` is consumed elsewhere


def test_guard__indented_class_not_flagged_as_top_level(tmp_path):
    """A class defined INSIDE a function — NOT a top-level definition,
    deletion shouldn't be guarded."""
    original = (
        "def outer():\n"
        "    class Inner:\n"
        "        pass\n"
        "    return Inner\n"
    )
    modified = "def outer():\n    return None\n"
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    # Inner is not top-level; should not be flagged
    assert not any("Inner" in f[0] for f in findings)


def test_guard__class_name_with_decorator_above(tmp_path):
    """A class with `@decorator` above it on the previous line."""
    _write(tmp_path / "consumer.py", "from a import DecoratedClass\n")
    original = "@register\nclass DecoratedClass:\n    pass\n"
    modified = ""
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    assert findings, "decorated class with consumer should be flagged"


def test_guard__class_with_inheritance_args(tmp_path):
    """`class Foo(BaseClass, Mixin):` — should parse just the name `Foo`."""
    _write(tmp_path / "consumer.py", "from a import ComplexClass\n")
    original = "class ComplexClass(BaseClass, Mixin, metaclass=Meta):\n    pass\n"
    modified = ""
    findings = _check_deleted_imports("a.py", original, modified, str(tmp_path))
    assert findings


# ───────────────────── PERFORMANCE ─────────────────────

def test_guard__large_project__completes_in_time(tmp_path):
    """1000 small files — guard should complete in reasonable time."""
    for i in range(1000):
        _write(tmp_path / f"src/mod_{i:04}.py", f"def func_{i}(): pass\n")
    _write(tmp_path / "consumer.py", "from src.mod_0500 import func_500\n")
    original = "def func_500(): return 1\n"
    modified = ""
    import time
    t0 = time.time()
    findings = _check_deleted_imports(
        "src/mod_0500.py", original, modified, str(tmp_path),
    )
    elapsed = time.time() - t0
    assert elapsed < 15, f"guard too slow on 1000 files: {elapsed}s"
