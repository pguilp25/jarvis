"""Shared fixtures + helpers for the aerospace-grade tool audit.

Conventions:
 - Each test file is named test_<tool>.py
 - Each test function asserts ONE specific property of the tool
 - Test names follow: test_<tool>__<scenario>__<expected>
 - Bugs found get a `bug_<short_id>` marker comment in the docstring so
   we can grep `grep -r 'bug_' tests_audit/` to enumerate findings.

Run all:    pytest tests_audit/ -q
Run one:    pytest tests_audit/test_line_prefix.py -q
Verbose:    pytest tests_audit/test_line_prefix.py -v
Stop first: pytest tests_audit/ -x
"""
import os
import sys
import tempfile
import textwrap
from pathlib import Path
import pytest

# Make `workflows`/`tools`/`core` importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def temp_project(tmp_path):
    """Empty temporary project root for tests that need a sandbox dir."""
    yield tmp_path


@pytest.fixture
def fake_python_project(tmp_path):
    """A minimal multi-file Python project — fake imports + tests.

    Layout:
      pkg/
        __init__.py            (re-exports `Widget` from .widgets)
        widgets.py             (defines Widget, Gadget; imports helpers)
        helpers.py             (defines `compute`, `_private`)
        sub/
          __init__.py          (re-exports `compute` from ..helpers)
          consumer.py          (calls Widget(), compute())
        tests/
          test_widget.py       (asserts Widget() raises specific error)
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .widgets import Widget, Gadget\n"
        "from .helpers import compute\n"
        "from .sub.consumer import call_widget\n"
    )
    (pkg / "widgets.py").write_text(
        "from .helpers import compute\n"
        "\n"
        "class Widget:\n"
        "    def __init__(self, x):\n"
        "        if x < 0:\n"
        "            raise ValueError(\n"
        "                'Widget rejected x=%r — expected non-negative' % x\n"
        "            )\n"
        "        self.x = x\n"
        "\n"
        "class Gadget:\n"
        "    def __init__(self):\n"
        "        self.val = compute(7)\n"
    )
    (pkg / "helpers.py").write_text(
        "def compute(n):\n"
        "    return n * 2\n"
        "\n"
        "def _private():\n"
        "    return 'private'\n"
    )
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text(
        "from ..helpers import compute\n"
    )
    (sub / "consumer.py").write_text(
        "from ..widgets import Widget\n"
        "from ..helpers import compute\n"
        "\n"
        "def call_widget(x):\n"
        "    return Widget(x).x + compute(x)\n"
    )
    tests = pkg / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_widget.py").write_text(
        "import pytest\n"
        "from pkg.widgets import Widget\n"
        "\n"
        "def test_widget_rejects_negative():\n"
        "    with pytest.raises(ValueError) as exc:\n"
        "        Widget(-1)\n"
        "    assert exc.value.args[0] == (\n"
        "        \"Widget rejected x=-1 — expected non-negative\"\n"
        "    )\n"
    )
    return tmp_path


@pytest.fixture
def fake_python_project_with_multiline_imports(tmp_path):
    """A project with PARENTHESIZED multi-line imports — known blind spot.

    pkg/__init__.py imports Widget on line 2 of a multi-line block:
        from .widgets import (
            Widget,            ← line 2 — this is the bug case
            Gadget,
        )
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from .widgets import (\n"
        "    Widget,\n"
        "    Gadget,\n"
        "    Helper,\n"
        ")\n"
        "from .nested import (Alpha,\n"
        "                    Beta,\n"
        "                    Gamma)\n"
    )
    (pkg / "widgets.py").write_text(
        "class Widget:\n    pass\n\n"
        "class Gadget:\n    pass\n\n"
        "class Helper:\n    pass\n"
    )
    (pkg / "nested.py").write_text(
        "class Alpha:\n    pass\n\n"
        "class Beta:\n    pass\n\n"
        "class Gamma:\n    pass\n"
    )
    return tmp_path


@pytest.fixture
def fake_test_heavy_project(tmp_path):
    """Project where the error-message string appears in MANY non-test files
    plus the actual test. Designed to defeat naive search caps.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # 10 noise files all containing the search string
    for i in range(10):
        (pkg / f"noise_{i}.py").write_text(
            "# this file mentions the magic phrase several times\n"
            * 5
            + "# magic phrase: WIDGET INVALID STATE\n" * 3
        )
    # The actual test file the agent should find
    tests = pkg / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_widget_state.py").write_text(
        "import pytest\n"
        "from pkg.widget import Widget\n"
        "\n"
        "def test_widget_invalid_state_message():\n"
        "    with pytest.raises(ValueError) as exc:\n"
        "        Widget().validate()\n"
        "    assert 'WIDGET INVALID STATE' in str(exc.value)\n"
    )
    return tmp_path


# Make bug log easy
BUGS_LOG = Path("/tmp/audit_bugs.log")


def record_bug(short_id: str, description: str):
    """Append a bug entry to /tmp/audit_bugs.log."""
    BUGS_LOG.write_text(
        (BUGS_LOG.read_text() if BUGS_LOG.exists() else "")
        + f"\n=== bug_{short_id} ===\n{description}\n"
    )
