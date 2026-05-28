"""Regression: the [VERIFY:]/[RUN:] sandbox must resolve the venv interpreter.

The reviewer's verification writes bare `python …` / `pytest …`. The bwrap
sandbox PATH used to expose only a bare system python3 (no pytest, no `python`),
so verification silently failed with "python not found" (exit 127) and never
actually ran. The fix binds the running interpreter's dir (the venv bin) and puts
it first on PATH. These tests pin that python/pytest resolve and run.
"""
import os
import pytest

import core.safe_exec as se
from core.safe_exec import run_sandboxed

_HAS_BWRAP = se._BWRAP is not None
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap not installed")
def test_sandbox_resolves_python():
    r = run_sandboxed("python --version", project_root=_REPO, cwd=_REPO)
    assert r.get("exit_code") == 0, r
    assert "Python" in (r.get("output") or "")


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap not installed")
def test_sandbox_resolves_python3():
    r = run_sandboxed("python3 --version", project_root=_REPO, cwd=_REPO)
    assert r.get("exit_code") == 0, r


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap not installed")
def test_sandbox_pytest_importable():
    # pytest must be importable inside the sandbox (the venv has it; bare system
    # python does not) — otherwise no verify command can ever run the tests.
    r = run_sandboxed("python -m pytest --version", project_root=_REPO, cwd=_REPO)
    assert r.get("exit_code") == 0, r


@pytest.mark.skipif(not _HAS_BWRAP, reason="bwrap not installed")
def test_sandbox_still_offline():
    # isolation must NOT be weakened by the PATH/bind change — no network.
    r = run_sandboxed("python -c \"import socket; socket.create_connection(('1.1.1.1',53),2)\"",
                      project_root=_REPO, cwd=_REPO)
    assert r.get("exit_code") != 0, ("network should be blocked in the sandbox", r)
