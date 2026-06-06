"""phase_implement no-steps fallback (ckpt-188).

When the merged plan has NO parseable `### STEP` blocks (the merger emitted an
empty `=== PLAN ===` → raw-prose salvage), IMPLEMENT must NOT dump every file on
one "implement all changes" blob (the coder then does only the easiest edit and
quits — a26 edited only uri.py, never urls.py where the tests lived → all fail).
It synthesizes ONE FOCUSED STEP PER FILE so the per-step loop forces an edit (or
explicit decline) for EACH file. These tests drive the real phase_implement with
_implement_one_step stubbed (no LLM), asserting the fallback's control flow.
"""
import asyncio
import os
import shutil
import tempfile

import pytest

import workflows.code as wc
from tools.sandbox import Sandbox


def _mkproj(files):
    root = tempfile.mkdtemp(prefix="fbtest_")
    for f in files:
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("x = 1\n")
    sb = Sandbox(root)
    sb.setup()
    for f in files:
        sb.load_file(f)
    return root, sb


def _run_fallback(files, step_impl):
    """Patch the heavy helpers so phase_implement reaches the no-steps fallback,
    record the per-step calls, and restore everything after."""
    saved = (wc._implement_one_step, wc._extract_new_files_from_plan,
             wc._extract_impl_steps, wc._extract_shared_interfaces)
    wc._implement_one_step = step_impl
    wc._extract_new_files_from_plan = lambda plan: []
    wc._extract_impl_steps = lambda plan: []          # force the no-steps fallback
    wc._extract_shared_interfaces = lambda plan: ""
    root, sb = _mkproj(files)
    try:
        asyncio.run(wc.phase_implement(
            task="fix it", plan="raw prose plan, no numbered steps; mentions urls.py uri.py",
            context="", sandbox=sb, project_root=root, files_to_modify=files,
            detailed_map="", purpose_map="", research_cache={}))
    finally:
        (wc._implement_one_step, wc._extract_new_files_from_plan,
         wc._extract_impl_steps, wc._extract_shared_interfaces) = saved
        shutil.rmtree(root, ignore_errors=True)


def _recorder():
    calls = []
    async def step(step_info, task, shared_interfaces, file_contents, sandbox,
                   project_root, plan, detailed_map="", purpose_map="",
                   research_cache=None, error_feedback=""):
        calls.append({"num": step_info["num"], "name": step_info["name"],
                      "files": list(step_info["files"])})
        return {}
    return calls, step


def test_multi_file_one_step_each_in_deterministic_order():
    calls, step = _recorder()
    _run_fallback(["a/urls.py", "b/uri.py", "c/get_url.py"], step)
    assert len(calls) == 3
    assert all(len(c["files"]) == 1 for c in calls)          # one file per step
    assert [c["num"] for c in calls] == [1, 2, 3]
    # deterministic: follows files_to_modify order, NOT set()-randomized
    assert [c["files"][0] for c in calls] == ["a/urls.py", "b/uri.py", "c/get_url.py"]


def test_single_file_uses_blob_fallback():
    calls, step = _recorder()
    _run_fallback(["only.py"], step)
    assert len(calls) == 1 and calls[0]["name"] == "implement all changes"


def test_zero_files_uses_blob_fallback_no_crash():
    calls, step = _recorder()
    _run_fallback([], step)
    assert len(calls) == 1 and calls[0]["name"] == "implement all changes"


def test_cap_at_ten_files():
    calls, step = _recorder()
    _run_fallback([f"f{i}.py" for i in range(15)], step)
    assert len(calls) == 10                                  # capped, the rest logged


def test_one_step_failure_does_not_abort_the_rest():
    calls = []
    async def flaky(step_info, **k):
        calls.append(step_info["files"][0])
        if step_info["num"] == 1:
            raise RuntimeError("boom")
        return {}
    _run_fallback(["x/a.py", "x/b.py", "x/c.py"], flaky)
    assert len(calls) == 3                                   # all attempted despite step-1 crash
