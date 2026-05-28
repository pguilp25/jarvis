#!/usr/bin/env python3
"""
run_app_task.py — isolated harness to measure JARVIS quality on real
app-building / improving tasks (greenfield + feature + refactor), as opposed
to swe_bench.py's bug-fix flow.

This is ADDITIVE scaffolding. It never modifies core JARVIS code; it only
imports `code_agent` + `new_state` the same way swe_bench.run_one_instance
does, and it works inside a throwaway temp project dir.

Usage
-----
  # dry-run (default): validate wiring + print what WOULD run. No LLM call.
  python behavioral_audit/app_tasks/run_app_task.py --task greenfield

  # list available tasks
  python behavioral_audit/app_tasks/run_app_task.py --list

  # live-run (calls the real LLM pipeline — costs API + time):
  python behavioral_audit/app_tasks/run_app_task.py --task greenfield --live

A live run:
  1. copies the task's fixture (if any) into a fresh temp project dir,
  2. invokes JARVIS via `_invoke_jarvis` (the single clearly-marked entry
     point — see below),
  3. applies the produced sandbox to the temp dir,
  4. captures the produced/modified files + JARVIS's final answer,
  5. runs the task's verify_cmd (e.g. pytest) and reports pass/fail,
  6. leaves everything under behavioral_audit/app_tasks/runs/<task>_<ts>/
     so a human or an LLM judge can score it against rubric.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
JARVIS_ROOT = HERE.parent.parent          # /home/pguilp25/jarvis
TASKS_DIR = HERE / "tasks"
FIXTURES_DIR = HERE / "fixtures"
RUNS_DIR = HERE / "runs"

sys.path.insert(0, str(JARVIS_ROOT))


# ── Task spec loading ──────────────────────────────────────────────────────────
def load_task(task_id: str) -> dict:
    spec_path = TASKS_DIR / f"{task_id}.json"
    if not spec_path.exists():
        avail = ", ".join(sorted(p.stem for p in TASKS_DIR.glob("*.json")))
        raise SystemExit(f"Unknown task '{task_id}'. Available: {avail}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["_spec_path"] = str(spec_path)
    return spec


def list_tasks() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.json"))


# ── Project dir setup (copy fixture into a temp working dir) ──────────────────
def setup_project_dir(spec: dict, dest: Path) -> Path:
    """Create the working project dir for a task, seeding any fixture."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fixture = spec.get("fixture")
    if fixture:
        src = FIXTURES_DIR / fixture
        if not src.exists():
            raise SystemExit(f"Fixture '{fixture}' not found at {src}")
        for item in src.iterdir():
            if item.is_dir():
                shutil.copytree(item, dest / item.name)
            else:
                shutil.copy2(item, dest / item.name)
    return dest


# ══════════════════════════════════════════════════════════════════════════════
#  JARVIS ENTRY POINT  —  the single, clearly-marked invocation.
#  Mirrors swe_bench.run_one_instance (swe_bench.py ~371-401). This is the ONE
#  function that touches the live LLM pipeline; everything else is plumbing.
# ══════════════════════════════════════════════════════════════════════════════
async def _invoke_jarvis(task_text: str, project_root: str, timeout: int) -> dict:
    """Run the JARVIS coding agent on a plain-English task in `project_root`.

    Returns {"final_answer", "applied", "sandbox", "error"}.
    The caller is responsible for reading the resulting files off disk.
    """
    # Load API keys the same way swe_bench does (settings.json is the UI's source
    # of truth) so a live run is self-sufficient for the NVIDIA/OpenRouter/etc.
    # keys. ZAI/MISTRAL/POLLINATIONS still come from the caller's env.
    try:
        _sf = Path.home() / ".jarvis" / "settings.json"
        if _sf.exists():
            _saved = json.loads(_sf.read_text(encoding="utf-8"))
            for _k in ("NVIDIA_API_KEY", "LIGHTNING_API_KEY", "DEEPINFRA_API_KEY",
                       "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY",
                       "OPENROUTER_API_KEY", "OPENROUTER_API_KEYS"):
                _v = _saved.get(_k, "")
                if _v:
                    os.environ[_k] = _v
    except Exception:
        pass

    # Import here (not at module top) so --dry / --list never load the heavy
    # LLM stack and never require API keys.
    from workflows.code import code_agent
    from core.state import new_state

    # Build state exactly like swe_bench does, minus the SWE-bench-specific
    # "fix a bug, don't touch tests" framing — here we WANT new files + tests.
    state = new_state(raw_input=task_text)
    state["processed_input"] = state["raw_input"]
    state["classification"] = {
        "complexity": 7,        # deep coding path: plan + parallel coders + review
        "domain": "code",
        "agent": "code",
        "intent": "build app",
    }
    state["forced_complexity"] = 7
    state["project_root"] = project_root

    error_msg = ""
    applied: list[str] = []
    try:
        state = await asyncio.wait_for(code_agent(state), timeout=timeout)
    except asyncio.TimeoutError:
        error_msg = f"timeout({timeout}s)"
    except Exception as e:  # noqa: BLE001
        error_msg = f"agent_exc: {type(e).__name__}: {e}"

    sandbox = state.get("pending_sandbox") if isinstance(state, dict) else None
    if sandbox is not None:
        try:
            applied = sandbox.apply()   # write sandbox edits onto project_root
        except Exception as e:  # noqa: BLE001
            error_msg = error_msg or f"apply_failed: {e}"

    return {
        "final_answer": state.get("final_answer", "") if isinstance(state, dict) else "",
        "applied": applied,
        "sandbox": sandbox,
        "error": error_msg,
    }


# ── Dry-run: validate wiring WITHOUT calling the LLM ──────────────────────────
def dry_run(spec: dict) -> int:
    """Validate everything needed for a live run, print the plan, exit 0."""
    print("=" * 72)
    print(f"DRY RUN — task '{spec['id']}' (mode={spec['mode']})")
    print("=" * 72)

    problems: list[str] = []

    # 1. fixture present (if required)
    fixture = spec.get("fixture")
    if fixture:
        fpath = FIXTURES_DIR / fixture
        if fpath.exists():
            files = sorted(p.name for p in fpath.rglob("*") if p.is_file())
            print(f"[ok] fixture '{fixture}' present: {files}")
        else:
            problems.append(f"fixture '{fixture}' missing at {fpath}")
    else:
        print("[ok] no fixture (greenfield — empty project dir)")

    # 2. verify command declared
    vcmd = spec.get("verify_cmd")
    print(f"[ok] verify_cmd: {vcmd!r}" if vcmd else "[warn] no verify_cmd declared")

    # 3. expected artifacts declared
    arts = spec.get("expected_artifacts", [])
    print(f"[ok] expected_artifacts: {arts}" if arts else "[warn] no expected_artifacts")

    # 4. JARVIS entry point importable (proves wiring without running it).
    #    We import lazily and only check the symbols exist + are callable.
    try:
        from workflows.code import code_agent  # noqa: F401
        from core.state import new_state       # noqa: F401
        s = new_state(raw_input="probe")
        s["processed_input"] = s["raw_input"]
        s["classification"] = {"complexity": 7, "domain": "code",
                               "agent": "code", "intent": "build app"}
        s["forced_complexity"] = 7
        s["project_root"] = "/tmp/__jarvis_app_task_probe__"
        assert asyncio.iscoroutinefunction(code_agent)
        print("[ok] entry point importable: workflows.code.code_agent (coroutine)")
        print("[ok] AgentState built: keys ="
              f" {sorted(k for k in s if s[k] not in (None, '', [], {}))}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"entry point import/build failed: {type(e).__name__}: {e}")

    # 5. show the exact task text JARVIS would receive
    print("-" * 72)
    print("TASK TEXT THAT WOULD BE SENT TO JARVIS:")
    print(spec["task"])
    print("-" * 72)
    print("WOULD: copy fixture -> temp dir; await code_agent(state); "
          "sandbox.apply(); run verify_cmd; save run dir.")
    print(f"To actually run it (costs API + time): "
          f"python {Path(__file__).name} --task {spec['id']} --live")

    if problems:
        print("\nWIRING PROBLEMS (would block a live run):")
        for p in problems:
            print(f"  [BLOCK] {p}")
        return 1
    print("\n[DRY RUN OK] wiring validated, no LLM was called.")
    return 0


# ── Live-run: actually invoke JARVIS ──────────────────────────────────────────
def live_run(spec: dict, timeout: int) -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{spec['id']}_{ts}"
    proj_dir = run_dir / "project"
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_project_dir(spec, proj_dir)
    print(f"[live] project dir: {proj_dir}")

    # Snapshot files BEFORE so we can report what JARVIS created/changed.
    before = {p.name for p in proj_dir.rglob("*") if p.is_file()}

    print(f"[live] invoking JARVIS (timeout={timeout}s) — this calls the LLM...")
    result = asyncio.run(_invoke_jarvis(spec["task"], str(proj_dir), timeout))

    after_files = sorted(
        str(p.relative_to(proj_dir))
        for p in proj_dir.rglob("*")
        if p.is_file() and ".jarvis_sandbox" not in p.parts
    )
    created = [f for f in after_files if Path(f).name not in before]

    # Run the verify command against the applied project.
    verify = {"cmd": spec.get("verify_cmd"), "exit_code": None, "output": ""}
    if spec.get("verify_cmd"):
        try:
            proc = subprocess.run(
                spec["verify_cmd"], shell=True, cwd=str(proj_dir),
                capture_output=True, text=True, timeout=300,
            )
            verify["exit_code"] = proc.returncode
            verify["output"] = (proc.stdout + "\n" + proc.stderr)[-4000:]
        except Exception as e:  # noqa: BLE001
            verify["exit_code"] = -1
            verify["output"] = f"verify run failed: {e}"

    report = {
        "task_id": spec["id"],
        "mode": spec["mode"],
        "timestamp": ts,
        "project_dir": str(proj_dir),
        "error": result["error"],
        "applied": result["applied"],
        "files_after": after_files,
        "files_created": created,
        "expected_artifacts": spec.get("expected_artifacts", []),
        "missing_artifacts": [
            a for a in spec.get("expected_artifacts", [])
            if not (proj_dir / a).exists()
        ],
        "verify": verify,
        "final_answer": result["final_answer"],
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print(f"LIVE RUN COMPLETE — {spec['id']}")
    print(f"  error            : {report['error'] or '(none)'}")
    print(f"  files created    : {report['files_created']}")
    print(f"  missing artifacts: {report['missing_artifacts'] or '(none)'}")
    print(f"  verify exit code : {verify['exit_code']}")
    print(f"  report           : {run_dir / 'report.json'}")
    print(f"  rubric           : {HERE / 'rubric.md'}")
    print("=" * 72)
    # Live exit code reflects verify pass/fail (0 = tests passed).
    return 0 if verify["exit_code"] in (0, None) and not report["error"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS app-task quality harness")
    ap.add_argument("--task", help="task id (see --list)")
    ap.add_argument("--live", action="store_true",
                    help="actually call JARVIS (costs API + time); default is dry-run")
    ap.add_argument("--list", action="store_true", help="list available tasks")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-task timeout for the live LLM run (seconds)")
    args = ap.parse_args()

    if args.list:
        for t in list_tasks():
            spec = load_task(t)
            print(f"  {t:12s}  mode={spec['mode']:10s}  fixture={spec.get('fixture')}")
        return 0

    if not args.task:
        ap.error("--task is required (or use --list)")

    spec = load_task(args.task)
    if args.live:
        return live_run(spec, args.timeout)
    return dry_run(spec)


if __name__ == "__main__":
    raise SystemExit(main())
