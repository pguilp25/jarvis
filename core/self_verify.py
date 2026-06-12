"""General self-verify execution backend for the REVIEW phase (ckpt-266).

The reviewer authors a minimal REPRODUCTION from the issue's own example values
(see SELFVERIFY_REPRO_PROMPT in prompts_v8). This module RUNS that repro against
the coder's EDITED tree and reports pass / fail / env-blocked — auto-detecting
WHERE it can run so the same loop works for a SWE-bench instance, a normal user's
repo, or a pure-logic snippet, with NOTHING hardcoded to one environment.

Backend escalation (cheapest first, escalate only when a dep is genuinely absent):
  1. LOCAL   — the existing bwrap stdlib+repo sandbox (core.safe_exec.run_sandboxed).
               Handles pure-logic code and any project whose deps are stdlib or
               live in the repo. ~5s.
  2. HOST    — same bwrap sandbox but with the HOST's installed site-packages bound
               read-only. This is the "works for any code" path: a normal user's
               project deps are installed in the running interpreter, so the repro
               imports them. (For a SWE-bench box that never pip-installed the
               instance's deps, this still can't import them → stays env-blocked,
               which is correct — we do NOT rubber-stamp.)
  3. CONTAINER — optional hook (container_runner) for running inside the instance's
               real image when one is registered. Off unless wired by the caller.
  4. none    — repro can't run anywhere → ENV_BLOCKED. The caller must fall back to
               STATIC review and NEVER auto-approve on this (the old review's fatal
               bug: "environment failure → APPROVED").

Why this does not reopen the ckpt-166/169 patch-pollution regression: that was the
CODER's edit loop manufacturing stub modules with site-packages mounted. Here it is
the REVIEWER running a THROWAWAY repro via inline `exec` (never written into the repo
tree, so it can't enter the diff), and site-packages is mounted READ-ONLY (nothing
can be shimmed into it). Different surface, both holes closed.
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile

from core.safe_exec import run_sandboxed

# pass=fixed, fail=real failure (assertion / repo-module missing / traceback),
# env_blocked=a 3rd-party dep is absent so we could not truly run it,
# error=inconclusive (timeout / sandbox could not start).
PASS = "pass"
FAIL = "fail"
ENV_BLOCKED = "env_blocked"
ERROR = "error"

_REPRO_TIMEOUT = 30   # seconds per backend attempt — fast; the loop caps cycles


class RunResult:
    __slots__ = ("status", "exit_code", "output", "backend", "missing_module")

    def __init__(self, status, exit_code=-1, output="", backend="none",
                 missing_module=""):
        self.status = status
        self.exit_code = exit_code
        self.output = output
        self.backend = backend
        self.missing_module = missing_module

    @property
    def ran(self) -> bool:
        """True iff the repro actually executed the target code (pass or fail) —
        i.e. the verdict is trustworthy, not an environment artifact."""
        return self.status in (PASS, FAIL)

    def __repr__(self):
        return (f"RunResult(status={self.status!r}, exit={self.exit_code}, "
                f"backend={self.backend!r}, missing={self.missing_module!r})")


# ── output classification ────────────────────────────────────────────────────

# "ModuleNotFoundError: No module named 'web'"  /  "...named 'foo.bar'"
_NO_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _missing_module(output: str) -> str:
    """The top-level package name of a ModuleNotFoundError, or '' if none.

    Only `No module named 'X'` counts — that is an ABSENT import target. An
    `ImportError: cannot import name N from pkg` is NOT returned here: pkg DID
    import, so it's a real failure (the coder didn't add symbol N), not an env gap."""
    m = _NO_MODULE_RE.search(output or "")
    if not m:
        return ""
    return m.group(1).split(".")[0]


def _is_repo_module(modname: str, sandbox_dir: str) -> bool:
    """Does `modname` resolve to something IN the edited repo? If a repro fails on
    `No module named 'X'` and X is a repo package/module, that is a REAL failure
    (the coder was meant to create it) — NOT an environment gap. Only an EXTERNAL
    missing module is env-blocked."""
    if not modname or not sandbox_dir:
        return False
    roots = [sandbox_dir]
    for sub in ("src", "lib"):
        d = os.path.join(sandbox_dir, sub)
        if os.path.isdir(d):
            roots.append(d)
    for root in roots:
        if os.path.isdir(os.path.join(root, modname)):
            return True
        if os.path.isfile(os.path.join(root, modname + ".py")):
            return True
    return False


def _classify(res: dict, sandbox_dir: str, backend: str) -> RunResult:
    """Turn a run_sandboxed() dict into a RunResult verdict."""
    if res.get("blocked"):
        # sandbox/policy could not even start it — inconclusive, never a verdict.
        return RunResult(ERROR, output=res.get("reason", "sandbox blocked"),
                         backend=backend)
    if res.get("timed_out"):
        return RunResult(ERROR, exit_code=-1, output=res.get("output", ""),
                         backend=backend)
    out = res.get("output", "") or ""
    code = res.get("exit_code", -1)
    if code == 0:
        # PASS requires the mandated success sentinel (prompt: end success with
        # print("REPRO_OK")). exit 0 WITHOUT it = a vacuous / short-circuited repro
        # → INCONCLUSIVE, not verified. This is the guard that stops a spurious
        # green from shipping a worse patch on the post-fix route (adversarial
        # review ckpt-266). An over-conservative miss only ever falls back to
        # deliver-as-is / revert — never a false PASS.
        if "REPRO_OK" in out:
            return RunResult(PASS, exit_code=0, output=out, backend=backend)
        return RunResult(ERROR, exit_code=0, output=out, backend=backend)
    missing = _missing_module(out)
    if missing and not _is_repo_module(missing, sandbox_dir):
        # an external 3rd-party dep is absent → we could not truly run it here.
        return RunResult(ENV_BLOCKED, exit_code=code, output=out,
                         backend=backend, missing_module=missing)
    # real failure: assertion, traceback, syntax, or a MISSING REPO module
    # (the coder failed to create the new symbol/file the addition needed).
    return RunResult(FAIL, exit_code=code, output=out, backend=backend,
                     missing_module=missing)


# ── the inline-exec command (no temp file, no escaping, never touches the repo) ─

def _b64exec_cmd(repro_code: str) -> str:
    b = base64.b64encode(repro_code.encode("utf-8")).decode("ascii")
    # -B: don't write .pyc (cwd is read-only). Decode+exec the repro inline so it
    # is never written into the repo tree (cannot pollute the diff).
    return (f'python -B -c "import base64;'
            f"exec(compile(base64.b64decode('{b}').decode('utf-8'),'<repro>','exec'))\"")


def _host_site_dirs() -> list[str]:
    """Existing site-packages dirs of the RUNNING interpreter — the deps a normal
    user's project is installed against. Empty list if none resolve."""
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        u = site.getusersitepackages()
        if isinstance(u, str):
            dirs.append(u)
    except Exception:
        pass
    # the venv's own site-packages (sys.executable's ../lib/pythonX/site-packages)
    for p in sys.path:
        if p and "site-packages" in p:
            dirs.append(p)
    out, seen = [], set()
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            out.append(d)
            seen.add(d)
    return out


def run_repro(repro_code: str, sandbox_dir: str,
              project_root: "str | None" = None, *,
              allow_host: bool = True,
              container_runner=None,
              timeout: int = _REPRO_TIMEOUT) -> RunResult:
    """Run `repro_code` against the edited tree at `sandbox_dir`, auto-detecting a
    backend that can actually execute it. Returns a RunResult; .ran is True only
    when the verdict is trustworthy (the code actually executed).

    container_runner, if given, is `callable(repro_code, timeout) -> RunResult`
    used as the last escalation step (e.g. run inside the instance's image)."""
    if not repro_code or not repro_code.strip():
        return RunResult(ERROR, output="empty repro")
    sandbox_dir = sandbox_dir or "/tmp"
    cmd = _b64exec_cmd(repro_code)

    # 1) LOCAL — stdlib + repo only.
    res = run_sandboxed(cmd, cwd=sandbox_dir, timeout=timeout,
                        project_root=project_root or sandbox_dir)
    local = _classify(res, sandbox_dir, "local")
    if local.ran:
        return local

    # 2) HOST — bind the host's installed site-packages read-only and retry, but
    #    only if there ARE host site-packages AND the local block was a missing
    #    EXTERNAL dep (no point re-running a timeout/sandbox error).
    if allow_host and local.status == ENV_BLOCKED:
        host_dirs = _host_site_dirs()
        if host_dirs:
            res = run_sandboxed(cmd, cwd=sandbox_dir, timeout=timeout,
                                project_root=project_root or sandbox_dir,
                                extra_ro_binds=host_dirs)
            host = _classify(res, sandbox_dir, "host")
            if host.ran:
                return host
            # keep the more-informative env_blocked (host still missing the dep)
            local = host if host.status == ENV_BLOCKED else local

    # 3) CONTAINER — optional real-env hook (off unless the caller wires it).
    if container_runner is not None and local.status in (ENV_BLOCKED, ERROR):
        try:
            cont = container_runner(repro_code, timeout)
            if isinstance(cont, RunResult) and cont.ran:
                return cont
            if isinstance(cont, RunResult):
                local = cont
        except Exception as e:  # a flaky backend must never crash the run
            local = RunResult(ERROR, output=f"container backend error: {e}",
                              backend="container")

    # 4) nothing could truly run it → return the best env_blocked/error verdict.
    return local


# ── repro authoring (one-shot completion) ────────────────────────────────────

# Non-anchored: grab the FIRST fenced block even when the model wraps it in prose
# (gpt-oss often adds an explanatory line despite the prompt). Anchoring to the
# whole string let prose leak into the "repro" → SyntaxError → spurious FAIL
# (adversarial review ckpt-266).
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.S)


def _strip_fence(text: str) -> str:
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # tolerate an opening ```python with no closing fence
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[: -3]
        return t.strip()
    # no fence at all → the prompt asks for raw script, so return as-is
    return t


async def author_repro(task: str, diff: str, changed_files_text: str,
                       model: str = "nvidia/gpt-oss-120b",
                       *, max_chars: int = 9000) -> "str | None":
    """Ask a model to write a runnable repro from the ISSUE. Returns the script,
    or None when the model says NO_REPRO / the call fails (caller treats None as
    'cannot verify' → deliver as-is, never a false bug)."""
    from clients.nvidia import call_nvidia
    from core.prompts_v8 import SELFVERIFY_REPRO_PROMPT

    _t = (task or "").strip()
    _d = (diff or "").strip()
    _cf = (changed_files_text or "").strip()
    if len(_t) > max_chars:
        _t = _t[: max_chars // 2] + "\n…\n" + _t[-max_chars // 2:]
    if len(_d) > max_chars:
        _d = _d[: max_chars] + "\n…[diff truncated]…"
    if len(_cf) > max_chars:
        _cf = _cf[: max_chars] + "\n…[files truncated]…"
    user = (f"=== ISSUE ===\n{_t}\n\n"
            f"=== DIFF JUST APPLIED ===\n{_d or '(no diff)'}\n\n"
            f"=== CHANGED FILES ===\n{_cf or '(none)'}\n")
    try:
        raw = await call_nvidia(model, prompt=user, system=SELFVERIFY_REPRO_PROMPT,
                                temperature=0.2, max_tokens=4096)
    except Exception:
        return None
    if not raw:
        return None
    code = _strip_fence(raw)
    if not code:
        return None
    # NO_REPRO sentinel anywhere in the first non-empty line → cannot verify.
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if first.strip().upper().startswith("NO_REPRO"):
        return None
    return code


def make_container_runner(image: str, changed_files: dict, sandbox_dir: str = "",
                          *, pull_if_missing: bool = True):
    """Phase-2 backend (ckpt-267): run the repro INSIDE the instance's real image
    (jefzda/sweap-images:<dockerhub_tag>), which has the project's true 3rd-party
    deps installed — so a repro for a dep-heavy instance (web.py, ansible, …) can
    actually execute, the case the local/host backends can't cover on a box that
    never pip-installed the instance.

    Returns callable(repro_code, timeout) -> RunResult, matching run_repro's
    container_runner hook. It overlays the coder's edited files onto /app (the
    image's repo root, per the eval harness) and runs the repro with the repo on
    PYTHONPATH. --network none (no leakage / no escape). FAIL-SAFE: any docker/
    image/timeout problem returns ERROR or ENV_BLOCKED, so the caller delivers the
    coder's patch as-is — the container backend can only help, never regress."""
    def _run(repro_code: str, timeout: int) -> "RunResult":
        if not image or not shutil.which("docker"):
            return RunResult(ERROR, output="docker/image unavailable", backend="container")
        try:
            insp = subprocess.run(["docker", "image", "inspect", image],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=30)
            if insp.returncode != 0:
                if not pull_if_missing:
                    return RunResult(ENV_BLOCKED, output=f"image not local: {image}",
                                     backend="container")
                pull = subprocess.run(["docker", "pull", image],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      timeout=900)
                if pull.returncode != 0:
                    return RunResult(ENV_BLOCKED, output=f"image unpullable: {image}",
                                     backend="container")
        except Exception as e:
            return RunResult(ERROR, output=f"docker inspect/pull error: {e}",
                             backend="container")

        ws = tempfile.mkdtemp(prefix="sv_ctr_")
        try:
            # tar the edited files at their repo-relative paths → overlaid onto /app
            with tarfile.open(os.path.join(ws, "edits.tar"), "w") as tf:
                for rel, content in (changed_files or {}).items():
                    rel = str(rel).lstrip("/")
                    if not rel or ".." in rel.split("/"):
                        continue   # never write outside the repo root
                    data = (content or "").encode("utf-8")
                    ti = tarfile.TarInfo(name=rel)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
            with open(os.path.join(ws, "repro.py"), "w", encoding="utf-8") as f:
                f.write(repro_code)
            inner = ("cd /app && tar xf /ws/edits.tar -C /app && "
                     "QT_QPA_PLATFORM=offscreen PYTHONPATH=/app:/app/src:/app/lib "
                     f"timeout {int(timeout)} python -B /ws/repro.py")
            # --entrypoint bash: MANY sweap images set ENTRYPOINT=[/bin/bash], so a plain
            # `docker run IMG bash -c …` becomes `/bin/bash bash -c …` → bash tries to exec
            # the binary `bash` as a script → "cannot execute binary file" (ckpt-268 root
            # cause of the container false-fails). Overriding the entrypoint runs `bash -c
            # inner` cleanly regardless of the image's default entrypoint.
            res = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash",
                 "-v", f"{ws}:/ws:ro", image, "-c", inner],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                errors="replace", timeout=int(timeout) + 90)
            out = res.stdout or ""
            # Container INFRASTRUCTURE failures (bad entrypoint, exec-format, OCI/runtime,
            # missing loader) are NOT repro failures — misclassifying them as FAIL triggers
            # spurious fix cycles. Treat them as env_blocked → deliver as-is (ckpt-268).
            _INFRA = ("cannot execute binary file", "exec format error", "OCI runtime",
                      "standard_init_linux", "executable file not found",
                      "docker: Error", "Unable to find image")
            if res.returncode != 0 and any(s in out for s in _INFRA):
                return RunResult(ENV_BLOCKED, exit_code=res.returncode, output=out,
                                 backend="container")
            d = {"blocked": False, "timed_out": False,
                 "exit_code": res.returncode, "output": out}
            # _is_repo_module checks the host edited tree (mirrors /app's layout) so a
            # missing REPO module the coder failed to create still classifies as FAIL.
            return _classify(d, sandbox_dir, "container")
        except subprocess.TimeoutExpired:
            return RunResult(ERROR, output="container run timed out", backend="container")
        except Exception as e:
            return RunResult(ERROR, output=f"container run error: {e}", backend="container")
        finally:
            shutil.rmtree(ws, ignore_errors=True)
    return _run


def fail_feedback(repro: str, result: "RunResult", step_num: "int | None") -> str:
    """Route message handed back to the coder when the repro genuinely FAILS:
    the verbatim repro + its traceback + a crisp instruction. Deterministic GPS,
    not a vague 'something's wrong'."""
    tail = (result.output or "")[-2500:]
    return (
        "SELF-VERIFY FAILED: a reproduction built from the issue's own example "
        "still does not pass against your change. Make THIS repro pass — do not "
        "edit the repro, fix the code it exercises.\n\n"
        "--- repro ---\n" + (repro or "")[:3000] + "\n--- end repro ---\n\n"
        "--- what happened when it ran ---\n" + tail + "\n--- end ---\n"
    )
