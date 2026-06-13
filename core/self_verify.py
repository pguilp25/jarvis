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

REPRO_SENTINEL = "REPRO_OK"   # success marker (must be the LAST stdout line) — language-agnostic

# ── language protocols (ckpt-275) ─────────────────────────────────────────────
# SWE-bench Pro and real-world use are MULTI-LANGUAGE (Go, TypeScript/JavaScript,
# Python, …). The self-verify loop is language-agnostic in SHAPE — only the repro
# LANGUAGE, the RUNNER, the success-sentinel print, and the error-string heuristics
# differ. We detect the language from the EDITED file's extension and follow its
# protocol. Python is the fully-validated default; the others are best-effort and
# degrade gracefully (an unknown language → run by exit-code + sentinel only).
_LANG_PROTOCOL = {
    "python": {
        "ext": "py", "runner": "python -B",
        # repro ERRORED because IT mis-called the code (wrong signature) — NOT a real
        # behavioural failure (a26: invented fetch_url(client_cert=...)):
        "malformed": ("unexpected keyword argument", "required positional argument",
                      "takes no arguments", "positional arguments but",
                      "got multiple values for"),
        "missing": ("No module named",),
    },
    "javascript": {
        "ext": "mjs", "runner": "node",
        "malformed": ("is not a function", "is not defined",
                      "cannot read properties of undefined"),
        "missing": ("Cannot find module", "ERR_MODULE_NOT_FOUND"),
    },
    "typescript": {
        "ext": "ts", "runner": "npx --yes tsx",
        "malformed": ("is not a function", "is not defined", "has no exported member"),
        "missing": ("Cannot find module", "Cannot find name"),
    },
    "go": {
        "ext": "go", "runner": "go run",
        "malformed": ("not enough arguments", "too many arguments",
                      "undefined:", "unknown field"),
        "missing": ("cannot find package", "no required module"),
    },
    "generic": {"ext": "txt", "runner": "", "malformed": (), "missing": ()},
}
# the success-sentinel print, per language (handed to the repro author)
_LANG_SENTINEL = {
    "python": 'print("REPRO_OK")', "javascript": 'console.log("REPRO_OK")',
    "typescript": 'console.log("REPRO_OK")', "go": 'fmt.Println("REPRO_OK")',
    "generic": "emit the line REPRO_OK",
}
_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go",
}


def detect_language(files) -> str:
    """Detect the repro language from the EDITED files' extensions (dominant wins).
    Defaults to python (the validated path + the most common SWE-bench-Pro language)."""
    import collections
    c = collections.Counter()
    for f in (files or []):
        lang = _EXT_TO_LANG.get(os.path.splitext(str(f))[1].lower())
        if lang:
            c[lang] += 1
    return c.most_common(1)[0][0] if c else "python"


def _proto(language: str) -> dict:
    return _LANG_PROTOCOL.get(language or "python", _LANG_PROTOCOL["python"])


# ── signature digest (ckpt-277) ───────────────────────────────────────────────
# The repro author must NOT be shown the patch (diff / post-patch bodies): seeing
# the implementation's control-flow primes a weak model to arrange invented internal
# state to satisfy it — dbbd9d53 set `web.ctx.method='POST'` ONLY because the patch it
# was shown gated on `web.ctx.method=='POST'`. Instead we hand it a SIGNATURE DIGEST:
# the callable API (def/class headers, no bodies) so it can ground its calls on the
# REAL argument shapes (the a26 lesson — no invented kwargs) while staying blind to the
# implementation. It then tests the ISSUE's contract, not the patch's invented one.
import ast as _ast
import re as _re

_NONPY_SIG_RE = {
    "go": _re.compile(r"^\s*func\s"),
    "javascript": _re.compile(
        r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w"
        r"|^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?\("
        r"|^\s*(export\s+)?(default\s+)?class\s+\w"),
    "typescript": _re.compile(
        r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w"
        r"|^\s*(export\s+)?(abstract\s+)?(class|interface|type|enum)\s+\w"
        r"|^\s*(public|private|protected|readonly|static)\s+\w"),
}


def _py_signatures(content: str) -> "list[str]":
    """def/class HEADERS + class FIELD NAMES + module CONSTANT names of a Python source —
    names, full arg lists (via ast.unparse), return annotation, decorators. NO bodies and
    NO docstrings: docstrings are coder-authored impl narration that can re-leak the very
    control-flow we hide (ckpt-277 review — a coder docstring 'parse the body when method
    is POST' re-primes the rig). Field/constant NAMES are structural (no values) so the
    author can ground attribute references (record.seeds) instead of guessing them."""
    try:
        tree = _ast.parse(content)
    except Exception:
        return []
    out: "list[str]" = []

    def _fn(node, indent=""):
        for d in getattr(node, "decorator_list", []):
            try:
                out.append(f"{indent}@{_ast.unparse(d)}")
            except Exception:
                pass
        try:
            args = _ast.unparse(node.args)
        except Exception:
            args = "..."
        ret = ""
        if getattr(node, "returns", None) is not None:
            try:
                ret = " -> " + _ast.unparse(node.returns)
            except Exception:
                ret = ""
        kw = "async def " if isinstance(node, _ast.AsyncFunctionDef) else "def "
        out.append(f"{indent}{kw}{node.name}({args}){ret}")

    def _names(body, upper_only=False):
        # AnnAssign / simple Assign TARGET names only — never the values (a value could
        # carry control-flow / literals that re-prime). upper_only → module constants.
        names = []
        for n in body:
            tgts = []
            if isinstance(n, _ast.AnnAssign) and isinstance(n.target, _ast.Name):
                tgts = [n.target.id]
            elif isinstance(n, _ast.Assign):
                tgts = [t.id for t in n.targets if isinstance(t, _ast.Name)]
            for nm in tgts:
                if (not upper_only) or (nm.isupper() and len(nm) > 1):
                    names.append(nm)
        return names

    def walk(body, indent, kind):
        if kind == "class":
            f = _names(body)
            if f:
                out.append(f"{indent}# fields: " + ", ".join(f[:40]))
        elif kind == "module":
            c = _names(body, upper_only=True)
            if c:
                out.append("# module constants: " + ", ".join(c[:40]))
        for node in body:
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                _fn(node, indent)
            elif isinstance(node, _ast.ClassDef):
                try:
                    bases = ", ".join(_ast.unparse(b) for b in node.bases)
                except Exception:
                    bases = ""
                out.append(f"{indent}class {node.name}({bases})" if bases
                           else f"{indent}class {node.name}")
                walk(node.body, indent + "    ", "class")
            elif isinstance(node, (_ast.If, _ast.Try, _ast.With, _ast.AsyncWith)):
                # defs hidden under `if TYPE_CHECKING:` / version gates / try-import
                sub = []
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    for item in getattr(node, attr, []) or []:
                        sub += getattr(item, "body", []) if isinstance(item, _ast.ExceptHandler) else [item]
                walk(sub, indent, "nested")
    walk(tree.body, "", "module")
    return out


def _regex_signatures(content: str, language: str) -> "list[str]":
    """Best-effort declaration headers for non-Python (Go/TS/JS). Line-based — the
    multi-language path is best-effort; the author also has the issue to ground on."""
    pat = _NONPY_SIG_RE.get(language)
    if pat is None:
        return []
    out: "list[str]" = []
    for ln in content.splitlines():
        if pat.match(ln):
            s = ln.strip().rstrip("{").strip()
            if s and s not in out:
                out.append(s[:160])
    return out


def extract_signatures(changed_files: dict, language: str = "python",
                       *, max_files: int = 8, max_per_file: int = 60) -> str:
    """SIGNATURE DIGEST of the changed files — the callable API (headers, no bodies)
    the repro author may use to ground its calls, WITHOUT seeing the implementation.
    See the module comment above for why the author is kept blind to the patch."""
    blocks = []
    for path, content in list((changed_files or {}).items())[:max_files]:
        if not content:
            continue
        # dispatch per-FILE by extension (a .py in a go-dominant changeset must still
        # use the AST path), falling back to the run's detected language (ckpt-277 review).
        flang = _EXT_TO_LANG.get(os.path.splitext(str(path))[1].lower(), language)
        if flang == "python":
            sigs = _py_signatures(content)
        else:
            sigs = _regex_signatures(content, flang)
        if sigs:
            body = "\n".join(sigs[:max_per_file])
            if len(sigs) > max_per_file:
                body += f"\n# …(+{len(sigs) - max_per_file} more)"
            blocks.append(f"# {path}\n{body}")
    return "\n\n".join(blocks)


_RE_ATTR = _re.compile(r"'(\w+)' object has no attribute '([A-Za-z_]\w*)'")  # AttributeError
_RE_NAME = _re.compile(r"name '([A-Za-z_]\w*)' is not defined")              # NameError
_RE_IMPORT = _re.compile(r"cannot import name '([A-Za-z_]\w*)'")             # ImportError
# builtin object types: `'NoneType' object has no attribute 'x'` is a WRONG-VALUE bug
# (the patch returned the wrong type), NOT an invented symbol → must still route a fix.
_BUILTIN_TYPES = {"NoneType", "str", "bytes", "bytearray", "int", "float", "bool",
                  "complex", "list", "tuple", "dict", "set", "frozenset", "range"}


def _is_repro_frame(path: str) -> bool:
    b = os.path.basename((path or "").strip())
    return (b.startswith("repro.") or b.startswith("sv_repro.")
            or (path or "").strip() in ("<repro>", "<string>"))


def hallucinated_symbol(output: str, known_text: str) -> "str | None":
    """If a repro FAILED because IT referenced a symbol/attribute the author INVENTED —
    the failure is raised IN THE REPRO (last traceback frame is the repro file), matches
    a hallucination pattern, and that name appears NOWHERE in the issue text or the
    signature digest — return the name. The caller then treats the run as INCONCLUSIVE
    and delivers the coder's patch as-is, rather than routing a fix that could mutate a
    possibly-correct patch to satisfy a hallucinated contract (ckpt-277 review, finding 3
    — blind authoring raises the shape-guessing rate; this caps the 'converged-on-
    hallucination ships worse' hole). A genuine failure raised in PROJECT code (last frame
    is a repo file) or a wrong-VALUE error (`'NoneType' object has no attribute …`) is
    NOT suppressed — it routes a fix as before."""
    out = output or ""
    known = known_text or ""
    frames = _re.findall(r'File "([^"]+)"', out)
    if frames and not _is_repro_frame(frames[-1]):
        return None   # failure originates in project code → a genuine bug, route it

    def _grounded(nm: str) -> bool:
        return bool(_re.search(rf"\b{_re.escape(nm)}\b", known))

    for m in _RE_ATTR.finditer(out):
        objtype, attr = m.group(1), m.group(2)
        if objtype in _BUILTIN_TYPES or attr.startswith("__"):
            continue                      # wrong-value bug / dunder → not an invented symbol
        if not _grounded(attr):
            return attr
    for pat in (_RE_NAME, _RE_IMPORT):
        for m in pat.finditer(out):
            nm = m.group(1)
            if not nm.startswith("__") and not _grounded(nm):
                return nm
    return None


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


def _classify(res: dict, sandbox_dir: str, backend: str, proto: dict = None) -> RunResult:
    """Turn a run_sandboxed() dict into a RunResult verdict, using the language
    protocol's error-string heuristics (ckpt-275). Defaults to the python protocol."""
    if proto is None:
        proto = _LANG_PROTOCOL["python"]
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
        # PASS requires the mandated success sentinel REPRO_OK as the LAST non-empty
        # line of stdout (ckpt-272: a substring check let a repro print REPRO_OK early
        # and then swallow its real assertion in a try/except → exit 0 + sentinel-present
        # → FALSE PASS shipping a broken patch). Requiring it to be the final line forces
        # it to come AFTER all assertions. An over-conservative miss (sentinel not last,
        # e.g. a trailing warning) only falls back to deliver-as-is / revert — never a
        # false PASS.
        _last = next((ln for ln in reversed(out.splitlines()) if ln.strip()), "")
        if _last.strip() == REPRO_SENTINEL:
            return RunResult(PASS, exit_code=0, output=out, backend=backend)
        return RunResult(ERROR, exit_code=0, output=out, backend=backend)
    # ckpt-275: the language RUNTIME itself is absent (go/node/tsx not installed in this
    # backend, e.g. the stdlib-only local sandbox) — NOT a patch failure. Treat as
    # env_blocked so run_repro ESCALATES to the instance container (which has it).
    if code == 127 or "command not found" in out or "executable file not found" in out:
        return RunResult(ENV_BLOCKED, exit_code=code, output=out, backend=backend,
                         missing_module="<runtime>")
    # missing DEPENDENCY (per-language patterns). For python we additionally distinguish
    # an absent EXTERNAL dep (env_blocked) from a missing REPO module (real FAIL — the
    # coder didn't create the symbol). Other languages: any missing-dep pattern → env_blocked.
    missing = _missing_module(out)
    if missing and not _is_repo_module(missing, sandbox_dir):
        return RunResult(ENV_BLOCKED, exit_code=code, output=out,
                         backend=backend, missing_module=missing)
    if proto.get("missing") and any(s in out for s in proto["missing"]) and not missing:
        return RunResult(ENV_BLOCKED, exit_code=code, output=out, backend=backend,
                         missing_module="<dep>")
    # ckpt-274: a MALFORMED repro — one that errors because IT called the code wrong
    # (hallucinated a parameter / wrong arity), not because a behavioural assertion
    # failed — is NOT a patch failure. Routing a fix burns cycles the coder can't use
    # (it can't fix a broken repro), as on a26 (the repro invented `fetch_url(...,
    # client_cert=...)`, a kwarg that doesn't exist and is unrelated to the task →
    # TypeError → 2 wasted fix cycles + 67min → revert). Treat it as INCONCLUSIVE
    # (→ deliver as-is, never a fix) UNLESS a genuine assertion failure is present (that
    # IS a real behavioural failure → route a fix). Strictly safe: inconclusive only
    # ever delivers the coder's patch as-is, never worse. Per-language signatures.
    _assert = ("AssertionError" in out or "assert.fail" in out  # py / generic
               or "--- FAIL:" in out)                            # go test-style
    if not _assert and proto.get("malformed") and any(s in out for s in proto["malformed"]):
        return RunResult(ERROR, exit_code=code, output=out, backend=backend)
    # real failure: assertion, traceback, syntax, or a MISSING REPO module
    # (the coder failed to create the new symbol/file the addition needed).
    return RunResult(FAIL, exit_code=code, output=out, backend=backend,
                     missing_module=missing)


# ── the run command: write the repro to a tmpfs scratch file and run it with the
#    language's runner (ckpt-275, replaces the python-only inline-exec). The scratch
#    file lives in /tmp (bwrap tmpfs / container ephemeral) — NEVER the repo tree, so
#    it can't pollute the diff. base64 keeps it shell-safe regardless of repro content.

def _run_cmd(repro_code: str, proto: dict) -> str:
    b = base64.b64encode(repro_code.encode("utf-8")).decode("ascii")
    ext = proto.get("ext", "txt")
    runner = proto.get("runner", "")
    if not runner:   # generic / unknown language — no way to run it here
        return "echo 'self-verify: no runner for this language' >&2; exit 97"
    return (f"printf %s '{b}' | base64 -d > /tmp/sv_repro.{ext} && "
            f"{runner} /tmp/sv_repro.{ext}")


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
              language: str = "python",
              allow_host: bool = True,
              container_runner=None,
              timeout: int = _REPRO_TIMEOUT) -> RunResult:
    """Run `repro_code` (in `language`) against the edited tree at `sandbox_dir`,
    auto-detecting a backend that can actually execute it. Returns a RunResult;
    .ran is True only when the verdict is trustworthy (the code actually executed).

    container_runner, if given, is `callable(repro_code, timeout) -> RunResult`
    used as the last escalation step (e.g. run inside the instance's image — which,
    for a non-python repo, is where the language runtime actually lives)."""
    if not repro_code or not repro_code.strip():
        return RunResult(ERROR, output="empty repro")
    sandbox_dir = sandbox_dir or "/tmp"
    proto = _proto(language)
    cmd = _run_cmd(repro_code, proto)

    # 1) LOCAL — stdlib + repo only (python runtime; other runtimes usually absent →
    #    env_blocked → escalates to the container that has them).
    res = run_sandboxed(cmd, cwd=sandbox_dir, timeout=timeout,
                        project_root=project_root or sandbox_dir)
    local = _classify(res, sandbox_dir, "local", proto)
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
            host = _classify(res, sandbox_dir, "host", proto)
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

# Prefer a PYTHON-tagged fence; fall back to the first fence of any language only if
# there's no python one (ckpt-272: the old `(?:python|py)?` optional tag grabbed the
# FIRST fence of ANY language — so a leading ```bash setup block or ```text note stole
# the match and the real python repro was dropped → SyntaxError → spurious FAIL on a
# possibly-correct patch). gpt-oss often emits a setup/prose fence before the repro.
_FENCE_PY = re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.S)
_FENCE_ANY = re.compile(r"```[^\n]*\n(.*?)\n```", re.S)


def _strip_fence(text: str) -> str:
    if not text:
        return ""
    m = _FENCE_PY.search(text) or _FENCE_ANY.search(text)
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


async def author_repro(task: str, signatures: str,
                       model: str = "nvidia/gpt-oss-120b",
                       *, language: str = "python", max_chars: int = 9000) -> "str | None":
    """Ask a model to write a runnable repro from the ISSUE, in `language`. It is given
    the ISSUE + a SIGNATURE DIGEST (callable API, no bodies) — NOT the patch/diff, so it
    cannot be primed to arrange invented internal state (ckpt-277). Returns the script,
    or None when the model says NO_REPRO / the call fails (caller treats None as 'cannot
    verify' → deliver as-is, never a false bug)."""
    from clients.nvidia import call_nvidia
    from core.prompts_v8 import SELFVERIFY_REPRO_PROMPT

    # python uses the validated prompt byte-identical; other languages get an override
    # header that re-targets the language + sentinel and tells the model to translate the
    # (python-flavoured) intent below (ckpt-275 — multi-language).
    system = SELFVERIFY_REPRO_PROMPT
    if language and language != "python":
        _sent = _LANG_SENTINEL.get(language, _LANG_SENTINEL["generic"])
        system = (f"⚠ TARGET LANGUAGE: {language} — write the reproduction in {language}, NOT "
                  f"Python. Run-as-a-script style. End every success path with `{_sent}` as the "
                  f"VERY LAST line printed. Where the rules below show Python syntax/wording, "
                  f"translate the INTENT to {language} (same faithfulness + no-leakage rules).\n\n"
                  + SELFVERIFY_REPRO_PROMPT)

    _t = (task or "").strip()
    _s = (signatures or "").strip()
    if len(_t) > max_chars:
        _t = _t[: max_chars // 2] + "\n…\n" + _t[-max_chars // 2:]
    if len(_s) > max_chars:
        _s = _s[: max_chars] + "\n…[signatures truncated]…"
    user = (f"=== ISSUE ===\n{_t}\n\n"
            f"=== CHANGED SYMBOLS — signatures only (the callable API; you are NOT shown "
            f"the implementation) ===\n{_s or '(none extracted — rely on the issue)'}\n")
    try:
        raw = await call_nvidia(model, prompt=user, system=system,
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
                          *, language: str = "python", pull_if_missing: bool = True):
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
    proto = _proto(language)
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
            _ext, _runner = proto.get("ext", "py"), proto.get("runner") or "python -B"
            with open(os.path.join(ws, f"repro.{_ext}"), "w", encoding="utf-8") as f:
                f.write(repro_code)
            # The container HAS the language runtime (it's the instance's own image), so
            # run with the per-language runner. The python-oriented env (PYTHONPATH, Qt
            # offscreen) is harmless for other runtimes.
            inner = ("cd /app && tar xf /ws/edits.tar -C /app && "
                     "QT_QPA_PLATFORM=offscreen PYTHONPATH=/app:/app/src:/app/lib "
                     f"timeout {int(timeout)} {_runner} /ws/repro.{_ext}")
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
            return _classify(d, sandbox_dir, "container", proto)
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
