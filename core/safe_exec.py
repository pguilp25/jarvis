"""
safe_exec — fail-proof sandboxed command execution for the planner/reviewer
DIAGNOSTIC command tool.

The model may run almost any command to LOOK at the codebase and OBSERVE how
the code behaves at runtime — but it must never edit code, delete files, reach
the network, escalate privilege, or damage the machine. We do not try to
achieve that with a clever blocklist (those are bypassable via `python -c`,
`$IFS`, base64, symlinks…). We achieve it with OS-level isolation, so that even
a command that TRIES something dangerous is physically prevented.

Defense in depth:
  1. POLICY (pre-exec): reject obviously-pointless-and-scary command heads
     (rm, sudo, curl, pip…) with a clear message. This is UX + a tripwire, NOT
     the guarantee.
  2. OS SANDBOX (the guarantee) — bubblewrap:
       • whole filesystem READ-ONLY  → no real file can be written or deleted
       • network namespace unshared   → no network at all
       • user namespace, no privilege → no sudo/setuid/mount/etc.
       • PID/IPC/UTS/cgroup unshared   → can't see or kill host processes
       • HOME=/tmp, /home NOT bound    → secrets (~/.bashrc keys, ~/.ssh) unseen
       • only /tmp (+ /run, /dev/shm) writable as tmpfs, ephemeral & discarded
  3. RESOURCE LIMITS — wall-clock timeout (+ process-group kill), CPU, address
     space, file size, no core dumps; output captured and truncated.

FAIL-SAFE: if bubblewrap is unavailable or the sandbox cannot be constructed,
the command is DENIED. We never fall back to running it unsandboxed.
"""
from __future__ import annotations
import os
import shlex
import shutil
import signal
import subprocess
import sys

_BWRAP = shutil.which("bwrap")

# Wall-clock + resource ceilings (generous enough for real diagnostics).
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 300
_CPU_SECONDS = 120          # RLIMIT_CPU
_ADDRESS_SPACE = 4 * 1024 ** 3   # RLIMIT_AS (4 GB)
_MAX_FILE_BYTES = 128 * 1024 ** 2  # RLIMIT_FSIZE (128 MB writes to /tmp)
_OUTPUT_CAP = 12000         # chars of combined stdout+stderr returned

# Read-only binds that make a normal toolchain usable. `_try` ones may be
# absent on some systems; bwrap's --ro-bind-try ignores those.
_RO_BINDS = ["/usr", "/etc"]
_RO_BINDS_TRY = ["/bin", "/sbin", "/lib", "/lib64", "/lib32", "/opt",
                 "/usr/local", "/snap"]

# ── Layer 1: policy blocklist (clarity + tripwire; the sandbox is the guard) ──
# Heads that are pointless or alarming inside a read-only, network-less sandbox.
# Blocking them up front gives the model a crisp reason instead of a confusing
# OS error, and stops accidental destructive intent before it even runs.
_DENY_HEADS = {
    # destructive filesystem
    "rm", "rmdir", "shred", "unlink", "mv", "dd", "truncate", "mkfs",
    "mke2fs", "mkswap", "fdisk", "sfdisk", "parted", "wipefs", "blkdiscard",
    "fallocate", "ln",
    # privilege / identity
    "sudo", "su", "doas", "pkexec", "chown", "chgrp", "chmod", "setcap",
    "setfacl", "passwd", "usermod", "useradd", "groupadd",
    # process / system / kernel control
    "kill", "pkill", "killall", "reboot", "shutdown", "poweroff", "halt",
    "init", "telinit", "systemctl", "service", "mount", "umount", "swapon",
    "swapoff", "sysctl", "insmod", "rmmod", "modprobe", "chroot", "nsenter",
    "unshare", "setarch", "iptables", "nft", "ip", "ifconfig", "route",
    # network clients (the sandbox blocks net anyway — explicit for clarity)
    "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "ftp",
    "telnet", "rsync", "socat", "ssh-keygen", "ssh-add", "ssh-agent",
    "openssl",
    # package managers / installers (mutate env, fetch code)
    "apt", "apt-get", "aptitude", "dpkg", "pip", "pip3", "pipx", "npm",
    "npx", "yarn", "pnpm", "gem", "cargo", "conda", "mamba", "brew", "snap",
    "flatpak", "poetry", "uv",
    # scheduling
    "crontab", "at", "batch",
}

# Wrappers that delegate to a following command — look past them to the real head.
_WRAPPERS = {"command", "env", "nice", "nohup", "stdbuf", "ionice", "setsid",
             "time", "exec", "xargs", "watch", "timeout"}

# git is allowed only for READ-ONLY subcommands (default-deny for git).
_GIT_READ_OK = {
    "log", "diff", "status", "show", "blame", "ls-files", "ls-tree",
    "cat-file", "rev-parse", "rev-list", "describe", "shortlog", "reflog",
    "grep", "name-rev", "merge-base", "symbolic-ref", "whatchanged",
    "annotate", "for-each-ref", "count-objects", "var", "help", "version",
    "show-ref", "verify-pack", "blame", "diff-tree", "diff-index",
}

# operators that separate one command from the next in a shell line. We do NOT
# split on `` ` `` / `$(` : that keeps a command-substitution in HEAD position as
# a single token (e.g. `` `echo rm` `` → token "`echo") which the indirect-head
# check below rejects. Substitutions in ARGUMENT position are left to the OS
# sandbox (the real guarantee).
_SEP_OPERATORS = ("&&", "||", "|&", "|", ";", "&", "\n")


def policy_check(cmd: str) -> "tuple[bool, str]":
    """Return (allowed, reason). Reject if any pipeline segment's effective head
    is destructive/network/privilege/installer, an indirect ($VAR / `...`)
    head, or a mutating `git` subcommand. Reason is '' when allowed.

    This is the tripwire/UX layer; the read-only + no-network OS sandbox is the
    actual guarantee, so imperfect shell parsing here cannot weaken safety —
    anything that slips past is still physically contained."""
    if not cmd or not cmd.strip():
        return False, "empty command"
    _RO = ("this is a READ-ONLY diagnostic sandbox (no editing, deleting, "
           "installing, network, or privilege) — run something that observes, "
           "not mutates.")
    work = cmd
    for op in _SEP_OPERATORS:
        work = work.replace(op, "\x00")
    for seg in work.split("\x00"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg, comments=False, posix=True)
        except ValueError:
            toks = seg.split()
        # find the effective head: skip env-assignments (FOO=bar) and wrappers
        head_tok = None
        rest: list[str] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            if "=" in t and "/" not in t.split("=", 1)[0] and t.split("=", 1)[0].isidentifier():
                i += 1
                continue
            if os.path.basename(t).lower() in _WRAPPERS:
                i += 1
                continue
            head_tok, rest = t, toks[i + 1:]
            break
        if head_tok is None:
            continue
        # indirect / runtime-expanded heads — can't be verified statically
        if head_tok[0] in "$`" or head_tok.startswith("${"):
            return False, (f"commands invoked through a shell variable or "
                           f"substitution ({head_tok}) aren't allowed — write "
                           f"the command name directly. {_RO}")
        head = os.path.basename(head_tok).lower()
        if head in _DENY_HEADS:
            return False, f"`{head}` is not allowed — {_RO}"
        if head == "git":
            sub = next((r.lower() for r in rest if not r.startswith("-")), "")
            if sub and sub not in _GIT_READ_OK:
                return False, (f"`git {sub}` can mutate the repo or hit the "
                               f"network — only read-only git (log/diff/status/"
                               f"show/blame) is allowed. {_RO}")
    return True, ""


def _set_rlimits():
    """preexec in the child: cap CPU, memory, file size, core dumps before exec.
    Inherited by bwrap and everything inside the sandbox."""
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS + 5))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_ADDRESS_SPACE, _ADDRESS_SPACE))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _bwrap_argv(cwd: str, project_root: "str | None") -> "list[str]":
    """Construct the bubblewrap argv for a read-only, network-less, privilege-
    less sandbox with a writable ephemeral /tmp and HOME hidden."""
    argv = [_BWRAP, "--unshare-all", "--die-with-parent", "--new-session"]
    for d in _RO_BINDS:
        argv += ["--ro-bind", d, d]
    for d in _RO_BINDS_TRY:
        argv += ["--ro-bind-try", d, d]
    # The interpreter JARVIS runs in (the venv) carries pytest + the repo's deps;
    # the bare system python3 on the default PATH does not, and there's no bare
    # `python` at all. Without this a [VERIFY:]/[RUN:] `python …` resolved to
    # nothing (exit 127 "python not found") so verification SILENTLY never ran.
    # Bind the venv bin read-only and put it first on PATH so python / python3 /
    # pytest resolve to it. (Read-only bind → no new write/network capability.)
    # NOTE: do NOT realpath() — a venv's bin/python is a symlink to the system
    # python, and resolving it would point _interp_dir back at /usr/bin (which
    # has no `python` and no pytest). We want the venv bin itself.
    _interp_dir = os.path.dirname(os.path.abspath(sys.executable)) if sys.executable else ""
    # the project + the working dir + the interpreter dir, read-only
    seen = set()
    for d in (project_root, cwd, _interp_dir):
        if d and d not in seen and os.path.isdir(d):
            argv += ["--ro-bind", d, d]
            seen.add(d)
    argv += ["--proc", "/proc", "--dev", "/dev"]
    # the only writable surfaces — tmpfs, ephemeral, discarded on exit
    for d in ("/tmp", "/run", "/var/tmp", "/dev/shm"):
        argv += ["--tmpfs", d]
    # clean env; HOME=/tmp so ~ resolves to writable tmp and ~/.bashrc, ~/.ssh,
    # ~/.aws etc. are simply absent (home is never bound).
    argv += [
        "--clearenv",
        "--setenv", "PATH",
        (f"{_interp_dir}:" if _interp_dir else "")
        + "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "LANG", os.environ.get("LANG", "C.UTF-8"),
        "--setenv", "LC_ALL", "C.UTF-8",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONUNBUFFERED", "1",
        "--setenv", "TERM", "dumb",
        "--chdir", cwd if (cwd and os.path.isdir(cwd)) else "/tmp",
    ]
    return argv


def run_sandboxed(cmd: str, cwd: str, timeout: int = _DEFAULT_TIMEOUT,
                  project_root: "str | None" = None) -> dict:
    """Run `cmd` in the fail-proof read-only sandbox.

    Returns a dict:
      blocked   — True if policy rejected it OR the sandbox couldn't be built
      reason    — why it was blocked ('' otherwise)
      exit_code — process exit code (or -1)
      output    — combined stdout+stderr, truncated (tail kept)
      timed_out — True if killed by the wall-clock timeout
      sandbox   — 'bwrap' on success, 'none' when denied for lack of sandbox
    """
    timeout = max(1, min(int(timeout or _DEFAULT_TIMEOUT), _MAX_TIMEOUT))

    ok, reason = policy_check(cmd)
    if not ok:
        return {"blocked": True, "reason": reason, "exit_code": -1,
                "output": "", "timed_out": False, "sandbox": "policy"}

    # FAIL-SAFE: no sandbox tool → refuse to run (never exec unsandboxed).
    if not _BWRAP:
        return {"blocked": True, "exit_code": -1, "output": "", "timed_out": False,
                "sandbox": "none",
                "reason": "command execution is disabled: the bubblewrap sandbox "
                          "(bwrap) is not available, and commands are never run "
                          "without it."}

    cwd = cwd or "/tmp"
    # non-login shell (`-c`, not `-lc`): avoids sourcing /etc/profile.d which
    # spews MOTD/notice noise into the captured output. The env we need (PATH,
    # HOME, TMPDIR…) is set explicitly via bwrap --setenv.
    argv = _bwrap_argv(cwd, project_root) + ["--", "/bin/bash", "-c", cmd]

    # Pass env={} so that bwrap's OWN process starts with an empty environment.
    # Without this, bwrap inherits the caller's env (including all API keys), and
    # the sandboxed process can read them via /proc/1/environ — because PID 1 inside
    # the PID namespace is bwrap itself, and /proc exposes bwrap's environ verbatim.
    # bwrap's --clearenv only clears the env of the CHILD it launches, not bwrap's own.
    # The writable env variables we need for the child (/proc/self/environ) are set
    # via bwrap's own --setenv flags and are unaffected by this change.
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, errors="replace",
            start_new_session=True,          # own process group → killpg on timeout
            preexec_fn=_set_rlimits,
            env={},                          # bwrap own env empty — see comment above
        )
    except Exception as e:
        return {"blocked": True, "reason": f"sandbox failed to start: {e}",
                "exit_code": -1, "output": "", "timed_out": False, "sandbox": "none"}

    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # kill the whole process group (bwrap + the fork bomb / hung child).
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except Exception:
                pass
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:
            out = ""
        out = (out or "") + f"\n[sandbox] killed after {timeout}s wall-clock timeout."

    out = out or ""
    if len(out) > _OUTPUT_CAP:
        head = out[: _OUTPUT_CAP // 3]
        tail = out[-(_OUTPUT_CAP * 2 // 3):]
        out = f"{head}\n…[output truncated, {len(out)} chars total]…\n{tail}"

    return {
        "blocked": False, "reason": "",
        "exit_code": (proc.returncode if proc.returncode is not None else -1),
        "output": out, "timed_out": timed_out, "sandbox": "bwrap",
    }
