"""
Sandbox — safe file editing environment for the coding agent.

All changes happen in a sandbox copy. Original files are NEVER touched
until the user explicitly approves the diff.
"""

import os
import shutil
import difflib
import subprocess
from pathlib import Path
from core.cli import step, status, warn, success


class Sandbox:
    """
    Manages a full copy of the project for safe editing.
    On setup, the entire project is copied to .jarvis_sandbox/.
    All reads and writes happen in the sandbox.
    Original files are NEVER touched until the user explicitly approves.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.sandbox_dir = self.project_root / ".jarvis_sandbox"
        self.original_files: dict[str, str] = {}   # rel_path → original content
        self.modified_files: dict[str, str] = {}    # rel_path → new content
        self.new_files: dict[str, str] = {}         # rel_path → content (files that don't exist yet)

    def setup(self):
        """Create sandbox as a full copy of the project."""
        # Clean previous sandbox
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Copy all project files to sandbox (skip hidden dirs, node_modules, etc.)
        skip_dirs = {'.jarvis_sandbox', '.git', 'node_modules', '__pycache__',
                     '.venv', 'venv', '.tox', '.mypy_cache', '.pytest_cache'}
        skip_exts = {'.pyc', '.pyo', '.so', '.o', '.a', '.dylib'}
        copied = 0
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            rel_root = Path(root).relative_to(self.project_root)

            for fname in files:
                if Path(fname).suffix in skip_exts:
                    continue
                src = Path(root) / fname
                rel_path = str(rel_root / fname)
                dest = self.sandbox_dir / rel_path

                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    copied += 1
                except Exception:
                    pass  # Skip files that can't be copied (binary, permissions, etc.)

        status(f"Sandbox created: {copied} files copied to {self.sandbox_dir}")

    def _norm(self, rel_path: str) -> str:
        """Normalize path separators (accept both / and \\) AND enforce CONTAINMENT: the
        path must resolve INSIDE the sandbox, never escape via `..` or an absolute path.
        A model-supplied `../../x.py` or `/abs/x.py` would otherwise be written OUTSIDE the
        repo (host pollution) and be silently ABSENT from the patch. Raise on escape so the
        caller surfaces a clean reject. (audit pass-6 / create_file_no_path_containment.)"""
        p = str(Path(rel_path))
        base = self.sandbox_dir.resolve()
        full = (self.sandbox_dir / p).resolve()
        if full != base and not full.is_relative_to(base):
            raise ValueError(
                f"path {rel_path!r} resolves outside the project — use a path INSIDE the "
                f"repo (no leading '/' and no '..' that escapes the root).")
        return p

    def load_file(self, rel_path: str) -> str | None:
        """Load a file from the sandbox (working copy). Returns content or None."""
        rel_path = self._norm(rel_path)

        # Read from sandbox (working copy)
        sandbox_src = self.sandbox_dir / rel_path
        if sandbox_src.exists():
            try:
                content = sandbox_src.read_text(encoding="utf-8", errors="replace")
                # Track original if not already tracked
                if rel_path not in self.original_files:
                    real_src = self.project_root / rel_path
                    if real_src.exists():
                        self.original_files[rel_path] = real_src.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    else:
                        self.original_files[rel_path] = ""
                return content
            except Exception as e:
                warn(f"Failed to load {rel_path} from sandbox: {e}")
                return None

        # Fallback: file not in sandbox, try project root
        src = self.project_root / rel_path
        if not src.exists():
            return None
        try:
            content = src.read_text(encoding="utf-8", errors="replace")
            self.original_files[rel_path] = content

            # Copy to sandbox for future reads
            dest = self.sandbox_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

            return content
        except Exception as e:
            warn(f"Failed to load {rel_path}: {e}")
            return None

    def write_file(self, rel_path: str, content: str):
        """Write modified content to sandbox.

        A write is only recorded as a "modification" if the content differs
        from the original. SEARCH/REPLACE blocks whose REPLACE body is
        byte-identical to the matched range produce no diff in git — and so
        should not be counted in `Applied N changes`. Observed failure mode
        on django-11551 / django-14631: workflow reported "Applied N
        changes" but final `git diff` was empty because every REPLACE body
        matched its SEARCH (no real edit).
        """
        rel_path = self._norm(rel_path)
        # Track whether this is a new file or modification
        if rel_path in self.original_files:
            if content == self.original_files[rel_path]:
                # No-op write — drop any stale modification record but do not
                # promote this to a "change".
                self.modified_files.pop(rel_path, None)
            else:
                self.modified_files[rel_path] = content
        else:
            src = self.project_root / rel_path
            if src.exists():
                # Load original first
                self.original_files[rel_path] = src.read_text(encoding="utf-8", errors="replace")
                if content == self.original_files[rel_path]:
                    self.modified_files.pop(rel_path, None)
                else:
                    self.modified_files[rel_path] = content
            else:
                if content == "":
                    # Empty new file is also a no-op for diff purposes.
                    self.new_files.pop(rel_path, None)
                else:
                    self.new_files[rel_path] = content

        # Always mirror to sandbox on disk, even for no-ops. Idempotent.
        dest = self.sandbox_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def get_diff(self, rel_path: str) -> str:
        """Get unified diff for a modified file."""
        original = self.original_files.get(rel_path, "")
        if rel_path in self.modified_files:
            modified = self.modified_files[rel_path]
        elif rel_path in self.new_files:
            original = ""
            modified = self.new_files[rel_path]
        else:
            return "(no changes)"

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return "\n".join(diff) or "(no changes)"

    def get_all_diffs(self) -> str:
        """Get diffs for all changed files."""
        diffs = []
        for rel_path in sorted(set(list(self.modified_files.keys()) + list(self.new_files.keys()))):
            diff = self.get_diff(rel_path)
            if diff != "(no changes)":
                diffs.append(f"═══ {rel_path} ═══\n{diff}")

        return "\n\n".join(diffs) if diffs else "(no changes)"

    def apply(self) -> list[str]:
        """Apply all sandbox changes to the actual project. Returns list of modified files."""
        applied = []
        _root = self.project_root.resolve()

        def _contained(rel_path):
            # bughunt ckpt-242: an ABSOLUTE rel_path ('/etc/passwd') or a '../'-escape would write
            # OUTSIDE the repo — Path's `/` lets an absolute RHS replace the LHS. Refuse anything
            # that doesn't resolve under project_root (keeps junk/escaping paths out of the patch).
            try:
                dest = (self.project_root / rel_path).resolve()
                dest.relative_to(_root)
                return dest
            except Exception:
                warn(f"  ⚠ sandbox.apply: refusing out-of-repo path {rel_path!r} — skipped")
                return None

        def _recover_if_empty(rel_path, content):
            """A real-life bug (R2): a non-converged self-verify revert / in-memory
            tracking glitch left modified_files[f]='' while the REAL edited content
            lived only on the sandbox disk — applying '' wiped a working file. Two
            safeguards: (a) if in-memory is empty but the sandbox's own edited file is
            non-empty, the genuine edit is on disk → use it; (b) NEVER overwrite an
            existing non-empty file with empty/whitespace content (return None = skip)."""
            if (content or "").strip():
                return content
            # (a) recover the real content from the sandbox's edited copy
            try:
                sb_file = self.sandbox_dir / rel_path
                if sb_file.is_file():
                    disk = sb_file.read_text(encoding="utf-8")
                    if disk.strip():
                        warn(f"  ⚠ sandbox.apply: in-memory {rel_path} was EMPTY — "
                             f"recovered {len(disk)}B from the sandbox copy")
                        return disk
            except Exception:
                pass
            # (b) backstop: refuse to destroy an existing non-empty file with empty
            try:
                if dest.exists() and dest.read_text(encoding="utf-8").strip():
                    warn(f"  ⚠ sandbox.apply: REFUSING to overwrite non-empty {rel_path} "
                         f"with EMPTY content — kept the existing file")
                    return None
            except Exception:
                pass
            return content

        for rel_path, content in self.modified_files.items():
            dest = _contained(rel_path)
            if dest is None:
                continue
            content = _recover_if_empty(rel_path, content)
            if content is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            applied.append(f"Modified: {rel_path}")

        for rel_path, content in self.new_files.items():
            dest = _contained(rel_path)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            applied.append(f"Created: {rel_path}")

        success(f"Applied {len(applied)} changes to project")
        return applied

    def run(self, cmd: str, timeout: int = 120) -> dict:
        """Run a shell command against the EDITED code (cwd = self.sandbox_dir).

        Returns {'exit_code': int, 'output': str, 'timed_out': bool}.
        output is stdout + '\\n' + stderr, capped to ~10000 chars (keeping the
        TAIL if longer, since errors are usually at the end, prefixed with a
        "...(truncated)..." marker). On timeout: exit_code=-1, timed_out=True.
        Never raises.
        """
        CAP = 10000
        # Make the EDITED repo importable from run_code (ckpt-162). The coder kept
        # trying to `import <repo_module>` to verify an edit and failing — the
        # subprocess had no PYTHONPATH to the sandbox, so even a pure-python module
        # it had just edited wouldn't import. Prepend the sandbox root + the common
        # source layouts (`lib/` for ansible-style, `src/` for src-layout) so
        # `import <pkg>` resolves by PATH against the edited tree. (Third-party deps
        # still come from the parent venv — this only fixes path resolution, not a
        # missing yaml/jinja2; the coder must NOT edit source to stub those.)
        env = dict(os.environ)
        _roots = [str(self.sandbox_dir)]
        for _sub in ("lib", "src"):
            _d = self.sandbox_dir / _sub
            if _d.is_dir():
                _roots.append(str(_d))
        _existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(_roots) + (os.pathsep + _existing if _existing else "")
        # Don't litter the sandbox with __pycache__ (keeps the tree = the patch, and
        # sidesteps the coder's "py_compile can't write bytecode" rabbit hole).
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self.sandbox_dir),
                timeout=timeout,
                env=env,
            )
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if len(output) > CAP:
                output = "...(truncated)...\n" + output[-CAP:]
            return {
                "exit_code": proc.returncode,
                "output": output,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "output": f"Command timed out after {timeout}s: {cmd}",
                "timed_out": True,
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "output": f"Command failed to run: {e}",
                "timed_out": False,
            }

    def cleanup(self):
        """Remove sandbox directory."""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def summary(self) -> str:
        """Human-readable summary of changes."""
        lines = []
        for p in self.modified_files:
            orig_lines = self.original_files.get(p, "").count("\n")
            new_lines = self.modified_files[p].count("\n")
            lines.append(f"  Modified: {p} ({orig_lines} → {new_lines} lines)")
        for p in self.new_files:
            new_lines = self.new_files[p].count("\n")
            lines.append(f"  New file: {p} ({new_lines} lines)")
        return "\n".join(lines) if lines else "  (no changes)"
