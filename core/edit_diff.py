"""
Claude-Code-TUI-style unified diff, rendered in JARVIS's `LINENO:` gutter form.

After the runtime applies a batch of edits it shows the coder exactly what
changed — not a prose summary ("84 → 112 lines") it has to trust, but the
real before/after lines, so it can VERIFY the edit landed where it meant
before signalling [DONE]. Format (mirrors the read view's `LINENO:` gutter):

    path/to/file.py
       11:     def deposit(self, amount):          ← context (new line #)
       12:+        if amount <= 0:                  ← ADDED   (`:+`, new line #)
       13:+            raise ValueError('amount')    ← ADDED
       14:         self.balance += amount           ← context
        9:-        self.balance -= amount           ← REMOVED (`:-`, old line #)

`:+` = line added, `:-` = line removed, `: ` = unchanged context. Added and
context lines carry their NEW line numbers (what a fresh re-read will show);
removed lines carry their OLD line number. Hunks are separated by `⋮`.
"""
import difflib


def render_diff(old_src: str, new_src: str, path: str,
                context: int = 3, max_lines: int = 240) -> str:
    """Unified diff of old_src→new_src in `LINENO:[+|-| ]content` form.

    Returns a string headed by `path`. If nothing changed, returns "" so the
    caller can skip it. Output is capped at `max_lines` body lines with a
    truncation note (a giant rewrite still produces a readable summary).
    """
    a = old_src.split("\n")
    b = new_src.split("\n")
    if a == b:
        return ""

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    groups = list(sm.get_grouped_opcodes(context))
    if not groups:
        return ""

    width = max(len(str(len(a))), len(str(len(b))), 2)
    out = [path]
    body = 0
    truncated = False

    for gi, group in enumerate(groups):
        if gi > 0:
            out.append(" " * width + " ⋮")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for k in range(i2 - i1):
                    out.append(f"{j1 + k + 1:>{width}}:  {b[j1 + k]}")
                    body += 1
            else:
                # removed lines (old numbers) first, then added (new numbers) —
                # matches how git / the CC TUI stack a replacement.
                for k in range(i1, i2):
                    out.append(f"{k + 1:>{width}}:- {a[k]}")
                    body += 1
                for k in range(j1, j2):
                    out.append(f"{k + 1:>{width}}:+ {b[k]}")
                    body += 1
            if body > max_lines:
                truncated = True
                break
        if truncated:
            break

    if truncated:
        out.append(" " * width + f" … diff truncated ({body} lines shown)")
    return "\n".join(out)


def render_multi(diffs: "list[tuple[str, str, str]]", **kw) -> str:
    """Render several files' diffs (list of (path, old_src, new_src)).

    Files with no change are skipped. Returns "" when nothing changed.
    """
    parts = []
    for path, old_src, new_src in diffs:
        d = render_diff(old_src, new_src, path, **kw)
        if d:
            parts.append(d)
    return "\n\n".join(parts)
