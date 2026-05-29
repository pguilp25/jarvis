"""Unit test: the native coder exposes the FULL text-coder toolset (parity).

Builds a real temp project + Sandbox, then dispatches every native tool and
checks the result is sane. Validates the n| (LINENO:INDENT|code) read format and
the replace_lines apply/reject path. The live SWE-bench smoke is the end-to-end
check; this proves each tool is wired to its executor and returns, not bare-tags.
"""
import asyncio, os, sys, tempfile, textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sandbox import Sandbox
from core.native_tools import _dispatch, CODER_TOOLS

SRC = textwrap.dedent('''\
    """A tiny module for testing the native coder tools."""


    def greet(name):
        """Return a greeting for name."""
        return "hello " + name


    class Counter:
        """Counts things."""

        def __init__(self):
            self.n = 0

        def bump(self):
            self.n = self.n + 1
            return self.n
''')


def _mk_ctx():
    root = tempfile.mkdtemp(prefix="nativetool_")
    rel = "mod.py"
    with open(os.path.join(root, rel), "w") as f:
        f.write(SRC)
    sb = Sandbox(root)
    sb.setup()
    sb.load_file(rel)  # seed original tracking
    ctx = {"file_contents": {rel: SRC}, "sandbox": sb, "project_root": root,
           "viewed_versions": {}, "purpose_map": "", "detailed_map": "",
           "files_changed": set()}
    return ctx, rel


async def main():
    ctx, rel = _mk_ctx()
    ok = 0; fail = 0
    def check(label, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1; print(f"  ✓ {label}")
        else:
            fail += 1; print(f"  ✗ {label}  -- {detail[:200]}")

    # 1. schema surface = full parity set
    names = {t["function"]["name"] for t in CODER_TOOLS}
    expect = {"read_file", "find_refs", "find_callers", "search_text",
              "file_purpose", "semantic_search", "depends_on",
              "replace_lines", "finish"}
    check("schemas cover full toolset", names == expect, f"got {sorted(names)}")

    # 2. read_file → LINENO:INDENT|code (n| prefix format)
    out = await _dispatch("read_file", {"path": rel}, ctx)
    check("read_file returns prefix n| format", ":0|" in out and "greet" in out, out)
    check("read_file shows indent count for body", ":4|" in out or ":8|" in out, out)

    # 3. read_file range
    out = await _dispatch("read_file", {"path": rel, "start_line": 4, "end_line": 6}, ctx)
    check("read_file range works", "greet" in out, out)

    # 4. replace_lines valid (n| INDENT|code body) — change the greeting
    #    line 6 is `    return "hello " + name` (4 spaces)
    out = await _dispatch("replace_lines",
                          {"path": rel, "start_line": 6, "end_line": 6,
                           "new_content": '4|return "hi " + name'}, ctx)
    check("replace_lines applies", out.startswith("✓ Applied"), out)
    check("indent-count expanded to spaces",
          ctx["file_contents"][rel].split("\n")[5] == '    return "hi " + name',
          repr(ctx["file_contents"][rel].split("\n")[5]))
    check("files_changed recorded", rel in ctx["files_changed"], str(ctx["files_changed"]))

    # 5. replace_lines bad range → clear reject, not crash
    out = await _dispatch("replace_lines",
                          {"path": rel, "start_line": 9999, "end_line": 99999,
                           "new_content": "0|x = 1"}, ctx)
    check("bad range rejected cleanly", out.startswith("✗"), out)

    # 6. find_refs (ripgrep) — should locate the symbol
    out = await _dispatch("find_refs", {"symbol": "greet"}, ctx)
    check("find_refs returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 7. search_text
    out = await _dispatch("search_text", {"pattern": "class Counter"}, ctx)
    check("search_text returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 8. file_purpose
    out = await _dispatch("file_purpose", {"path": rel}, ctx)
    check("file_purpose returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 9. semantic_search
    out = await _dispatch("semantic_search", {"query": "counting things"}, ctx)
    check("semantic_search returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 10. depends_on
    out = await _dispatch("depends_on", {"symbol": "Counter"}, ctx)
    check("depends_on returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 11. find_callers (no real #tag map here — must not crash)
    out = await _dispatch("find_callers", {"tag": "nope"}, ctx)
    check("find_callers returns a result string", isinstance(out, str) and len(out) > 0, out)

    # 12. finish
    out = await _dispatch("finish", {"summary": "done"}, ctx)
    check("finish signals __FINISH__", isinstance(out, tuple) and out[0] == "__FINISH__", str(out))

    # 13. unknown tool
    out = await _dispatch("bogus", {}, ctx)
    check("unknown tool message", isinstance(out, str) and out.startswith("✗ Unknown"), out)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
