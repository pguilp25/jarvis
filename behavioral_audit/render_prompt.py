#!/usr/bin/env python3
"""Render the FULL assembled artifact the coder model actually receives — offline, in seconds.

CLAUDE.md doctrine: VERIFY THE RENDERED ARTIFACT. We do not edit a prompt string and run a 30-60
min live job to discover a bug; we render the *assembled* result (system + tool schemas + user turn
+ injected files + the growing view) and read it. This calls the SAME functions the live coder uses
(workflows.code.build_implement_native_prompt + core.native_tools.finalize_coder_system + the real
_dispatch view renderer), so what you read here is byte-identical to what gpt-oss sees — no replica.

Usage:
  python3 behavioral_audit/render_prompt.py                 # all built-in cases (read each!)
  python3 behavioral_audit/render_prompt.py --case big      # one case: small|big|multi|reject|grow
  python3 behavioral_audit/render_prompt.py --stats         # char/section sizes only (sweep view)
  python3 behavioral_audit/render_prompt.py --file PATH --reads 1300-1320,1480-1500   # real file,
                                                            # show the growing view after those reads
  python3 behavioral_audit/render_prompt.py --tools         # dump the tool schemas the model sees
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.code import build_implement_native_prompt
from core.native_tools import build_json_ops_system, CODER_TOOLS, _dispatch

_BAR = "═" * 100


def assemble(step_instructions, iface_block, files, to_create=None, error_feedback=""):
    """Return the FULL coder artifact dict {system, user, tools, injected, overflow} EXACTLY as the
    live PRIMARY coder builds + sends it. The live json-ops coder is ALL-SYSTEM (user 2026-06-11):
    one system message (instructions + indent block + JSON-OPS protocol + the [USER REQUEST]-fenced
    step) and NO user turn. We assemble it through the SAME build_json_ops_system the coder uses, so
    the audited artifact can never drift from the live one (CLAUDE.md doctrine)."""
    nat_system, nat_user, injected, overflow = build_implement_native_prompt(
        step_instructions, iface_block, files, to_create=to_create or [],
        error_feedback=error_feedback)
    return {
        "system": build_json_ops_system(nat_system, nat_user),
        "user": "",   # all-system: there is no user turn
        "tools": [],   # json-ops PRIMARY coder runs in TEXT mode — NO tools array is sent
                       # (ops are emitted as flat JSON lines). CODER_TOOLS is dumped via --tools.
        "injected": list(injected),
        "overflow": list(overflow),
    }


def _toklen(s):
    return f"{len(s):,} chars (~{len(s)//4:,} tok)"


def print_artifact(label, art, stats_only=False):
    print(f"\n{_BAR}\n  CASE: {label}\n{_BAR}")
    print(f"  injected-in-full: {art['injected'] or '(none)'}")
    print(f"  read-on-demand  : {art['overflow'] or '(none)'}")
    print(f"  SYSTEM {_toklen(art['system'])} | USER {_toklen(art['user'])} | "
          f"TOOLS {len(art['tools'])} schemas")
    if stats_only:
        return
    print(f"\n┌─ SYSTEM (the full final system prompt the model receives) {'─'*30}")
    print(art["system"])
    print(f"\n┌─ USER (the step turn: instructions + interfaces + injected view) {'─'*23}")
    print(art["user"])


def render_growing_view(content, reads, path="big.py"):
    """Drive the REAL read path (_dispatch) on `content` through a sequence of range reads, then
    print the accumulating 'growing view' the model would hold — exactly as the live coder sees it."""
    ctx = {"file_contents": {path: content}, "sandbox": None, "viewed_versions": {},
           "project_root": ".", "files_changed": set(), "step_num": 1}
    last = ""
    for (s, e) in reads:
        args = {"path": path} if (s is None) else {"path": path, "start_line": s, "end_line": e}
        last = asyncio.run(_dispatch("read_file", args, ctx))
    print(f"\n{_BAR}\n  GROWING VIEW after reads {reads} of {path} "
          f"({content.count(chr(10))+1} lines)\n{_BAR}")
    print(f"  revealed ranges held: {ctx.get('_served_ranges', {}).get(path)}")
    print(f"  view size: {_toklen(last)}")
    print(last)


# ── built-in cases (the shapes the coder actually meets) ────────────────────────────────────────
def _small_file():
    return ("STEP 1: rename `foo` to `bar` in helpers.py and update its one caller.",
            "(no shared interfaces)",
            {"pkg/helpers.py": "def foo(x):\n    return x + 1\n\n\ndef use():\n    return foo(3)\n"})


def _big_file():
    big = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(900))  # ~2700 lines
    return ("STEP 1: thread a `use_netrc=True` parameter through f10 and f20.",
            "(no shared interfaces)", {"pkg/big.py": big})


def _multi_file():
    return ("STEP 2: add `use_netrc` to url_get (get_url.py) and forward it to fetch_url.",
            "use_netrc: bool = True threads main()→url_get()→fetch_url()",
            {"mod/get_url.py": "def url_get(m, u):\n    return fetch_url(m, u)\n",
             "mod/uri.py": "def uri(m, u):\n    return fetch_url(m, u)\n"})


def _reject_reentry():
    return ("STEP 1: gate the .netrc block behind `if use_netrc:` in Request.open.",
            "(no shared interfaces)",
            {"m/urls.py": "class Request:\n    def open(self, use_netrc=None):\n        rc = netrc()\n"},
            "Your last edit was REJECTED — `old` did not match. The file is UNCHANGED; copy the "
            "view line VERBATIM (with its `LINENO ⇥INDENT|`) for `old`.")


CASES = {
    "small": _small_file, "big": _big_file, "multi": _multi_file, "reject": _reject_reentry,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES) + ["grow", "all"], default="all")
    ap.add_argument("--stats", action="store_true", help="section sizes only (sweep)")
    ap.add_argument("--tools", action="store_true", help="dump tool schemas the model sees")
    ap.add_argument("--file", help="render the growing view of a REAL file")
    ap.add_argument("--reads", default="", help="comma ranges e.g. 1300-1320,1480-1500 (empty=whole)")
    args = ap.parse_args()

    if args.tools:
        import json
        print(json.dumps(CODER_TOOLS, indent=2))
        return

    if args.file:
        content = open(args.file, encoding="utf-8", errors="replace").read()
        reads = []
        for chunk in (args.reads.split(",") if args.reads else []):
            a, b = chunk.split("-"); reads.append((int(a), int(b)))
        if not reads:
            reads = [(None, None)]   # whole-file read → def-index (nothing revealed)
        render_growing_view(content, reads, path=args.file)
        return

    if args.case == "grow":
        big = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(900))
        render_growing_view(big, [(1, 6), (1300, 1320), (1480, 1500), (1300, 1310)])
        return

    names = list(CASES) if args.case == "all" else [args.case]
    for nm in names:
        c = CASES[nm]()
        art = assemble(c[0], c[1], c[2], error_feedback=(c[3] if len(c) > 3 else ""))
        print_artifact(nm, art, stats_only=args.stats)


if __name__ == "__main__":
    main()
