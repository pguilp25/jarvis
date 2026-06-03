"""Generative STABILITY / fuzz suite for the native-tool coder (ckpt-149).

The native loop's hard contract: NOTHING the model emits — any tool, any garbage
arguments, any way it phrases its call — may crash the coder. Specifically:

  • _dispatch(name, args, ctx) must NEVER raise. For every (tool, args) it returns
    a usable value (a str the coder reacts to, or the ("__FINISH__", summary) tuple).
  • the pure parsers the coder's calls flow through — the inline-call salvage, the
    balanced-JSON extractor, the indent-expander, the view-lineno reader, the block
    locator — must handle ANY input without raising and without corrupting valid
    content.

This is a SEEDED fuzzer (deterministic → reproducible failures), generating ~15k
cases across all 11 tools, every malformation (missing/extra/null/wrong-type/huge/
unicode/nested), and every way a call can arrive (structured args, leaked-as-text,
the various edit line forms). Counts are tunable via JARVIS_FUZZ_SCALE.
"""
import asyncio
import os
import random
import shutil
import string
import tempfile

import pytest

from core.native_tools import (
    _dispatch, CODER_TOOLS,
    _salvage_inline_tool_call, _balanced_json_objs, _infer_tool_from_args,
    _expand_indent_lines, _view_lineno, _locate_block,
)

_SCALE = float(os.environ.get("JARVIS_FUZZ_SCALE", "1.0"))
def _n(base): return max(1, int(base * _SCALE))

TOOL_NAMES = [t["function"]["name"] for t in CODER_TOOLS]
# real tools + names a confused/native model reaches for + junk
ALL_NAMES = TOOL_NAMES + ["", " ", "replace_lines", "REPLACE", "edit", "grep",
                          "READ", "do_thing", "finish ", "read_file\n", "Edit_File",
                          "🙂", "a" * 200, None]

# value fragments the model realistically produces
_CODE = ["def foo():", "    return x", "x = 1", "import os", "class C:", "    pass",
         "ns.coll", "_is_fqcn(name)", "return all(...)", "}", "{", "[", "()",
         '"""doc"""', "# comment", "if x: {", "a = {'k': 1}", ""]
_EDIT_FORMS = ["4|def f():", "8|    return x", "286 ⇥4|    def setvalue", "0|x = 1",
               "286:4|x", "12:+    return 2", "  3:- old line", "def bare():",
               "", "   ", '{"a":1}', "返回值", "a" * 4000, "\n", "\t\tx", "⇥4|y"]
KEYS = ["path", "old", "new", "start_line", "end_line", "pattern", "symbol", "tag",
        "query", "command", "content", "summary", "hunks", "goal", "traced", "check",
        "random_key", "深", "", "PATH"]


def _scalar(rng):
    return rng.choice([
        None, True, False, 0, 1, -1, 999999, -999999, 2 ** 50,
        rng.randint(-10 ** 6, 10 ** 6), 1.5, -3.14, 1e308,
        "", " ", "x", rng.choice(_CODE), rng.choice(_EDIT_FORMS),
        "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 80))),
        "🙂" * 50, "\x00\x01\x02", "a" * 6000, "\n" * 30,
    ])


def _val(rng, depth=0):
    r = rng.random()
    if depth >= 4 or r < 0.55:
        return _scalar(rng)
    if r < 0.78:
        return [_val(rng, depth + 1) for _ in range(rng.randint(0, 6))]
    return {str(rng.choice(KEYS)): _val(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def _rargs(rng):
    """A random arguments dict (sometimes empty, sometimes valid-ish keys, sometimes junk)."""
    return {str(rng.choice(KEYS)): _val(rng) for _ in range(rng.randint(0, 6))}


def _mk_ctx(root):
    rel = "m.py"
    with open(os.path.join(root, rel), "w") as f:
        f.write("def greet(name):\n    return 'hi ' + name\n\nclass C:\n    def bump(self):\n        self.n += 1\n")
    return {"file_contents": {rel: open(os.path.join(root, rel)).read()},
            "sandbox": None, "viewed_versions": {}, "project_root": root,
            "purpose_map": "", "detailed_map": "", "files_changed": set(),
            "view_at": {}}


# ── 1. THE core invariant: _dispatch never raises, for any tool × any args ──────
def test_fuzz_dispatch_never_raises():
    rng = random.Random(20240603)
    root = tempfile.mkdtemp(prefix="fuzz_")
    N = _n(4000)
    failures = []

    # semantic_search makes real embedding network calls on a valid string query
    # (slow + needs auth) — its arg-handling is covered by the deterministic matrix
    # below (which never reaches the embed path), so keep it out of the random fuzz.
    pool = [n for n in ALL_NAMES if n != "semantic_search"]

    async def run():
        for i in range(N):
            name = rng.choice(pool)
            args = _rargs(rng)
            ctx = _mk_ctx(root)            # fresh ctx: edit_file mutates file_contents
            try:
                out = await _dispatch(name if name is not None else "", args, ctx)
            except Exception as e:           # noqa: BLE001 — ANY raise is a stability bug
                failures.append((name, args, type(e).__name__, str(e)[:160]))
                continue
            # contract: a str, or the finish sentinel tuple
            ok = isinstance(out, str) or (isinstance(out, tuple) and out and out[0] == "__FINISH__")
            if not ok:
                failures.append((name, args, "BAD_RETURN", repr(out)[:160]))
    try:
        asyncio.run(run())
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert not failures, (f"{len(failures)}/{N} dispatch calls raised/misbehaved. "
                          f"First 5:\n" + "\n".join(repr(f) for f in failures[:5]))


# ── 2. edit_file is the most complex tool — fuzz its args hard ─────────────────
def test_fuzz_edit_file_never_raises():
    rng = random.Random(777)
    N = _n(4000)
    src = "def greet(name):\n    return 'hi ' + name\n\nclass C:\n    def bump(self):\n        self.n += 1\n"
    failures = []

    async def run():
        for i in range(N):
            ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
                   "project_root": ".", "files_changed": set(), "view_at": {}}
            # mix flat {old,new} form and the legacy {hunks:[...]} form
            if rng.random() < 0.5:
                args = {"path": rng.choice(["m.py", "other.py", "", 5, None]),
                        "old": _val(rng), "new": _val(rng)}
            else:
                args = {"path": rng.choice(["m.py", "", None]),
                        "hunks": [{"old": _val(rng), "new": _val(rng),
                                   "start_line": rng.choice([None, 1, 2, -5, "3", 9999, 4.0])}
                                  for _ in range(rng.randint(0, 3))]}
            try:
                out = await _dispatch("edit_file", args, ctx)
                assert isinstance(out, str)
            except Exception as e:           # noqa: BLE001
                failures.append((args, type(e).__name__, str(e)[:160]))
    asyncio.run(run())
    assert not failures, (f"{len(failures)}/{N} edit_file calls raised. First 5:\n"
                          + "\n".join(repr(f) for f in failures[:5]))


# ── 3. the inline-call salvage must never raise & never false-salvage a non-tool ─
def test_fuzz_salvage_never_raises():
    rng = random.Random(42)
    N = _n(5000)
    frags = _CODE + _EDIT_FORMS + [
        '{"path":"x.py","start_line":40,"end_line":70}',
        '{"name":"read_file","arguments":{"path":"a.py"}}',
        '{"old":["if x: {"],"new":["y"],"path":"a.py"}', '{"command":"pytest -q"}',
        '{"pattern":"foo"}', '{"symbol":"Bar"}', '{', '}', '{"a":', '{{{', '}}}',
        '{"x": {"y": {"z": 1}}}', 'no json here at all', '{"summary":"done"}',
        '{"name":"rm_rf","arguments":{}}', '{"path":', '{"old": "not a list"}',
        '"path": "x"', "{'single':'quotes'}", '{"deep":[[[[[1]]]]]}',
    ]
    for i in range(N):
        text = " ".join(rng.choice(frags) for _ in range(rng.randint(0, 8)))
        if rng.random() < 0.3:
            text += "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 120)))
        try:
            r = _salvage_inline_tool_call(text, i)
        except Exception as e:               # noqa: BLE001
            pytest.fail(f"salvage raised on {text[:120]!r}: {type(e).__name__}: {e}")
        if r is not None:
            assert isinstance(r, dict)
            fn = r.get("function", {})
            assert fn.get("name") in TOOL_NAMES, f"salvaged a non-tool: {fn.get('name')!r}"
            assert isinstance(fn.get("arguments"), str)   # must be JSON string for the loop


# ── 4. balanced-JSON extractor: never raises, never hangs, returns valid substrings ─
def test_fuzz_balanced_json_objs_never_raises():
    rng = random.Random(99)
    for i in range(_n(3000)):
        text = "".join(rng.choice('{}[]"\'\\:,abc 0\n' + string.printable[:30])
                        for _ in range(rng.randint(0, 200)))
        try:
            objs = _balanced_json_objs(text)
        except Exception as e:               # noqa: BLE001
            pytest.fail(f"_balanced_json_objs raised on {text[:80]!r}: {e}")
        assert isinstance(objs, list)
        for o in objs:
            assert o.startswith("{") and o.endswith("}")


# ── 5. indent-expander: never raises, never DROPS or DUPLICATES lines ──────────
def test_fuzz_expand_indent_never_corrupts():
    rng = random.Random(2024)
    for i in range(_n(3000)):
        lines = [rng.choice(_EDIT_FORMS + _CODE + [
            str(rng.choice(KEYS)) + ": value", "443:  description", "    data[5:10]",
            "flags = 0o755 | 0o644", "5: 'value',", _scalar(rng) if rng.random() < .2 else "x"])
            for _ in range(rng.randint(0, 8))]
        lines = [l if isinstance(l, str) else str(l) for l in lines]
        try:
            out = _expand_indent_lines(lines)
        except Exception as e:               # noqa: BLE001
            pytest.fail(f"_expand_indent_lines raised on {lines!r}: {e}")
        assert isinstance(out, list) and len(out) == len(lines)  # 1:1, never drop/dup


# ── 6. view-lineno + locate_block: never raise on arbitrary input ──────────────
def test_fuzz_view_lineno_and_locate_never_raise():
    rng = random.Random(5)
    cur = ["def greet(name):", "    return x", "    return x", "class C:", "    pass"]
    for i in range(_n(2000)):
        raw = [rng.choice(_EDIT_FORMS + _CODE) for _ in range(rng.randint(0, 4))]
        try:
            _view_lineno(raw)
            _locate_block(cur, _expand_indent_lines(raw))
        except Exception as e:               # noqa: BLE001
            pytest.fail(f"view_lineno/locate raised on {raw!r}: {e}")


# ── 7. DETERMINISTIC exhaustive matrix: every tool × every malformation class ──
@pytest.mark.parametrize("name", TOOL_NAMES + ["totally_unknown"])
def test_each_tool_handles_every_malformation(name):
    src = "def f():\n    return 1\n"
    malformations = [
        {},                                              # no args at all
        {"path": None}, {"path": 123}, {"path": ""},     # bad path
        {"path": "m.py"},                                # path only
        {"path": "m.py", "old": None, "new": None},
        {"path": "m.py", "old": "str-not-list", "new": "str-not-list"},
        {"path": "m.py", "old": [], "new": []},
        {"path": "m.py", "old": [1, 2, 3], "new": [None]},
        {"path": "m.py", "hunks": "not-a-list"},
        {"path": "m.py", "hunks": [None, 5, {}]},
        {"start_line": "x", "end_line": []},
        {"pattern": None}, {"symbol": 5}, {"tag": []}, {"query": None},
        {"command": None}, {"content": 5}, {"summary": None},
        {"random": "junk", "深": [1, {"x": None}]},
        {"path": "m.py", "old": ["a" * 5000], "new": ["b" * 5000]},
        {"path": "../../etc/passwd", "old": ["x"], "new": ["y"]},  # path traversal attempt
    ]

    async def run():
        for args in malformations:
            ctx = {"file_contents": {"m.py": src}, "sandbox": None, "viewed_versions": {},
                   "project_root": ".", "files_changed": set(), "view_at": {}}
            try:
                out = await _dispatch(name, args, ctx)
            except Exception as e:           # noqa: BLE001
                pytest.fail(f"{name}({args}) RAISED {type(e).__name__}: {e}")
            assert isinstance(out, str) or (isinstance(out, tuple) and out[0] == "__FINISH__"), \
                f"{name}({args}) returned non-string: {out!r}"
    asyncio.run(run())
