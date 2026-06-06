"""JSON-ops coder tests (core.native_tools._parse_json_ops / _json_ops_stop /
call_with_json_ops — flag JARVIS_JSON_OPS).

The JSON-ops path is gpt-oss in TEXT mode (no native tool-calling): it emits FLAT
JSON-line ops, one per line, terminated by a `done` op. These tests pin two things:
  1. the PARSER is tolerant by design (weak model) and NEVER raises, across the
     full zoo of shapes a weak model actually emits (fences, <think>, prose,
     concatenation, nesting reversion, garbage);
  2. the LOOP drives a scripted streaming model through read→edit→verify→done,
     honours the verify gate, nudges then bails on no-ops, and stops when stuck.
"""
import asyncio
import json
import os
import shutil
import tempfile
import textwrap

import core.native_tools as nt
from tools.sandbox import Sandbox


# ── parser shapes ────────────────────────────────────────────────────────────

def _ops_names(ops):
    return [o["tool"] for o in ops]


def test_parse_clean_lines():
    txt = ('{"tool":"read_file","args":{"path":"a.py"}}\n'
           '{"tool":"search_text","args":{"pattern":"foo"}}')
    ops, done, summary = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file", "search_text"]
    assert ops[0]["args"] == {"path": "a.py"}
    assert done is False and summary == ""


def test_parse_done_op():
    ops, done, summary = nt._parse_json_ops(
        '{"tool":"done","args":{"summary":"fixed the bug"}}')
    assert ops == []
    assert done is True
    assert summary == "fixed the bug"


def test_parse_ops_then_done_same_response():
    txt = ('{"tool":"read_file","args":{"path":"a.py"}}\n'
           '{"tool":"done","args":{"summary":"ok"}}')
    ops, done, summary = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file"]
    assert done is True and summary == "ok"


def test_parse_top_level_done_flag():
    ops, done, summary = nt._parse_json_ops('{"done": true, "summary": "all set"}')
    assert done is True
    assert summary == "all set"


def test_parse_strips_think_block():
    txt = ('<think>I should read the file first, then edit it.</think>\n'
           '{"tool":"read_file","args":{"path":"a.py"}}')
    ops, done, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file"]
    # a `{...}` that lives only inside <think> must NOT be parsed as an op
    txt2 = '<think>maybe {"tool":"finish"} ? no.</think>\n{"tool":"read_file","args":{"path":"a.py"}}'
    ops2, _, _ = nt._parse_json_ops(txt2)
    assert _ops_names(ops2) == ["read_file"]


def test_parse_salvages_ops_leaked_into_think():
    # gpt-oss harmony empty-turn: the WHOLE response leaked into the reasoning
    # channel. Visible part is empty → salvage the ops from inside <think>.
    txt = '<think>I will read the file.\n{"tool":"read_file","args":{"path":"a.py"}}</think>'
    ops, done, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file"]
    # but when there IS a visible op, a different op merely CONSIDERED in <think>
    # must NOT be salvaged (visible wins, no fallback)
    txt2 = ('<think>maybe {"tool":"create_file","args":{"path":"z.py","content":""}} ? no.</think>\n'
            '{"tool":"read_file","args":{"path":"a.py"}}')
    ops2, _, _ = nt._parse_json_ops(txt2)
    assert _ops_names(ops2) == ["read_file"]


def test_parse_strips_code_fences():
    txt = ('```json\n'
           '{"tool":"read_file","args":{"path":"a.py"}}\n'
           '{"tool":"edit_file","args":{"path":"a.py","edits":[]}}\n'
           '```')
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file", "edit_file"]


def test_parse_concatenated_no_newlines():
    txt = '{"tool":"read_file","args":{"path":"a.py"}}{"tool":"read_file","args":{"path":"b.py"}}'
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file", "read_file"]
    assert ops[1]["args"]["path"] == "b.py"


def test_parse_prose_between_objects():
    txt = ('Sure, here is what I will do.\n'
           '{"tool":"read_file","args":{"path":"a.py"}}\n'
           'and then I will search:\n'
           '{"tool":"search_text","args":{"pattern":"x"}}\n'
           'done for now.')
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file", "search_text"]


def test_parse_nested_ops_array_flattened():
    # model reverts to the nested form — flatten it rather than drop everything
    txt = '{"ops":[{"tool":"read_file","args":{"path":"a.py"}},{"tool":"search_text","args":{"pattern":"y"}}]}'
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file", "search_text"]


def test_parse_nested_calls_array_flattened():
    txt = '{"calls":[{"tool":"read_file","args":{"path":"a.py"}}]}'
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["read_file"]


def test_parse_malformed_object_skipped():
    # first object is broken JSON (trailing comma / missing brace); second is valid
    txt = ('{"tool":"read_file","args":{"path":}}\n'
           '{"tool":"search_text","args":{"pattern":"ok"}}')
    ops, _, _ = nt._parse_json_ops(txt)
    assert _ops_names(ops) == ["search_text"]


def test_parse_flat_no_args_wrapper():
    # model puts args inline rather than under "args"
    ops, _, _ = nt._parse_json_ops('{"tool":"read_file","path":"a.py"}')
    assert _ops_names(ops) == ["read_file"]
    assert ops[0]["args"] == {"path": "a.py"}


def test_parse_name_alias_for_tool():
    ops, _, _ = nt._parse_json_ops('{"name":"read_file","args":{"path":"a.py"}}')
    assert _ops_names(ops) == ["read_file"]


def test_parse_garbage_returns_empty():
    for junk in ["", None, "no json here at all", "}{ broken", "[1,2,3]", 42, "{not json"]:
        ops, done, summary = nt._parse_json_ops(junk)
        assert ops == [] and done is False and summary == ""


def test_parse_caps_runaway_op_list():
    txt = "\n".join('{"tool":"read_file","args":{"path":"f%d.py"}}' % i for i in range(60))
    ops, _, _ = nt._parse_json_ops(txt)
    assert len(ops) == 24            # bounded


def test_parse_never_raises_fuzz():
    import random
    random.seed(7)
    frags = ['{', '}', '"tool"', ':', '"read_file"', '"args"', '[', ']', ',',
             '"path"', '\\', '"', 'done', 'true', '<think>', '</think>', '```', '\n', ' ']
    for _ in range(2000):
        s = "".join(random.choice(frags) for _ in range(random.randint(0, 40)))
        nt._parse_json_ops(s)        # must not raise


# ── _coalesce_edit_ops (flat edit ops → merged edits batch) ──────────────────

def test_coalesce_flat_single_edit_becomes_edits_list():
    ops = [{"tool": "edit_file", "args": {"path": "f.py", "old": "1 ⇥0|a", "new": "0|b"}}]
    out = nt._coalesce_edit_ops(ops)
    assert len(out) == 1
    assert out[0]["args"]["path"] == "f.py"
    assert out[0]["args"]["edits"] == [{"old": "1 ⇥0|a", "new": "0|b"}]


def test_coalesce_merges_same_path_keeps_order():
    ops = [
        {"tool": "read_file", "args": {"path": "a.py"}},
        {"tool": "edit_file", "args": {"path": "f.py", "old": "1 ⇥0|a", "new": "0|b"}},
        {"tool": "edit_file", "args": {"path": "g.py", "old": "1 ⇥0|x", "new": "0|y"}},
        {"tool": "edit_file", "args": {"path": "f.py", "old": "5 ⇥0|c", "new": "0|d"}},
    ]
    out = nt._coalesce_edit_ops(ops)
    assert [o["tool"] for o in out] == ["read_file", "edit_file", "edit_file"]
    # f.py's two hunks merged into the FIRST f.py op (order preserved: f before g)
    assert out[1]["args"]["path"] == "f.py"
    assert len(out[1]["args"]["edits"]) == 2
    assert out[2]["args"]["path"] == "g.py"
    assert len(out[2]["args"]["edits"]) == 1


def test_coalesce_preserves_explicit_edits_list():
    ops = [{"tool": "edit_file", "args": {"path": "f.py",
            "edits": [{"old": ["1 ⇥0|a"], "new": ["0|b"]}, {"old": ["5 ⇥0|c"], "new": ["0|d"]}]}}]
    out = nt._coalesce_edit_ops(ops)
    assert len(out[0]["args"]["edits"]) == 2


def test_coalesce_passes_through_non_edits_and_pathless():
    ops = [{"tool": "search_text", "args": {"pattern": "x"}},
           {"tool": "edit_file", "args": {"new": "0|b"}}]   # no path
    out = nt._coalesce_edit_ops(ops)
    assert out[0]["tool"] == "search_text"
    assert out[1]["tool"] == "edit_file" and out[1]["args"].get("path") is None


# ── loop tests (scripted streaming model) ────────────────────────────────────

SRC = textwrap.dedent('''\
    def greet(name):
        return "hello " + name
''')


def _mk_ctx():
    root = tempfile.mkdtemp(prefix="jsonops_")
    rel = "m.py"
    with open(os.path.join(root, rel), "w") as f:
        f.write(SRC)
    sb = Sandbox(root)
    sb.setup()
    sb.load_file(rel)
    ctx = {"file_contents": {rel: SRC}, "sandbox": sb, "project_root": root,
           "viewed_versions": {}, "purpose_map": "", "detailed_map": "",
           "files_changed": set()}
    return ctx, rel, root


class _ScriptedStream:
    """Pops one canned text reply per call. call_with_json_ops calls
    call_nvidia_stream(...) and expects a STRING back (the visible content)."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen = []

    async def __call__(self, model_id, prompt="", system="", messages_override=None,
                       stop_check=None, max_tokens=0, log_label="", **kw):
        self.calls += 1
        # deep snapshot per call — messages_override is the LIVE list and keeps
        # mutating after the call, so a reference would only show the final state
        self.seen.append([dict(m) for m in (messages_override or [])])
        if not self.script:
            return ""                       # exhausted → no ops → nudge/bail
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _run_json(script, ctx, **loop_kw):
    model = _ScriptedStream(script)
    import clients.nvidia as cn
    orig = getattr(cn, "call_nvidia_stream", None)
    cn.call_nvidia_stream = model
    try:
        res = asyncio.run(nt.call_with_json_ops(
            "nvidia/gpt-oss-120b", "sys", "user", ctx, **loop_kw))
    finally:
        if orig is not None:
            cn.call_nvidia_stream = orig
    return res, model


def test_loop_read_edit_verify_done():
    ctx, rel, root = _mk_ctx()
    try:
        edit = ('{"tool":"edit_file","args":{"path":"%s","edits":'
                '[{"old":["2 ⇥4|    return \\"hello \\" + name"],'
                '"new":["4|    return \\"hi \\" + name"]}]}}' % rel)
        script = [
            '{"tool":"read_file","args":{"path":"%s"}}' % rel,   # round 1: look
            edit,                                                # round 2: edit
            '{"tool":"done","args":{"summary":"changed greeting"}}',  # round 3: done → verify gate
            '{"tool":"done","args":{"summary":"verified"}}',     # round 4: done accepted
        ]
        res, model = _run_json(script, ctx)
        assert res["done"] is True
        assert res["reason"] == "finished"
        assert rel in res["files_changed"]
        assert ctx["file_contents"][rel].split("\n")[1] == '    return "hi " + name'
        assert model.calls == 4             # the verify gate forced the extra round
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_flat_edit_applies():
    # the FLAT edit form (old/new as strings, no edits array) must apply end-to-end
    ctx, rel, root = _mk_ctx()
    try:
        flat_edit = ('{"tool":"edit_file","args":{"path":"%s",'
                     '"old":"2 ⇥4|    return \\"hello \\" + name",'
                     '"new":"4|    return \\"hi \\" + name"}}' % rel)
        script = [
            '{"tool":"read_file","args":{"path":"%s"}}' % rel,
            flat_edit,
            '{"tool":"done","args":{"summary":"flat edit"}}',
            '{"tool":"done","args":{"summary":"verified"}}',
        ]
        res, _ = _run_json(script, ctx)
        assert res["done"] is True
        assert rel in res["files_changed"]
        assert ctx["file_contents"][rel].split("\n")[1] == '    return "hi " + name'
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_two_flat_edits_same_file_one_round_both_apply():
    # a 2-site change emitted as TWO flat edit ops in ONE round → coalesced → both land
    from tools.sandbox import Sandbox
    src = 'def f():\n    a = 1\n    b = 2\n    return a + b\n'
    root = tempfile.mkdtemp(prefix="jsonops2_")
    rel = "m.py"
    with open(os.path.join(root, rel), "w") as fh:
        fh.write(src)
    sb = Sandbox(root); sb.setup(); sb.load_file(rel)
    ctx = {"file_contents": {rel: src}, "sandbox": sb, "project_root": root,
           "viewed_versions": {rel: src}, "purpose_map": "", "detailed_map": "",
           "files_changed": set()}
    try:
        e1 = '{"tool":"edit_file","args":{"path":"%s","old":"2 ⇥4|    a = 1","new":"4|    a = 10"}}' % rel
        e2 = '{"tool":"edit_file","args":{"path":"%s","old":"3 ⇥4|    b = 2","new":"4|    b = 20"}}' % rel
        script = ['{"tool":"read_file","args":{"path":"%s"}}' % rel,
                  e1 + "\n" + e2,
                  '{"tool":"done","args":{"summary":"two sites"}}',
                  '{"tool":"done","args":{"summary":"ok"}}']
        res, _ = _run_json(script, ctx)
        body = ctx["file_contents"][rel]
        assert "a = 10" in body and "b = 20" in body, body
        assert res["done"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_multi_op_round():
    ctx, rel, root = _mk_ctx()
    try:
        # two read ops in ONE round — both must run, results fed back. (The two
        # `done`s cover the no-edit nudge: a done after only lookups is pushed back
        # once, then accepted.)
        script = [
            '{"tool":"read_file","args":{"path":"%s"}}\n'
            '{"tool":"search_text","args":{"pattern":"greet"}}' % rel,
            '{"tool":"done","args":{"summary":"looked"}}',
            '{"tool":"done","args":{"summary":"looked, no change needed"}}',
        ]
        res, model = _run_json(script, ctx)
        assert res["done"] is True
        # round 1 ran 2 ops; the RESULTS turn fed into round 2 must list both.
        # (Inspect round 2's input snapshot — seen[1] — not the final state.)
        round2_users = [m for m in model.seen[1] if m["role"] == "user"]
        results_msg = round2_users[-1]["content"]
        assert "op[1]" in results_msg and "op[2]" in results_msg
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_no_ops_nudges_then_bails():
    ctx, rel, root = _mk_ctx()
    try:
        res, model = _run_json(["just chatting, no json", "still nothing", "nope"], ctx)
        assert res["done"] is False
        assert res["reason"] == "no-ops"
        assert model.calls == 3             # round1 + 2 nudges
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_no_edit_done_nudged_then_accepted():
    # a done with ZERO edits is NOT a real completion — nudge ONCE to make the
    # change; if the coder insists (done again), accept it (fail-soft).
    ctx, rel, root = _mk_ctx()
    try:
        res, model = _run_json(
            ['{"tool":"done","args":{"summary":"nothing to do"}}',
             '{"tool":"done","args":{"summary":"confirmed: no change needed"}}'], ctx)
        assert res["done"] is True
        assert res["reason"] == "finished"
        assert model.calls == 2          # round1 done → nudge → round2 done accepted
        assert res["files_changed"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_api_error_breaks_clean():
    ctx, rel, root = _mk_ctx()
    try:
        res, _ = _run_json([RuntimeError("boom")], ctx)
        assert res["done"] is False
        assert res["reason"] == "api-error"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_stuck_on_repeated_rejects():
    # an edit_file for a file never read → rejected every round; after 12 rejects, stop
    ctx, rel, root = _mk_ctx()
    try:
        bad = ('{"tool":"edit_file","args":{"path":"%s","edits":'
               '[{"old":["9 ⇥4|    nonexistent"],"new":["4|x"]}]}}' % rel)
        res, model = _run_json([bad] * 20, ctx)
        assert res["done"] is False
        assert res["reason"] == "stuck-repeating"
        assert res["files_changed"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_failing_lookups_do_not_trip_stuck_bail():
    # P1-2: a big speculative LOOK round with many ✗ lookups (semantic_search is
    # unavailable here → every one ✗) must NOT trip the stuck-repeating bail —
    # only EDIT-op rejects count. The step should proceed to finish, not fall over.
    ctx, rel, root = _mk_ctx()
    try:
        many = "\n".join('{"tool":"semantic_search","args":{"query":"q%d"}}' % i
                         for i in range(15))
        res, _ = _run_json([many,
                            '{"tool":"done","args":{"summary":"nothing applicable"}}',
                            '{"tool":"done","args":{"summary":"confirmed"}}'], ctx)
        assert res["reason"] != "stuck-repeating"
        assert res["done"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_finish_tool_alias_triggers_done():
    # model emits a `finish` op (native-style) instead of `done` → still ends
    # (second finish clears the no-edit nudge, same as a second done)
    ctx, rel, root = _mk_ctx()
    try:
        res, _ = _run_json(['{"tool":"finish","args":{"summary":"done via finish"}}',
                            '{"tool":"finish","args":{"summary":"confirmed"}}'], ctx)
        assert res["done"] is True
        assert res["reason"] == "finished"
    finally:
        shutil.rmtree(root, ignore_errors=True)
