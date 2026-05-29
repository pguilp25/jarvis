"""Native tool-use LOOP behavior tests (core.native_tools.call_with_native_tools).

Drives the loop with a SCRIPTED fake model (canned assistant messages) instead of
a live LLM, so loop-control behavior is testable fast and hermetically:
read-before-edit recovery, no-explicit-finish exit, budget exhaustion, malformed
tool args, and transient-API-error retry/give-up. This is the fast-iteration
harness for hardening the native coder's stability.
"""
import asyncio
import json
import os
import shutil
import tempfile
import textwrap

import core.native_tools as nt
from tools.sandbox import Sandbox

SRC = textwrap.dedent('''\
    def greet(name):
        return "hello " + name
''')


def _mk_ctx():
    root = tempfile.mkdtemp(prefix="natloop_")
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


def _tc(call_id, name, **args):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _msg(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


class _ScriptedModel:
    """Pops one canned assistant message per call. Records the messages it was
    handed so tests can assert the loop fed tool results back."""
    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []
        self.calls = 0

    async def __call__(self, model_id, messages, tools, **kw):
        self.calls += 1
        self.seen_messages.append([dict(m) for m in messages])
        if not self.script:
            return _msg("(no more script)")     # graceful: ends loop
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _run(script, ctx, monkeypatch=None, **loop_kw):
    model = _ScriptedModel(script)
    # call_with_native_tools does `from clients.nvidia import call_nvidia_tools`
    # lazily inside _call_tools_with_retry, so patching the attribute works.
    import clients.nvidia as cn
    orig = getattr(cn, "call_nvidia_tools", None)
    cn.call_nvidia_tools = model
    # No real backoff sleeps in tests.
    orig_sleep = nt.asyncio.sleep
    async def _fast_sleep(*a, **k):
        return None
    nt.asyncio.sleep = _fast_sleep
    try:
        res = asyncio.run(nt.call_with_native_tools(
            "nvidia/gpt-oss-120b", "sys", "user", ctx, **loop_kw))
    finally:
        if orig is not None:
            cn.call_nvidia_tools = orig
        nt.asyncio.sleep = orig_sleep
    return res, model


# ── happy path: read → edit → finish ─────────────────────────────────────────
def test_loop_read_edit_finish():
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            _msg(tool_calls=[_tc("1", "read_file", path=rel)]),
            _msg(tool_calls=[_tc("2", "replace_lines", path=rel,
                                 start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(tool_calls=[_tc("3", "finish", summary="changed greeting")]),
            # finish-with-edits triggers ONE self-check pass → finish again
            _msg(tool_calls=[_tc("3b", "finish", summary="verified")]),
        ]
        res, model = _run(script, ctx)
        assert res["done"] is True
        assert res["files_changed"] == [rel]
        assert ctx["file_contents"][rel].split("\n")[1] == '    return "hi " + name'
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── edit-before-read is rejected, then the model recovers ────────────────────
def test_loop_edit_before_read_recovers():
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            # blind edit (no prior read) → rejected by viewed_versions gate
            _msg(tool_calls=[_tc("1", "replace_lines", path=rel,
                                 start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(tool_calls=[_tc("2", "read_file", path=rel)]),
            _msg(tool_calls=[_tc("3", "replace_lines", path=rel,
                                 start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(tool_calls=[_tc("4", "finish")]),
            _msg(tool_calls=[_tc("4b", "finish")]),   # self-check pass → finish again
        ]
        res, model = _run(script, ctx)
        assert res["done"] is True
        assert res["files_changed"] == [rel]
        # the first (blind) edit's tool result must be a rejection string
        # surfaced back into the conversation
        flat = [m for conv in model.seen_messages for m in conv]
        tool_results = [m["content"] for m in flat if m.get("role") == "tool"]
        assert any(r.startswith("✗") for r in tool_results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── model stops emitting tool calls (no explicit finish) ─────────────────────
def test_loop_exits_on_no_tool_call_but_keeps_edit():
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            _msg(tool_calls=[_tc("1", "read_file", path=rel)]),
            _msg(tool_calls=[_tc("2", "replace_lines", path=rel,
                                 start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(content="I'm done, the greeting is updated."),   # no tool_calls
        ]
        res, model = _run(script, ctx)
        # current contract: done=False (no explicit finish) but edit is captured
        assert res["done"] is False
        assert res["files_changed"] == [rel]
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── budget exhaustion: model never finishes ──────────────────────────────────
def test_loop_budget_exhaustion():
    ctx, rel, root = _mk_ctx()
    try:
        # always asks to read, never finishes
        script = [_msg(tool_calls=[_tc(str(i), "read_file", path=rel)])
                  for i in range(20)]
        res, model = _run(script, ctx, max_rounds=3)
        assert res["done"] is False
        assert res["rounds"] == 3
        assert model.calls == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── malformed tool arguments must not crash the loop ─────────────────────────
def test_loop_malformed_arguments_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        bad = {"id": "1", "type": "function",
               "function": {"name": "read_file", "arguments": "{not valid json"}}
        script = [
            _msg(tool_calls=[bad]),                # → args {} → ✗ string, no crash
            _msg(tool_calls=[_tc("2", "finish")]),
        ]
        res, model = _run(script, ctx)
        assert res["done"] is True   # loop survived the malformed call
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── transient API error retried, then succeeds ──────────────────────────────
def test_loop_transient_error_then_success():
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            RuntimeError("NVIDIA openai/gpt-oss-120b:free HTTP 503: bad gateway"),
            _msg(tool_calls=[_tc("1", "finish")]),
        ]
        res, model = _run(script, ctx)
        assert res["done"] is True
        assert model.calls == 2   # one failure + one success
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── non-transient API error gives up cleanly (no crash, no edits) ────────────
def test_loop_hard_error_gives_up():
    ctx, rel, root = _mk_ctx()
    try:
        script = [RuntimeError("NVIDIA HTTP 400: malformed request")]
        res, model = _run(script, ctx)
        assert res["done"] is False
        assert res["files_changed"] == []
        assert res["reason"] == "api-error"
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── non-object tool arguments must not crash the loop ────────────────────────
def test_loop_nonobject_arguments_no_crash():
    ctx, rel, root = _mk_ctx()
    try:
        # arguments is a JSON scalar (not an object) — would AttributeError on
        # args.get(...) without the isinstance guard
        bad = {"id": "1", "type": "function",
               "function": {"name": "read_file", "arguments": "5"}}
        script = [_msg(tool_calls=[bad]), _msg(tool_calls=[_tc("2", "finish")])]
        res, model = _run(script, ctx)
        assert res["done"] is True
        # the bad call's tool result tells the model its args weren't an object
        flat = [m for conv in model.seen_messages for m in conv]
        tool_results = [m["content"] for m in flat if m.get("role") == "tool"]
        assert any("not a valid JSON object" in r for r in tool_results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── an executor that raises must degrade to a ✗ result, not kill the loop ────
def test_loop_executor_exception_survived():
    ctx, rel, root = _mk_ctx()
    try:
        # force search_text's executor to blow up
        import core.native_tools as _nt
        orig = _nt._do_search
        async def _boom(args, c):
            raise RuntimeError("kaboom in executor")
        _nt._do_search = _boom
        try:
            script = [_msg(tool_calls=[_tc("1", "search_text", pattern="x")]),
                      _msg(tool_calls=[_tc("2", "finish")])]
            res, model = _run(script, ctx)
        finally:
            _nt._do_search = orig
        assert res["done"] is True   # loop survived the executor raise
        flat = [m for conv in model.seen_messages for m in conv]
        tool_results = [m["content"] for m in flat if m.get("role") == "tool"]
        assert any("failed internally" in r for r in tool_results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── reason field distinguishes outcomes ──────────────────────────────────────
def test_loop_reason_finished():
    ctx, rel, root = _mk_ctx()
    try:
        res, _ = _run([_msg(tool_calls=[_tc("1", "finish")])], ctx)
        assert res["reason"] == "finished"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_reason_budget():
    ctx, rel, root = _mk_ctx()
    try:
        script = [_msg(tool_calls=[_tc(str(i), "read_file", path=rel)]) for i in range(10)]
        res, _ = _run(script, ctx, max_rounds=2)
        assert res["reason"] == "budget-exhausted"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_breaker_escalates_repeated_failing_edit():
    ctx, rel, root = _mk_ctx()
    try:
        # the same blind edit (no prior read → rejected by the viewed_versions
        # gate) sent twice, then finish. The 2nd reject must carry an escalation.
        bad = _tc("x", "replace_lines", path=rel, start_line=2, end_line=2,
                  new_content='4|return "hi " + name')
        script = [_msg(tool_calls=[bad]), _msg(tool_calls=[bad]),
                  _msg(tool_calls=[_tc("f", "finish")])]
        res, model = _run(script, ctx)
        flat = [m for conv in model.seen_messages for m in conv]
        tool_results = [m["content"] for m in flat if m.get("role") == "tool"]
        assert any("EXACT" in r and "rejected" in r for r in tool_results), tool_results
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_nudge_fires_once_before_finish():
    # coder edits then finishes → must be nudged for ONE self-check pass before
    # the finish is accepted, then finish again is honored.
    ctx, rel, root = _mk_ctx()
    try:
        ctx["viewed_versions"][rel] = ctx["file_contents"][rel]
        edit = _tc("e", "replace_lines", path=rel, start_line=2, end_line=2,
                   new_content='4|return "hi " + name')
        script = [_msg(tool_calls=[edit]),
                  _msg(tool_calls=[_tc("f1", "finish")]),   # 1st finish → nudged
                  _msg(tool_calls=[_tc("f2", "finish")])]   # 2nd finish → accepted
        res, model = _run(script, ctx)
        assert res["done"] is True
        assert model.calls == 3   # the nudge forced the extra round
        flat = [m for conv in model.seen_messages for m in conv]
        assert any("SELF-CHECK before you finish" in str(m.get("content"))
                   for m in flat if m.get("role") == "user")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_nudge_skipped_when_no_edits():
    # finishing with ZERO edits must NOT be nudged (nothing to self-check).
    ctx, rel, root = _mk_ctx()
    try:
        res, model = _run([_msg(tool_calls=[_tc("f", "finish")])], ctx)
        assert res["done"] is True
        assert model.calls == 1   # no extra round
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_reason_empty_turn():
    ctx, rel, root = _mk_ctx()
    try:
        res, _ = _run([_msg(content="")], ctx)   # empty assistant turn
        assert res["reason"] == "empty-turn"
        assert res["done"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)
