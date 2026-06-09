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
            return _msg("(no more script)")     # graceful: ends loop (no-tool-call)
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
            _msg(tool_calls=[_tc("2", "finish")]),  # nudged (zero edits)
            _msg(tool_calls=[_tc("3", "finish")]),  # accepted
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
            _msg(tool_calls=[_tc("1", "finish")]),   # nudged (zero edits, ckpt-121)
            _msg(tool_calls=[_tc("2", "finish")]),   # accepted
        ]
        res, model = _run(script, ctx)
        assert res["done"] is True
        # one failure + two finishes (first nudged, second accepted)
        assert model.calls == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── non-transient API error gives up cleanly (no crash, no edits) ────────────
def test_loop_hard_error_gives_up():
    ctx, rel, root = _mk_ctx()
    try:
        # A permanent error (HTTP 400) on every call → the loop gives up.
        # (gpt-oss-120b pins ONE endpoint now; the chain ORDER is in code.)
        script = [RuntimeError("NVIDIA HTTP 400: malformed request")] * 3
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
        script = [_msg(tool_calls=[bad]),
                  _msg(tool_calls=[_tc("2", "finish")]),   # nudged (zero edits)
                  _msg(tool_calls=[_tc("3", "finish")])]   # accepted
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
                      _msg(tool_calls=[_tc("2", "finish")]),   # nudged (zero edits)
                      _msg(tool_calls=[_tc("3", "finish")])]   # accepted
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
        # ckpt-121: a zero-edit finish is nudged once; the second finish is accepted.
        res, _ = _run([_msg(tool_calls=[_tc("1", "finish")]),
                       _msg(tool_calls=[_tc("2", "finish")])], ctx)
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
        # ckpt-224 Cluster H routes an EXPLICIT `finish` tool call through the same
        # verify gate; the nudge is then delivered as the TOOL result answering that
        # finish call (API-correct — a tool_call must be answered), not a free `user`
        # message. Accept it in either role. (bughunt ckpt-242: test was stale.)
        assert any("SELF-CHECK before you finish" in str(m.get("content"))
                   for m in flat if m.get("role") in ("user", "tool"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_edit_finish_is_nudged_once_then_accepted():
    # ckpt-121: finishing with ZERO edits is a BAIL, not a completion — the FIRST
    # such finish is rejected + nudged ("make the change first"); a SECOND finish is
    # accepted (fail-soft → the chain falls over to the next coder). So one finish does
    # NOT complete the step; two finishes do, taking one extra round.
    ctx, rel, root = _mk_ctx()
    try:
        res, model = _run([_msg(tool_calls=[_tc("f", "finish")]),
                           _msg(tool_calls=[_tc("f2", "finish")])], ctx)
        assert res["done"] is True
        assert model.calls == 2   # nudged once, accepted on the second finish
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_bullet_cot_flag_gates_prompt_in_native_loop():
    # ckpt-185: JARVIS_BULLET_COT=1 appends the tight-bullets style block to the
    # native coder's system prompt; flag off → byte-for-byte absent.
    ctx, rel, root = _mk_ctx()
    try:
        _, model = _run([_msg(tool_calls=[_tc("f", "finish")]),
                         _msg(tool_calls=[_tc("f2", "finish")])], ctx)
        sys_off = model.seen_messages[0][0]["content"]
        assert "REASONING STYLE — tight bullets" not in sys_off
    finally:
        shutil.rmtree(root, ignore_errors=True)
    ctx, rel, root = _mk_ctx()
    os.environ["JARVIS_BULLET_COT"] = "1"
    try:
        _, model = _run([_msg(tool_calls=[_tc("f", "finish")]),
                         _msg(tool_calls=[_tc("f2", "finish")])], ctx)
        sys_on = model.seen_messages[0][0]["content"]
        assert "REASONING STYLE — tight bullets" in sys_on
        assert "correctness always beats brevity" in sys_on
        # the always-on INDENT block must still be there (bullets append, never replace)
        assert "## INDENTATION" in sys_on
    finally:
        os.environ.pop("JARVIS_BULLET_COT", None)
        shutil.rmtree(root, ignore_errors=True)


def test_loop_reason_empty_turn():
    ctx, rel, root = _mk_ctx()
    try:
        # Persistent empty assistant turn: round 1 + 2 retries all empty.
        # (One scripted msg isn't enough — the model's exhaustion sentinel
        # "(no more script)" would leak into `final` and flip the terminal
        # reason to no-tool-call. Supply enough empties to cover the retries.)
        res, _ = _run([_msg(content=""), _msg(content=""), _msg(content="")], ctx)
        assert res["reason"] == "empty-turn"
        assert res["done"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_hard_stops_on_repeated_identical_reject():
    # same rejected edit 3x → hard-stop (fall over) instead of burning the budget
    ctx, rel, root = _mk_ctx()
    try:
        bad = _tc("x", "replace_lines", path=rel, start_line=2, end_line=2,
                  new_content='4|return "hi " + name')  # no prior read → rejected
        script = [_msg(tool_calls=[bad]) for _ in range(10)]   # would spin forever
        res, model = _run(script, ctx, max_rounds=16)
        assert res["reason"] == "stuck-repeating"
        assert res["rounds"] < 16   # stopped early, didn't burn the budget
        assert model.calls <= 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_keep_evicts_old_view_from_history():
    # ckpt-179: keep(path, ranges) rewrites the file's prior view messages in history down to
    # just the kept ranges, so context shrinks. Drive the real loop and inspect what the model
    # was handed on the call AFTER keep.
    root = tempfile.mkdtemp(prefix="natloopkeep_")
    big = "".join(f"def f{i}():\n    return {i}\n" for i in range(800))   # 1600 lines (>1000)
    rel = "big.py"
    with open(os.path.join(root, rel), "w") as f:
        f.write(big)
    sb = Sandbox(root); sb.setup(); sb.load_file(rel)
    ctx = {"file_contents": {rel: big}, "sandbox": sb, "project_root": root,
           "viewed_versions": {}, "purpose_map": "", "detailed_map": "", "files_changed": set(),
           "step_num": 1, "_first_seen": {}}
    try:
        script = [
            _msg(tool_calls=[_tc("1", "read_file", path=rel)]),                     # def-index
            _msg(tool_calls=[_tc("2", "read_file", path=rel, start_line=10, end_line=30)]),  # a range
            _msg(tool_calls=[_tc("3", "keep", path=rel, ranges=[[10, 30]])]),       # keep → evict
            _msg(tool_calls=[_tc("4", "finish", summary="done")]),
        ]
        res, model = _run(script, ctx)
        # the finish call (last) saw the rewritten history
        final_msgs = model.seen_messages[-1]
        blob = "\n".join(str(m.get("content")) for m in final_msgs if m.get("role") == "tool")
        assert "⟪KEPT only lines 10-30" in blob          # the view was rewritten to kept ranges
        assert "TOO LARGE" not in blob                    # the big def-index view is gone
        assert "10 ⇥0|def f4" in blob or "11 ⇥" in blob   # kept range content is present
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_loop_batch_runs_lookups_then_edits():
    # ckpt-181: a `batch` of reads runs in ONE round, its results feed the next round, then a
    # normal edit lands. Verifies the loop handles a batch tool_call end-to-end (results fed
    # back, no API-pairing break, reject-counter not tripped).
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            _msg(tool_calls=[_tc("1", "batch", calls=[
                {"tool": "read_file", "args": {"path": rel}},
                {"tool": "find_refs", "args": {"symbol": "greet"}}])]),
            _msg(tool_calls=[_tc("2", "replace_lines", path=rel, start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(tool_calls=[_tc("3", "finish", summary="done")]),
            _msg(tool_calls=[_tc("3b", "finish", summary="verified")]),
        ]
        res, model = _run(script, ctx)
        # the round-2 (edit) call saw the batch results in its history
        after_batch = model.seen_messages[1]
        blob = "\n".join(str(m.get("content")) for m in after_batch if m.get("role") == "tool")
        assert "op[1] read_file" in blob and "greet" in blob   # both lookups returned, one round
        assert res["done"] is True and res["files_changed"] == [rel]  # then the edit landed
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── A1: read-spin backstop (ckpt-205) ────────────────────────────────────────
def test_read_spin_backstop_breaks_on_reads_without_edit():
    # 25 read ops, never an edit → the loop must STOP (reason='read-budget') around 20 reads
    # instead of burning the whole round budget exploring (the a26/f327 pre-edit storm).
    ctx, rel, root = _mk_ctx()
    try:
        script = [_msg(tool_calls=[_tc(str(i), "read_file", path=rel)]) for i in range(25)]
        res, model = _run(script, ctx, max_rounds=40)
        assert res["reason"] == "read-budget", res["reason"]
        assert not res["files_changed"]                 # nothing edited
        assert model.calls <= 21                         # stopped ~20, didn't run all 40 rounds
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_spin_backstop_not_tripped_once_an_edit_lands():
    # an edit early resets the counter → many subsequent reads do NOT trip read-budget (a
    # productive multi-edit step is bounded by the wall-clock A2, never killed by A1).
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            _msg(tool_calls=[_tc("0", "read_file", path=rel)]),                       # read first
            _msg(tool_calls=[_tc("e", "replace_lines", path=rel, start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),              # edit lands
        ] + [_msg(tool_calls=[_tc(str(i), "read_file", path=rel)]) for i in range(25)] \
          + [_msg(tool_calls=[_tc("d", "finish", summary="done")]),
             _msg(tool_calls=[_tc("d2", "finish", summary="verified")])]
        res, model = _run(script, ctx, max_rounds=40)
        assert res["reason"] != "read-budget", res["reason"]   # the edit reset the spin counter
        assert res["files_changed"]                            # the edit landed
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── A2: wall-clock deadline (ckpt-205) ───────────────────────────────────────
def test_wall_clock_deadline_stops_cleanly():
    # a deadline already in the past → the loop stops on the FIRST round with reason='time-budget'
    # (and never calls the model), so a slow step can't consume the whole instance timeout.
    import time
    ctx, rel, root = _mk_ctx()
    try:
        script = [_msg(tool_calls=[_tc("1", "read_file", path=rel)])]   # never reached
        res, model = _run(script, ctx, max_rounds=40, deadline=time.monotonic() - 1)
        assert res["reason"] == "time-budget", res["reason"]
        assert model.calls == 0                          # bailed before any model call
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_deadline_is_byte_identical_default():
    # deadline=None (the default) must not change behavior — a normal read→edit→finish still works.
    ctx, rel, root = _mk_ctx()
    try:
        script = [
            _msg(tool_calls=[_tc("1", "read_file", path=rel)]),
            _msg(tool_calls=[_tc("2", "replace_lines", path=rel, start_line=2, end_line=2,
                                 new_content='4|return "hi " + name')]),
            _msg(tool_calls=[_tc("3", "finish", summary="done")]),
            _msg(tool_calls=[_tc("3b", "finish", summary="verified")]),
        ]
        res, model = _run(script, ctx)                   # no deadline kwarg
        assert res["reason"] in ("finished", "budget-exhausted")
        assert res["files_changed"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
