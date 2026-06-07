# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What JARVIS is

A multi-brain AI agent that coordinates many **free/weak** LLMs (NVIDIA NIM, OpenRouter, DeepInfra, Groq, Gemini) to approximate frontier-model quality at $0, slower. The bet: several models plan independently, critique, and merge, beating one weak model thinking alone. The hardest and most actively developed part is the **coding agent** — a 4-role pipeline (UNDERSTAND → PLAN → IMPLEMENT → REVIEW) whose target is to score well on SWE-bench / SWE-bench Pro with weak models.

## Engineering doctrine — VERIFY THE RENDERED ARTIFACT (read this before touching anything)

The reachable standard for this project is **not "zero bugs ever"** — it is **every catchable bug caught in the first pass, offline, before a live run.** The recurring failure mode, proven over and over in this repo's own history, is *editing a string and never looking at the whole thing the model actually receives*: dead code (the legacy `PLAN_COT_EXISTING`/`MERGE_PROMPT_TEMPLATE` strings silently overwritten by their `_V8` versions — edits to them are no-ops), stale prompt anchors, leaked protocol tokens, an unclosed `[think]` that swallowed a correct plan. Each was invisible at edit time and cost a 30–60 min live run to surface. **We do not work that way.** Spending more tokens, more thinking, and more verification up front is *always* correct here — it is far cheaper than a wrong assumption found after the tenth run. Token budget is not a constraint; rigor is the constraint.

**THE RULE: before any live run, render and read the FULL assembled artifact the model actually receives — end to end.** Not the source string — the assembled result: system prompt + tool schemas + user turn + injected files + the growing view + reject/nudge messages, exactly as the API sees it. If you changed a prompt, a tool description, a view renderer, or a reject message, you render it and you read all of it. A bug you could have seen in the rendered artifact must never reach a live run.

Non-negotiables:
- **Edits must land on the LIVE artifact.** The `_V8` constants in `core/prompts_v8.py` are what run; same-named legacy strings elsewhere are DEAD. After any prompt/format edit, render the assembled prompt and confirm your change is *in it* (use `behavioral_audit/render_prompt.py`).
- **Render across MANY cases, not one.** A prompt is correct only across the shapes it meets in production: small/large files, a view with holes, post-edit diffs, rejects, multi-file steps, empty-turn retries, create-file. Render a representative spread and read them.
- **Offline-first.** A live SWE run (30–60 min, partial logs) is the WRONG loop for finding mechanics/prompt bugs. Reproduce offline in seconds (`render_prompt.py`, `behavioral_audit/bridge.py`, `jarvis_emu.py`, trace replay). Live runs are for *measuring resolution*, not debugging.
- **Capture the whole story.** When a run fails, dump the COMPLETE trace (every prompt, tool result, reject) and audit ALL causes at once — never grep one symptom and ship one fix.
- **Adversarial review BEFORE commit, not after the run.** Spawn independent agents to attack new prompts/code. A 2-minute adversarial read beats a 40-minute run that finds the same bug. Use the tokens.
- **No dead code, no stale instructions, no contradictions.** A model-facing instruction that contradicts the live format silently degrades a weak model — treat prompt/format drift as a P0 bug, not cosmetic.

## Commands

```bash
# Run the agent
python3 main.py            # terminal CLI (main pipeline entry)
python3 ui_main.py         # web UI (aiohttp WebSocket server + ui/index.html)

# Render the FULL artifact the coder model receives (offline, seconds) — DO THIS before any run
# that touches a prompt/tool-schema/view (see the doctrine above). Calls the real assembly fns.
python3 behavioral_audit/render_prompt.py            # all cases (small/big/multi/reject) — READ them
python3 behavioral_audit/render_prompt.py --stats    # section sizes only (sweep)
python3 behavioral_audit/render_prompt.py --case grow # the accumulating big-file "growing view"
python3 behavioral_audit/render_prompt.py --file <path> --reads 1300-1320,1480-1500  # real file view

# Tests (no pytest-asyncio — async tools are driven via asyncio.run inside tests)
python3 -m pytest tests_audit/test_native_tools.py -q        # native-coder regression suite
python3 -m pytest tests_audit/test_native_tools.py::test_name -q   # a single test
JARVIS_FUZZ_SCALE=1.0 python3 -m pytest tests_audit/test_native_tools_fuzz.py -q  # ~15k stability cases (scale tunable)

# SWE-bench evaluation (writes one JSON line per instance to the predictions file)
python3 swe_bench.py --instances-json behavioral_audit/<set>.json \
    --predictions preds_<name>.jsonl --parallel 1 --timeout 1800
# coder-only (skip UNDERSTAND/PLAN, reuse a cached plan) — much faster, isolates the coder:
JARVIS_PLAN_CACHE=behavioral_audit/plan_cache python3 swe_bench.py ...

# Real pass/fail grading (ScaleAI harness + local Docker, jefzda/sweap-images per instance)
behavioral_audit/grade_pro.sh <preds.jsonl> [num_workers]
# or directly: behavioral_audit/SWE-bench_Pro-os/swe_bench_pro_eval.py
#   --raw_sample_path=<csv> --patch_path=<patches.json> --use_local_docker --dockerhub_username=jefzda
```

API keys are read from env vars (no config file): `OPENROUTER_API_KEY`/`OPENROUTER_API_KEYS` (comma-list, round-robined), `NVIDIA_API_KEY`, `DEEPINFRA_API_KEY`, `LIGHTNING_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`/`GEMINI_API_KEYS`, `TAVILY_API_KEY`. A `venv/` exists (`venv/bin/python3`); the grading harness uses it explicitly.

## Architecture — the parts that span multiple files

**Two distinct coder paths share the same applier but speak different protocols.** This is the single most important thing to internalize before editing coder code:

- **Native (primary): `core/native_tools.py`** — `gpt-oss-120b` via structured **function calling** (`call_with_native_tools` loop, `_dispatch` to per-tool `_do_*` functions). Tools: read_file, edit_file, create_file, search_text, find_refs, find_callers, file_purpose, semantic_search, depends_on, run_code, finish.
- **Text protocol: `core/tool_call.py`** — fallback coders (qwen3-coder, glm-5.1) that don't do native function calling; they emit `[CODE:]`/`[edit]`/`[STOP]`/`[VIEW:]`/`[KEEP:]` bracket tags parsed from text. The **reviewer** also runs this protocol.

Both render file views and apply edits, but with **different gutter formats** — do not conflate them:
- Native view (`display_mode="prefix_ws"`): `LINENO ⇥INDENT|<real spaces>code` (the `⇥` marks the indent number). Edits: copy the view line verbatim for `old` (anchors on **both** line number AND content), write `new` as `INDENT|code` (the harness re-emits the leading spaces from the number).
- Text view (`display_mode="prefix"`/else): `LINENO:INDENT|code` (colon). Changing one coder's format must not touch the other's.

**The edit applier is shared.** Native `edit_file` and the text `[edit]` blocks both funnel through `workflows/code.py:_apply_extracted_code` (number-first, content-verified). Validation gates there: parse check (reject + revert on `SyntaxError`/`ValueError`) and an undefined-name/dangling-reference check (reject-and-revert — deleting a def while it's still called is blocked). `_expand_indent_lines` in `native_tools.py` turns the `INDENT|`/`LINENO ⇥INDENT|` forms back into real spaces.

**Provider routing: `clients/nvidia.py`.** `call_nvidia_tools` + `_route_provider` send the OpenAI-compatible request; `gpt-oss-120b` is pinned to DeepInfra@bf16 via OpenRouter's `provider` field. The coder fallback chain is `_CODER_CHAIN` in `workflows/code.py` (gpt-oss-120b native → qwen3-coder text → mistral/medium native → gpt-oss-nim native → glm-5.1 text); **gpt-oss-nim must stay last** (it hangs ~5 min then 504s instead of failing fast).

**Empty-turn handling (a recurring real failure mode).** DeepInfra intermittently returns `finish_reason=stop` with no structured `tool_calls` even under `tool_choice="required"` — it leaves the tool call as TEXT in the harmony reasoning channel. The native loop handles this in two layers: **salvage** (`_salvage_inline_tool_call` parses the leaked JSON, infers the tool, synthesizes the call — no extra API call) then a **retry** with a "emit a structured tool call" nudge. `tool_choice="required"` is *not* reliably enforced by all providers; some providers (Google Vertex) don't expose tools at all.

**Stability invariant:** `_dispatch(name, args, ctx)` must never raise for any tool × any args — a malformed/wrong-typed call returns a clean `✗` string the coder reacts to. `tests_audit/test_native_tools_fuzz.py` enforces this with generated cases; tool entry points coerce/validate via `_str_or_err`.

**Prompts: `core/prompts_v8.py`.** `IMPLEMENT_NATIVE_PROMPT_V8` (≈ lines 1902–1936) is the native coder's system prompt; `REVIEW_*`/`SELF_CHECK_*` are the reviewer's (text protocol). The native coder's full system prompt is `IMPLEMENT_NATIVE_PROMPT` plus an always-on indent-format block appended in `native_tools.py`. Keep model-facing instructions (schema descriptions, prompt prose, reject messages) consistent with the live edit format — stale instructions silently confuse weak coders.

**Code indexing & sandbox.** `tools/code_index.py` builds 3 maps (purpose / detailed / symbol) used by find_refs/file_purpose/semantic_search; `tools/codebase.py` renders views and runs searches (ripgrep); `tools/sandbox.py` is a read-only/no-network bwrap sandbox where edits land and `run_code` executes.

## Working in this repo (project conventions)

- **SWE-bench runs:** exactly one at a time; long-running. Kill background runs **by PID** — never `pkill`/`pgrep -f` a pattern that matches the current shell, and never `kill -- -PID` (both exit 144).
- **Never reuse an existing `preds_*.jsonl` filename** for a new run — the grader can read stale rows before `swe_bench.py` truncates it, poisoning the cumulative score. Use a fresh name; graded count must never exceed the preds line count.
- **Commit messages via `-F <file>`**, not `-m "..."` — backticks and `!` in `-m` get shell-command-substituted/expanded and mangle the message.
- `tests_audit/test_native_tools.py` should be fully green (88/88 as of ckpt-159). The three formerly-failing prompt-drift tests (`test_native_prompt_tools_match_schema`, `test_coder_prompts_ground_on_spec_literals`, `test_native_prompt_has_thinking_toolkit_reflexes`) were fixed in ckpt-159 — one was a false-failing regex; two needed dropped prompt anchors restored. A NEW failure in this suite is a real regression; don't dismiss it as pre-existing.
- A separate persistent memory lives under `memory/MEMORY.md` (mission state, hard-won gotchas) — distinct from this file.



to develop this, you need to be verry carfull, all the bug that you put will do delay, and make us do bad assuption, you need to test all your code with a lot of care, you can do test code to test for bug and spawn suagent to directly probe the code for bug and a lot of other technic. when you code use the mesur twice cut once technic, think more, to be shure you don't do misstake, and take the best decision. taking more time when coding will save us a lot of time at debugging and testing. 

if you see i am wrong or i did bad assumtion, call it out, if you see batter alternative, call it out, if you have doupt, ask me question, i don't just want you to always agree with me, it is better that you call the wrong assuption and wrong decision then to let them trough. don't be afraid to call me when you think i am wrong. 
