# JARVIS — a multi-brain coding agent built on free/cheap models

JARVIS coordinates a handful of **free and very cheap** LLMs — NVIDIA NIM, OpenRouter, DeepInfra, Groq, Gemini — to approximate frontier-model coding quality for **about 2¢ a task**, just slower. The bet behind it: instead of one weak model thinking alone, have several models plan independently, critique each other, merge the best plan, implement it, and then *verify the result by running it* — and push as much of the hard, error-prone bookkeeping as possible into a **deterministic harness** so the weak model only ever has to make a small, local decision.

The most developed part — and the part worth using — is the **coding agent ("Deep Code")**: a four-stage pipeline (UNDERSTAND → PLAN → IMPLEMENT → REVIEW) aimed at solving real GitHub issues on real repositories. It's good enough that, on a hard task, it produces a working patch a frontier model would be happy with — on models that individually are nowhere near that level.

---

## ⚠️ Honest status — what to actually use

This is a research project, not a finished product. Be selective:

- ✅ **Use the coding agent (Deep Code).** This is where almost all the engineering went. It plans, edits real files, runs its own checks, and ships a patch. It's the real deal.
- ⚠️ **Everything else works but is far from optimal** — general chat, web search, image gen, formatting, the conversational flow. They run, but they haven't had the attention the coder has.
- 🚫 **Don't rely on the auto-router.** The classifier that decides "is this chat / a question / a coding task" is the weakest link and will sometimes route a coding task to the wrong place. Skip it: go straight to Deep Code.

**The recommended way to use JARVIS:**

```bash
python ui_main.py      # opens the web UI
```

Then in the UI: paste your **free API keys** in Settings → API Keys, set your **project path** in Settings → Project, and run your coding task in **Deep Code** mode. That's the path that's been tuned and tested.

API keys are read from env vars (or the UI settings): `OPENROUTER_API_KEY(S)`, `NVIDIA_API_KEY`, `DEEPINFRA_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY(S)`. You need OpenRouter + NVIDIA at minimum; the rest add redundancy.

---

## How it works

A single free/weak model is decent but not frontier-level, and — more importantly — it's *unreliable*: it loses track of indentation, edits the wrong line, forgets which file it's in, reinvents a format. JARVIS's whole design is two ideas stacked together.

### 1. Multi-brain: plan independently, debate, merge

```
UNDERSTAND  →  PLAN  →  IMPLEMENT  →  REVIEW
              (4 drafts          (one coder,    (run a repro,
               → 1 merger)        function-      route a fix)
                                  calling)
```

- **PLAN** — four planners draft a plan *independently* and in parallel (Nemotron-3-Ultra, Owl-Alpha, Gemma-4, Nemotron-3-Super — deliberately diverse, open, non-frontier models). A merger (Owl-Alpha) then consolidates them into one plan, taking the most *correct and complete* approach rather than the longest. Disagreement between drafts is a feature: it surfaces the parts that are actually hard.
- **IMPLEMENT** — the coder (**gpt-oss-120b**) executes one plan step at a time through **native function calling** — `read_file`, `edit_file`, `search_text`, `find_refs`, `run_code`, `finish`, etc. — not by dumping a blob of text.
- **REVIEW (self-verify)** — after the patch, JARVIS *writes a small reproduction from the issue's own description, runs it*, and if it fails, routes a corrective fix back to the coder. It never reads the project's hidden test suite (that would be cheating); it tests what the issue actually says.

### 2. The harness computes the global state; the model only acts locally

This is the idea that makes weak models reliable. Anything global, stateful, or easy to get wrong is handled deterministically by the harness, so the model is left with a single small move:

- **Edits are number-first and content-verified.** The coder copies a line from the view (which carries both its line number *and* its content) and the harness applies the change by anchoring on both — so a stale line number self-corrects and a wrong anchor is rejected instead of silently corrupting the file.
- **Indentation is computed, not guessed.** The file view encodes each line's indent as a number; the harness re-emits the real whitespace. The model never counts spaces.
- **Edits run through safety gates** before they land: a parse check (reject + revert on a syntax error the edit introduced), an undefined-name / dangling-reference check (you can't delete a symbol that's still called), a duplicate-block guard, and an apply-time guard that **refuses to overwrite a working file with empty content** (recovering the real content from the sandbox if the in-memory copy was lost).
- **The code map, symbol lookups, stale-read detection, and "which file am I in" are all the harness's job.** The model asks; the harness answers with ground truth.

The slogan, from the project's own notes: *the harness computes the global, the model acts local.*

---

## What's been built (the last few hundred commits)

Much of this repo's history is hard-won, offline-verified fixes to make weak models behave. The big themes:

- **A reflex library for the coder.** Instead of vague advice, the coder's prompt carries concrete, *triggered* reflexes — fire-this-the-moment-it-applies rules drawn from real observed failures: read from the source you gated on (don't gate on `web.data()` then read `web.input()`); "all / every / collect" means *accumulate*, don't overwrite; produce the exact type/literal a test expects; bytes stay bytes until you decode them; use the stdlib serializer for wire formats instead of hand-rolling; remove a symbol and fix every call site in one edit; a missing third-party import is the environment, not your bug. These are woven into a step-by-step "how to think" reasoning flow and grouped by category so the model can find the one it needs. (A repeatedly-validated lesson: for weak models, *concrete* reflexes beat elegant abstract principles — when we tried replacing them with general principles, the score dropped.)
- **A self-verifying reviewer.** It authors a repro from the issue, runs it (local sandbox → host → the instance's real Docker image as needed), and routes a fix — under a strict *snapshot-and-revert invariant*: the review can only help or be neutral, **never ship a patch worse than the coder's original.** The repro author is shown the changed symbols' *signatures only* — not the implementation — so it can't be primed into rubber-stamping a buggy patch. Import-time crashes in the repro are treated as inconclusive rather than as bug signals.
- **Correct plan routing.** The planner now correctly distinguishes a *greenfield* build from *modifying an existing project* by actual file presence (a small existing project used to be misread as "new" and get rebuilt from scratch, drifting conventions and regressing adjacent behavior).
- **Robust provider routing.** gpt-oss-120b is pinned to DeepInfra's bf16 endpoint through OpenRouter's provider field; planners and reviewer run on free models with multi-provider fallback chains; API keys are round-robined, dead/over-quota keys are skipped automatically, and 402/429 responses are retried down the chain. The result is that a single provider hiccup doesn't sink a run.
- **Empty-turn recovery.** Some providers occasionally return no structured tool call even when one is required, leaving the call as text; the harness salvages the inline call and retries, so the coder doesn't silently stall.
- **A read-only, no-network sandbox** (bwrap) where all edits land and `run_code` executes, so nothing the agent does touches your real files until you approve.

---

## Cost

A typical Deep Code run costs **roughly 2¢**. The expensive part — the coder — runs on **gpt-oss-120b via DeepInfra (cheap, paid)**, while the planners, merger, and reviewer run on **free** model tiers, with free models also configured as fallbacks. So you're paying cents for the one model that does the heavy editing, and nothing for the rest.

(Earlier versions were fully $0 on free tiers; the current setup spends a little to get a much more reliable coder.)

---

## Benchmarks

JARVIS is developed against **SWE-bench Pro** — real GitHub issues, graded by actually running the project's hidden tests in Docker (the ScaleAI harness + per-instance images). Pro is multi-language (Python, Go, TypeScript/JavaScript) and ships the issue's behavioral spec, which is a fair, hard target.

We don't want to quote a cherry-picked number, so the honest statement is: on a hard hand-picked subset the agent reliably solves the genuinely-winnable instances and converges to working patches, with the residual failures being either oscillation on a couple of stochastic instances or contracts that aren't derivable from the issue text alone. **A full 75-instance run is in progress to measure the true SWE-bench score** — that number will go here when it lands.

Separately, as a real-world sanity check we had JARVIS build and then *iteratively extend* a small application from scratch over many rounds (add features, fix bugs, refactor), running the app after each round. It built a working CLI app, debugged a regression from just a symptom report, and shipped features end-to-end — and that exercise surfaced (and led to fixes for) bugs that pure benchmark runs never would, because benchmarks only read the final patch and never iterate on a living repo.

---

## Layout

- `main.py` — terminal CLI entry point
- `ui_main.py` — web UI (recommended)
- `workflows/code.py` — the coding pipeline (plan → implement → review)
- `core/native_tools.py` — the coder's function-calling loop and tools
- `core/self_verify.py` — the self-verifying reviewer
- `core/prompts_v8.py` — the live prompts (planner / coder / reviewer)
- `tools/` — code index, sandbox, codebase views
- `clients/` — provider routing and fallback

---

JARVIS is an experiment in getting frontier-ish coding out of models that, alone, aren't close — by making them collaborate and by doing the hard bookkeeping for them. The coding agent is the part that delivers on that today; the rest is catching up.
