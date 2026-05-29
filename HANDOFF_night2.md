# Night 2 handoff — prompt engineering + rigorous diagnosis (2026-05-28→29)

Branch `overnight-stability`, local commits only. Suite 15,559 green throughout.
Goal: make JARVIS (weak FREE models) perform near frontier via prompts; test on
SWE-bench AND real app-building (anti-overfit); iterate.

## Headline result
On the 3 hard instances (django-14053, matplotlib-25332, pylint-4551):
- **ckpt-26 baseline (regression reverted + Mistral-Large merger): 0/3 resolved.**
- **ckpt-27 (lean prompt pass): 2/3 resolved** (django + matplotlib). A real,
  measured improvement from prompt engineering alone.

## What landed
- **ckpt-26**: reverted the merger-prompt overhaul that caused the django/pylint
  regression (forensically proven: heavy prompt → glm-5.1 emits empty plan →
  bad salvage). Merger model → **mistral/large** (fallback glm-5.1). Salvage
  capped (8K). Native loop-breaker hard-stops a stuck coder.
- **ckpt-27**: a 15-agent design workflow (proposers → synthesize → adversarial
  anti-bloat/anti-overfit vet) → applied 17 surgical CUTS (prompts ~186 lines
  leaner): merger dedup; planner cut SWE-bench war-stories + dup-of-SYSTEM_KNOWLEDGE;
  coder dedup, merged Q-checks → Q-IMPACT, cut niche SWE-bench checks, UNGATED the
  SCENARIO-TRACE self-check to all task shapes, fixed a native-coder instruction
  CONTRADICTION ("read before editing" vs "already loaded"). Contract tokens intact.
- **ckpt-28**: component-isolation harnesses (the key methodology):
  - CODER isolation: `JARVIS_PLAN_CACHE=behavioral_audit/plan_cache` + gold plans
    → run the coder against a KNOWN-GOOD plan (isolates coder from planner).
  - PLAN isolation: `JARVIS_PLAN_ONLY=<dir>` → stop after planning, write plan +
    file-scope → iterate planner/merger prompt against the SCOPE metric, no coder.

## The diagnosis (hard evidence, this is the valuable part)
TWO real, separable levers — proven by isolation, not guessed:
1. **Planning scope/correctness.** django + matplotlib scoped the right file →
   resolved. **pylint mis-scopes** (hits __init__/main or wrong utils path, never
   the 4 pyreverse logic files) → fails, every run. The planner is the limiter here.
2. **Coder detail-consistency on very-subtle fixes.** Given the GOLD plan:
   matplotlib resolves reliably; **django resolves ~50%** (1 of 2 isolation runs) —
   the exact yield-ordering structure is hard, and the coder sometimes deviates
   from the plan's nuance ("collect ALL" vs "collect only adjustable"). The
   ungated self-check fired but didn't reliably catch it.

## App-building (anti-overfit)
The app LOGIC is built correctly; the GENERATED TESTS are buggy/inconsistent:
- Greenfield: app sound, generated test cwd'd to tmp_path where todo.py isn't
  copied (errno 2) → verify=1.
- Feature: 5 original tests PASS (feature didn't break anything); the 3 NEW
  generated tests fail because the test asserts numeric priority (`== 0`) while
  the code (correctly, per the spec's "high/med/low") uses strings. Test ⊥ code.
=> Gap #3: JARVIS writes correct app code but inconsistent/unrunnable TESTS.
   Addressable via a coder nudge (tests must match the implementation's contract
   and run as written); respect the lean law; validate before keeping.

## Where to push next (ranked)
1. **Coder plan-adherence on subtle structural fixes** — strengthen the self-check
   to verify the patch matches the PLAN's stated structure (would catch django's
   collect-all deviation). Validate with coder-isolation (django consistency).
2. **Planner scope for multi-file/data-flow fixes** (pylint) — hardest; iterate
   with PLAN_ONLY against the scope metric. Beware: heavy merger prompts regress.
3. **App test-quality** — nudge generated tests to be runnable as written.

## Reproduce
- SWE run: keys from ~/.bashrc, `JARVIS_PREFER_OPENROUTER=1`, HF offline,
  `swe_bench.py --instance-ids <ids> --parallel 1 --predictions X.jsonl`.
- Coder isolation: add `JARVIS_PLAN_CACHE=$PWD/behavioral_audit/plan_cache`.
- Eval (Docker): `REQUESTS_CA_BUNDLE=$(certifi)`, `run_evaluation --dataset_name
  princeton-nlp/SWE-bench_Verified --clean False --max_workers 2`.
- Reviewer stays OFF (JARVIS_ENABLE_REVIEW unset).

## ⚠ CORRECTION / honest final read (after full-trio re-measures)
The full-pipeline resolve number on these 3 instances is DOMINATED BY PLANNER
VARIANCE, not a stable improvement. Same code scored 2/3 (ckpt-27) and 0/3
(ckpt-30) on different runs — the swing is the planner inconsistently scoping
(ckpt-30: matplotlib scoped lib/matplotlib/figure.py, the WRONG file; gold is
cbook.py, which ckpt-27 scoped correctly). So do NOT read "0/3→2/3" as a fixed win.

WHAT IS PROVEN AND STABLE (measured under ISOLATION, which is why isolation was
built — the full pipeline is too noisy to measure a change):
- The CODER, given a correct plan, is now RELIABLE: django-on-gold-plan went
  60% (3/5) → 100% (4/4) after the plan-adherence self-check (ckpt-30). Validated
  A/B; matplotlib within its noise. The coder executes a correct plan correctly.
- THE BOTTLENECK IS PLANNER SCOPE CONSISTENCY. It names the right files
  inconsistently (matplotlib cbook.py vs figure.py; pylint 2/4 vs wrong). The
  coder cannot rescue a wrong/mis-scoped plan.

NEXT LEVER (clear, but do it carefully): planner/merger SCOPE CONSISTENCY. Use the
JARVIS_PLAN_ONLY harness to measure "does the plan name the gold files?" across N
runs per instance, iterate the planner prompt against THAT metric (not the noisy
end-to-end resolve), and A/B it. CAUTION: planner/merger prompt changes caused
tonight's regression — keep them lean, validate via PLAN_ONLY before trusting.

DURABLE DELIVERABLE: the component-isolation methodology (coder via
JARVIS_PLAN_CACHE + gold plans; planner via JARVIS_PLAN_ONLY). It lets you measure
and improve ONE component at a time even when the end-to-end metric is noise.
