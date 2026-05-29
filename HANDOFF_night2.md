# Night 2 handoff — prompt engineering + rigorous diagnosis (2026-05-28→29)

Branch `overnight-stability`, local commits only. Suite 15,559 green throughout.
Goal: make JARVIS (weak FREE models) perform near frontier via prompts; test on
SWE-bench AND real app-building (anti-overfit); iterate.

## Headline result (read the ⚠ CORRECTION at the bottom — it supersedes this)
On the 3 hard instances the FULL-PIPELINE number SWINGS with planner luck (0/3 to
2/3 on the SAME code across runs) — it is NOT a stable metric. The real, PROVEN,
stable win is the CODER under isolation: django-on-gold-plan 60%→100% (plan-
adherence self-check). The bottleneck is PLANNER SCOPE CONSISTENCY. See bottom.

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
THE bottleneck is PLANNER SCOPE CONSISTENCY — the planner names the right files
INCONSISTENTLY run-to-run (matplotlib scoped cbook.py in one run, figure.py in
another; pylint hits __init__/main or 2-of-4 pyreverse files). When the plan is
right the rest of the pipeline resolves; when it's mis-scoped, it fails. The coder
(given a correct plan) is now reliable (see CORRECTION). So end-to-end resolve is
gated — and made noisy — by planner scope.

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

## Precise planner scope-hit (JARVIS_PLAN_ONLY, 2 runs/instance) — sharper diagnosis
gold: django=storage.py | matplotlib=cbook.py | pylint=diagrams+inspector+utils+writer
- django 2/2: always names storage.py (+ css/test NOISE). Scope FINE → its failures
  are plan STRUCTURE (merger "collect-all") + coder, not scope.
- matplotlib 2/2: cbook.py present both runs. Scope mostly fine; ckpt-30's
  figure.py-only was an unlucky drop.
- pylint 2/2 hits 3/4 (diagrams+inspector+writer) but ALWAYS MISSES utils.py
  (+ noise: __init__/diadefslib/main).

=> Refined: the planner's SCOPE is fairly consistent (gold file usually present).
   End-to-end noise is more plan-STRUCTURE + coder than scope roulette.

ACTIONABLE: pylint's utils.py miss is specific — utils.py holds NEW functions that
don't exist yet, so the planner can't REFS/find them; exploration never surfaces
the file. The TEST references utils.get_annotation, so test-derived scope (#2 in
core/plan_scope.py) SHOULD flag it.

## ckpt 33 — ROOT-CAUSED + FIXED the pylint utils.py miss
Traced it end to end (NOT the sibling-filter I first guessed — that check actually
PASSES, utils.py is same-dir as the scoped inspector.py):
- Gold test `tests/unittest_pyreverse_writer.py` does a clean module import
  `from pylint.pyreverse.utils import get_annotation, get_visibility, infer_node`
  (verified against the SWE-bench test_patch — even the PRE-patch test imports
  pyreverse.utils). So imported_modules→modules_to_files maps it to utils.py and
  _same_pkg(utils.py)=True. The backstop SHOULD add it.
- REAL BUG: the backstop ranks candidate tests by stem-match to pick which 12 to
  LOAD, but matched generic scope stems (`__init__`, `main` — both in pylint's
  scope) as WHOLE-PATH substrings. On pylint (hundreds of test files) nearly every
  path contains 'main'/'init' → ranking collapses to alphabetical → the real gold
  test sits past the cap and is NEVER LOADED → utils.py never surfaces.
- FIX (committed): extracted `plan_scope.rank_relevant_tests()` — rank by BASENAME
  against MEANINGFUL stems only (≥4 chars; drop main/__init__/base/utils/… catch-
  alls); cap 12→16. Pure, unit-tested (pylint scenario pinned). suite 15,562 green.
- VALIDATION (honest): the deterministic unit test proves the ranking fix surfaces
  the gold test #1 within the cap WHEN the writer/inspector/diagrams files are in
  scope. BUT a live JARVIS_PLAN_ONLY re-run did NOT surface utils.py — because that
  run's PLANNER scoped only 2 files (diagrams.py + inspector.py), DROPPING writer.py
  (the ckpt-32 sample had 6 incl. writer). The utils-importing gold test is the
  WRITER test; with writer.py unscoped, its basename doesn't match the meaningful
  stems {diagrams,inspector}, so it isn't prioritized. CONCLUSION: ckpt-33 fixes a
  GENUINELY BROKEN ranking (whole-path substring let main/__init__ rank everything),
  but it is NOT a silver bullet for pylint — the dominant blocker is UPSTREAM
  PLANNER SCOPE VARIANCE (this 3rd run scoped 2 files and wrote NO STEPs for them).
  The backstop can only amplify a signal when the right scope file is already
  present. So the lever remains: make the PLANNER's own scope consistent. The
  ranking fix is correct hygiene that helps on the runs where scope is right.
## ckpt 34 — FULL pylint root-cause chain (traced end to end from the live log)
The pylint planout was a 978-char PREAMBLE with 0 STEPs. Reading the live log gave
the COMPLETE causal chain (each layer ruled in/out by evidence, not guessed):
1. NOT the scope-backstop ranking (ckpt-33 fixed a real ranking bug — correct
   hygiene — but it can only amplify a signal when the right scope file is present).
2. NOT plan-selection: the substance-over-structure fallback (code.py ~8784) FIRED
   correctly; it ranked all 3 drafts + merger + salvage and the 978-char merger
   preamble still won → meaning EVERY layer-1 draft was ALSO thin/preamble.
3. THE MECHANISM: the merger (`_call_with_tools max_rounds=6`) spent ALL 6 rounds
   INVESTIGATING (reading files) and on its final round was STILL requesting
   inspector.py ("I need to understand … before I can merge. Let me investigate the
   key files. [tool use][PURPOSE: inspector.py][/tool use][STOP]") → ran out of
   rounds → never emitted the plan. ([STOP] does NOT end the loop — it runs tools +
   continues; max_rounds is what ended it.)
4. THE FLOOR: pylint-4551 (multi-file + NEW functions get_annotation/infer_node) is
   at/over the weak models' PLANNING-CAPABILITY ceiling.

FIX (ckpt 34, committed): merger FORCED-COMMIT. When the merged plan is unusable,
make ONE forced NO-TOOLS merger turn that must emit === PLAN === + ### STEPs,
BEFORE the draft-salvage fallback. Gated on the existing unusable test → never
touches the healthy django/matplotlib path. Prompt leads with the POSITIVE
directive after observing the glm-5.1 fallback RUMINATE on "do NOT use tools"
instead of writing.

HONEST VALIDATION: the forced-commit fired + gated correctly, but did NOT rescue
pylint-4551 — mistral/large was rate-limited so the call fell to glm-5.1, which
can't one-shot this hard instance (it looped on the meta-instruction). So ckpt-34
is a SOUND, GENERAL, gated robustness fix (helps when the model just needs a commit
nudge and mistral/large is up) — NOT a proven pylint win. pylint-4551 needs a
stronger planner model, not more prompt surgery.

RANKED NEXT LEVERS (post-ckpt-34):
1. django merger plan STRUCTURE (collect-all vs collect-adjustable) — coder-isolation
   shows the coder is reliable GIVEN a correct plan, so getting the merger to emit
   the right structure is the django lever. Validate via JARVIS_PLAN_CACHE + a
   structure check.
2. A STRONGER planner/merger model for hard multi-file instances (pylint class) —
   the weak free models hit a real ceiling there; no prompt fix clears it.
3. App test-quality nudge (generated tests must run as written / match the impl
   contract).
All planner/merger changes: validate via PLAN_ONLY (~45-300s/run, free-model speed
varies) BEFORE trusting; they are the regression-prone area.
