export const meta = {
  name: 'f12-fail-diagnose-and-hunt',
  description: 'Diagnose the 5 fresh12 FAILS (gold vs produced) + round-by-round bug hunt across all 12',
  phases: [
    { title: 'Diagnose', detail: 'per fail: gold-vs-produced → why it failed, winnable, the lever' },
    { title: 'Hunt', detail: 'one agent per slice across all 12 → harness bugs, all severities' },
    { title: 'Synthesize', detail: 'per-fail verdicts + deduped harness bugs ranked by run cost' },
  ],
}

const DIAG = [
  { tag: 'f631cd44', diag: '/tmp/f12fails/f631cd44_diag.txt' },
  { tag: 'dbbd9d53', diag: '/tmp/f12fails/dbbd9d53_diag.txt' },
  { tag: '111347e9', diag: '/tmp/f12fails/111347e9_diag.txt' },
  { tag: '395e5e20', diag: '/tmp/f12fails/395e5e20_diag.txt' },
  { tag: 'b748edea', diag: '/tmp/f12fails/b748edea_diag.txt' }
]
const HUNT = [
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/coder_step_1.txt' },
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/coder_step_2.txt' },
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/coder_step_3.txt' },
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/prompt_round0.txt' },
  { tag: 'qute_f91ace96', path: '/tmp/f12all/qute_f91ace96_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'ansi_f327e65d', path: '/tmp/f12all/ansi_f327e65d_groups/coder_step_1.txt' },
  { tag: 'ansi_f327e65d', path: '/tmp/f12all/ansi_f327e65d_groups/coder_step_2.txt' },
  { tag: 'ansi_f327e65d', path: '/tmp/f12all/ansi_f327e65d_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'ansi_f327e65d', path: '/tmp/f12all/ansi_f327e65d_groups/prompt_round0.txt' },
  { tag: 'ansi_f327e65d', path: '/tmp/f12all/ansi_f327e65d_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'open_4a5d2a7d', path: '/tmp/f12all/open_4a5d2a7d_groups/coder_step_1.txt' },
  { tag: 'open_4a5d2a7d', path: '/tmp/f12all/open_4a5d2a7d_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'open_4a5d2a7d', path: '/tmp/f12all/open_4a5d2a7d_groups/prompt_round0.txt' },
  { tag: 'open_4a5d2a7d', path: '/tmp/f12all/open_4a5d2a7d_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'qute_c580ebf0', path: '/tmp/f12all/qute_c580ebf0_groups/coder_step_1.txt' },
  { tag: 'qute_c580ebf0', path: '/tmp/f12all/qute_c580ebf0_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'qute_c580ebf0', path: '/tmp/f12all/qute_c580ebf0_groups/prompt_round0.txt' },
  { tag: 'qute_c580ebf0', path: '/tmp/f12all/qute_c580ebf0_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_1.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_2.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_3.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_4.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_5.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_6.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_7.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_8.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/coder_step_9.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/prompt_round0.txt' },
  { tag: 'ansi_a26c325b', path: '/tmp/f12all/ansi_a26c325b_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'open_dbbd9d53', path: '/tmp/f12all/open_dbbd9d53_groups/coder_step_1.txt' },
  { tag: 'open_dbbd9d53', path: '/tmp/f12all/open_dbbd9d53_groups/coder_step_2.txt' },
  { tag: 'open_dbbd9d53', path: '/tmp/f12all/open_dbbd9d53_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'open_dbbd9d53', path: '/tmp/f12all/open_dbbd9d53_groups/prompt_round0.txt' },
  { tag: 'open_dbbd9d53', path: '/tmp/f12all/open_dbbd9d53_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/coder_step_1.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/coder_step_2.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/coder_step_3.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/prompt_round0.txt' },
  { tag: 'qute_f631cd44', path: '/tmp/f12all/qute_f631cd44_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/coder_step_1.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/coder_step_2.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/coder_step_3.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/prompt_round0.txt' },
  { tag: 'ansi_395e5e20', path: '/tmp/f12all/ansi_395e5e20_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/coder_step_1.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/coder_step_2.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/coder_step_3.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/prompt_round0.txt' },
  { tag: 'open_111347e9', path: '/tmp/f12all/open_111347e9_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'qute_96b99780', path: '/tmp/f12all/qute_96b99780_groups/coder_step_1.txt' },
  { tag: 'qute_96b99780', path: '/tmp/f12all/qute_96b99780_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'qute_96b99780', path: '/tmp/f12all/qute_96b99780_groups/prompt_round0.txt' },
  { tag: 'qute_96b99780', path: '/tmp/f12all/qute_96b99780_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/coder_step_1.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/coder_step_2.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/coder_step_3.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/coder_step_4.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/prompt_round0.txt' },
  { tag: 'ansi_b748edea', path: '/tmp/f12all/ansi_b748edea_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/coder_step_1.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/coder_step_2.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/coder_step_3.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/planner_merging_plans__final__owl-alpha.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/prompt_round0.txt' },
  { tag: 'open_8a5a63af', path: '/tmp/f12all/open_8a5a63af_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' }
]

const ALREADY_FIXED = `Recently FIXED (do NOT re-report): JSON-OPS read-view supersession + the quadratic context
blowup; intra-round dup-read dedup; the empty-turn storm; the read-refusal trap; search_text path
scope (now with a **/ glob anchor) + quote strip + searching the EDITED sandbox; run_code read-only
__pycache__ OSError labelling + python -B hint; docstring-insert warning; the merger CORE
double-prepend; dead-merger no-PLAN nudge; scaffold-hallucination dash tolerance; per-plan merge
truncation; def-index gap cap scaled to gap size; verify-gate skip when run_code already passed;
blank-line INSERT anchor; [DETAIL:]/[LS:] added to the tool table; <think>/[SYSTEM NOTE] stripped
from merge inputs; degenerate-draft filter; dangling-ref double-indent; plan-line-number caveat.`

const DIAG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['classification','winnable','missed','harness_cause','fix'],
  properties: {
    classification: { type: 'string', enum: ['under-scoped','wrong-edit','incomplete-edit','unwinnable','other'] },
    winnable: { type: 'boolean', description: 'could a better harness plausibly have solved it (NOT unwinnable)?' },
    missed: { type: 'string', description: 'the SPECIFIC scope/site/symbol the produced patch missed or got wrong vs gold' },
    harness_cause: { type: 'string', description: 'the harness lever: planner under-scoped? coder dropped a step? edit landed wrong? or none (model incapacity / unwinnable)' },
    fix: { type: 'string', description: 'concrete harness change that would help (or "none — unwinnable")' },
  },
}

phase('Diagnose')
const diagnoses = await parallel(DIAG.map(d => () =>
  agent(`You are diagnosing WHY a fresh12 SWE-bench-Pro instance FAILED grading. Read this file FULLY:
  ${d.diag}
It has the PROBLEM, REQUIREMENTS, INTERFACE, fail_to_pass tests, the GOLD reference patch, and the
JARVIS PRODUCED patch (which FAILED). Compare gold vs produced precisely: what scope/site/symbol did
the produced patch MISS or get WRONG? Classify: under-scoped (missed whole files/symbols the gold
changed), incomplete-edit (right files, missing pieces), wrong-edit (edited but incorrectly),
unwinnable (the gold bundles an incidental refactor the hidden test pins, unsolvable from the issue
alone), or other. State whether a better HARNESS could plausibly have solved it, and name the
specific harness lever (planner under-scoped the plan? coder skipped/half-did a step?). Be precise and
quote the exact missing/wrong lines.`, { label: `diag:${d.tag}`, phase: 'Diagnose', schema: DIAG_SCHEMA, agentType: 'Explore' })
    .then(v => ({ tag: d.tag, verdict: v })).catch(() => null)))
log(`Diagnosed ${diagnoses.filter(Boolean).length}/${DIAG.length} fails`)

const FIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['findings'],
  properties: { findings: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['title','severity','where','evidence','why_bug','fix'],
    properties: {
      title: { type: 'string' }, severity: { type: 'string', enum: ['large','medium','small'] },
      where: { type: 'string', description: 'best-guess JARVIS source file+function' },
      evidence: { type: 'string' }, why_bug: { type: 'string' }, fix: { type: 'string' },
    } } } },
}

phase('Hunt')
const hunted = await parallel(HUNT.map(h => () =>
  agent(`Round-by-round HARNESS bug hunt on ONE slice of a fresh12 JSON-OPS coder/planner trace
(instance ${h.tag}). The CODER is JSON-OPS (gpt-oss text mode emits flat {"tool":...,"args":...} ops;
harness runs each, feeds results back, until {"tool":"done"}). Read FULLY, round by round:
  ${h.path}
Find HARNESS bugs at EVERY severity (large/medium/small) — the harness fed WRONG/STALE/MISLEADING
input or mishandled output: a wasted round, a confusing message, an off-by-one, a stale prompt
phrase, a dropped op, a mis-scoped read, a plan that under-covers, a contradictory instruction.
${ALREADY_FIXED}
Do NOT report pure model-incapacity on CORRECT input. Quote verbatim evidence + best-guess source
location. Empty list if clean.`, { label: `hunt:${h.tag}`, phase: 'Hunt', schema: FIND_SCHEMA }).catch(() => null)))
const raw = hunted.filter(Boolean).flatMap(a => a.findings || [])
log(`Hunt: ${raw.length} raw findings across ${hunted.filter(Boolean).length} slices`)

phase('Synthesize')
const SYN = {
  type: 'object', additionalProperties: false, required: ['fail_verdicts','bugs','summary'],
  properties: {
    summary: { type: 'string' },
    fail_verdicts: { type: 'array', items: { type:'object', additionalProperties:false,
      required:['tag','classification','winnable','missed','fix'],
      properties:{ tag:{type:'string'}, classification:{type:'string'}, winnable:{type:'boolean'},
        missed:{type:'string'}, fix:{type:'string'} } } },
    bugs: { type: 'array', items: { type:'object', additionalProperties:false,
      required:['title','severity','where','evidence','fix'],
      properties:{ title:{type:'string'}, severity:{type:'string'}, where:{type:'string'},
        evidence:{type:'string'}, fix:{type:'string'} } } },
  },
}
const synth = await agent(`Lead auditor. Below are (A) per-fail diagnoses (gold vs produced) and (B)
round-by-round harness-bug findings across all 12 fresh12 instances. Produce: (1) fail_verdicts — for
each of the 5 fails: classification, winnable, the exact missed scope, and the harness fix that would
convert it to a pass; (2) bugs — dedup the harness findings, merge same-root-cause, rank by run cost
(large→small), keep ALL severities. Concrete file:line fixes.

DIAGNOSES:
${JSON.stringify(diagnoses.filter(Boolean), null, 1).slice(0, 40000)}

HARNESS FINDINGS:
${JSON.stringify(raw, null, 1).slice(0, 110000)}`, { label: 'synthesize', phase: 'Synthesize', schema: SYN })

return { diagnosed: diagnoses.filter(Boolean).length, raw_findings: raw.length, synthesis: synth }
