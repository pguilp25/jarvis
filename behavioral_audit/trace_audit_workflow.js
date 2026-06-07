export const meta = {
  name: 'a26-round-audit',
  description: 'Round-by-round bug audit of a JARVIS coder+planner trace (a26)',
  phases: [
    { title: 'Audit', detail: 'one agent per planner draft / coder step / prompt' },
    { title: 'Synthesize', detail: 'dedup + rank harness/prompt bugs by severity' },
  ],
}

// args: { groups: [{kind, path}, ...] }  — produced by split_trace.py + a `ls`.
// Hardcoded fallback (workflow scripts can't glob the fs) so the audit runs even if args
// plumbing drops the value.
const DEFAULT_GROUPS = [
  { kind: 'coder_step_1', path: '/tmp/a26_groups/coder_step_1.txt' },
  { kind: 'coder_step_2', path: '/tmp/a26_groups/coder_step_2.txt' },
  { kind: 'coder_step_3', path: '/tmp/a26_groups/coder_step_3.txt' },
  { kind: 'coder_step_4', path: '/tmp/a26_groups/coder_step_4.txt' },
  { kind: 'coder_step_5', path: '/tmp/a26_groups/coder_step_5.txt' },
  { kind: 'coder_step_6', path: '/tmp/a26_groups/coder_step_6.txt' },
  { kind: 'coder_step_7', path: '/tmp/a26_groups/coder_step_7.txt' },
  { kind: 'planner_merge_final_owl', path: '/tmp/a26_groups/planner_merging_plans__final__owl-alpha.txt' },
  { kind: 'planner_L1_gemma-4', path: '/tmp/a26_groups/planner_planning__Layer_1__gemma-4-31b-it.txt' },
  { kind: 'planner_L1_nemotron-super', path: '/tmp/a26_groups/planner_planning__Layer_1__nemotron-3-super-120b-a12b.txt' },
  { kind: 'planner_L1_nemotron-ultra', path: '/tmp/a26_groups/planner_planning__Layer_1__nemotron-3-ultra-550b-a55b.txt' },
  { kind: 'planner_L1_owl', path: '/tmp/a26_groups/planner_planning__Layer_1__owl-alpha.txt' },
  { kind: 'planner_step7_uri', path: '/tmp/a26_groups/planner_step_7__lib_ansible_modules_uri.py__Add__use_netrc__paramete_owl-alpha.txt' },
  { kind: 'planner_step8_uri', path: '/tmp/a26_groups/planner_step_8__lib_ansible_modules_uri.py__Add__use_netrc__to__argu_owl-alpha.txt' },
  { kind: 'prompt_round0', path: '/tmp/a26_groups/prompt_round0.txt' },
]
const groups = (args && args.groups && args.groups.length) ? args.groups : DEFAULT_GROUPS
if (!groups.length) { log('no groups'); return { error: 'no groups' } }

const FINDINGS = {
  type: 'object',
  additionalProperties: false,
  required: ['group', 'findings'],
  properties: {
    group: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'round', 'category', 'severity', 'evidence', 'why_bug'],
        properties: {
          title: { type: 'string' },
          round: { type: 'string', description: 'round number(s) where it shows' },
          category: { type: 'string', enum: ['harness', 'prompt', 'model-incapacity'] },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          evidence: { type: 'string', description: 'verbatim quote of what the model SAW or DID' },
          why_bug: { type: 'string', description: 'what is wrong — esp. did the harness feed it bad/stale/contradictory input' },
          fix_hint: { type: 'string' },
        },
      },
    },
  },
}

const AUDIT_PROMPT = (g) => `You are auditing ONE slice of a JARVIS coding-agent trace for the ansible a26c325b SWE-bench instance.
JARVIS is a 4-role agent (PLAN -> IMPLEMENT) running on weak/free LLMs. The point of THIS audit is to find HARNESS bugs:
places where the harness fed the model WRONG, STALE, TRUNCATED, CONTRADICTORY, or MISLEADING input, or mis-handled the model's output — NOT places where a weak model simply reasoned poorly despite correct inputs.

Read this file FULLY and analyze it ROUND BY ROUND:
  ${g.path}    (group kind: ${g.kind})

For a CODER step file each round shows: the model's reasoning, then each tool call's args and the result the harness returned.
For a PLANNER file each round shows: the prompt the model saw (it carries prior rounds' tool results), then the model's response.
For the prompt_round0 file: it is the FULL assembled coder system+user prompt — hunt for stale/contradictory/duplicated instructions, wrong format docs, dead references.

Look specifically for, round by round:
- a tool RESULT that is wrong, empty, truncated, a malformed view, wrong line numbers, a stale view served after an edit, or an arg-name alias the harness failed to read (e.g. line_start vs start_line)
- the model asking for something reasonable and getting a confusing/rejected/✗ result it then loops on
- repeated re-reads of the same file/region (read-storm) and WHY the harness made it re-read
- the model citing a line/symbol the view never actually showed it
- contradictory or stale instructions in the prompt
- the harness accepting a broken edit or rejecting a correct one
- any wasted round (empty turn, dead planner, salvage, retry) and its harness cause

Quote VERBATIM what the model saw or did as evidence. Assign category (harness/prompt/model-incapacity) and severity honestly. If the slice is clean, return an empty findings list. Be concrete and skeptical — this trace is being used to fix real bugs.`

phase('Audit')
const audits = await parallel(groups.map(g => () =>
  agent(AUDIT_PROMPT(g), { label: `audit:${g.kind}`, phase: 'Audit', schema: FINDINGS })
    .catch(() => null)
))

const all = audits.filter(Boolean)
const flat = all.flatMap(a => (a.findings || []).map(f => ({ ...f, group: a.group })))
log(`collected ${flat.length} raw findings across ${all.length} slices`)

phase('Synthesize')
const SYNTH = {
  type: 'object', additionalProperties: false,
  required: ['bugs', 'summary'],
  properties: {
    summary: { type: 'string' },
    bugs: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['title', 'category', 'severity', 'where', 'evidence', 'fix'],
        properties: {
          title: { type: 'string' },
          category: { type: 'string' },
          severity: { type: 'string' },
          where: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
          confidence: { type: 'string' },
        },
      },
    },
  },
}

const synth = await agent(
  `You are the lead auditor. Below are raw round-by-round findings from a JARVIS a26 trace audit (JSON).
Dedup them, drop any that are really model-incapacity (keep only if the harness contributed), MERGE duplicates that point at the same root cause, and RANK by how much each one costs the run (read-storms, wasted rounds, wrong edits, dead planners rank high).
For each surviving bug give: title, category, severity, where (file/loop/round), verbatim evidence, and a concrete fix.
Be ruthless about separating "harness fed it bad input" from "model is just weak".

RAW FINDINGS:
${JSON.stringify(flat, null, 1).slice(0, 120000)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH }
)

return { raw_count: flat.length, slices: all.length, synthesis: synth }
