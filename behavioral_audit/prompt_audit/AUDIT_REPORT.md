The report is the deliverable here. Writing it directly as my final response.

# JARVIS Role-Prompt Audit — Prioritized Engineering Report

## 1. EXECUTIVE SUMMARY (highest-impact, priority order)

1. **One 481-line SYSTEM_KNOWLEDGE block is prepended byte-identically to 7 of 10 roles, and is NOT given to the 3 standalone roles** (MERGER, IMPROVE, REVIEWER) — this single architecture decision is the root cause of almost every other finding.
2. **The 3 standalone tool-using roles have NO tool-turn protocol**: MERGER, IMPROVE, REVIEWER contain zero `[tool use]` / `[STOP][CONFIRM_STOP]` instruction, yet MERGER fires DEPENDENCY/SEARCH and REVIEWER fires RUN — this is the literal root cause of MERGER's "runs out of 6 rounds without committing a plan" failure.
3. **Four mutually incompatible edit syntaxes coexist in the shared block** (`[REPLACE LINES]`, `[edit:N]`, `[SEARCH]`-anchor, `=== REVISE EDIT ===`) and propagate verbatim into 7 prompts — a weak model cannot tell which edit format is live.
4. **The ~100-line coder edit tutorial is dead weight in 5 non-editing roles** (UNDERSTAND, both PLANNERs, REVIEW_ROUTE, mostly SELF_CHECK) because it lives in the shared block and cannot be trimmed per-role without breaking CODER.
5. **`[KNOWLEDGE: topic]` is canon but appears in ZERO of the 10 prompts** — every role's tool table is missing a tool the roles may legally call (or canon is stale).
6. **`[FORCE DONE][CONFIRM_FORCE_DONE]` is table-defined and uniformly inherited across the suite** — strong evidence it is REAL and the audit canon (STOP/DONE/PLAN DONE/CONTINUE only) is the stale party; needs to be added to canon, then gated to coder roles.
7. **MERGER has no round-budget stop rule** — its known failure mode is over-investigation, yet the prompt never states there is a budget or a "COMMIT by round N" directive.
8. **IMPROVE (06) is dead code** — never `.format()`-ed since the 2026-05-27 2-layer cutover; it duplicates MERGER plan-discipline and is a future drift hazard. Delete it.

---

## 2. CRITICAL & HIGH FINDINGS BY ROLE

### SYSTEM_KNOWLEDGE (01) — affects 7 roles
- **[contradictory / CRITICAL]** lines 227-229 vs 359-445 — VIEW/KEEP header says line numbers are "accurate for `[REPLACE LINES N-M]`" while the edit tutorial teaches `[edit:N]` with `LINENO:+INDENT|` diff lines; two edit primitives presented as live with no reconciliation. **Fix:** pick ONE (the `[edit:N]` block is the real one per the worked examples); rewrite the KEEP/VIEW header to reference `[edit:N]` and drop `[REPLACE LINES]`.
- **[undefined / HIGH]** lines 46, 303 — references a `[SEARCH]` tag living *inside* `=== EDIT ===` (warned about as "unterminated") but never taught. **Fix:** delete the `[SEARCH]`-anchor mentions; rewrite line 303 to only disambiguate the `[SEARCH:]` text-search tool.
- **[undefined / HIGH]** line 69 — `=== REVISE EDIT === … === END REVISE ===` introduced as a retract primitive, but line 402 uses `[undo edit: N]` for the same job; two unreconciled retract mechanisms. **Fix:** keep `[undo edit:]` only; delete `=== REVISE EDIT ===`.
- **[missing / CRITICAL→see cross-role]** TOOL TABLE 134-188 — no `[KNOWLEDGE:]` entry. **Fix:** add to CORE table or remove from canon.
- **[redundant / HIGH]** 190-229 / 352-357 / 440-445 — the `LINENO:INDENT|content` prefix + "indent is a count" rule explained 3-4 times. **Fix:** state once near the top of the edit section; delete repeats (~40-60 lines saved).
- **[fitness / HIGH]** 190-445 (~255 of 481 lines) — over half the shared block is edit/output-format minutiae, prepended to roles that never edit. **Fix:** move the coder edit tutorial into the CODER role; shared block keeps only tool table, signal protocol, `[tool use]` wrapping, prefix-read format.

### UNDERSTAND (02) — read-only analyst
- **[fitness / CRITICAL]** 350-446 + Examples A/B/C — ~100 lines of edit-block format injected into a role that produces a findings map and never edits. **Fix:** delete the whole edit section; keep only the read PREFIX format for line citation.
- **[fitness / HIGH]** 308-347 — post-edit diff-verification ("read every `:-` before `[DONE]`") and edit-rejection reasons, all coder-only. **Fix:** remove for this role.
- **[redundant / HIGH]** 56-78 — `=== REVISE EDIT ===` / `[REVERT FILE:]` edit-retract machinery for a role that commits a findings map, not edits. **Fix:** trim to a 2-line `[continue from: -N]` note.

### PLANNER — existing (03)
- **[fitness / CRITICAL]** 350-446 — ~100 lines of edit-block mechanics in a role whose own rule (line 497 "No code in the plan", 872 "NO CODE in the STEP body") forbids exactly this. **Fix:** delete the edit subsection + Examples A/B/C; keep only the read-format note.
- **[fitness / HIGH]** 308-347 — post-edit DIFF feedback; planner never applies edits. **Fix:** remove.
- **[fitness / HIGH]** 190-241 — PREFIX explanation over-built with `[edit]`-copy guidance the planner never uses. **Fix:** collapse to "CODE/VIEW/KEEP show `LINENO:INDENT|content`; cite the LINENO."
- **[redundant / HIGH]** 110-131 vs 668-680 vs 682-699 — blast-radius / `|appears N (#tag)` handoff stated three times. **Fix:** keep ONE planner-facing block.

### PLANNER — new/greenfield (04)
- **[fitness / CRITICAL]** 350-445 — full coder edit format + 3 examples in a role where "the coder writes the code" and code tools "return nothing for a new project." **Fix:** strip entirely.
- **[fitness / HIGH]** 308-347 (edit feedback), 190-298 (CODE/VIEW/KEEP/REFS/DEPENDENCY output formats) — all describe tools that return nothing on an empty project. **Fix:** collapse to WEBSEARCH + PLAN contract.
- **[stale / HIGH]** 110-131 — BLAST RADIUS tells the planner to fold `|appears N (#tag)` into STEPs; no callsites exist on greenfield. **Fix:** exclude for this build.
- **[contract / HIGH]** line 36 `[FORCE DONE]` — coder concept in a role that finishes with `[PLAN DONE]`. **Fix:** remove from planner's signal menu.
- **[contradictory / HIGH]** 503-505 vs 166-189 — "No code tools" lists only CODE/REFS/DEPENDENCY/SEARCH as dead, but the table also teaches PURPOSE/SEMANTIC/VIEW/KEEP/DEPENDSON (equally dead). **Fix:** make the dead-tool list exhaustive, or omit code tools from this build's table.

### MERGER (05)
- **[missing / CRITICAL]** "Investigation discipline" — the role's KNOWN failure (runs out of rounds without a plan) yet there is no stated round budget and no hard COMMIT directive. **Fix:** add "You get up to 6 tool rounds. By round 4, STOP and WRITE the `=== PLAN ===` block with your best understanding, marking gaps in `## CONFIDENCE`. A committed plan with a noted gap beats no plan."
- **[missing / CRITICAL]** whole prompt — no `[tool use]…[/tool use]` wrapper and no `[STOP][CONFIRM_STOP]` end-of-turn instruction, despite firing DEPENDENCY/SEARCH/REFS. **Fix:** inject CORE (which carries these).
- **[contradictory / HIGH]** line 4 "3 independent Layer-1 plans" vs line 205 "3 Layer-2 plans to merge." **Fix:** one canonical name — "the 3 Layer-1 drafts."
- **[stale / HIGH]** 19-24, 130-136 — invents `[think]`/`[continue from: -N]` not in canon (canon continue is `[CONTINUE][CONFIRM_CONTINUE]`). **Fix:** confirm `[think]` is runtime-stripped or route reasoning through the canonical signal.
- **[redundant / HIGH]** 10-27 / 117-127 / 129-136 — "orient in think, exit, write visible, never empty plan" stated 3x. **Fix:** state once near the top.

### IMPROVE (06) — DEAD ROLE
- **[stale / CRITICAL]** whole file — never `.format()`-ed since the 2-layer cutover (code.py:8501-8514); only references are the definition and V8 reassignment. **Fix:** delete `IMPROVE_PROMPT_TEMPLATE_V8` (prompts_v8.py 1124-1251), the re-export (2410), the dead local def (code.py:3715), and the V8 reassignment (code.py:12946).
- **[contradictory / CRITICAL]** line 70 "Reasoning lives in `[think]`" vs 99-100 "`[think]` is stripped, all output must be VISIBLE." **Fix:** moot on deletion; otherwise pick one channel rule.
- **[tool / HIGH]** 46-47 — bare tool name list with no SYSTEM_KNOWLEDGE prepended, so the model gets NO tool definitions. **[missing / HIGH]** no `[tool use]`/`[STOP][CONFIRM_STOP]`. **Fix:** moot on deletion.

### CODER (07)
- **[fitness / CRITICAL]** whole prompt — 881 lines / 38.7K chars before the STEP is injected. **Fix:** target 40-50% cut.
- **[redundant / CRITICAL]** edit format taught 4x (192-229, 350-445, 494-540, 745-856); diff legend printed 3x (315-317, 548-551, 830-834); INDENT-count rule 7x. **Fix:** ONE "How to edit" section with one worked example; delete the rest.
- **[redundant / HIGH]** 308-347 / 542-583 / 745-756 — verify→done loop explained 3x. **Fix:** merge into one section.
- **[contract / HIGH]** line 36 / 677-683 `[FORCE DONE]` — taught as coder completion but not in audit canon. **Fix:** verify it's a live signal (cross-role evidence says yes); add to canon.
- **[missing / MEDIUM→HIGH]** native `create_file`/`replace_lines` never named though the role spec says native function-calling is primary. **Fix:** add a text→native mapping note.

### REVIEWER (08) — disabled by default
- **[contract / HIGH]** line 154 — bare `[STOP]` with no `[CONFIRM_STOP]`; runtime treats bare half as malformed. **Fix:** inject CORE and write `[STOP][CONFIRM_STOP]` everywhere.
- **[fitness / HIGH]** ~280 lines — among the heaviest prompts; verify checklist + root-cause + anti-orphan + completion all overlap. **Fix:** cut to a ~60-90 line spine (review patch → trace one input per branch and RUN it → check callers/imports/signatures → one verdict tag).

### REVIEW_ROUTE (09) — verdict step
- **[contradictory / HIGH]** line 502 "EXIT CODE: 0" hard-coded vs 529-531 rules keying off tracebacks/failures — the reviewer is told the command succeeded even when it failed. **Fix:** make exit code a runtime placeholder `EXIT CODE: <<exit_code>>`.
- **[fitness / CRITICAL]** 350-446 — ~95 lines of edit-block tutorial for a role that cannot edit. **Fix:** remove entirely.
- **[fitness / HIGH]** 308-347 (edit feedback) + 110-131 (BLAST RADIUS) — coder/planner concerns; line 117-118 explicitly addresses other roles. **Fix:** drop both.
- **[redundant / HIGH]** 56-92 — revise-verb catalog (`continue from`, `REVISE EDIT`, `REVERT FILE`) inapplicable to a single-shot router. **Fix:** trim to one `[think]` sentence.

### SELF_CHECK (10)
- **[fitness / CRITICAL]** 1-481 preamble before a ~140-line role body — full edit tutorial + 3 examples + every tool's output format for a role that mostly reads. **Fix:** slim preamble to signal protocol + read format + ONE edit example.
- **[redundant / HIGH]** 350-445 then 595-607 — full edit format taught, then re-taught as "Edit-block constraints." **Fix:** delete the 595-607 duplicate.

---

## 3. CROSS-ROLE ISSUES

- **Shared-block monolith (CRITICAL).** Lines 1-481 are byte-identical across UNDERSTAND, both PLANNERs, CODER, REVIEW_ROUTE, SELF_CHECK. The ~100-line edit tutorial cannot be trimmed per-role without breaking CODER — this is why every per-role audit independently flagged "bloat" but couldn't fix it.
- **Standalone-role protocol gap (CRITICAL).** MERGER, IMPROVE, REVIEWER each have 0 occurrences of `CONFIRM_STOP` and 0 of `[tool use]` — they receive none of the shared block, so the very roles that fire tools have no firing/ending protocol. Closing this simultaneously fixes MERGER's round-budget failure and REVIEWER's bare-`[STOP]` bug.
- **Four edit syntaxes propagate to 7 prompts (CRITICAL).** `[REPLACE LINES]` / `[edit:N]` / `[SEARCH]`-anchor / `=== REVISE EDIT ===` all live in the shared block, so the contradiction replicates identically.
- **`[KNOWLEDGE:]` absent suite-wide (CRITICAL).** Zero occurrences in all 10 files; resolve once in CORE.
- **`[FORCE DONE]` canon resolution (HIGH).** Table-defined and uniformly inherited → it is REAL; the audit canon is the stale party. Add to canon, then gate to coder roles only (EDIT_MECHANICS, not CORE) so it stops bleeding into PLANNER/UNDERSTAND/REVIEW_ROUTE.
- **`[PLAN DONE]` attribution drift (HIGH).** Line 37 says "(planner/merger)" — UNDERSTAND (which inherits the block and must emit `[PLAN DONE]`) is omitted. **Fix:** "(understand/planner/merger)".
- **Tool-description drift (MEDIUM).** 7 roles get the full annotated tool table; MERGER/IMPROVE get bare comma-lists with no semantics for the same tools (e.g. MERGER told to "prefer DEPENDSON" but never told it's the reverse of DEPENDENCY).
- **RUN leakage (MEDIUM).** The ~12-line RUN sandbox block sits in CORE, so it injects into UNDERSTAND, greenfield PLANNER, REVIEW_ROUTE, SELF_CHECK — all roles canon says cannot use RUN.

---

## 4. RECOMMENDED OVERHAUL PLAN (ordered, safe-to-execute)

### Phase 0 — Resolve canon questions FIRST (these gate everything; format-contract risk)
1. **Confirm `[FORCE DONE][CONFIRM_FORCE_DONE]` is a live runtime signal** (grep the runtime). Evidence says yes. If yes, add it to the canonical SIGNAL PROTOCOL. ⚠️ FORMAT CONTRACT.
2. **Confirm `[KNOWLEDGE: topic]` is live.** If yes it goes in CORE; if dead, remove from canon. ⚠️ FORMAT CONTRACT.
3. **Confirm the single live edit primitive** (`[edit:N]` per worked examples) and the single live retract verb (`[undo edit:]`). Verify `[SEARCH]`-anchor and `=== REVISE EDIT ===` are dead against `tool_call.py`. ⚠️ FORMAT CONTRACT — do not delete a live syntax.
4. **Confirm `[think]` strip behavior** and whether `[continue from: -N]` is a real runtime feature (MERGER/IMPROVE rely on it).

### Phase 1 — Split SYSTEM_KNOWLEDGE (touches every role — do once, carefully)
5. Split the 481-line block into **CORE** (signal protocol incl. confirmed `[FORCE DONE]`/`[KNOWLEDGE]`, tool table with one-line semantics, `[tool use]` wrapper, `LINENO:INDENT|content` read format, blast-radius) and **EDIT_MECHANICS** (the single `[edit:N]` envelope, INDENT-count rule stated once, anchor rules, ONE worked example, `[undo edit:]`).
6. Inject **CORE into ALL 10 roles** (this is the fix for MERGER/IMPROVE/REVIEWER protocol gap).
7. Inject **EDIT_MECHANICS only into CODER and SELF_CHECK**.
8. Move the full **RUN sandbox detail into a planner/reviewer-only module**; CORE keeps a one-line gated mention.

### Phase 2 — Fix contradictions inside the now-split CORE/EDIT_MECHANICS
9. Delete the `[REPLACE LINES]` claim at 227-229 (or restate via `[edit:N]`); delete `[SEARCH]`-anchor (46, 303) and `=== REVISE EDIT ===` (69).
10. State the INDENT-count + prefix-read rule ONCE in each module.
11. Fix `[PLAN DONE]` attribution to "(understand/planner/merger)".

### Phase 3 — Per-role cleanup (now safe; edit tutorial no longer shared)
12. **MERGER:** add the round-budget COMMIT directive; fix Layer-1/Layer-2 naming; delete the 3x-repeated "never empty plan" paragraphs; merge DEEP THINK + "How you reason" into one visible analysis block.
13. **REVIEWER:** fix bare `[STOP]`→`[STOP][CONFIRM_STOP]`; cut to ~60-90 line spine; fold anti-orphan into one line; state the exact terminal sequence (one verdict tag, then `[DONE][CONFIRM_DONE]`).
14. **REVIEW_ROUTE:** ⚠️ FORMAT CONTRACT — make `EXIT CODE` a real injected placeholder; collapse the 3x GO-TO-STEP definitions; state how the verdict tag terminates the round.
15. **CODER:** collapse the 4 edit-format expositions and 3 verify-loop sections into one each; add the text→native (`create_file`/`replace_lines`) mapping; make all examples use numbered `[edit:1]`.
16. **PLANNERs (both):** remove `[FORCE DONE]` from the signal menu; greenfield — make the dead-tool list exhaustive and add `[KNOWLEDGE:]` as a useful tool.

### Phase 4 — Delete dead code
17. **Delete IMPROVE** (06): `IMPROVE_PROMPT_TEMPLATE_V8` at prompts_v8.py 1124-1251, re-export 2410, dead def code.py:3715, reassignment code.py:12946. MERGER becomes the sole post-draft prompt.

### Verification after each phase
Re-grep all 10 rendered prompts for: exactly one edit syntax, `CONFIRM_STOP`/`[tool use]` present in all 10, `[KNOWLEDGE]` present iff confirmed live, zero `[REPLACE LINES]`/`[SEARCH]`-anchor/`=== REVISE EDIT ===` if confirmed dead. Run the 3 test suites named in `test_invariants.md` after any change touching the format contract.

---

## 5. MEDIUM / LOW APPENDIX

**SYSTEM_KNOWLEDGE:** [overdefined/M] 153-164 RUN entry restates sandbox 3x — compress to ~3 lines. [overdefined/L] 40-54 + 231-241 + 328-347 overlapping failure-mode prose. [unclear/M] 49-54 "no signal → treated COMPLETE" undercuts line 30's "EVERY ROUND MUST END WITH A CLOSING SIGNAL" — reframe as a hazard. [unclear/L] line 131 garbled "break unrelated tests / breakage". [redundant/L] 222-223 `#tag` mechanism stated 3x. [tone/L] 60-63 meta-asides about model "default behavior."

**UNDERSTAND:** [overdefined/H] RUN entry (planner/reviewer-only) is the longest table entry. [tool/M] 113-115 lists KEEP as a `#tag` annotation source (canon: CODE/VIEW only). [stale/M] 35-36 coder-only terminal signals in the analyst's signal menu. [overdefined/M] BLAST RADIUS A/B/C strategy is off-role. [redundant/L] PREFIX format explained 4x. [overdefined/L] 455-481 lists coder/reviewer input blocks the analyst never receives.

**PLANNER existing:** [contract/M] `[FORCE DONE]` in menu. [missing/M] `[KNOWLEDGE]` absent, `[DISCARD]` only at line 305 not in table. [overdefined/M] 709-746 two worked ambiguity examples → keep one. [overdefined/M] 563-599 ROOT-CAUSE TRACE says the same thing 3x. [redundant/M] contract-shape taxonomy taught 4x. [unclear/L] 538-541 stray "the [SEARCH: <ClassName>] under tests/". [missing/L] planner never told to USE `[RUN:]` to confirm pre-fix behavior. [tone/L] repeated consequence-threats.

**PLANNER new:** [missing/M] `[KNOWLEDGE]` absent; `[RUN:]` pointless pre-code. [redundant/M] 562-565 "How to use [think]" stub. [contradictory/M] 567-571 "[think] stripped" vs 56-57 "use freely." [contract/M] 540-560 uses "## STEPS" but never shows the canonical `### STEP N` token. [overdefined/M] RUN entry. [missing/L] 492-493 no guidance on what makes a plan WIN merger selection (substance/specificity).

**MERGER:** [overdefined/M] DEEP THINK A-D and "How you reason" PART A-D are duplicative frameworks → collapse. [unclear/M] ambiguous whether PART A-D is visible output. [tone/L] "FATAL MISTAKE"/"never do that" alarmism. [tool/L] SEMANTIC/VIEW named without semantics. [undefined/L] 185 "any axis < 6" references undefined confidence axes. [contract/L] "## STEPS" vs "### STEP N" relationship implicit.

**CODER:** [overdefined/M] 153-164 RUN block (coder can't use RUN). [overdefined/M] 40-54 + 95-107 overlapping failure-mode prose. [redundant/M] skeleton rule 2x, re-read ban 3x. [redundant/M] Q-checks re-explain rules. [unclear/M] 819 bare `[edit]` vs 510 "number each edit." [unclear/L] 538 "never rewrite whole function" vs `[REPLACE LINES]` unreconciled. [tone/L] accumulated "Why:" threat tails. [unclear/L] scattered negative-token warnings → one "closers" cheat-line.

**REVIEWER:** [redundant/M] post-coder-vs-manifest re-read rule 3x (99-102, 105-108, 242-248). [redundant/M] 82-85 vs 204-211 "value not no-exception" 2x. [overdefined/M] 53-69 rigid INPUT/STEP/OUTPUT trace template invites fake traces. [overdefined/M] 130-172 42-line anti-orphan section. [tool/L] "LSP-precise" stale wording → "AST-precise." [contract/L] `=== REVISE EDIT ===`/`=== END REVISE ===` asymmetric closer footgun. [unclear/M] 28-31 cross-refs "the IMPLEMENT prompt" the reviewer can't see. [missing/L] DEPENDSON/PURPOSE never offered. [unclear/L] verdict-tag vs `[DONE]` terminator muddle. [redundant/L] shortcut-patterns dup checklist #5. [tone/L] "False approval is your worst outcome" preachy.

**REVIEW_ROUTE:** [tool/M] presents 9 non-RUN tools to a verdict-only step. [contract/M] verdict tags absent from the closing-signal list (34-38). [unclear/M] unclear if reviewer may issue its own `[RUN:]` before verdict. [overdefined/M] 190-298 per-tool output catalog. [missing/M] cycle budget (≤3) never surfaced. [redundant/L] READ THE LOG off-scope. [unclear/L] STEP N source depends on injected steps being numbered. [tone/L] scare-framing + "tests / breakage" slip.

**SELF_CHECK:** [redundant/M] read-format 2x (192-210, 351-357). [tool/M] `[KNOWLEDGE]`/`[DISCARD]` absent from table. [tool/L] full RUN block for a role that can't run it. [contract/M] `[FORCE DONE]` in menu but role body only uses `[DONE]`. [overdefined/M] BLAST RADIUS + DEPENDENCY walls vs role's "this step only" scope. [contradictory/L] 489-491 "per-step only" vs 556 "shared interfaces honored." [missing/M] core plan-adherence check (over/under-implementation vs the STEP) only implied — add explicitly. [overdefined/L] 231-241 VIEW out-of-bounds example. [unclear/L] 558 "value is non-empty/correct" reads as leftover from a past task. [redundant/L] no-blind-retry stated 3x; indent rule 2x (380-387, 440-445).