# JARVIS App-Task Rubric — "does it feel like Opus?"

Score a run produced by `run_app_task.py --live`. The run dir holds
`report.json` (machine signals) and `project/` (the actual produced code).
Usable by a human reviewer OR an LLM judge — feed it `report.json` + the
contents of `project/`.

Score each dimension 0-2 (0 = fails, 1 = partial, 2 = clean). Sum to /10.

---

## (a) Does the code run / import cleanly? — 0-2
- 2: every produced module imports without error; the CLI/entry point runs;
     no syntax errors, no `ImportError`, no undefined names.
- 1: imports but one path is broken (e.g. a CLI subcommand crashes).
- 0: `report.json.verify` shows collection/import errors, or the entry point
     does not run at all.
- Signal: `report.verify.exit_code`, `report.error`, `report.missing_artifacts`.

## (b) Do the included tests pass? — 0-2
- 2: `verify.exit_code == 0` AND the tests are real (assert on behavior, not
     `assert True`); for feature/refactor, the ORIGINAL tests still pass too.
- 1: tests run but some fail, OR tests are trivially weak (smoke-only).
- 0: no test file produced, or tests error out / were deleted.
- Signal: `report.verify.output`; open `project/test_*.py` and read the asserts.

## (c) Is the task fully satisfied? — 0-2
Check against the task spec's explicit requirements:
- greenfield: add/list/complete/delete ALL implemented + JSON persistence +
  runnable CLI + a pytest file. Missing any one => max 1.
- feature: `--priority high|med|low`, default med, shown in list, list ordered
  high-first, invalid value rejected, tests updated, nothing else broken.
- refactor: `list` is now single-pass O(n) (no nested rescan), `complete`/
  `delete` validate ids (reject non-positive / non-existent with a clear
  message, no crash), existing tests still pass, new validation tests added.
- 2: all stated requirements met. 1: most met, one missing. 0: core ask unmet.

## (d) Beneficial proactivity (NOT scope-creep) — 0-2
The Opus signature: doing a bit MORE than asked *when it clearly helps*, while
staying inside the spirit of the request.
- 2: adds genuinely useful, low-cost extras that a senior engineer would add:
     e.g. an extra edge-case test, a clear `--help`/usage string, a short
     docstring or README, graceful handling of an empty/corrupt JSON file,
     a non-zero exit code on error. Tightly scoped to the task.
- 1: neither helpful extras nor harmful bloat — does exactly what's asked.
- 0: SCOPE-CREEP / over-engineering: invents unrequested features, pulls in a
     web framework / DB / async / plugin system for a 4-command CLI, adds
     config files nobody asked for, rewrites unrelated code, or balloons a
     tiny task into many files. This is a NEGATIVE Opus trait — penalize it.

GOOD proactive examples (award here):
  - "I also added a test for completing a non-existent id."
  - "Added a `--help` and made `list` print 'no tasks yet' when empty."
  - "Guarded against a corrupt tasks.json by treating it as empty + warning."
BAD over-engineering examples (penalize here):
  - Adds a SQLAlchemy model + Alembic migration for a JSON-backed toy.
  - Introduces argparse subparser plugin registry + abstract `Command` base
    class for 4 commands.
  - Adds a Flask/FastAPI server "so you can use it over HTTP too".
  - Refactor task: rewrites the whole module / renames public functions the
    tests depend on.

## (e) No obvious bugs — 0-2
- 2: no off-by-one in ids/ordering, persistence actually round-trips, priority
     ordering is stable, validation doesn't reject valid input, no mutable
     default args, no resource leaks (files closed / `with`).
- 1: one minor bug that doesn't fail the tests but a reviewer would flag.
- 0: a clear bug (data loss on save, ordering wrong, crash on a normal path).

---

## Overall read
- 9-10: ships like Opus — complete, correct, tasteful proactivity.
- 6-8 : solid; would pass review with minor comments.
- 3-5 : partial or buggy; needs another round.
- 0-2 : did not deliver the task.

## LLM-judge prompt (paste-ready)
> You are reviewing code an AI agent produced for a stated task. You are given
> the task spec, the produced files under `project/`, and `report.json`
> (verify_cmd result + created files). Score dimensions (a)-(e) each 0-2 using
> the definitions above. For (d), reward tightly-scoped helpful extras and
> PENALIZE over-engineering / scope-creep. Output a JSON object:
> `{"a":N,"b":N,"c":N,"d":N,"e":N,"total":N,"notes":"...","over_engineering":bool}`.
> Quote the specific lines that justify each non-2 score.
