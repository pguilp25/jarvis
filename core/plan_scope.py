"""Deterministic plan-scope helpers.

The point: get planning to the FULL, CORRECT set of files/areas to change
WITHOUT depending on a weak free model choosing to explore. The planner ensemble
already produces several drafts and the merger can call dependency tools — but
the merge PICKs one baseline (and is told to narrow), nothing cross-checks the
plan's files against the dependency graph or a relevant test, and nothing lints
the plan for completeness. These helpers make those checks deterministic.

Everything here is PURE (no I/O) so it's trivially testable; the wiring (calling
the file extractor, REFS, reading test files) lives in workflows/code.py.

General to coding, not SWE-bench-specific: under-scoping a feature (missed a
layer), a refactor (missed callers), or a bug (fixed the symptom not the cause)
are the same failure — "missed a file/area" — and these catch all of them.
"""
from __future__ import annotations
import re


# ── #1 UNION scope across drafts (vs PICK one) ───────────────────────────────
def union_file_scopes(per_draft_files: list) -> tuple:
    """Union the file sets the Layer-1 drafts identified.

    `per_draft_files`: one iterable of file paths per surviving draft.
    Returns (union_sorted_by_votes, votes) where votes[path] = how many DISTINCT
    drafts named it. A file ≥2 drafts independently named is high-confidence; a
    file only 1 named is a candidate to verify. This turns the ensemble into
    scope-VOTING instead of pick-one."""
    votes: dict = {}
    for files in per_draft_files:
        for f in set(files or []):
            if f:
                votes[f] = votes.get(f, 0) + 1
    union = sorted(votes, key=lambda f: (-votes[f], f))
    return union, votes


def majority_files(votes: dict, n_drafts: int, threshold: int = 2) -> list:
    """Files named by >= threshold drafts — high-confidence scope to consider
    even if the final plan dropped them. Needs >= threshold drafts to have a
    real agreement signal (with 1 draft there is none → []). Majority (not
    union) keeps a single draft's hallucinated file from forcing scope creep."""
    if n_drafts < threshold:
        return []
    return sorted(f for f, v in votes.items() if v >= threshold)


def format_candidate_block(union: list, votes: dict, n_drafts: int) -> str:
    """The 'CANDIDATE FILES' block fed to the merger: account for EACH."""
    if not union:
        return ""
    lines = "\n".join(f"  - {f}  (named by {votes[f]}/{n_drafts} drafts)" for f in union)
    return (
        "CANDIDATE FILES — the UNION of what the input plans identified. For a "
        "complete fix you must account for EACH: put it in a STEP, or state in one "
        "line why it needs NO change. Never silently drop a file that ≥2 drafts "
        "agreed on.\n" + lines
    )


# ── #2 test-derived scope ────────────────────────────────────────────────────
_IMPORT_RE = re.compile(
    r'^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import|import[ \t]+([\w.]+))', re.M)


def imported_modules(test_source: str) -> set:
    """Dotted module paths a test imports (`from a.b import c` / `import a.b`).
    A relevant test's imports point at the modules whose behaviour it pins — so
    those modules are in scope for a change that must satisfy the test."""
    mods: set = set()
    for m in _IMPORT_RE.finditer(test_source or ""):
        mod = (m.group(1) or m.group(2) or "").split(" as ")[0].strip()
        if mod:
            mods.add(mod)
    return mods


# Generic stems that, as substrings, match nearly every test path — using them
# to rank tests destroys the ranking (this silently buried the real gold test
# past the cap on big repos like pylint, whose scope routinely includes
# __init__/main). A meaningful stem is ≥4 chars and not a catch-all.
_NOISE_STEMS = frozenset({"__init__", "main", "base", "test", "tests",
                          "conftest", "utils", "core", "api", "app", "init"})


def rank_relevant_tests(test_files, scope_files, cap: int = 16) -> list:
    """Order candidate test files so the ones most likely to exercise the plan's
    scope come first, then return the first `cap`. Ranking is by BASENAME match
    against the scope files' (meaningful) stems — NOT a whole-path substring,
    which let generic stems like 'main' rank everything 0. The caller loads only
    the returned files (a cap to bound I/O), so a good ranking is what surfaces a
    test that pins a not-yet-scoped sibling (e.g. pyreverse/utils.py)."""
    import os
    stems = {os.path.splitext(os.path.basename(s))[0] for s in (scope_files or [])}
    stems = {st for st in stems if st and len(st) >= 4 and st not in _NOISE_STEMS}

    def _rank(f):
        bn = os.path.basename(f)
        return 0 if any(st in bn for st in stems) else 1

    tests = [f for f in (test_files or [])
             if "test" in f.lower() and f.endswith(".py")]
    tests.sort(key=_rank)
    return tests[:cap]


# Capture `from MODULE import a, b as c, d` → {MODULE: {a, b, d}}. Same-line and
# the first line of a parenthesised import; aliases reduced to the imported name.
_FROM_IMPORT_RE = re.compile(r'^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(.+)$', re.M)


def imported_symbols(test_source: str) -> dict:
    """{module: {symbol,...}} for `from module import …` lines. The symbols a test
    imports from a project module ARE the interface that module must expose — if
    one isn't defined yet, the test can't even be collected (ImportError), so
    creating it is non-optional. (`import a.b` binds no specific symbol → ignored
    here; `imported_modules` already covers module-level scope.)"""
    out: dict = {}
    for m in _FROM_IMPORT_RE.finditer(test_source or ""):
        mod = m.group(1).strip()
        rhs = m.group(2)
        if "(" in rhs:                      # `from x import (` … take this line's part
            rhs = rhs.split("(", 1)[1]
        rhs = rhs.replace(")", "").split("#", 1)[0]
        for piece in rhs.split(","):
            name = piece.strip().split(" as ")[0].strip()
            if name and name != "*" and name.isidentifier():
                out.setdefault(mod, set()).add(name)
    return out


def missing_symbols(symbols, module_source: str) -> list:
    """Of `symbols`, the ones NOT defined anywhere in `module_source` (no
    `def`/`class`/assignment/`import … as name`). Conservative — any plausible
    binding form counts as 'present' so we never invent a phantom create-step for
    a symbol that's actually there (e.g. re-exported)."""
    src = module_source or ""
    miss: list = []
    for name in sorted(symbols or []):
        n = re.escape(name)
        present = re.search(
            rf'(?m)^[ \t]*(?:def|class)[ \t]+{n}\b'      # def/class name
            rf'|^[ \t]*{n}[ \t]*[:=]'                     # name = … / name: type
            rf'|[ \t]+as[ \t]+{n}\b'                      # import … as name
            rf'|^[ \t]*{n}[ \t]*,'                        # name, in a tuple-LHS / __all__
            , src)
        if not present:
            miss.append(name)
    return miss


def modules_to_files(modules, project_files) -> list:
    """Map dotted modules (pkg.sub.mod) to existing project file paths
    (pkg/sub/mod.py), matching by exact path or path-suffix (project paths may
    carry a repo-relative prefix)."""
    fileset = list(project_files or [])
    out: set = set()
    for mod in (modules or []):
        cand = mod.replace(".", "/") + ".py"
        for f in fileset:
            if f == cand or f.endswith("/" + cand):
                out.add(f)
    return sorted(out)


# ── #3 dependency backstop ───────────────────────────────────────────────────
_PYPATH_RE = re.compile(r'[\w./-]+\.py')


def referenced_files_outside_scope(tool_output: str, scope_files,
                                   project_files) -> list:
    """File paths mentioned in a REFS/DEPENDENCY output that are real project
    files but NOT in the plan's scope — i.e. callers/dependents the change may
    need to touch but the plan didn't name."""
    scope = set(scope_files or [])
    fileset = set(project_files or [])
    found = set(_PYPATH_RE.findall(tool_output or ""))
    return sorted(p for p in found if p in fileset and p not in scope)


# ── #4 static completeness lint ──────────────────────────────────────────────
_STEP_RE = re.compile(r'^###?[ \t]*STEP[ \t]*\d+', re.M | re.I)
_REQ_HEADER_RE = re.compile(r'^##[ \t]*REQUIREMENTS\b', re.M | re.I)


def completeness_lint(plan_text: str, scope_files, required_files) -> list:
    """Deterministic plan gaps — returns human-readable gap strings (empty =
    clean). The load-bearing check: a file that multiple independent sources
    (draft votes + a relevant test) flagged is missing from the plan's scope.
    Also flags a plan that has REQUIREMENTS but no STEPs."""
    gaps: list = []
    scope = set(scope_files or [])
    plan = plan_text or ""
    for rf in (required_files or []):
        if rf and rf not in scope:
            gaps.append(
                f"{rf}: identified by multiple sources (draft consensus and/or a "
                f"test that references it) but NOT covered by any STEP — add a "
                f"STEP for it, or confirm in one line it needs no change.")
    if _REQ_HEADER_RE.search(plan) and not _STEP_RE.search(plan):
        gaps.append("the plan lists REQUIREMENTS but has no `### STEP` blocks — "
                    "the coder needs concrete steps to execute.")
    return gaps


def format_plan_gaps(gaps: list) -> str:
    """Append-able note the coder will see when the static lint found gaps.

    Framed IMPERATIVELY (ckpt-166): these are files an independent check says the
    change must touch (the spec names them, or several planner drafts agreed on
    them) but the merged plan dropped — the dominant under-scope failure (a merger
    that prunes behavior-bearing files ships a patch that applies cleanly yet fails
    every test; openlibrary-111347e9 lost exactly this way). The old wording was a
    buried conditional ("if you must, add a step") and the coder ignored it. Now it
    is a directive WITH an escape hatch (read-and-decline) so it can't over-scope a
    file that genuinely needs no change."""
    if not gaps:
        return ""
    return ("\n\n## ⛔ REQUIRED SCOPE THE PLAN OMITTED — treat each as an ADDITIONAL STEP\n"
            "An independent scope check flagged these (the spec names them, or several planner "
            "drafts agreed on them), but the plan's steps don't cover them. A plan that drops a "
            "behavior-bearing file is the #1 reason a clean-looking patch still fails the tests. "
            "For EACH item: OPEN the file, decide what THIS task needs there, and edit it as if it "
            "were an explicit numbered step — UNLESS reading it proves it genuinely needs no change "
            "(then say so briefly and move on; do not edit a file that doesn't need it).\n"
            + "\n".join(f"- {g}" for g in gaps))


# ── coverage enforcement: consensus file MENTIONED but never STEPPED ───────────
# Distinct from completeness_lint / the _required backstop: those key on scope
# MEMBERSHIP (a file the merged plan never mentions). This keys on step COVERAGE.
# A merger can NAME a file in prose (so it counts as "in scope") yet emit NO
# `### STEP` for it — and the per-step coder, which iterates STEPS, then never
# edits it. That silently drops a behavior-bearing file the drafts agreed on
# (ansible-395e5e20: strategy/__init__.py + linear.py were named but unstepped →
# missed sites → fail; the run that DID step them passed). So: for each ≥2-draft
# consensus file with no step, synthesize a real STEP the parser will pick up.
_STEP_NUM_RE = re.compile(r'###\s*STEP\s*(\d+)', re.I)


def coverage_steps(best_plan: str, consensus_files, stepped_files,
                   proj_files, votes: dict, n_drafts: int = 0, max_add: int = 4):
    """Return (append_text, added_files): real `### STEP` blocks for ≥2-draft
    consensus files that have NO step in `best_plan`. `stepped_files` = the files
    the plan's parsed steps already cover. Gated hard (≥2 consensus + real project
    file + not a test) and capped, so it can't over-scope; each step carries an
    explicit read-and-decline escape hatch. Empty when coverage is already complete."""
    proj = set(proj_files or [])
    stepped = set(stepped_files or [])
    uncovered = [f for f in (consensus_files or [])
                 if f and f not in stepped and f in proj and "test" not in f.lower()]
    if not uncovered:
        return "", []
    nums = [int(m.group(1)) for m in _STEP_NUM_RE.finditer(best_plan or "")]
    maxstep = max(nums) if nums else 0
    blocks, added = [], []
    for f in uncovered[:max_add]:
        maxstep += 1
        v = votes.get(f, 2) if isinstance(votes, dict) else 2
        blocks.append(
            f"### STEP {maxstep}: Apply the required change to {f}\n"
            f"FILES: {f}\n"
            f"{v} of the planner drafts independently flagged {f} as needing changes, but the "
            f"merged plan emitted no step for it. Read {f} and make the change THIS task requires "
            f"there — consistent with the other steps (the same new symbols / signatures / enums "
            f"they introduce) and wire it in. If, after reading it, you are certain it needs NO "
            f"change, say so in one line and make none (do not edit a file that doesn't need it).")
        added.append(f)
    return "\n\n" + "\n\n".join(blocks), added
