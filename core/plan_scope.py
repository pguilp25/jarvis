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
    """Append-able note the coder will see when the static lint found gaps."""
    if not gaps:
        return ""
    return ("\n\n## ⚠ PLAN GAPS (auto-detected — resolve before/while coding)\n"
            + "\n".join(f"- {g}" for g in gaps))
