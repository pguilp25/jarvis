"""Audit `_apply_revise_edits`, `_push_revert_state`, `_pop_revert_state`,
`_clear_revert_history`, `_reindent_replace`.

These are the edit-time safety nets:
  • REVISE EDIT lets the coder retract a still-pending edit and replace
    it before [STOP] applies anything.
  • Revert stack lets the runtime undo a just-applied edit if it broke
    syntax or tests.
  • _reindent_replace shifts the model's REPLACE body to match the
    file's actual indentation when the search matched anyway.

Bugs cause:
  • REVISE keeps the prior bad edit alive → both versions applied.
  • Revert pops the wrong snapshot → restores stale content.
  • Reindent applies wrong delta → splices block at wrong column.
"""
import pytest
from workflows.code import (
    _apply_revise_edits,
    _push_revert_state,
    _pop_revert_state,
    _clear_revert_history,
    _reindent_replace,
)


# ───────────────────── _apply_revise_edits ─────────────────────


def test_revise__rewrites_to_regular_edit():
    """REVISE EDIT body should be rewritten as a regular EDIT block."""
    response = (
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    assert "REVISE EDIT" not in out
    assert "=== EDIT: a.py ===" in out
    assert "[SEARCH]" in out
    assert "old" in out
    assert "new" in out


def test_revise__removes_prior_edit_on_same_path():
    """If a prior EDIT on the same path exists, it's removed."""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nFIRST_BAD\n[/SEARCH]\n[REPLACE]\nbad_replace\n[/REPLACE]\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nGOOD\n[/SEARCH]\n[REPLACE]\ngood_replace\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    # Prior bad EDIT is removed
    assert "FIRST_BAD" not in out
    assert "bad_replace" not in out
    # New EDIT body present
    assert "GOOD" in out
    assert "good_replace" in out
    # Only ONE === EDIT: a.py === in the output
    assert out.count("=== EDIT: a.py ===") == 1


def test_revise__only_affects_matching_path():
    """A prior EDIT on `b.py` should NOT be removed when revising `a.py`."""
    response = (
        "=== EDIT: b.py ===\n"
        "[SEARCH]\nB_OLD\n[/SEARCH]\n[REPLACE]\nB_NEW\n[/REPLACE]\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nA_GOOD\n[/SEARCH]\n[REPLACE]\nA_REPLACE\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    # b.py's edit is untouched
    assert "B_OLD" in out
    assert "B_NEW" in out
    # a.py's revised edit is in place
    assert "A_GOOD" in out
    assert "A_REPLACE" in out


def test_revise__no_prior_edit_still_becomes_edit():
    """If no prior EDIT exists, the REVISE is still turned into an EDIT."""
    response = (
        "=== REVISE EDIT: new.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    assert "=== EDIT: new.py ===" in out


def test_revise__no_revise_block_passthrough():
    """No REVISE EDIT blocks → text returned verbatim."""
    response = "=== EDIT: a.py ===\n[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]"
    out = _apply_revise_edits(response)
    assert out == response


def test_revise__multiple_revisions():
    """Multiple REVISE EDIT blocks should all be processed."""
    response = (
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nA1\n[/SEARCH]\n[REPLACE]\nA2\n[/REPLACE]\n"
        "=== END REVISE EDIT ===\n"
        "=== REVISE EDIT: b.py ===\n"
        "[SEARCH]\nB1\n[/SEARCH]\n[REPLACE]\nB2\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    assert "=== EDIT: a.py ===" in out
    assert "=== EDIT: b.py ===" in out
    assert "REVISE" not in out


def test_revise__retracts_most_recent_only():
    """If TWO prior edits on same path exist, only the MOST RECENT is removed.
    (The earlier one stays because the coder may have intended to keep both,
    and the REVISE only retracts the latest draft.)"""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nFIRST\n[/SEARCH]\n[REPLACE]\nfirst_replace\n[/REPLACE]\n"
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nSECOND\n[/SEARCH]\n[REPLACE]\nsecond_replace\n[/REPLACE]\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nFINAL\n[/SEARCH]\n[REPLACE]\nfinal_replace\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(response)
    # FIRST edit kept (earlier prior)
    assert "FIRST" in out
    assert "first_replace" in out
    # SECOND removed
    assert "SECOND" not in out or "second_replace" not in out
    # FINAL present
    assert "FINAL" in out
    assert "final_replace" in out


# ───────────────────── Revert stack ─────────────────────


def test_revert__push_pop_lifo():
    """Pop returns last pushed."""
    _clear_revert_history()
    _push_revert_state("a.py", "v1")
    _push_revert_state("a.py", "v2")
    assert _pop_revert_state("a.py") == "v2"
    assert _pop_revert_state("a.py") == "v1"
    assert _pop_revert_state("a.py") is None


def test_revert__per_file_isolation():
    """Push on a.py shouldn't affect b.py's stack."""
    _clear_revert_history()
    _push_revert_state("a.py", "v_a1")
    _push_revert_state("b.py", "v_b1")
    assert _pop_revert_state("a.py") == "v_a1"
    assert _pop_revert_state("b.py") == "v_b1"


def test_revert__empty_stack_returns_none():
    _clear_revert_history()
    assert _pop_revert_state("never_pushed.py") is None


def test_revert__stack_cap_32():
    """Stack is capped at 32 entries — older entries dropped."""
    _clear_revert_history()
    for i in range(50):
        _push_revert_state("a.py", f"v{i}")
    # Most recent 32 retained; oldest 18 dropped
    out = []
    while True:
        v = _pop_revert_state("a.py")
        if v is None:
            break
        out.append(v)
    assert len(out) == 32
    # First popped is v49, last popped is v18
    assert out[0] == "v49"
    assert out[-1] == "v18"


def test_revert__clear_all():
    _clear_revert_history()
    _push_revert_state("a.py", "x")
    _push_revert_state("b.py", "y")
    _clear_revert_history()
    assert _pop_revert_state("a.py") is None
    assert _pop_revert_state("b.py") is None


def test_revert__clear_single_file():
    _clear_revert_history()
    _push_revert_state("a.py", "x")
    _push_revert_state("b.py", "y")
    _clear_revert_history("a.py")
    assert _pop_revert_state("a.py") is None
    # b.py still has its snapshot
    assert _pop_revert_state("b.py") == "y"


# ───────────────────── _reindent_replace ─────────────────────


def test_reindent__no_shift_needed():
    """REPLACE indent matches file indent — no shift."""
    rep = "    x = 1\n    y = 2"
    matched = ["    z = 3"]
    out = _reindent_replace(rep, matched)
    assert out == ["    x = 1", "    y = 2"]


def test_reindent__add_indent():
    """REPLACE has 0 indent, file has 4 → shift +4."""
    rep = "x = 1\ny = 2"
    matched = ["    z = 3"]
    out = _reindent_replace(rep, matched)
    assert out[0] == "    x = 1"
    assert out[1] == "    y = 2"


def test_reindent__remove_indent():
    """REPLACE has 8, file has 4 → shift -4."""
    rep = "        x = 1\n        y = 2"
    matched = ["    z = 3"]
    out = _reindent_replace(rep, matched)
    assert out[0] == "    x = 1"
    assert out[1] == "    y = 2"


def test_reindent__preserves_relative_indent():
    """Nested indentation within REPLACE block is preserved."""
    rep = "def f():\n    if x:\n        return 1"
    matched = ["    def existing():"]
    out = _reindent_replace(rep, matched)
    # Shift +4 so def f() → 4 indent, if x → 8, return 1 → 12
    assert out[0] == "    def f():"
    assert out[1] == "        if x:"
    assert out[2] == "            return 1"


def test_reindent__blank_lines_preserved():
    rep = "x = 1\n\ny = 2"
    matched = ["    z = 3"]
    out = _reindent_replace(rep, matched)
    # Blank line remains blank
    assert out[1] == "" or out[1].strip() == ""
    # Non-blank lines shifted
    assert out[0] == "    x = 1"
    assert out[2] == "    y = 2"


def test_reindent__skip_leading_blank_in_matched():
    """If matched_lines starts with blank lines, indent is taken from the
    first non-blank line."""
    rep = "x = 1"
    matched = ["", "", "    real_line"]
    out = _reindent_replace(rep, matched)
    assert out[0] == "    x = 1"


def test_reindent__all_blank_no_shift():
    """If REPLACE is all blank lines, return as-is."""
    rep = "\n\n"
    matched = ["    z = 3"]
    out = _reindent_replace(rep, matched)
    # Nothing to align — should return verbatim
    assert out == ["", "", ""]


def test_reindent__no_negative_indent():
    """Delta calculation should not produce negative indentation."""
    rep = "x = 1"  # 0 indent
    matched = ["    z"]  # 4 indent → would shift +4, fine
    # But what if we asked the inverse: matched has 0, REPLACE has 4 → delta = -4
    rep2 = "    x"
    matched2 = ["z"]
    out = _reindent_replace(rep2, matched2)
    # `    x` shifted by -4 → `x`
    assert out[0] == "x"


def test_reindent__matched_as_single_string_legacy():
    """Legacy: matched_lines can be a single str."""
    out = _reindent_replace("x = 1", "    z = 3")
    assert out[0] == "    x = 1"
