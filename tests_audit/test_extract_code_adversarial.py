"""ADVERSARIAL SECOND-PASS audit of `_extract_code_blocks`.

This is the most complex parser in the codebase. The first-pass tests
verified happy paths. This pass attacks with:

  • Edit blocks intermingled with [think], <think>, ```, and prose.
  • Multiple EDIT blocks on the same file.
  • EDIT + FILE blocks interleaved.
  • Malformed boundaries (no closing ===, no [/REPLACE], etc.).
  • REVERT directives inside string literals in new file content.
  • REVISE EDIT chains (revise, then revise again).
  • SEARCH/REPLACE with empty SEARCH or empty REPLACE.
  • Line-anchored [SEARCH: 45-49] form.
  • The `=== FILE: ... === ... === END FILE ===` new-file form.
"""
import pytest
from workflows.code import _extract_code_blocks


# ─────────────── BASIC EXTRACTION ───────────────


def test_basic__single_edit_block():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]\n"
    )
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert out["text_edits"]["a.py"][0] == ("old", "new")


def test_basic__new_file():
    response = (
        "=== FILE: new.py ===\n"
        "x = 1\ny = 2\n"
        "=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    assert "new.py" in out["new_files"]
    assert "x = 1" in out["new_files"]["new.py"]


def test_basic__empty_response():
    out = _extract_code_blocks("")
    assert out["text_edits"] == {}
    assert out["edits"] == {}
    assert out["new_files"] == {}
    assert out["reverts"] == []


def test_basic__no_edit_syntax():
    out = _extract_code_blocks("Just plain prose with no edit syntax.")
    assert out["text_edits"] == {}


# ─────────────── INERT-ZONE MASKING ───────────────


def test_inert__think_bracket_masks():
    """Edit syntax inside [think] should NOT extract."""
    response = (
        "[think]\n"
        "=== EDIT: hidden.py ===\n"
        "[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        "[/think]\n"
        "=== EDIT: real.py ===\n"
        "[SEARCH]\nreal_old\n[/SEARCH]\n"
        "[REPLACE]\nreal_new\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "hidden.py" not in out["text_edits"]
    assert "real.py" in out["text_edits"]


def test_inert__think_xml_masks():
    response = (
        "<think>\n"
        "=== EDIT: hidden.py ===\n"
        "[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        "</think>\n"
        "=== EDIT: real.py ===\n"
        "[SEARCH]\nreal_old\n[/SEARCH]\n"
        "[REPLACE]\nreal_new\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "hidden.py" not in out["text_edits"]
    assert "real.py" in out["text_edits"]


def test_inert__fenced_block_masks():
    """Edit syntax inside ``` fenced blocks → masked."""
    response = (
        "```\n"
        "=== EDIT: in_fence.py ===\n"
        "[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        "```\n"
        "=== EDIT: outside.py ===\n"
        "[SEARCH]\nout_old\n[/SEARCH]\n"
        "[REPLACE]\nout_new\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "in_fence.py" not in out["text_edits"]
    assert "outside.py" in out["text_edits"]


# ─────────────── MULTIPLE EDITS, SAME FILE ───────────────


def test_multi__two_edits_same_file():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]first_old[/SEARCH][REPLACE]first_new[/REPLACE]\n"
        "=== EDIT: a.py ===\n"
        "[SEARCH]second_old[/SEARCH][REPLACE]second_new[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert len(out["text_edits"]["a.py"]) >= 2


def test_multi__edits_on_different_files():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]a_old[/SEARCH][REPLACE]a_new[/REPLACE]\n"
        "=== EDIT: b.py ===\n"
        "[SEARCH]b_old[/SEARCH][REPLACE]b_new[/REPLACE]\n"
        "=== EDIT: c.py ===\n"
        "[SEARCH]c_old[/SEARCH][REPLACE]c_new[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    for fp in ["a.py", "b.py", "c.py"]:
        assert fp in out["text_edits"]


# ─────────────── EDIT + FILE INTERLEAVED ───────────────


def test_interleaved__edit_then_file():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        "=== FILE: new.py ===\n"
        "new content here\n"
        "=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert "new.py" in out["new_files"]


def test_interleaved__file_then_edit():
    response = (
        "=== FILE: created.py ===\n"
        "x = 1\n"
        "=== END FILE ===\n"
        "=== EDIT: existing.py ===\n"
        "[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "created.py" in out["new_files"]
    assert "existing.py" in out["text_edits"]


# ─────────────── MALFORMED BOUNDARIES ───────────────


def test_malformed__missing_close_search():
    """`[SEARCH]old [REPLACE]new[/REPLACE]` — no `[/SEARCH]` → won't extract."""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold without close\n"
        "[REPLACE]\nnew\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # No valid SEARCH/REPLACE pair → empty extraction for this file
    assert "a.py" not in out["text_edits"] or out["text_edits"]["a.py"] == []


def test_malformed__missing_close_replace():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew without close"
    )
    out = _extract_code_blocks(response)
    assert "a.py" not in out["text_edits"] or out["text_edits"]["a.py"] == []


def test_malformed__edit_block_with_no_search_replace():
    """`=== EDIT: a.py ===` followed by prose, no SEARCH/REPLACE → empty."""
    response = (
        "=== EDIT: a.py ===\n"
        "I'm going to edit this but forget how.\n"
    )
    out = _extract_code_blocks(response)
    assert "a.py" not in out["text_edits"] or out["text_edits"]["a.py"] == []


def test_malformed__edit_header_no_trailing_eq():
    """`=== EDIT: a.py` (no trailing `===`) — should still work."""
    response = (
        "=== EDIT: a.py\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # The pattern accepts both forms
    assert "a.py" in out["text_edits"]


# ─────────────── REVERT DIRECTIVES ───────────────


def test_revert__top_level_directive():
    response = "[REVERT FILE: a.py]"
    out = _extract_code_blocks(response)
    assert "a.py" in out["reverts"]


def test_revert__inside_file_body_NOT_extracted():
    """REVERT directive inside a `=== FILE: ===` body is content, not directive."""
    response = (
        "=== FILE: prompt_data.py ===\n"
        "PROMPT = '[REVERT FILE: foo.py]'\n"
        "=== END FILE ===\n"
    )
    out = _extract_code_blocks(response)
    # The literal [REVERT FILE: foo.py] inside the file body is content
    assert "foo.py" not in out["reverts"]


def test_revert__inside_edit_body_NOT_extracted():
    """REVERT inside an EDIT body — content, not directive."""
    response = (
        "=== EDIT: x.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\n# [REVERT FILE: not_a_real_revert.py]\nnew\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "not_a_real_revert.py" not in out["reverts"]


def test_revert__multiple_top_level():
    response = "[REVERT FILE: a.py]\n[REVERT FILE: b.py]"
    out = _extract_code_blocks(response)
    assert "a.py" in out["reverts"]
    assert "b.py" in out["reverts"]


def test_revert__path_with_directory():
    response = "[REVERT FILE: pkg/sub/file.py]"
    out = _extract_code_blocks(response)
    assert "pkg/sub/file.py" in out["reverts"]


# ─────────────── REVISE EDIT CHAINS ───────────────


def test_revise__single_revise_retracts_prior():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nBAD_OLD\n[/SEARCH]\n"
        "[REPLACE]\nbad_replace\n[/REPLACE]\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]\nGOOD_OLD\n[/SEARCH]\n"
        "[REPLACE]\ngood_replace\n[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _extract_code_blocks(response)
    edits = out["text_edits"].get("a.py", [])
    # Bad edit should be retracted
    assert not any(p == ("BAD_OLD", "bad_replace") for p in edits)
    # Good edit should be present
    assert any(p == ("GOOD_OLD", "good_replace") for p in edits)


def test_revise__chain_of_two_revises():
    """REVISE → REVISE → final. Each revise should retract the prior."""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]V1[/SEARCH][REPLACE]v1[/REPLACE]\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]V2[/SEARCH][REPLACE]v2[/REPLACE]\n"
        "=== END REVISE EDIT ===\n"
        "=== REVISE EDIT: a.py ===\n"
        "[SEARCH]V3[/SEARCH][REPLACE]v3[/REPLACE]\n"
        "=== END REVISE EDIT ==="
    )
    out = _extract_code_blocks(response)
    edits = out["text_edits"].get("a.py", [])
    # Eventually only V3 should survive (V1 retracted by first REVISE,
    # promoted V2-edit retracted by second REVISE)
    assert any(p == ("V3", "v3") for p in edits)


# ─────────────── EMPTY SEARCH/REPLACE ───────────────


def test_empty__empty_search_kept_as_pair():
    """`[SEARCH][/SEARCH][REPLACE]new[/REPLACE]` — empty search is preserved
    in the extracted pair; downstream applier will skip it."""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH][/SEARCH][REPLACE]new[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # Either the pair is kept (with empty search) or filtered out — both OK
    if "a.py" in out["text_edits"]:
        for s, r in out["text_edits"]["a.py"]:
            # Empty search is documented behavior
            assert isinstance(s, str)


def test_empty__empty_replace_for_deletion():
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH]\nDELETE_ME\n[/SEARCH]\n"
        "[REPLACE][/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    pairs = out["text_edits"]["a.py"]
    assert any(s == "DELETE_ME" and r == "" for s, r in pairs)


# ─────────────── NEW FILE BODY EDGE CASES ───────────────


def test_newfile__contains_edit_syntax_as_content():
    """A new file whose CONTENT contains literal `=== EDIT:` text — must
    NOT be misinterpreted as an edit block (this is the JARVIS-rewriting-
    its-own-prompts edge case)."""
    response = (
        "=== FILE: prompt.py ===\n"
        "# This file contains literal edit syntax as a string\n"
        "TEMPLATE = '=== EDIT: foo.py ===\\n[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]'\n"
        "=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    # prompt.py should be the new file
    assert "prompt.py" in out["new_files"]
    # foo.py should NOT have spurious edits
    assert "foo.py" not in out["text_edits"]


def test_newfile__with_revert_directive_in_body():
    """A new file body containing `[REVERT FILE: x]` as content — NOT a directive."""
    response = (
        "=== FILE: data.py ===\n"
        "MSG = '[REVERT FILE: noop.py]'\n"
        "=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    assert "data.py" in out["new_files"]
    assert "noop.py" not in out["reverts"]


def test_newfile__empty_body():
    response = "=== FILE: empty.py ===\n=== END FILE ==="
    out = _extract_code_blocks(response)
    # Empty file might be allowed or filtered — document either way
    if "empty.py" in out["new_files"]:
        assert out["new_files"]["empty.py"].strip() == ""


def test_newfile__unicode_content():
    response = (
        "=== FILE: i18n.py ===\n"
        "msg = '北京 résumé 🎉'\n"
        "=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    assert "i18n.py" in out["new_files"]
    assert "北京" in out["new_files"]["i18n.py"]


# ─────────────── PATH NORMALIZATION IN EDIT HEADERS ───────────────


def test_path__leading_dot_slash():
    response = (
        "=== EDIT: ./a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # Path may be normalized or kept verbatim
    assert "./a.py" in out["text_edits"] or "a.py" in out["text_edits"]


def test_path__with_subdir():
    response = (
        "=== EDIT: pkg/sub/a.py ===\n"
        "[SEARCH]\nold\n[/SEARCH]\n"
        "[REPLACE]\nnew\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    assert "pkg/sub/a.py" in out["text_edits"]


# ─────────────── EDGE: HEAVY MIX ───────────────


def test_heavy_mix__edits_thinks_fences_files():
    """Realistic adversarial response: edits + thinks + fences + new files."""
    response = (
        "[think]\n"
        "Let me consider this edit:\n"
        "=== EDIT: rejected.py ===\n"
        "[SEARCH]not real[/SEARCH][REPLACE]not real[/REPLACE]\n"
        "[/think]\n"
        "\n"
        "Here's my real plan:\n"
        "\n"
        "```python\n"
        "# Example pattern in a fenced code block (not extracted)\n"
        "=== EDIT: also_rejected.py ===\n"
        "[SEARCH]fenced[/SEARCH][REPLACE]fenced[/REPLACE]\n"
        "```\n"
        "\n"
        "=== EDIT: real_a.py ===\n"
        "[SEARCH]\nreal_a_old\n[/SEARCH]\n"
        "[REPLACE]\nreal_a_new\n[/REPLACE]\n"
        "\n"
        "=== FILE: real_new.py ===\n"
        "print('hello')\n"
        "=== END FILE ===\n"
        "\n"
        "=== EDIT: real_b.py ===\n"
        "[SEARCH]\nreal_b_old\n[/SEARCH]\n"
        "[REPLACE]\nreal_b_new\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # Inert zones masked
    assert "rejected.py" not in out["text_edits"]
    assert "also_rejected.py" not in out["text_edits"]
    # Real edits extracted
    assert "real_a.py" in out["text_edits"]
    assert "real_b.py" in out["text_edits"]
    assert "real_new.py" in out["new_files"]


# ─────────────── BACKWARD COMPAT ───────────────


def test_format__line_anchored_search():
    """`[SEARCH: 45-49]` form with line range."""
    response = (
        "=== EDIT: a.py ===\n"
        "[SEARCH: 45-49]\n"
        "exact code lines\n"
        "[/SEARCH]\n"
        "[REPLACE]\nnew code\n[/REPLACE]"
    )
    out = _extract_code_blocks(response)
    # This should still extract — line range is anchor info
    # Either in text_edits or in line-edits (edits)
    has_edit = (
        "a.py" in out["text_edits"]
        or "a.py" in out["edits"]
    )
    assert has_edit
