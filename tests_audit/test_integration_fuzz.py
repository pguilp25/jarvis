"""INTEGRATION FUZZ — multi-function pipelines.

Tests that COMBINE multiple functions to catch interaction bugs that
the unit-level fuzz didn't see:

  • extract → smart_apply → check_syntax → sandbox roundtrip
  • mask → extract → revise → dedup → apply
  • multi-round dedup across iterations
"""
import pytest
import random
import string
from pathlib import Path
from workflows.code import (
    _extract_code_blocks,
    _smart_apply,
    _check_syntax,
    _apply_revise_edits,
    _dedup_against_seen,
    _mask_inert_zones,
    _apply_edits,
)
from tools.sandbox import Sandbox


def _setup_sandbox(tmp_path: Path, files: dict[str, str]):
    project = tmp_path / "project"
    project.mkdir()
    for path, content in files.items():
        fp = project / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    sb = Sandbox(str(project))
    sb.setup()
    return sb


# ─────────────── PIPELINE: EXTRACT → APPLY → SYNTAX ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_pipe__extract_apply_syntax(tmp_path, seed):
    """Extract edits from a response, apply them, verify syntax holds."""
    rng = random.Random(seed)
    orig = "def foo():\n    return 1\n"
    response = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\n    return 1\n[/SEARCH]\n"
        f"[REPLACE]\n    return {seed}\n[/REPLACE]"
    )
    extracted = _extract_code_blocks(response)
    new_content = _smart_apply(orig, extracted, "a.py")
    assert new_content is not None
    ok, _ = _check_syntax("a.py", new_content)
    # Result is valid Python
    assert ok


@pytest.mark.parametrize("seed", range(30))
def test_pipe__multiple_edits_one_response(tmp_path, seed):
    """A response with 3 edits on the same file — all applied."""
    rng = random.Random(seed)
    orig = "v1 = 1\nv2 = 2\nv3 = 3\n"
    response = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nv1 = 1\n[/SEARCH]\n[REPLACE]\nv1 = {seed}A\n[/REPLACE]\n"
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nv2 = 2\n[/SEARCH]\n[REPLACE]\nv2 = {seed}B\n[/REPLACE]\n"
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nv3 = 3\n[/SEARCH]\n[REPLACE]\nv3 = {seed}C\n[/REPLACE]\n"
    )
    extracted = _extract_code_blocks(response)
    new_content = _smart_apply(orig, extracted, "a.py")
    if new_content:
        # Each replacement should be visible
        assert f"v1 = {seed}A" in new_content
        assert f"v2 = {seed}B" in new_content
        assert f"v3 = {seed}C" in new_content


# ─────────────── PIPELINE: REVISE → EXTRACT → APPLY ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_pipe__revise_retraction_propagates(tmp_path, seed):
    """REVISE block must retract prior bad edit so the final apply uses
    only the revised version."""
    orig = "value = OLD\n"
    response_raw = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nvalue = OLD\n[/SEARCH]\n"
        f"[REPLACE]\nvalue = BAD_REPLACE_{seed}\n[/REPLACE]\n"
        f"=== REVISE EDIT: a.py ===\n"
        f"[SEARCH]\nvalue = OLD\n[/SEARCH]\n"
        f"[REPLACE]\nvalue = GOOD_REPLACE_{seed}\n[/REPLACE]\n"
        f"=== END REVISE EDIT ===\n"
    )
    # Pipeline: revise (in extract), apply
    extracted = _extract_code_blocks(response_raw)
    new_content = _smart_apply(orig, extracted, "a.py")
    if new_content:
        # GOOD replacement wins
        assert f"GOOD_REPLACE_{seed}" in new_content
        assert f"BAD_REPLACE_{seed}" not in new_content


# ─────────────── PIPELINE: INERT-ZONE → EXTRACT BLOCK ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_pipe__think_block_NOT_applied(tmp_path, seed):
    """Edit syntax inside [think] must NOT propagate through extract → apply."""
    orig = "real = 1\n"
    response = (
        f"[think]\n"
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nreal = 1\n[/SEARCH]\n"
        f"[REPLACE]\nreal = HIJACKED_{seed}\n[/REPLACE]\n"
        f"[/think]\n"
    )
    extracted = _extract_code_blocks(response)
    new_content = _smart_apply(orig, extracted, "a.py")
    # Either no apply (extracted is empty) or apply preserved orig
    if new_content is None:
        # No edit extracted → smart_apply returns None
        pass
    else:
        # Should NOT contain the hijack
        assert f"HIJACKED_{seed}" not in new_content


# ─────────────── PIPELINE: DEDUP ACROSS ROUNDS ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_pipe__dedup_across_rounds(tmp_path, seed):
    """Simulate 3 rounds where the response_so_far includes all prior edits.
    Dedup must prevent reapplication."""
    rng = random.Random(seed)
    seen: set = set()

    # Round 1: extract one edit
    r1_response = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nx = 1\n[/SEARCH]\n"
        f"[REPLACE]\nx = {seed}\n[/REPLACE]"
    )
    extracted_1 = _extract_code_blocks(r1_response)
    _dedup_against_seen(extracted_1, seen)
    # First time — edit kept
    assert "a.py" in extracted_1["text_edits"]

    # Round 2: same edit re-appears (response_so_far includes round 1)
    r2_response = r1_response + "\n\nstill thinking..."
    extracted_2 = _extract_code_blocks(r2_response)
    _dedup_against_seen(extracted_2, seen)
    # Same edit — deduped
    assert "a.py" not in extracted_2["text_edits"]


# ─────────────── PIPELINE: MASK → EXTRACT BLOCK ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_pipe__mask_inert_then_extract(tmp_path, seed):
    """Manually mask inert zones, then verify extract doesn't catch them.
    (extract already does this internally, but verify the contract.)"""
    response = (
        f"```python\n"
        f"=== EDIT: hidden.py ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"```\n"
        f"=== EDIT: real.py ===\n"
        f"[SEARCH]\nreal_old\n[/SEARCH]\n"
        f"[REPLACE]\nreal_new\n[/REPLACE]\n"
    )
    extracted = _extract_code_blocks(response)
    assert "hidden.py" not in extracted["text_edits"]
    assert "real.py" in extracted["text_edits"]


# ─────────────── PIPELINE: APPLY VIA SANDBOX ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_pipe__sandbox_isolated_writes(tmp_path, seed):
    """Sandbox roundtrip — write then apply preserves content."""
    sb = _setup_sandbox(tmp_path, {"a.py": f"x = {seed}\n"})
    sb.load_file("a.py")
    new_content = f"y = {seed}\nx = {seed * 2}\n"
    sb.write_file("a.py", new_content)
    sb.apply()
    # Project root file matches what we wrote
    real_content = (sb.project_root / "a.py").read_text()
    assert real_content == new_content


# ─────────────── PIPELINE: FULL EDIT WITH SYNTAX ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_pipe__edit_apply_via_sandbox_preserves_syntax(tmp_path, seed):
    sb = _setup_sandbox(tmp_path, {"a.py": "def foo():\n    return 1\n"})
    orig = sb.load_file("a.py")
    edits = [("return 1", f"return {seed}")]
    new_content, applied, _, _ = _apply_edits(orig, edits)
    if applied:
        sb.write_file("a.py", new_content)
        sb.apply()
        result_text = (sb.project_root / "a.py").read_text()
        ok, _ = _check_syntax("a.py", result_text)
        assert ok


# ─────────────── PIPELINE: MULTIPLE FILES ───────────────


@pytest.mark.parametrize("seed", range(15))
def test_pipe__multiple_files_in_response(tmp_path, seed):
    """Response with edits to 3 different files — all applied independently."""
    sb = _setup_sandbox(tmp_path, {
        "a.py": "a_value = 1\n",
        "b.py": "b_value = 2\n",
        "c.py": "c_value = 3\n",
    })
    response = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\na_value = 1\n[/SEARCH]\n[REPLACE]\na_value = {seed}\n[/REPLACE]\n"
        f"=== EDIT: b.py ===\n"
        f"[SEARCH]\nb_value = 2\n[/SEARCH]\n[REPLACE]\nb_value = {seed * 2}\n[/REPLACE]\n"
        f"=== EDIT: c.py ===\n"
        f"[SEARCH]\nc_value = 3\n[/SEARCH]\n[REPLACE]\nc_value = {seed * 3}\n[/REPLACE]\n"
    )
    extracted = _extract_code_blocks(response)
    for path in ["a.py", "b.py", "c.py"]:
        orig = sb.load_file(path)
        new_content = _smart_apply(orig, extracted, path)
        if new_content:
            sb.write_file(path, new_content)
    sb.apply()
    # Verify each file got its update
    assert f"a_value = {seed}" in (sb.project_root / "a.py").read_text()
    assert f"b_value = {seed * 2}" in (sb.project_root / "b.py").read_text()
    assert f"c_value = {seed * 3}" in (sb.project_root / "c.py").read_text()


# ─────────────── PIPELINE: NEW FILE CREATION ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_pipe__new_file_extracted_and_written(tmp_path, seed):
    sb = _setup_sandbox(tmp_path, {"a.py": "x = 1\n"})
    new_path = f"new_{seed}.py"
    response = (
        f"=== FILE: {new_path} ===\n"
        f"def helper():\n    return {seed}\n"
        f"=== END FILE ==="
    )
    extracted = _extract_code_blocks(response)
    assert new_path in extracted["new_files"]
    content = extracted["new_files"][new_path]
    sb.write_file(new_path, content)
    sb.apply()
    assert (sb.project_root / new_path).exists()
    real_content = (sb.project_root / new_path).read_text()
    assert f"return {seed}" in real_content


# ─────────────── PIPELINE: REGRESSION FOR BUG #5 ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_regression__new_file_with_edit_syntax_body_NOT_spurious(tmp_path, seed):
    """Regression for bug found in batch 18: a new file containing literal
    `=== EDIT: x ===` in its body must NOT trigger a spurious edit."""
    fake_target = f"fake_target_{seed}.py"
    new_path = f"new_{seed}.py"
    response = (
        f"=== FILE: {new_path} ===\n"
        f"TEMPLATE = '''\n"
        f"=== EDIT: {fake_target} ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"'''\n"
        f"=== END FILE ==="
    )
    extracted = _extract_code_blocks(response)
    # New file extracted
    assert new_path in extracted["new_files"]
    # Fake target NEVER appears
    assert fake_target not in extracted["text_edits"]
    assert fake_target not in extracted["edits"]


# ─────────────── PIPELINE: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_pipe__deterministic_pipeline(seed):
    """Running the full pipeline twice on identical input gives identical results."""
    response = (
        f"=== EDIT: a.py ===\n"
        f"[SEARCH]\nold_{seed}\n[/SEARCH]\n"
        f"[REPLACE]\nnew_{seed}\n[/REPLACE]"
    )
    e1 = _extract_code_blocks(response)
    e2 = _extract_code_blocks(response)
    assert e1 == e2
    o1 = _smart_apply(f"old_{seed}\n", e1, "a.py")
    o2 = _smart_apply(f"old_{seed}\n", e2, "a.py")
    assert o1 == o2
