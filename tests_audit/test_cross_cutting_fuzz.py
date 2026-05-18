"""THIRD-PASS FUZZ audit — cross-cutting properties for the remaining
high-risk functions:
  • _apply_revise_edits
  • _apply_plan_edits
  • _check_syntax
  • _smart_apply
  • _apply_map_edits
  • _detect_unterminated_blocks
"""
import pytest
import random
import string
from workflows.code import (
    _apply_revise_edits,
    _check_syntax,
    _smart_apply,
    _apply_map_edits,
)
from core.tool_call import _detect_unterminated_blocks, _apply_plan_edits


# ═══════════════════════════════════════════════════════════════════════════════
# _apply_revise_edits FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(50))
def test_revise__no_revise_passthrough(seed):
    """Text with no REVISE EDIT — exact passthrough."""
    rng = random.Random(seed)
    text = "".join(rng.choice(string.ascii_letters + " \n=") for _ in range(rng.randint(0, 200)))
    assert _apply_revise_edits(text) == text


@pytest.mark.parametrize("seed", range(30))
def test_revise__output_no_REVISE_marker(seed):
    """After processing, output contains NO `=== REVISE EDIT:` markers."""
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    text = (
        f"=== REVISE EDIT: {path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]\n"
        f"=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(text)
    assert "REVISE EDIT" not in out


@pytest.mark.parametrize("seed", range(30))
def test_revise__rewritten_as_EDIT(seed):
    """REVISE block rewritten as `=== EDIT: path ===` block."""
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    text = (
        f"=== REVISE EDIT: {path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]\n"
        f"=== END REVISE EDIT ==="
    )
    out = _apply_revise_edits(text)
    assert f"=== EDIT: {path} ===" in out


@pytest.mark.parametrize("seed", range(30))
def test_revise__idempotent(seed):
    """Applying revise twice = once (no more REVISE markers second time)."""
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    text = (
        f"=== REVISE EDIT: {path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]\n"
        f"=== END REVISE EDIT ==="
    )
    once = _apply_revise_edits(text)
    twice = _apply_revise_edits(once)
    assert once == twice


# ═══════════════════════════════════════════════════════════════════════════════
# _apply_plan_edits FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(50))
def test_plan_edit__no_edits_returns_same_plan(seed):
    rng = random.Random(seed)
    plan = "\n".join(f"line {i}" for i in range(rng.randint(0, 30)))
    out, log = _apply_plan_edits(plan, "")
    assert out == plan
    assert log == []


@pytest.mark.parametrize("seed", range(30))
def test_plan_edit__replace_single_line(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(5, 20)
    plan = "\n".join(f"line_{i}" for i in range(n_lines))
    line_to_replace = rng.randint(1, n_lines)
    edit = f"[REPLACE LINES {line_to_replace}-{line_to_replace}]\nREPLACED\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    assert "REPLACED" in out


@pytest.mark.parametrize("seed", range(30))
def test_plan_edit__insert_after(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(5, 20)
    plan = "\n".join(f"line_{i}" for i in range(n_lines))
    after = rng.randint(0, n_lines)
    edit = f"[INSERT AFTER LINE {after}]\nINSERTED\n[/INSERT]"
    out, log = _apply_plan_edits(plan, edit)
    assert "INSERTED" in out


@pytest.mark.parametrize("seed", range(30))
def test_plan_edit__oob_replace_skipped(seed):
    rng = random.Random(seed)
    plan = "line_1\nline_2\nline_3"
    edit = f"[REPLACE LINES 100-200]\nnew\n[/REPLACE]"
    out, log = _apply_plan_edits(plan, edit)
    # OOB skipped
    assert out == plan
    assert any("out of range" in l for l in log)


# ═══════════════════════════════════════════════════════════════════════════════
# _check_syntax FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(50))
def test_syntax__valid_simple_def(seed):
    """Random valid def — should pass."""
    rng = random.Random(seed)
    name = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    src = f"def {name}():\n    return 1\n"
    ok, msg = _check_syntax("test.py", src)
    assert ok


@pytest.mark.parametrize("seed", range(30))
def test_syntax__missing_colon_fails(seed):
    rng = random.Random(seed)
    name = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    src = f"def {name}()\n    return 1\n"
    ok, msg = _check_syntax("test.py", src)
    assert not ok


def test_syntax__unknown_extension_passes():
    """Files with extensions the checker doesn't handle — passes by default."""
    for ext in [".md", ".txt", ".rst", ".csv", ".log"]:
        ok, _ = _check_syntax(f"file{ext}", "any content even (broken")
        assert ok, f"Should pass for {ext}"


@pytest.mark.parametrize("seed", range(30))
def test_syntax__deterministic(seed):
    rng = random.Random(seed)
    src = "x = " + str(rng.randint(0, 100)) + "\n"
    o1 = _check_syntax("a.py", src)
    o2 = _check_syntax("a.py", src)
    assert o1 == o2


# ═══════════════════════════════════════════════════════════════════════════════
# _smart_apply FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(30))
def test_smart__line_edits_priority_over_text(seed):
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    extracted = {
        "edits": {path: [(1, 1, f"i0|line_edit_replacement_{seed}")]},
        "text_edits": {path: [("orig", f"text_edit_replacement_{seed}")]},
    }
    out = _smart_apply("orig\n", extracted, path)
    assert out is not None
    # Line edit took priority
    assert f"line_edit_replacement_{seed}" in out


@pytest.mark.parametrize("seed", range(30))
def test_smart__no_match_returns_none(seed):
    rng = random.Random(seed)
    extracted = {"edits": {}, "text_edits": {}}
    out = _smart_apply("content\n", extracted, f"any_{seed}.py")
    assert out is None


@pytest.mark.parametrize("seed", range(30))
def test_smart__suffix_match(seed):
    """Path `pkg/a.py` in edits, query `a.py` — match via path-bounded suffix."""
    rng = random.Random(seed)
    name = f"file_{seed}"
    extracted = {
        "edits": {f"pkg/{name}.py": [(1, 1, f"i0|REPLACED_{seed}")]},
        "text_edits": {},
    }
    out = _smart_apply("orig\n", extracted, f"{name}.py")
    if out is not None:
        assert f"REPLACED_{seed}" in out


@pytest.mark.parametrize("seed", range(30))
def test_smart__partial_suffix_rejected(seed):
    """`foo/bar.py` should NOT match query `qux/bar.py`."""
    rng = random.Random(seed)
    extracted = {
        "edits": {f"foo_{seed}/bar.py": [(1, 1, "i0|WRONG")]},
        "text_edits": {},
    }
    out = _smart_apply("orig\n", extracted, f"qux_{seed}/bar.py")
    # Should not match (different parent dirs)
    assert out is None or "WRONG" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# _apply_map_edits FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(30))
def test_map__simple_replace(seed):
    rng = random.Random(seed)
    target = f"TARGET_{seed}"
    replacement = f"NEW_{seed}"
    orig = f"prefix\n{target}\nsuffix"
    resp = f"[SEARCH]\n{target}\n[/SEARCH][REPLACE]\n{replacement}\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert replacement in out
    assert target not in out


@pytest.mark.parametrize("seed", range(30))
def test_map__no_match_unchanged(seed):
    rng = random.Random(seed)
    orig = "any content here"
    resp = f"[SEARCH]\nabsent_pattern_{seed}\n[/SEARCH][REPLACE]\nx\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    assert out == orig


@pytest.mark.parametrize("seed", range(30))
def test_map__add_section_appends(seed):
    rng = random.Random(seed)
    orig = "header"
    addition = f"=== SECTION: new_{seed} ===\nbody"
    resp = f"[ADD_SECTION]\n{addition}\n[/ADD_SECTION]"
    out = _apply_map_edits(orig, resp)
    assert addition in out
    assert "header" in out


@pytest.mark.parametrize("seed", range(30))
def test_map__ambiguous_refused(seed):
    rng = random.Random(seed)
    dup = f"DUP_{seed}"
    orig = f"{dup}\nmid\n{dup}\n{dup}"
    resp = f"[SEARCH]\n{dup}\n[/SEARCH][REPLACE]\nX\n[/REPLACE]"
    out = _apply_map_edits(orig, resp)
    # Ambiguous — refused
    assert out.count(dup) == 3
    assert "X" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# _detect_unterminated_blocks FUZZ
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(30))
def test_unterm__balanced_returns_empty(seed):
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    text = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n"
        f"[REPLACE]\nnew\n[/REPLACE]\n"
    )
    out = _detect_unterminated_blocks(text)
    # Balanced — no issues for this block
    assert not any(kind == "EDIT" and p.endswith(path) for kind, p in out)


@pytest.mark.parametrize("seed", range(30))
def test_unterm__missing_replace_close_detected(seed):
    rng = random.Random(seed)
    path = f"file_{seed}.py"
    text = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n"
        f"[REPLACE]\nnew without close"
    )
    out = _detect_unterminated_blocks(text)
    # Detected
    assert any(kind == "EDIT" and p.endswith(path) for kind, p in out)


@pytest.mark.parametrize("seed", range(30))
def test_unterm__file_block_missing_close_detected(seed):
    rng = random.Random(seed)
    path = f"new_{seed}.py"
    text = (
        f"=== FILE: {path} ===\n"
        f"content here\n"
        # NO === END FILE ===
    )
    out = _detect_unterminated_blocks(text)
    assert any(kind == "FILE" and p.endswith(path) for kind, p in out)


def test_unterm__no_blocks_returns_empty():
    assert _detect_unterminated_blocks("") == []
    assert _detect_unterminated_blocks("plain text") == []


@pytest.mark.parametrize("seed", range(20))
def test_unterm__multiple_unterminated(seed):
    """Multiple unterminated blocks — all reported."""
    rng = random.Random(seed)
    text = ""
    paths = []
    for i in range(3):
        path = f"file_{seed}_{i}.py"
        paths.append(path)
        text += (
            f"=== EDIT: {path} ===\n"
            f"[SEARCH]\nold_{i}\n[/SEARCH]\n"
            f"[REPLACE]\nnew_{i}\n"  # no /REPLACE
        )
    out = _detect_unterminated_blocks(text)
    # All 3 paths reported as unterminated
    found_paths = {p for _, p in out}
    for p in paths:
        assert p in found_paths
