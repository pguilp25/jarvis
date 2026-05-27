"""THIRD-PASS FUZZ audit of `_extract_code_blocks`.

Properties:
  I1. Result is always a dict with keys: `edits`, `text_edits`, `new_files`, `reverts`.
  I2. Each value matches its type contract.
  I3. Inert-zone masking is bulletproof: NO edit syntax inside [think]/<think>/
      fenced blocks ever gets extracted, regardless of how many or how nested.
  I4. REVERT directives inside === FILE: === or === EDIT: === bodies are NEVER
      extracted (treated as content).
  I5. Empty / malformed inputs return empty result, no crash.
"""
import pytest
import random
import string
from workflows.code import _extract_code_blocks


def _random_path(rng: random.Random) -> str:
    name = "".join(rng.choice(string.ascii_lowercase) for _ in range(5))
    return f"{name}.py"


def _random_body(rng: random.Random) -> str:
    """Random code-like body."""
    chars = string.ascii_letters + string.digits + " ()={}"
    n_lines = rng.randint(1, 5)
    lines = ["".join(rng.choice(chars) for _ in range(rng.randint(5, 30)))
             for _ in range(n_lines)]
    return "\n".join(lines)


# ─────────────── PROPERTY: RESULT SHAPE ───────────────


@pytest.mark.parametrize("seed", range(100))
def test_inv__result_has_required_keys(seed):
    rng = random.Random(seed)
    response = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 500)))
    out = _extract_code_blocks(response)
    assert "edits" in out
    assert "text_edits" in out
    assert "new_files" in out
    assert "reverts" in out


@pytest.mark.parametrize("seed", range(100))
def test_inv__types_correct(seed):
    rng = random.Random(seed)
    response = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    out = _extract_code_blocks(response)
    assert isinstance(out["edits"], dict)
    assert isinstance(out["text_edits"], dict)
    assert isinstance(out["new_files"], dict)
    assert isinstance(out["reverts"], list)


# ─────────────── PROPERTY: INERT-ZONE NEVER EXTRACTED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inert__think_bracket_isolates_inner_edits(seed):
    """Random edit syntax inside [think] never extracts to text_edits/edits."""
    rng = random.Random(seed)
    hidden_path = f"hidden_{seed}.py"
    response = (
        f"[think]\n"
        f"=== EDIT: {hidden_path} ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"[/think]\n"
    )
    # Optionally add real edit
    if rng.random() < 0.5:
        real_path = f"real_{seed}.py"
        response += (
            f"=== EDIT: {real_path} ===\n"
            f"[SEARCH]\nreal_old\n[/SEARCH]\n"
            f"[REPLACE]\nreal_new\n[/REPLACE]\n"
        )
    out = _extract_code_blocks(response)
    # Hidden path NEVER appears
    assert hidden_path not in out["text_edits"]
    assert hidden_path not in out["edits"]


@pytest.mark.parametrize("seed", range(50))
def test_inert__think_xml_isolates_inner_edits(seed):
    rng = random.Random(seed)
    hidden_path = f"hidden_{seed}.py"
    response = (
        f"<think>\n"
        f"=== EDIT: {hidden_path} ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"</think>\n"
    )
    out = _extract_code_blocks(response)
    assert hidden_path not in out["text_edits"]
    assert hidden_path not in out["edits"]


@pytest.mark.parametrize("seed", range(50))
def test_inert__fence_isolates_inner_edits(seed):
    rng = random.Random(seed)
    hidden_path = f"hidden_{seed}.py"
    response = (
        f"```\n"
        f"=== EDIT: {hidden_path} ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"```\n"
    )
    out = _extract_code_blocks(response)
    assert hidden_path not in out["text_edits"]
    assert hidden_path not in out["edits"]


# ─────────────── PROPERTY: REAL EDITS EXTRACTED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_real__simple_edit_extracted(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nold_content\n[/SEARCH]\n"
        f"[REPLACE]\nnew_content\n[/REPLACE]\n"
    )
    out = _extract_code_blocks(response)
    assert path in out["text_edits"]
    assert any(s == "old_content" and r == "new_content" for s, r in out["text_edits"][path])


@pytest.mark.parametrize("seed", range(30))
def test_real__multiple_edits_same_file(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nfirst_old\n[/SEARCH]\n"
        f"[REPLACE]\nfirst_new\n[/REPLACE]\n"
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nsecond_old\n[/SEARCH]\n"
        f"[REPLACE]\nsecond_new\n[/REPLACE]\n"
    )
    out = _extract_code_blocks(response)
    assert path in out["text_edits"]
    assert len(out["text_edits"][path]) == 2


# ─────────────── PROPERTY: REVERT IN INERT ZONE NOT EXTRACTED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_inert__revert_inside_file_body_ignored(seed):
    """[REVERT FILE: ...] inside a === FILE: === body is content, not a directive."""
    rng = random.Random(seed)
    new_path = f"new_{seed}.py"
    fake_revert_path = f"fake_revert_{seed}.py"
    response = (
        f"=== FILE: {new_path} ===\n"
        f"PROMPT = '[REVERT FILE: {fake_revert_path}]'\n"
        f"=== END FILE ===\n"
    )
    out = _extract_code_blocks(response)
    # The fake revert path inside the new-file body should NOT be in reverts
    assert fake_revert_path not in out["reverts"]
    # The new file IS extracted
    assert new_path in out["new_files"]


@pytest.mark.parametrize("seed", range(50))
def test_inert__revert_inside_edit_body_ignored(seed):
    rng = random.Random(seed)
    edit_path = f"edit_{seed}.py"
    fake_revert_path = f"fake_revert_{seed}.py"
    response = (
        f"=== EDIT: {edit_path} ===\n"
        f"[SEARCH]\nold\n[/SEARCH]\n"
        f"[REPLACE]\n# [REVERT FILE: {fake_revert_path}] documentation\nnew\n[/REPLACE]\n"
    )
    out = _extract_code_blocks(response)
    assert fake_revert_path not in out["reverts"]


# ─────────────── PROPERTY: REAL REVERT DIRECTIVE EXTRACTED ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_real__top_level_revert_extracted(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = f"[REVERT FILE: {path}]"
    out = _extract_code_blocks(response)
    assert path in out["reverts"]


# ─────────────── PROPERTY: NEW-FILE EXTRACTION ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_new_file__extracted(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    body = _random_body(rng)
    response = f"=== FILE: {path} ===\n{body}\n=== END FILE ==="
    out = _extract_code_blocks(response)
    assert path in out["new_files"]
    # Body content present
    assert body.strip() in out["new_files"][path]


# ─────────────── PROPERTY: EMPTY INPUT ───────────────


def test_empty__no_response():
    out = _extract_code_blocks("")
    assert out["text_edits"] == {}
    assert out["edits"] == {}
    assert out["new_files"] == {}
    assert out["reverts"] == []


@pytest.mark.parametrize("seed", range(30))
def test_random__no_edit_syntax_no_extractions(seed):
    """Random prose with no edit syntax → no extractions."""
    rng = random.Random(seed)
    # Use chars that don't accidentally form edit syntax
    chars = string.ascii_letters + " \n.,!?"
    response = "".join(rng.choice(chars) for _ in range(rng.randint(0, 500)))
    out = _extract_code_blocks(response)
    assert out["text_edits"] == {}
    assert out["edits"] == {}
    assert out["new_files"] == {}


# ─────────────── PROPERTY: MALFORMED EDITS DROPPED ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_malformed__missing_search_close_no_extract(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nold but no close\n"
        f"[REPLACE]\nnew\n[/REPLACE]\n"
    )
    out = _extract_code_blocks(response)
    # No valid pair → empty extraction
    assert path not in out["text_edits"] or out["text_edits"].get(path, []) == []


@pytest.mark.parametrize("seed", range(30))
def test_malformed__no_search_replace_at_all(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = f"=== EDIT: {path} ===\nJust some prose, no edit syntax.\n"
    out = _extract_code_blocks(response)
    assert path not in out["text_edits"] or out["text_edits"].get(path, []) == []


# ─────────────── PROPERTY: NEW FILE BODY CONTAINING EDIT SYNTAX ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_new_file__body_with_edit_syntax_NOT_spurious(seed):
    """A new file whose body literally contains edit syntax — must NOT
    trigger spurious extractions of those nested patterns."""
    rng = random.Random(seed)
    new_path = f"new_{seed}.py"
    fake_target = f"fake_{seed}.py"
    response = (
        f"=== FILE: {new_path} ===\n"
        f"TEMPLATE = '''\n"
        f"=== EDIT: {fake_target} ===\n"
        f"[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]\n"
        f"'''\n"
        f"=== END FILE ==="
    )
    out = _extract_code_blocks(response)
    # The new file is extracted
    assert new_path in out["new_files"]
    # The fake target is NOT
    assert fake_target not in out["text_edits"]


# ─────────────── PROPERTY: DETERMINISM ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_det__same_input_same_output(seed):
    rng = random.Random(seed)
    path = _random_path(rng)
    response = (
        f"=== EDIT: {path} ===\n"
        f"[SEARCH]\nold_{seed}\n[/SEARCH]\n"
        f"[REPLACE]\nnew_{seed}\n[/REPLACE]\n"
    )
    o1 = _extract_code_blocks(response)
    o2 = _extract_code_blocks(response)
    assert o1 == o2


# ─────────────── PROPERTY: NO CRASH ON ADVERSARIAL ───────────────


def test_adv__massive_input_no_crash():
    """1MB random input — should complete."""
    rng = random.Random(99)
    response = "".join(rng.choice(string.printable) for _ in range(1_000_000))
    out = _extract_code_blocks(response)
    assert isinstance(out, dict)


def test_adv__unbalanced_brackets_no_crash():
    response = "[[[[[]]]]] === EDIT: x ===  [SEARCH][/REPLACE]"
    out = _extract_code_blocks(response)
    assert isinstance(out, dict)


def test_adv__null_bytes_no_crash():
    response = "\x00\x00\x00 normal text \x01\x02"
    out = _extract_code_blocks(response)
    assert isinstance(out, dict)


def test_adv__unicode_in_paths():
    response = "=== EDIT: 北京.py ===\n[SEARCH]\nold\n[/SEARCH]\n[REPLACE]\nnew\n[/REPLACE]"
    out = _extract_code_blocks(response)
    assert isinstance(out, dict)


# ─────────────── PROPERTY: HEAVY MIX ───────────────


@pytest.mark.parametrize("seed", range(20))
def test_heavy__realistic_response(seed):
    """A realistic adversarial response with thinks, fences, multiple edits,
    multiple file creations, reverts — extracted correctly."""
    rng = random.Random(seed)
    fake_path = f"fake_{seed}.py"
    real_path1 = f"real1_{seed}.py"
    real_path2 = f"real2_{seed}.py"
    new_file_path = f"new_{seed}.py"
    revert_path = f"revert_{seed}.py"
    response = f"""
[think]
Let me consider this:
=== EDIT: {fake_path} ===
[SEARCH]old[/SEARCH][REPLACE]new[/REPLACE]
[/think]

```python
# Documentation example:
=== EDIT: {fake_path}_2 ===
[SEARCH]fenced[/SEARCH][REPLACE]fenced[/REPLACE]
```

[REVERT FILE: {revert_path}]

=== EDIT: {real_path1} ===
[SEARCH]
real1_old
[/SEARCH]
[REPLACE]
real1_new
[/REPLACE]

=== FILE: {new_file_path} ===
print('hello')
=== END FILE ===

=== EDIT: {real_path2} ===
[SEARCH]
real2_old
[/SEARCH]
[REPLACE]
real2_new
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    # Fake (in think / fence) NOT extracted
    assert fake_path not in out["text_edits"]
    assert f"{fake_path}_2" not in out["text_edits"]
    # Real edits extracted
    assert real_path1 in out["text_edits"]
    assert real_path2 in out["text_edits"]
    # New file extracted
    assert new_file_path in out["new_files"]
    # Real revert extracted
    assert revert_path in out["reverts"]
