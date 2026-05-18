"""DEEP FUZZ audit of remaining utility functions:
  • `_format_research_cache`
  • `extract_relevant_sections`
  • `_render_plan_with_line_numbers`
  • `_strip_label`
  • `_parse_code_arg`
  • `_strip_think`
  • `_mask_inert_zones`
"""
import pytest
import random
import string
from workflows.code import (
    _format_research_cache,
    _mask_inert_zones,
)
from tools.codebase import extract_relevant_sections
from core.tool_call import (
    _strip_label, _parse_code_arg, _strip_think,
    _render_plan_with_line_numbers,
)


# ─────────────── _format_research_cache ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_format_cache__handles_random(seed):
    rng = random.Random(seed)
    n_entries = rng.randint(0, 20)
    cache = {}
    for i in range(n_entries):
        key_type = rng.choice(["REFS", "PURPOSE", "SEARCH", "DETAIL"])
        cache[f"{key_type}:item_{i}"] = "".join(rng.choice(string.ascii_letters + " \n") for _ in range(rng.randint(10, 100)))
    out = _format_research_cache(cache)
    assert isinstance(out, str)


def test_format_cache__none_returns_empty():
    assert _format_research_cache(None) == ""


def test_format_cache__empty_dict_returns_empty():
    assert _format_research_cache({}) == ""


@pytest.mark.parametrize("seed", range(30))
def test_format_cache__has_banner_when_content(seed):
    cache = {"REFS:foo": "content"}
    out = _format_research_cache(cache)
    assert "PRE-LOADED RESEARCH" in out


@pytest.mark.parametrize("seed", range(20))
def test_format_cache__max_chars_respected(seed):
    rng = random.Random(seed)
    # Very long values
    cache = {
        f"REFS:item_{i}": "x" * 5000
        for i in range(20)
    }
    out = _format_research_cache(cache, max_chars=1000)
    # Output should not be wildly larger than max_chars
    assert len(out) < 3000


# ─────────────── extract_relevant_sections ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_extract_relevant__handles_random(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(0, 500)
    src = "\n".join(
        "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(0, 40)))
        for _ in range(n_lines)
    )
    hints = "".join(rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(0, 100)))
    out = extract_relevant_sections(src, hints, context_lines=10, max_short_file=200)
    assert isinstance(out, str)


def test_extract_relevant__empty_source():
    out = extract_relevant_sections("", "any hints", context_lines=10, max_short_file=200)
    assert isinstance(out, str)


@pytest.mark.parametrize("seed", range(30))
def test_extract_relevant__short_file_full_content(seed):
    """Short files (under max_short_file) return whole content."""
    rng = random.Random(seed)
    n_lines = rng.randint(1, 50)
    lines = [f"line_{i}_marker" for i in range(n_lines)]
    src = "\n".join(lines)
    out = extract_relevant_sections(src, "anything", context_lines=10, max_short_file=200)
    # All lines should be present
    for line in lines:
        assert line in out


# ─────────────── _render_plan_with_line_numbers ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_render__line_count_preserved(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(0, 50)
    plan = "\n".join(
        "".join(rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(0, 40)))
        for _ in range(n_lines)
    )
    out = _render_plan_with_line_numbers(plan)
    assert out.count('\n') == plan.count('\n')


@pytest.mark.parametrize("seed", range(30))
def test_render__all_lines_numbered(seed):
    rng = random.Random(seed)
    n_lines = rng.randint(1, 30)
    plan = "\n".join(f"content_{i}" for i in range(n_lines))
    out = _render_plan_with_line_numbers(plan)
    # Each line number should be present
    for i in range(1, n_lines + 1):
        assert str(i) in out


def test_render__empty_plan():
    assert _render_plan_with_line_numbers("") == ""


# ─────────────── _strip_label ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_strip_label__no_label_passes_through(seed):
    rng = random.Random(seed)
    arg = "".join(rng.choice(string.ascii_letters + "/.-_") for _ in range(rng.randint(1, 30)))
    clean, label = _strip_label(arg)
    if "#" not in arg:
        assert label is None
        assert clean == arg.strip()


@pytest.mark.parametrize("seed", range(30))
def test_strip_label__hash_label_extracted(seed):
    rng = random.Random(seed)
    arg = f"file_{seed}.py"
    label = f"label_{seed}"
    full = f"{arg} #{label}"
    clean, extracted = _strip_label(full)
    assert clean == arg
    assert extracted == label


# ─────────────── _parse_code_arg ───────────────


@pytest.mark.parametrize("seed", range(30))
def test_parse_code_arg__path_only(seed):
    rng = random.Random(seed)
    path = f"pkg/file_{seed}.py"
    p, ranges = _parse_code_arg(path)
    assert p == path
    assert ranges is None


@pytest.mark.parametrize("seed", range(30))
def test_parse_code_arg__with_range(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 100)
    b = a + rng.randint(0, 100)
    arg = f"file.py {a}-{b}"
    p, ranges = _parse_code_arg(arg)
    assert p == "file.py"
    assert (a, b) in ranges


@pytest.mark.parametrize("seed", range(30))
def test_parse_code_arg__multi_range(seed):
    rng = random.Random(seed)
    ranges_list = [(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(rng.randint(1, 5))]
    ranges_list = [(a, b) if a <= b else (b, a) for a, b in ranges_list]
    range_str = ", ".join(f"{a}-{b}" for a, b in ranges_list)
    arg = f"file.py {range_str}"
    p, ranges = _parse_code_arg(arg)
    assert p == "file.py"
    if ranges is not None:
        for r in ranges_list:
            assert r in ranges


# ─────────────── _strip_think ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_strip_think__removes_brackets(seed):
    rng = random.Random(seed)
    secret = f"SECRET_{seed}_REASONING"
    text = f"before [think]{secret}[/think] after"
    out = _strip_think(text)
    assert secret not in out
    assert "before" in out
    assert "after" in out


@pytest.mark.parametrize("seed", range(50))
def test_strip_think__removes_xml(seed):
    rng = random.Random(seed)
    secret = f"SECRET_{seed}_REASONING"
    text = f"before <think>{secret}</think> after"
    out = _strip_think(text)
    assert secret not in out


@pytest.mark.parametrize("seed", range(50))
def test_strip_think__no_think_passthrough(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(string.ascii_letters + " ") for _ in range(rng.randint(0, 200)))
    out = _strip_think(text)
    # If no think markers, output is identical (or stripped)
    if "[think]" not in text and "<think>" not in text:
        assert text in out or out == text


@pytest.mark.parametrize("seed", range(30))
def test_strip_think__multiple_think_blocks(seed):
    """Multiple think blocks — all stripped."""
    rng = random.Random(seed)
    text = (
        f"prelude\n"
        f"[think]secret_1_{seed}[/think]\n"
        f"middle\n"
        f"<think>secret_2_{seed}</think>\n"
        f"end"
    )
    out = _strip_think(text)
    assert f"secret_1_{seed}" not in out
    assert f"secret_2_{seed}" not in out
    assert "prelude" in out
    assert "middle" in out
    assert "end" in out


# ─────────────── _mask_inert_zones ───────────────


@pytest.mark.parametrize("seed", range(50))
def test_mask_inert__length_preserved(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    out = _mask_inert_zones(text)
    assert len(out) == len(text)


@pytest.mark.parametrize("seed", range(50))
def test_mask_inert__newlines_preserved(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 200)))
    out = _mask_inert_zones(text)
    assert out.count('\n') == text.count('\n')


@pytest.mark.parametrize("seed", range(30))
def test_mask_inert__think_content_blanked(seed):
    rng = random.Random(seed)
    secret = f"SECRET_INSIDE_{seed}"
    text = f"real [think]{secret}[/think] more"
    out = _mask_inert_zones(text)
    assert secret not in out


@pytest.mark.parametrize("seed", range(30))
def test_mask_inert__fenced_blanked(seed):
    rng = random.Random(seed)
    secret = f"FENCED_{seed}"
    text = f"real\n```\n{secret}\n```\nmore"
    out = _mask_inert_zones(text)
    assert secret not in out


@pytest.mark.parametrize("seed", range(30))
def test_mask_inert__regular_content_unchanged(seed):
    rng = random.Random(seed)
    # No think/fence markers
    safe = string.ascii_letters + " "
    text = "".join(rng.choice(safe) for _ in range(rng.randint(0, 200)))
    out = _mask_inert_zones(text)
    assert out == text
