"""Precision tests for the semantic-search code chunker (tools/embeddings.py).

Semantic search indexes these chunks; if a chunk is truncated, drops decorators,
duplicates a class body, or loses a unit, retrieval quality silently degrades.
These tests pin the chunker's invariants so that can't happen unnoticed.
"""
import os
import tempfile

import pytest

from tools.embeddings import (
    parse_code_chunks, _line_windows, _unit_start_line,
    _EMBED_WINDOW_CHARS,
)
import ast


# ─── helpers ──────────────────────────────────────────────────────────────────

def _chunk_dir(files: dict) -> str:
    """Write {relpath: source} into a fresh temp dir, return its path."""
    d = tempfile.mkdtemp(prefix="chunktest_")
    for rel, src in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
    return d


def _by_name(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c["name"], []).append(c)
    return out


# ─── _line_windows: the splitting primitive ─────────────────────────────────────

def test_window_short_input_is_single_complete_window():
    lines = ["a = 1", "b = 2", "c = 3"]
    wins = _line_windows(lines, max_chars=1000)
    assert wins == ["\n".join(lines)]          # one window, verbatim, no loss


def test_window_never_cuts_mid_line():
    lines = [f"line_{i} = {i}" for i in range(200)]
    wins = _line_windows(lines, max_chars=120)
    assert len(wins) > 1
    for w in wins:
        for ln in w.splitlines():
            assert ln in lines                 # every window line is an original line


def test_window_union_covers_every_line():
    lines = [f"x{i} = {i}" for i in range(300)]
    wins = _line_windows(lines, max_chars=150)
    covered = set()
    for w in wins:
        covered.update(w.splitlines())
    assert covered == set(lines)               # nothing dropped


def test_window_has_overlap_between_consecutive():
    lines = [f"v{i} = {i}" for i in range(120)]
    wins = _line_windows(lines, max_chars=100)
    assert len(wins) >= 2
    # consecutive windows share at least one boundary line (overlap, no gap)
    for a, b in zip(wins, wins[1:]):
        assert set(a.splitlines()) & set(b.splitlines())


def test_window_each_fits_budget_when_lines_are_small():
    lines = [f"s{i} = {i}" for i in range(500)]
    maxc = 200
    wins = _line_windows(lines, max_chars=maxc)
    # a window may exceed only if a SINGLE line already exceeds the budget;
    # here every line is tiny, so all windows must respect the budget.
    for w in wins:
        assert len(w) <= maxc + max(len(x) for x in lines)


# ─── parse_code_chunks: completeness ────────────────────────────────────────────

SAMPLE = '''\
"""Module docstring for sample."""
import os
import sys

CONST = 42


def top_level(a, b):
    """A top-level function."""
    return a + b


async def async_fn(x):
    return await x


class Widget:
    """A widget class."""
    kind = "gadget"

    def __init__(self, n):
        self.n = n

    @property
    def doubled(self):
        return self.n * 2

    @staticmethod
    def make():
        def helper():        # nested function — stays inside make()
            return 1
        return helper()


if __name__ == "__main__":
    print(top_level(1, 2))
'''


def test_captures_every_unit():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    names = {c["name"].split(":")[0] for c in chunks}  # strip :line / #win suffix-ish
    # qualnames present (path::qual)
    quals = {c["name"].split("::", 1)[1].split(":")[0] for c in chunks}
    for expected in ["top_level", "async_fn", "Widget", "Widget.__init__",
                     "Widget.doubled", "Widget.make", "<module>"]:
        assert expected in quals, f"missing chunk for {expected}; got {sorted(quals)}"


def test_nested_function_not_emitted_separately():
    d = _chunk_dir({"sample.py": SAMPLE})
    quals = {c["name"].split("::", 1)[1].split(":")[0] for c in parse_code_chunks(d)}
    # helper() lives inside make() — must NOT be its own chunk
    assert not any(q.endswith("helper") for q in quals)


def test_function_chunk_is_full_body_not_truncated():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    fn = next(c for c in chunks if "::top_level:" in c["name"])
    # full body present: def line AND the return line
    assert "def top_level(a, b):" in fn["text"]
    assert "return a + b" in fn["text"]
    assert "A top-level function." in fn["text"]


def test_decorators_are_included():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    prop = next(c for c in chunks if "::Widget.doubled:" in c["name"])
    assert "@property" in prop["text"], "decorator dropped — fidelity loss"
    made = next(c for c in chunks if "::Widget.make:" in c["name"])
    assert "@staticmethod" in made["text"]


def test_decorator_line_is_the_reported_start():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    prop = next(c for c in chunks if "::Widget.doubled:" in c["name"])
    src_lines = SAMPLE.splitlines()
    # the reported line points at the @property decorator, not the def
    assert src_lines[prop["line"] - 1].strip() == "@property"


def test_class_chunk_is_header_only():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    cls = next(c for c in chunks if "::Widget:" in c["name"]
               and "." not in c["name"].split("::", 1)[1].split(":")[0])
    assert "class Widget:" in cls["text"]
    assert 'kind = "gadget"' in cls["text"]        # class-level attr kept
    assert "A widget class." in cls["text"]
    # methods must NOT appear in the class header chunk (no duplication)
    assert "def __init__" not in cls["text"]
    assert "def doubled" not in cls["text"]


def test_module_chunk_has_toplevel_code():
    d = _chunk_dir({"sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    mod = next(c for c in chunks if "::<module>" in c["name"])
    assert "import os" in mod["text"]
    assert "CONST = 42" in mod["text"]
    assert "Module docstring" in mod["text"]
    assert '__main__' in mod["text"]
    # but NOT function/class bodies (those are their own chunks)
    assert "return a + b" not in mod["text"]


def test_text_begins_with_path_and_qual():
    d = _chunk_dir({"pkg/sample.py": SAMPLE})
    chunks = parse_code_chunks(d)
    fn = next(c for c in chunks if "::top_level:" in c["name"])
    assert fn["text"].startswith("pkg/sample.py::top_level\n")
    assert fn["file"] == "pkg/sample.py"


# ─── flexibility: large units window, small units don't ─────────────────────────

def test_large_function_is_windowed_not_truncated():
    big_body = "\n".join(f"    step_{i} = compute({i})" for i in range(1200))
    src = f"def huge():\n{big_body}\n    return step_0\n"
    d = _chunk_dir({"big.py": src})
    chunks = [c for c in parse_code_chunks(d) if "::huge:" in c["name"]]
    assert len(chunks) > 1, "huge function must split into multiple windows"
    # every original body line survives across the windows (no truncation)
    all_text = "\n".join(c["text"] for c in chunks)
    for i in range(1200):
        assert f"step_{i} = compute({i})" in all_text
    # windows are labelled #k/N
    assert all("#" in c["name"] for c in chunks)


def test_small_unit_is_single_window():
    d = _chunk_dir({"s.py": "def tiny():\n    return 1\n"})
    chunks = [c for c in parse_code_chunks(d) if "::tiny:" in c["name"]]
    assert len(chunks) == 1
    assert "#" not in chunks[0]["name"]


# ─── robustness ─────────────────────────────────────────────────────────────────

def test_syntax_error_file_still_indexed():
    d = _chunk_dir({"broken.py": "def x(:\n    this is not valid python @@@\n"})
    chunks = parse_code_chunks(d)
    assert chunks, "unparseable file must still produce chunks (fallback)"
    assert all(c["file"] == "broken.py" for c in chunks)
    assert any("not valid python" in c["text"] for c in chunks)


def test_skips_vendor_dirs():
    d = _chunk_dir({
        "real.py": "def keep():\n    return 1\n",
        ".git/x.py": "def gone():\n    return 1\n",
        "node_modules/y.py": "def gone2():\n    return 1\n",
        "__pycache__/z.py": "def gone3():\n    return 1\n",
    })
    quals = {c["name"].split("::", 1)[1].split(":")[0] for c in parse_code_chunks(d)}
    assert "keep" in quals
    assert "gone" not in quals and "gone2" not in quals and "gone3" not in quals


def test_empty_and_comment_only_files():
    d = _chunk_dir({"empty.py": "", "comments.py": "# just a comment\n"})
    chunks = parse_code_chunks(d)
    # empty.py yields nothing; comments.py has a top-level comment (module chunk
    # only if there's a real statement — a bare comment is not an AST statement)
    assert all(c["file"] != "empty.py" for c in chunks)


def test_deterministic():
    d = _chunk_dir({"a.py": SAMPLE, "b.py": "def f():\n    return 2\n"})
    a = parse_code_chunks(d)
    b = parse_code_chunks(d)
    assert [c["name"] for c in a] == [c["name"] for c in b]
    assert [c["text"] for c in a] == [c["text"] for c in b]


def test_names_unique():
    d = _chunk_dir({"sample.py": SAMPLE})
    names = [c["name"] for c in parse_code_chunks(d)]
    assert len(names) == len(set(names)), "chunk names must be unique (embed/rank keys)"


def test_overloaded_same_name_methods_disambiguated_by_line():
    # property getter + setter share the qualname `x` — line suffix keeps them distinct
    src = (
        "class C:\n"
        "    @property\n"
        "    def x(self):\n"
        "        return self._x\n"
        "    @x.setter\n"
        "    def x(self, v):\n"
        "        self._x = v\n"
    )
    d = _chunk_dir({"c.py": src})
    names = [c["name"] for c in parse_code_chunks(d) if "::C.x:" in c["name"]]
    assert len(names) == 2 and len(set(names)) == 2


def test_unit_start_line_includes_decorators():
    src = "@deco\n@deco2\ndef f():\n    return 1\n"
    tree = ast.parse(src)
    fn = tree.body[0]
    assert _unit_start_line(fn) == 1          # the first @deco, not the def at line 3


def test_chunk_text_never_exceeds_window_plus_header():
    # synthetic repo with assorted sizes — no chunk text blows past the budget
    # by more than one (long) line + the path header.
    d = _chunk_dir({"sample.py": SAMPLE})
    for c in parse_code_chunks(d):
        assert len(c["text"]) <= _EMBED_WINDOW_CHARS + 4000
