"""Precision tests for [DEPENDSON:] — extract_dependencies (core/exploration_tools).

DEPENDSON answers "what does this symbol depend ON" (its callees), the reverse of
DEPENDENCY's "what depends on this". It must be PRECISE: a false dependency (a
builtin or an unrelated same-named project def) misleads the agent. These tests
pin true-positive recall AND false-positive exclusion.
"""
import os
import tempfile

from core.exploration_tools import extract_dependencies


def _repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="depson_")
    for rel, src in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p) or d, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
    return d


SAMPLE = '''\
def helper_a(x):
    return x + 1

def helper_b(x):
    return x * 2

class Engine:
    def run(self, data):
        # depends on two project functions + a builtin (len) + a method (.append)
        out = []
        for d in data:
            out.append(helper_a(d))
        total = helper_b(len(out))
        return total

    def unrelated(self):
        return 0
'''


def test_finds_real_project_dependencies():
    d = _repo({"m.py": SAMPLE})
    res = extract_dependencies("Engine.run", d)
    assert "helper_a" in res
    assert "helper_b" in res
    assert "m.py:" in res            # def sites reported


def test_excludes_builtins():
    d = _repo({"m.py": SAMPLE})
    res = extract_dependencies("Engine.run", d)
    # len() is a builtin — must NOT be reported as a dependency
    assert "len" not in res.replace("(only builtins", "")  # guard against the stopword note


def test_excludes_common_methods_even_if_same_named_project_def_exists():
    # a project function literally named `append` and `get` exist; a symbol that
    # calls list.append()/dict.get() must NOT resolve to them (false positive).
    files = {
        "util.py": "def append(x):\n    return x\n\ndef get(k):\n    return k\n",
        "main.py": (
            "def process(items, cfg):\n"
            "    items.append(1)\n"          # list method, not util.append
            "    return cfg.get('k')\n"       # dict method, not util.get
        ),
    }
    d = _repo(files)
    res = extract_dependencies("process", d)
    assert "no project-internal dependencies" in res or "append" not in res
    assert "util.py" not in res, f"false positive on append/get:\n{res}"


def test_recursion_not_reported_as_dependency():
    d = _repo({"r.py": "def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)\n"})
    res = extract_dependencies("fact", d)
    # fact calls itself (recursion, excluded) + builtins only → no project deps
    assert "no project-internal dependencies" in res
    assert "  • " not in res              # no dependency bullets listed


def test_dotted_form_scopes_to_class():
    d = _repo({"m.py": SAMPLE})
    res = extract_dependencies("Engine.run", d)
    assert "DEPENDSON: Engine.run" in res
    # `unrelated` is a sibling method NOT called by run → not a dependency
    assert "unrelated" not in res


def test_pure_builtin_function_reports_none():
    d = _repo({"m.py": "def f(s):\n    return s.strip().upper().split(',')\n"})
    res = extract_dependencies("f", d)
    assert "no project-internal dependencies" in res


def test_not_found_symbol_is_helpful():
    d = _repo({"m.py": "def f():\n    return 1\n"})
    res = extract_dependencies("does_not_exist", d)
    assert "No `def does_not_exist`" in res or "not" in res.lower()
    assert "REFS" in res                # points at the right alternative tool


def test_invalid_identifier():
    d = _repo({"m.py": "def f():\n    return 1\n"})
    assert "not a valid identifier" in extract_dependencies("a b c!", d)


def test_deterministic():
    d = _repo({"m.py": SAMPLE})
    assert extract_dependencies("Engine.run", d) == extract_dependencies("Engine.run", d)


def test_reports_definition_sites_with_line_numbers():
    files = {"lib.py": "def target():\n    return 1\n",
             "use.py": "def caller():\n    return target()\n"}
    d = _repo(files)
    res = extract_dependencies("caller", d)
    assert "target" in res
    assert "lib.py:1" in res            # exact def site
