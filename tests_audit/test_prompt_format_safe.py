"""Guard against the `.format()`-crash class of prompt bug.

A literal `{}` or `{0}` left in a prompt template (e.g. a code example like
`format_html('<script>{}</script>', ...)`) makes `TEMPLATE.format(**kwargs)`
raise `IndexError: Replacement index 0 out of range` — it crashes the coder on
EVERY instance, yet unit tests on the appliers never exercise it. Literal braces
in examples MUST be escaped as `{{ }}`.

This test parses every `*_PROMPT*` / `*_COT*` template and asserts it has no
auto-numbered or positional replacement fields (only named ones, which the
call sites supply). It is intentionally cheap and dependency-free.
"""
import string
import pytest

import core.prompts_v8 as P


def _templates():
    for name in dir(P):
        if not (name.endswith("_PROMPT") or "_PROMPT_V8" in name
                or "_COT" in name):
            continue
        val = getattr(P, name)
        if isinstance(val, str) and "{" in val:
            yield name, val


@pytest.mark.parametrize("name,template",
                         list(_templates()),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_no_auto_or_positional_format_fields(name, template):
    autos = [t for t in string.Formatter().parse(template)
             if t[1] == "" or (t[1] is not None and t[1].isdigit())]
    assert not autos, (
        f"{name} has {len(autos)} auto/positional {{}} field(s) — these crash "
        f".format() with IndexError. Escape literal braces in examples as {{{{ }}}}."
    )


def test_at_least_one_template_scanned():
    # Sanity: the discovery actually found templates (so the guard isn't vacuous).
    assert list(_templates()), "no prompt templates discovered to scan"
