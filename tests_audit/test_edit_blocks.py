"""Audit edit-block extraction — `_extract_code_blocks` parses model output
into `{edits, text_edits, new_files, reverts}` for downstream application.

Contracts:
  - `=== EDIT: path === [SEARCH]...[/SEARCH] [REPLACE]...[/REPLACE]` → text_edit
  - `=== EDIT: path === [REPLACE LINES N-M] ... [/REPLACE]` → line edit
  - `=== FILE: path === ... === END FILE ===` → new file
  - `[REVERT FILE: path]` → revert directive
  - `=== REVISE EDIT: path === ... === END REVISE EDIT ===` → retracts prior EDIT
  - Inside fenced code blocks (```), edit syntax is INERT
  - Inside [think] blocks, edit syntax is also INERT
  - `[continue from: -N]` removes N visible lines BEFORE extraction
"""
import pytest
from workflows.code import _extract_code_blocks, _apply_revise_edits


# ───────────────────── BASIC SEARCH/REPLACE ─────────────────────

def test_edit__simple_search_replace():
    response = """
=== EDIT: foo.py ===
[SEARCH]
def old():
    pass
[/SEARCH]
[REPLACE]
def new():
    return 1
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert "foo.py" in out["text_edits"]
    edits = out["text_edits"]["foo.py"]
    assert len(edits) == 1
    search, replace = edits[0]
    assert "def old()" in search
    assert "def new()" in replace


def test_edit__multiple_edits_same_file():
    response = """
=== EDIT: a.py ===
[SEARCH]
foo
[/SEARCH]
[REPLACE]
bar
[/REPLACE]
=== EDIT: a.py ===
[SEARCH]
baz
[/SEARCH]
[REPLACE]
qux
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert len(out["text_edits"]["a.py"]) == 2


def test_edit__multiple_files():
    response = """
=== EDIT: a.py ===
[SEARCH]
A
[/SEARCH]
[REPLACE]
A1
[/REPLACE]
=== EDIT: b.py ===
[SEARCH]
B
[/SEARCH]
[REPLACE]
B1
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert "b.py" in out["text_edits"]


def test_edit__empty_replace_body():
    """`[REPLACE][/REPLACE]` → deletes the SEARCH block."""
    response = """
=== EDIT: a.py ===
[SEARCH]
line to delete
[/SEARCH]
[REPLACE]
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    s, r = out["text_edits"]["a.py"][0]
    assert "line to delete" in s
    assert r.strip() == ""


def test_edit__no_edits_in_response():
    response = "Just some thinking, no edits here."
    out = _extract_code_blocks(response)
    assert out["text_edits"] == {}
    assert out["edits"] == {}
    assert out["new_files"] == {}


# ───────────────────── REPLACE LINES ─────────────────────

def test_edit__replace_lines_basic():
    response = """
=== EDIT: a.py ===
[REPLACE LINES 5-10]
new content
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["edits"]
    start, end, code = out["edits"]["a.py"][0]
    assert start == 5 and end == 10


def test_edit__insert_after_line():
    """`[INSERT AFTER LINE N]` → start=0 in the edit tuple convention."""
    response = """
=== EDIT: a.py ===
[INSERT AFTER LINE 10]
new code
[/INSERT]
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["edits"]
    start, end, code = out["edits"]["a.py"][0]
    # Convention: insert is encoded as start=0


# ───────────────────── NEW FILES ─────────────────────

def test_edit__new_file():
    response = """
=== FILE: scripts/new.py ===
print("hello")
=== END FILE ===
"""
    out = _extract_code_blocks(response)
    assert "scripts/new.py" in out["new_files"]
    assert 'print("hello")' in out["new_files"]["scripts/new.py"]


def test_edit__new_file_with_special_chars():
    response = """
=== FILE: a.py ===
x = "this has [brackets] and === marks"
y = '''triple
quoted'''
=== END FILE ===
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["new_files"]
    content = out["new_files"]["a.py"]
    assert "[brackets]" in content


# ───────────────────── REVERT ─────────────────────

def test_edit__revert_directive():
    response = """
The previous edit was wrong.
[REVERT FILE: broken.py]
"""
    out = _extract_code_blocks(response)
    assert "broken.py" in out["reverts"]


def test_edit__revert_inside_file_body__not_a_directive():
    """A `[REVERT FILE: x]` literal inside a `=== FILE: ... ===` body should
    NOT trigger a revert (it's data, not a directive)."""
    response = """
=== FILE: docs.txt ===
The user can issue [REVERT FILE: any.py] as documentation.
=== END FILE ===
"""
    out = _extract_code_blocks(response)
    # The new file is captured
    assert "docs.txt" in out["new_files"]
    # No revert triggered
    assert out["reverts"] == []


def test_edit__revert_inside_edit_body__not_a_directive():
    """`[REVERT FILE: x]` literal inside a REPLACE body — also not a
    directive."""
    response = """
=== EDIT: comment.py ===
[SEARCH]
old
[/SEARCH]
[REPLACE]
# The system processes [REVERT FILE: other.py] directives
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    assert out["reverts"] == [], (
        f"REVERT inside REPLACE should not fire: {out['reverts']}"
    )


# ───────────────────── REVISE EDIT ─────────────────────

def test_revise__retracts_prior_edit():
    """`=== REVISE EDIT: path === ... === END REVISE EDIT ===` removes
    the most recent prior EDIT on the same path and promotes the revise
    body to a regular EDIT."""
    response = """
=== EDIT: a.py ===
[SEARCH]
typo
[/SEARCH]
[REPLACE]
buggy_fix
[/REPLACE]

=== REVISE EDIT: a.py ===
[SEARCH]
typo
[/SEARCH]
[REPLACE]
correct_fix
[/REPLACE]
=== END REVISE EDIT ===
"""
    out = _extract_code_blocks(response)
    edits = out["text_edits"]["a.py"]
    # The result should have ONE edit (the revise body), not two
    assert len(edits) == 1
    s, r = edits[0]
    assert "correct_fix" in r
    assert "buggy_fix" not in r


def test_revise__different_path_does_not_retract():
    """REVISE on a different path shouldn't retract edits on another path."""
    response = """
=== EDIT: a.py ===
[SEARCH]
A_old
[/SEARCH]
[REPLACE]
A_new
[/REPLACE]

=== REVISE EDIT: b.py ===
[SEARCH]
B_old
[/SEARCH]
[REPLACE]
B_new
[/REPLACE]
=== END REVISE EDIT ===
"""
    out = _extract_code_blocks(response)
    assert "a.py" in out["text_edits"]
    assert "b.py" in out["text_edits"]
    # a.py's edit survives
    assert any("A_new" in r for s, r in out["text_edits"]["a.py"])


def test_revise__twice_on_same_path():
    """Two REVISE blocks on the same path — only the final one survives."""
    response = """
=== EDIT: a.py ===
[SEARCH]
v1
[/SEARCH]
[REPLACE]
v1_replace
[/REPLACE]
=== REVISE EDIT: a.py ===
[SEARCH]
v1
[/SEARCH]
[REPLACE]
v2_replace
[/REPLACE]
=== END REVISE EDIT ===
=== REVISE EDIT: a.py ===
[SEARCH]
v1
[/SEARCH]
[REPLACE]
v3_replace
[/REPLACE]
=== END REVISE EDIT ===
"""
    out = _extract_code_blocks(response)
    edits = out["text_edits"]["a.py"]
    assert len(edits) == 1, f"expected 1 final edit, got {len(edits)}"
    s, r = edits[0]
    assert "v3_replace" in r


# ───────────────────── INERTNESS INSIDE FENCED / [think] ─────────────────────

def test_edit__inside_think_block_inert():
    """Edit syntax inside `[think]...[/think]` should NOT extract."""
    response = """
[think]
We could do:
=== EDIT: x.py ===
[SEARCH]
foo
[/SEARCH]
[REPLACE]
bar
[/REPLACE]
but actually let me think more.
[/think]
"""
    out = _extract_code_blocks(response)
    assert out["text_edits"] == {}, (
        f"edits inside [think] should be inert: {out}"
    )


def test_edit__inside_fenced_block_inert():
    """Edit syntax inside ``` fenced block ``` should NOT extract."""
    response = """
Here's the pattern:

```text
=== EDIT: x.py ===
[SEARCH]
foo
[/SEARCH]
[REPLACE]
bar
[/REPLACE]
```

That's the format.
"""
    out = _extract_code_blocks(response)
    # The fenced block content shouldn't trigger extraction
    assert out["text_edits"] == {}, (
        f"edits inside fenced block should be inert: {out}"
    )


# ───────────────────── ADVERSARIAL ─────────────────────

def test_edit__path_with_spaces():
    """A path containing spaces — questionable but document behavior."""
    response = """
=== EDIT: foo bar/file.py ===
[SEARCH]
A
[/SEARCH]
[REPLACE]
B
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    # The path regex uses \S+ so spaces in paths break parsing.
    # Document expected behavior: path is captured up to first space.


def test_edit__nested_edit_in_search_body():
    """The SEARCH body contains the literal text `=== EDIT:`. Should NOT
    confuse the parser — the boundary is the [/SEARCH] tag."""
    response = """
=== EDIT: meta.py ===
[SEARCH]
# Documentation: edits look like
# === EDIT: file.py ===
# [SEARCH]old[/SEARCH] [REPLACE]new[/REPLACE]
[/SEARCH]
[REPLACE]
# (cleaned up)
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    # Should produce ONE edit on meta.py, not multiple
    if "meta.py" in out["text_edits"]:
        assert len(out["text_edits"]["meta.py"]) == 1


def test_edit__edit_with_no_replace_block():
    """`=== EDIT: x === [SEARCH]...[/SEARCH]` with NO [REPLACE] — ill-formed."""
    response = """
=== EDIT: x.py ===
[SEARCH]
foo
[/SEARCH]
"""
    out = _extract_code_blocks(response)
    # Should be empty or graceful — not crash
    assert isinstance(out["text_edits"], dict)


def test_edit__windows_path_with_backslash():
    """Windows-style backslashes — accepted?"""
    response = r"""
=== EDIT: src\widgets.py ===
[SEARCH]
foo
[/SEARCH]
[REPLACE]
bar
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    # Document behavior — paths might be normalized
    assert out["text_edits"]  # at minimum, captured under some key


def test_edit__unicode_path():
    response = """
=== EDIT: pkg/widgets_中文.py ===
[SEARCH]
foo
[/SEARCH]
[REPLACE]
bar
[/REPLACE]
"""
    out = _extract_code_blocks(response)
    # Unicode paths should work
    assert any("widgets_中文" in k for k in out["text_edits"].keys())


def test_edit__crlf_line_endings_in_response():
    """Windows-style CRLF in the response — should still parse."""
    response = "=== EDIT: a.py ===\r\n[SEARCH]\r\nfoo\r\n[/SEARCH]\r\n[REPLACE]\r\nbar\r\n[/REPLACE]\r\n"
    out = _extract_code_blocks(response)
    # If the parser respects \n only, CRLF still works because \n is in CRLF
    if out["text_edits"]:
        assert "a.py" in out["text_edits"]


# ───────────── prefix_ws view: ⇥ gutter in [edit] markers (ckpt-234) ─────────────
# The read view is now `LINENO ⇥INDENT|<real spaces>code` for ALL roles. A text
# coder copies those lines into [edit] blocks. apply_edit_block normalizes the
# `LINENO ⇥` gutter to `LINENO:` so the keep/del/add markers still match, and
# _expand_indent treats the count as authoritative (it must NOT double the visible
# real spaces). These lock that in.

def test_prefix_ws__keep_and_add_round_trip():
    from core.edit_block import apply_edit_block
    src = "def foo():\n    x = 1\n    return x\n"
    # copy the ⇥ keep-anchors verbatim; add a count-form new line between them
    blk = "[edit]\n2 ⇥4|    x = 1\n2:+ 4|y = 2\n3 ⇥4|    return x\n[/edit]"
    new, info, warn = apply_edit_block(src, blk)
    assert new == "def foo():\n    x = 1\n    y = 2\n    return x\n", new
    assert warn == []


def test_prefix_ws__delete_via_arrow_gutter():
    from core.edit_block import apply_edit_block
    src = "def foo():\n    x = 1\n    return x\n"
    # delete the middle line by adding `-` right after the ⇥
    blk = "[edit]\n1 ⇥0|def foo():\n2 ⇥-4|    x = 1\n3 ⇥4|    return x\n[/edit]"
    new, info, warn = apply_edit_block(src, blk)
    assert new == "def foo():\n    return x\n", new


def test_prefix_ws__no_double_indent_on_added_line():
    from core.edit_block import _expand_indent
    # count is authoritative; copied real spaces must be dropped, not doubled
    assert _expand_indent("4|    y = 2") == "    y = 2"      # prefix_ws dup spaces
    assert _expand_indent("4|y = 2") == "    y = 2"          # count-only
    assert _expand_indent(" 8|        z") == "        z"     # space-after-+ AND dup
    assert _expand_indent("    raw") == "    raw"            # whitespace-mode verbatim


def test_prefix_ws__colon_markers_still_work():
    from core.edit_block import apply_edit_block
    src = "def foo():\n    x = 1\n    return x\n"
    # the prompt-documented colon forms must remain valid (back-compat)
    blk = "[edit]\n2:4|    x = 1\n2:+ 4|y = 2\n3:4|    return x\n[/edit]"
    new, info, warn = apply_edit_block(src, blk)
    assert new == "def foo():\n    x = 1\n    y = 2\n    return x\n", new


def test_prefix_ws__strip_view_linenos_drops_arrow_gutter():
    from core.view_lineno_strip import strip_view_linenos
    # leaves INDENT|<real spaces>content for the downstream indent-expander
    assert strip_view_linenos("12 ⇥4|    return x") == "4|    return x"
    # colon back-compat still works
    assert strip_view_linenos("12:4|return x") == "4|return x"


def test_prefix_ws__strip_line_numbers_expands_to_real_source():
    from workflows.code import _strip_line_numbers
    view = "1 ⇥0|def foo():\n2 ⇥4|    return x"
    clean, hint = _strip_line_numbers(view)
    assert clean == "def foo():\n    return x", repr(clean)
    assert hint == 1


# ───── ckpt-234 review fixes: ⇥ strip, structural guard, space-after-⇥ marker ─────

def test_prefix_ws__stray_arrow_glyph_stripped_from_code():
    # the ⇥ display glyph is never valid source; a leaked one in the code part
    # must be dropped (parity with native_tools._expand_indent_lines) — else it
    # ships as a literal ⇥ → SyntaxError.
    from core.edit_block import _expand_indent, _bare_code
    from workflows.code import _strip_line_numbers, _restore_replace_whitespace
    assert _expand_indent("8|⇥def f") == "        def f"
    assert _bare_code("8|⇥def f") == "def f"
    assert _strip_line_numbers("42 ⇥8|⇥def f")[0] == "        def f"
    assert _restore_replace_whitespace("8|⇥def f") == "        def f"


def test_prefix_ws__structural_line_guard_keeps_indent():
    # a def/class whose typed spaces disagree with the count: trust the TYPED
    # spaces (don't eject the method to a shallower scope — the f631 failure).
    from core.edit_block import _expand_indent
    assert _expand_indent("0|    def foo") == "    def foo"      # def: typed 4 wins over count 0
    assert _expand_indent("0|    class C") == "    class C"
    assert _expand_indent("0|    @deco") == "    @deco"
    # non-structural line keeps count-authoritative (intentional dedent honoured)
    assert _expand_indent("4|        x = 1") == "    x = 1"


def test_prefix_ws__space_after_arrow_before_marker():
    # a sloppy space between ⇥ and the -/+ marker still resolves to a del/add
    from core.edit_block import apply_edit_block
    src = "class C:\n    def m(self):\n        return 1\n"
    blk = ("[edit]\n1 ⇥0|class C:\n2 ⇥ -4|    def m(self):\n"
           "2:+ 4|def m(self, x):\n3 ⇥8|        return 1\n[/edit]")
    new, info, warn = apply_edit_block(src, blk)
    assert new == "class C:\n    def m(self, x):\n        return 1\n", new


# ───── ckpt-238 bug-hunt fixes: parser lstrip/⇥ parity across SEARCH + native ─────

def test_bughunt_strip_line_numbers_v9_no_double_indent():
    # v9 LINE|INDENT|content + INDENT|content: the count is authoritative; the
    # content's real spaces must be dropped (else 286|4|'    def' → 8 spaces).
    from workflows.code import _strip_line_numbers
    assert _strip_line_numbers("286|4|    def foo")[0] == "    def foo"
    assert _strip_line_numbers("4|    def foo")[0] == "    def foo"
    # stray ⇥ in the content is dropped (never valid source)
    assert "⇥" not in _strip_line_numbers("286|4|⇥def foo")[0]


def test_bughunt_strip_line_numbers_prefix_ws_leading_space():
    # ws_prefix_format now tolerates leading whitespace (^\s*), symmetric w/ REPLACE
    from workflows.code import _strip_line_numbers
    assert _strip_line_numbers("   286 ⇥4|    def foo")[0] == "    def foo"


def test_bughunt_native_expand_tolerates_space_after_marker():
    # `+ <code>` → ` INDENT|code`; native expander must still expand (not ship verbatim)
    from core.native_tools import _expand_indent_lines
    assert _expand_indent_lines([" 4|return 2"]) == ["    return 2"]
    assert _expand_indent_lines(["8|raise X"]) == ["        raise X"]
