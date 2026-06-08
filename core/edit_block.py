"""
Single-block edit — a small numbered DIFF over a contiguous window of the file.

An [edit] block mirrors the read view (every line is `LINENO:...`) and marks
each line's fate, exactly like a unified diff:

    N: <content>     KEEP line N unchanged — copy it from the read view verbatim.
    N:- <content>    DELETE line N. The content is shown so the removal is
                     EXPLICIT and can be verified against the file.
    N:+ <content>    ADD a new / changed line here (content is the new code).
                     A bare `+<content>` (no number) is also accepted.
    M-N:-            BULK-DELETE original lines M..N — a shorthand so you don't
                     retype a long run you're removing (no content needed).

A line you DON'T mention is KEPT. Deletion is never silent: it needs a `-`.
To CHANGE a line, delete the old (`N:-`) and add the new (`N:+`).

Why every line keeps its number: the model COPIES the numbers from the read
view instead of computing them, so they stay trustworthy — and an explicit `-`
means it never has to renumber the lines below an insert. That lets the applier
work NUMBER-FIRST:

  1. Parse the block top-to-bottom into keep / del / add / bulk-del ops.
  2. Resolve each keep/del to a file line: try its line number and VERIFY the
     content matches there; only if it doesn't, fall back to locating by
     content (weak models occasionally misnumber). Reject a keep/del whose
     content is nowhere in the file (stale view).
  3. Reconstruct the region [first ref … last ref]: walk the ops in order —
     keep -> emit the ORIGINAL file line (real whitespace preserved), add ->
     emit the new content verbatim, del / bulk-del -> emit nothing. Original
     lines inside the region the block didn't mention are KEPT.

Returns (new_src, info, warnings). new_src is None on rejection (info = why).
"""
import re

# Every line carries its read-view number. A stray `=` after the colon (old
# habit) is tolerated on kept lines. A `+` / `-` immediately after `N:` marks an
# added / deleted line; otherwise `N:` is a kept line.
_KEEP_RE = re.compile(r'^(\d+):=?(?![+\-])(.*)$')              # N:content      (kept)
_ADD_RE  = re.compile(r'^(?:(\d+):)?\+(.*)$')                  # N:+content / +content (new)
_DEL_RE  = re.compile(r'^(\d+):-(.*)$')                        # N:-content     (delete one line)
_BULKDEL_RE = re.compile(r'^\s*(\d+)\s*-\s*(\d+)\s*:\s*-\s*$') # M-N:-          (delete a run)
# Terminator is lenient AND line-anchored: a missing `[/edit]` (model closed with
# `=== END EDIT ===` or just stopped) still parses, but a `[/edit]` / `=== END
# EDIT ===` that appears MID-LINE inside a string/comment is content, not a
# terminator (re.M + `^[ \t]*`), so it can't silently truncate the edit.
_BLOCK_RE  = re.compile(
    r'\[edit(?::\s*\d+)?\](.*?)(?:^[ \t]*\[/edit\]|^[ \t]*===\s*END\s+EDIT\s*===|\Z)',
    re.S | re.I | re.M)
_EDIT_NUM_RE = re.compile(r'\[edit:\s*(\d+)\]', re.I)
_GUTTER_LEAK_RE = re.compile(r'^\s*\d+:')   # a bare-`+` line whose content still carries a gutter
# prefix_ws view gutter `LINENO ⇥` (the native view now shared by all roles). The
# keep/del/add markers below all anchor on `LINENO:` (colon); normalize a copied
# `LINENO ⇥` gutter to `LINENO:` FIRST so a line the coder copied verbatim from the
# view (`12 ⇥4|code`) is recognized as a keep, and a `12 ⇥-…`/`12 ⇥+…` resolves to
# the `:-`/`:+` markers. The prompt-instructed colon forms pass through untouched.
# The trailing `\s*` also EATS a stray space the coder may leave after the ⇥ and
# before a `-`/`+` marker (`12 ⇥ -4|` → `12:-4|`), which would otherwise split the
# marker from its colon and fall through unmatched.
_PREFIX_WS_GUTTER_RE = re.compile(r'^(\d+)\s*⇥\s*')

# Prefix-mode (v12) bodies carry indentation as a COUNT: `INDENT|code` (e.g.
# `16|raise TypeError(other)` = 16 leading spaces). The view shows
# `LINENO:INDENT|content`, so after the `N:`/`N:-`/`N:+` marker the content is
# `INDENT|code`. We EXPAND that to real spaces for an added line, and STRIP it
# to bare code for keep/del matching (the file is matched by content). Plain
# whitespace content (no `INDENT|`) passes through unchanged, so both formats
# work. This is what lets the model specify indent as a number it computes
# (else: at 12 → its body at 16) instead of typing exact spaces.
_INDENT_PREFIX_RE = re.compile(r'^(\d+)\|(.*)$')


def _expand_indent(content: str) -> str:
    """`INDENT|code` → `<INDENT spaces>code` (for an ADDED line). Else verbatim.

    The COUNT is authoritative: any real leading spaces still typed in the code
    part are dropped (`lstrip`) before the count's spaces are re-emitted. This is
    what makes the prefix_ws view (`INDENT|<real spaces>code`) safe — the coder
    copying a line keeps both the count AND the visible spaces, and we must not
    DOUBLE them (`4|    x` is 4 spaces, not 8). Mirrors core/native_tools.py:
    _expand_indent_lines, which already lstrips. A count-only `4|x` is unaffected.

    A leading space BEFORE the count is also tolerated — the documented add form
    is `N:+ <code>` (a space after the marker), so `+ 4|y = 2` arrives here as
    ` 4|y = 2`; match against the space-stripped form so the count still fires.
    If there is NO count prefix (whitespace-mode add), return the ORIGINAL content
    verbatim so genuine leading indentation is preserved.

    Two further guards mirror core/native_tools.py:_expand_indent_lines so the text
    and native coders behave IDENTICALLY on the shared prefix_ws view:
      • the ⇥ tab-glyph is the harness's display marker and is NEVER valid source —
        a stray one copied into the code part is stripped (else it ships as a literal
        `⇥` → `SyntaxError: invalid character`).
      • STRUCTURAL-line guard: if the code is a `def`/`class`/decorator AND the typed
        leading spaces disagree with the count, trust the TYPED spaces — trusting the
        number would EJECT a method to a shallower scope (the f631 failure)."""
    m = _INDENT_PREFIX_RE.match(content.lstrip(' '))
    if not m:
        return content
    ind = int(m.group(1))
    code = m.group(2).replace("⇥", "")
    typed = len(code) - len(code.lstrip(' '))
    if typed > 0 and typed != ind:
        _cs = code.lstrip(' ')
        if re.match(r'(?:async\s+def|def|class)\b', _cs) or _cs.startswith('@'):
            ind = typed
    return " " * ind + code.lstrip(' ')


def _bare_code(content: str) -> str:
    """Strip a leading `INDENT|` so keep/del match the file by CODE, not the
    count. Else return content unchanged (whitespace mode). A stray ⇥ display
    glyph is dropped either way (it's never in the real file, so it would only
    break the content match)."""
    m = _INDENT_PREFIX_RE.match(content)
    return (m.group(2) if m else content).replace("⇥", "")


def edit_label(block_text: str) -> "str | None":
    """Return the edit's number from `[edit:N]`, or None for a bare `[edit]`."""
    m = _EDIT_NUM_RE.search(block_text or "")
    return m.group(1) if m else None


_TRIVIAL_ANCHORS = {
    "", "pass", "return", "break", "continue", "else:", "try:", "finally:",
    ")", "]", "}", "{", "(", "[", "):", "],", "},", "});", "})",
    "end", "done", "...", "fi", "esac",
}


def _distinctive(content: str) -> bool:
    """True if `content` is specific enough to anchor a region reliably."""
    s = content.strip()
    return len(s) >= 4 and s not in _TRIVIAL_ANCHORS


def _locate(lines, content, hint, after=0, upto=None):
    """Find the line whose content matches `content`. Prefer an EXACT rstrip
    match; else a fully-STRIPPED match (weak models mangle leading indentation).
    Disambiguate repeats by the nearest line-number hint."""
    upto = len(lines) if upto is None else upto
    lo = max(0, after)
    want_r, want_s = content.rstrip(), content.strip()
    exact = [i for i in range(lo, upto) if lines[i].rstrip() == want_r]
    cands = exact if exact else [i for i in range(lo, upto) if lines[i].strip() == want_s]
    if not cands:
        return None
    return min(cands, key=lambda i: abs((i + 1) - hint)) if hint else cands[0]


def apply_edit_block(src, block_text):
    """Apply one [edit] block (a numbered unified diff). Returns
    (new_src|None, info, warnings:list).

    Numbers are the primary locator but are CONTENT-VERIFIED: a kept/deleted
    line is matched at its number only if the content agrees, else located by
    content. Kept anchors re-emit the ORIGINAL file line (so a mis-indented copy
    still yields correct whitespace); `+` lines are emitted verbatim."""
    m = _BLOCK_RE.search(block_text)
    if not m:
        return None, "no [edit]...[/edit] block found", []
    # CRLF safety (C#3): a CRLF file split on "\n" leaves "\r" on every kept
    # line, while `+` lines are emitted verbatim (bare "\n") → the result mixes
    # endings on edited lines. Normalize to "\n" for processing and restore the
    # file's original CRLF on the way out so endings stay uniform.
    _had_crlf = "\r\n" in src
    if _had_crlf:
        src = src.replace("\r\n", "\n")
    lines = src.split("\n")

    # ── 1. parse ───────────────────────────────────────────────────────────
    # ops: ("keep",num,content) | ("del",num,content) | ("add",num|None,content)
    #      | ("bulk", start, end)
    ops = []
    for ln in m.group(1).strip("\n").split("\n"):
        ln = _PREFIX_WS_GUTTER_RE.sub(r'\1:', ln)   # prefix_ws `N ⇥` gutter → `N:` so markers match
        mb = _BULKDEL_RE.match(ln)
        if mb:
            ops.append(("bulk", int(mb.group(1)), int(mb.group(2)))); continue
        ma = _ADD_RE.match(ln)
        if ma:
            content = ma.group(2)
            if _GUTTER_LEAK_RE.match(content):
                return (None,
                        f"a `+` line carries a stray line-number gutter: "
                        f"{content.strip()!r}. The number goes BEFORE the `+` "
                        f"(`N:+code`) or is dropped (`+code`) — never inside the "
                        f"code you're adding.", [])
            ops.append(("add", int(ma.group(1)) if ma.group(1) else None, _expand_indent(content))); continue
        md = _DEL_RE.match(ln)
        if md:
            ops.append(("del", int(md.group(1)), _bare_code(md.group(2)))); continue
        mk = _KEEP_RE.match(ln)
        if mk:
            ops.append(("keep", int(mk.group(1)), _bare_code(mk.group(2)))); continue
        if ln.strip():                       # unmarked non-empty line → tolerate as added
            ops.append(("add", None, ln))

    if not any(o[0] in ("keep", "del", "bulk") for o in ops):
        if not src.strip():
            return (None, "the file is empty — there are no lines to anchor on. "
                    "Write the initial content with `=== FILE: <path> === … "
                    "=== END FILE ===`, not an [edit] block.", [])
        return (None, "no kept or deleted line to anchor on — copy at least one "
                "surrounding line VERBATIM as `LINENO ⇥INDENT|code` (e.g. "
                "`43 ⇥8|        raise ValueError(\"x\")`) so the edit's location is "
                "unambiguous. A block of only `+` adds has nowhere to attach.", [])

    # ── 2. resolve keep/del positions (number-first, content-verified) ───────
    # Returns (pos, verified, matches). The line NUMBER is the PRIMARY locator:
    # when the content also appears elsewhere (ambiguous), pick the occurrence
    # CLOSEST to the requested number instead of rejecting — the coder gave a
    # number for a reason, and views drift only a few lines across edits, so the
    # nearest match is what they meant. (Previously this rejected as AMBIGUOUS,
    # which turned a 1-line drift into a 0-byte instance failure — see the v15
    # audit: 3 of 5 empties were duplicate-content anchors like `else:` / a
    # repeated `return`/`yield`.) Only a ZERO-match (content nowhere in the
    # file) stays unresolved; a wrong pick is still caught by the parse/verify
    # gates downstream.
    def _resolve(num, content):
        pos = num - 1
        if 0 <= pos < len(lines) and (
                lines[pos].rstrip() == content.rstrip()
                or lines[pos].strip() == content.strip()):
            return pos, True, [pos]          # number verified by content
        want_r, want_s = content.rstrip(), content.strip()
        matches = [i for i in range(len(lines)) if lines[i].rstrip() == want_r]
        if not matches:
            matches = [i for i in range(len(lines)) if lines[i].strip() == want_s]
        if not matches:
            return None, False, []
        if len(matches) == 1:
            return matches[0], False, matches
        # ambiguous content → disambiguate by the requested number (closest)
        return min(matches, key=lambda i: abs(i - pos)), False, matches

    resolved = []                            # parallel to ops
    ref_positions = []
    used_fallback = False
    ambig_picks = []                         # (requested_num, chosen_line, n_matches)
    for o in ops:
        if o[0] in ("keep", "del"):
            pos, ok, matches = _resolve(o[1], o[2])
            if pos is None:
                kind = "kept" if o[0] == "keep" else "deleted"
                if len(matches) > 1:
                    where = ", ".join(str(i + 1) for i in matches[:6])
                    return (None, f"the {kind} line {o[1]} ({o[2].strip()!r}) is "
                            f"AMBIGUOUS — that content is on lines {where}, and the "
                            f"number {o[1]} doesn't match any of them, so I can't "
                            f"tell which one you mean. Re-read and copy the line at "
                            f"its correct number.", [])
                if o[2].strip():
                    # Show what line N ACTUALLY is now (bug-hunt #3): a weak
                    # model otherwise GUESSES variations of the missing line
                    # instead of re-reading. Hand it the real line to copy.
                    n = o[1]
                    actual = ""
                    if 1 <= n <= len(lines):
                        cur_ln = lines[n - 1]
                        ind = len(cur_ln) - len(cur_ln.lstrip(" "))
                        if cur_ln.strip():
                            actual = (f" Line {n} CURRENTLY reads: "
                                      f"`{n} ⇥{ind}|{' ' * ind}{cur_ln.strip()}` — anchor on THAT "
                                      f"(or the correct line), do not re-type your "
                                      f"old version.")
                        else:
                            actual = f" Line {n} is currently BLANK."
                    return (None, f"the {kind} line {n} isn't in the file "
                            f"(stale view?): {o[2].strip()!r}.{actual}", [])
                pos, ok = o[1] - 1, True      # blank line — trust the number
            elif len(matches) > 1:
                # closest-match disambiguated an anchor whose content repeats
                ambig_picks.append((o[1], pos + 1, len(matches)))
            resolved.append(pos); ref_positions.append(pos)
            used_fallback = used_fallback or not ok
        elif o[0] == "bulk":
            if o[1] > o[2]:
                return (None, f"backwards bulk-delete range {o[1]}-{o[2]} — write "
                        f"M-N:- with M ≤ N (you meant {o[2]}-{o[1]}:-).", [])
            a, b = o[1] - 1, o[2] - 1
            resolved.append((a, b)); ref_positions.append(a); ref_positions.append(b)
        else:
            resolved.append(None)

    fp, lp = min(ref_positions), max(ref_positions)
    if not (0 <= fp <= lp < len(lines)):
        return (None, f"edit line numbers {fp + 1}-{lp + 1} fall outside the file "
                f"(1-{len(lines)}). Re-read and use the current line numbers.", [])

    # ── 3. reconstruct [fp..lp], preserving unmentioned original lines ───────
    out = []
    cursor = fp
    for o, r in zip(ops, resolved):
        if o[0] == "keep":
            if r == cursor - 1:
                return None, (f"line {r + 1} is listed twice — to CHANGE it use only "
                              f"`{r + 1}:-`/`{r + 1}:+`, don't also keep it with "
                              f"`{r + 1}:`."), []
            if r < cursor:
                return None, ("edit lines are out of order — list them "
                              "top-to-bottom in the same order as the file."), []
            out.extend(lines[cursor:r])       # unmentioned originals before this line → kept
            out.append(lines[r] if 0 <= r < len(lines) else o[2])
            cursor = r + 1
        elif o[0] == "del":
            if r == cursor - 1:
                return None, (f"line {r + 1} is both KEPT and DELETED — to change it "
                              f"use only `{r + 1}:-` then `{r + 1}:+`, not a `{r + 1}:` "
                              f"keep as well."), []
            if r < cursor:
                return None, ("edit lines are out of order — list them "
                              "top-to-bottom in the same order as the file."), []
            out.extend(lines[cursor:r])       # kept before the deleted line
            cursor = r + 1                    # skip the deleted line
        elif o[0] == "bulk":
            a, b = r
            if a < cursor:
                return None, "bulk-delete range overlaps an earlier edit line.", []
            out.extend(lines[cursor:a])       # kept before the run
            cursor = b + 1                    # skip the run
        else:                                 # add
            out.append(o[2])
    out.extend(lines[cursor:lp + 1])          # trailing unmentioned originals → kept

    # ── 4. advisory warnings (non-fatal) ─────────────────────────────────────
    warnings = []
    if used_fallback:
        warnings.append("some line numbers didn't match the file — located by "
                        "content instead (your read view may be stale; re-read "
                        "if the diff looks off)")
    for req, chosen, nm in ambig_picks:
        warnings.append(
            f"anchor content appears on {nm} lines — used the one NEAREST your "
            f"number {req} (line {chosen}). VERIFY the diff changed the spot you "
            f"meant; if not, re-anchor on a DISTINCTIVE (unique) code line.")
    plus = [o[2] for o in ops if o[0] == "add" and o[2].strip()]
    keep_strs = {o[2].strip() for o in ops if o[0] == "keep" and len(o[2].strip()) >= 3}
    dups = [p.strip() for p in plus if p.strip() in keep_strs]
    if dups:
        warnings.append("possible DUPLICATION — added (+) line(s) match a kept "
                        "line (" + "; ".join(d[:40] for d in dups[:3])
                        + "); to MOVE/CHANGE a line, delete the old one with `N:-`.")

    lines[fp:lp + 1] = out
    info = f"matched original lines {fp + 1}-{lp + 1}, replaced with {len(out)} lines"
    new_src = "\n".join(lines)
    if _had_crlf:
        new_src = new_src.replace("\n", "\r\n")
    return new_src, info, warnings
