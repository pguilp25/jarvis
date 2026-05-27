"""Conservative indentation-slip checker for Python edits.

Public entry point: verify_indent(old_src, new_src) -> list[str].

Flags only clear, objective indentation problems on lines that an edit
ADDED or CHANGED (never on pre-existing lines). Pure stdlib. Designed to
fail safe: if anything is ambiguous or can't be analyzed, returns [].
"""
import difflib
import re
import tokenize
import io
from collections import Counter

_MAX_ITEMS = 8
_LEADING_WS = re.compile(r"^[ \t]*")


def _leading_ws(line):
    """Return the leading-whitespace string of a line."""
    m = _LEADING_WS.match(line)
    return m.group(0) if m else ""


def _file_uses_tabs(old_src):
    """Determine whether the existing file indents with tabs or spaces.

    Returns 'tab', 'space', or None (undetectable / mixed beyond use).
    Decision is based on which character begins the leading indentation of
    indented lines in the baseline.
    """
    tab_lines = 0
    space_lines = 0
    for line in old_src.splitlines():
        if not line.strip():
            continue
        ws = _leading_ws(line)
        if not ws:
            continue
        if ws[0] == "\t":
            tab_lines += 1
        elif ws[0] == " ":
            space_lines += 1
    if tab_lines == 0 and space_lines == 0:
        return None
    if tab_lines and space_lines:
        # Genuinely mixed baseline — don't claim a single convention.
        return None
    return "tab" if tab_lines else "space"


def _detect_indent_unit(old_src):
    """Detect the file's indent step: the most common positive leading-space
    delta between consecutive non-blank lines. Default 4 if undetectable."""
    deltas = Counter()
    prev = None
    for line in old_src.splitlines():
        if not line.strip():
            continue
        ws = _leading_ws(line)
        # Only consider purely-space indentation for step detection.
        if "\t" in ws:
            prev = None
            continue
        count = len(ws)
        if prev is not None:
            d = count - prev
            if d > 0:
                deltas[d] += 1
        prev = count
    if not deltas:
        return 4
    return deltas.most_common(1)[0][0]


def _continuation_lines(src):
    """Return a set of 1-based line numbers that are physical continuation
    lines (inside open brackets/parens/braces or spanned by a multi-line
    string).

    Tries tokenize first (most accurate). If tokenize fails — which commonly
    happens precisely because of an indentation slip we WANT to flag — falls
    back to a simple bracket-depth scan so analysis can still proceed.
    """
    cont = set()
    try:
        readline = io.StringIO(src).readline
        depth = 0
        prev_end_row = None
        for tok in tokenize.generate_tokens(readline):
            ttype = tok.type
            tstr = tok.string
            srow = tok.start[0]
            erow = tok.end[0]
            if ttype == tokenize.OP:
                if tstr in "([{":
                    depth += 1
                elif tstr in ")]}":
                    if depth > 0:
                        depth -= 1
            # A token that starts on a row while a bracket is open means that
            # row is a continuation row.
            if depth > 0 and srow != prev_end_row:
                cont.add(srow)
            # Multi-line tokens (strings) span continuation rows too.
            if erow > srow:
                for r in range(srow + 1, erow + 1):
                    cont.add(r)
            prev_end_row = erow
        return cont
    except IndentationError:
        # Indentation slips are exactly what we want to analyze. tokenize
        # bails on them, so fall back to a bracket-depth scan and keep going.
        return _bracket_continuation_fallback(src)
    except Exception:
        # TokenError (unterminated bracket/string) or anything else means the
        # source is genuinely unparseable — signal "unknown" so the caller
        # fails safe and returns [] rather than guessing.
        return None


def _bracket_continuation_fallback(src):
    """Best-effort continuation detection without tokenize.

    Counts unmatched (, [, { across physical lines, naively ignoring string
    contents and comments. A line is a continuation if a bracket was open at
    its start. Conservative: any error → return None (caller fails safe).
    """
    try:
        cont = set()
        depth = 0
        in_str = None  # quote char of an open string, or None
        for i, line in enumerate(src.splitlines()):
            lineno = i + 1
            if depth > 0:
                cont.add(lineno)
            j = 0
            n = len(line)
            while j < n:
                ch = line[j]
                if in_str:
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == in_str:
                        in_str = None
                    j += 1
                    continue
                if ch == "#":
                    break  # rest of line is a comment
                if ch in ("'", '"'):
                    in_str = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    if depth > 0:
                        depth -= 1
                j += 1
            in_str = None  # don't carry naive string state across lines
        return cont
    except Exception:
        return None


def _explicit_continuation_lines(src):
    """Return 1-based line numbers that immediately follow a backslash
    line-continuation. These are continuations regardless of tokenize."""
    cont = set()
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip("\n").endswith("\\"):
            cont.add(i + 2)  # next physical line (1-based)
    return cont


def _introduced_line_numbers(old_lines, new_lines):
    """Return the set of 1-based line numbers in new_lines that an edit
    added or changed (insert/replace opcode new ranges)."""
    introduced = set()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for j in range(j1, j2):
                introduced.add(j + 1)  # 1-based
    return introduced


def verify_indent(old_src, new_src):
    """Python source only. Compare new_src against old_src and flag
    INDENTATION problems on lines that the edit ADDED or CHANGED — never
    on lines that were already there. Returns a list of human-readable
    problem strings (empty list = clean). Be CONSERVATIVE: only flag clear,
    objective slips, never stylistic choices."""
    try:
        # Brand-new file: no baseline to compare against.
        if not old_src or not old_src.strip():
            return []
        if new_src is None:
            return []

        old_lines = old_src.splitlines()
        new_lines = new_src.splitlines()

        introduced = _introduced_line_numbers(old_lines, new_lines)
        if not introduced:
            return []

        # Determine continuation lines in new_src so we don't flag wrapped
        # expressions. If tokenize fails on new_src, fail safe (return []).
        cont = _continuation_lines(new_src)
        if cont is None:
            return []
        cont |= _explicit_continuation_lines(new_src)

        convention = _file_uses_tabs(old_src)  # 'tab' / 'space' / None
        unit = _detect_indent_unit(old_src)
        if not unit or unit <= 0:
            unit = 4

        problems = []
        seen = set()

        for lineno in sorted(introduced):
            if lineno in cont:
                continue
            idx = lineno - 1
            if idx < 0 or idx >= len(new_lines):
                continue
            line = new_lines[idx]
            if not line.strip():
                continue  # blank line
            ws = _leading_ws(line)
            if not ws:
                continue  # top-level, no indent to judge

            has_tab = "\t" in ws
            has_space = " " in ws

            # 1) TAB/SPACE MIXING relative to the file's convention.
            if convention == "space" and has_tab:
                msg = f"line {lineno} mixes tabs with the file's space indentation"
                if msg not in seen:
                    seen.add(msg)
                    problems.append(msg)
                if len(problems) >= _MAX_ITEMS:
                    break
                continue
            if convention == "tab" and has_space:
                # Spaces inside a tab-indented file's leading whitespace.
                msg = f"line {lineno} mixes spaces with the file's tab indentation"
                if msg not in seen:
                    seen.add(msg)
                    problems.append(msg)
                if len(problems) >= _MAX_ITEMS:
                    break
                continue

            # 2) OFF-STEP INDENT — only meaningful for space-indented files
            #    and pure-space leading whitespace.
            if convention == "space" and not has_tab:
                count = len(ws)
                if count % unit != 0:
                    msg = (
                        f"line {lineno} indented {count} space"
                        f"{'s' if count != 1 else ''} "
                        f"(not a multiple of the file's {unit}-space step)"
                    )
                    if msg not in seen:
                        seen.add(msg)
                        problems.append(msg)
                    if len(problems) >= _MAX_ITEMS:
                        break
                    continue

        return problems[:_MAX_ITEMS]
    except Exception:
        # Fail safe: never raise out of the checker.
        return []
