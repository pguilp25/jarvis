"""
View-format line-number stripper (v9).

The [CODE:] / [VIEW:] view formatter renders each source line as:
    <LINENO>|<INDENT>|<code>
where LINENO is the 1-indexed line number and INDENT is the number of
leading spaces.

The model copies these view lines into SEARCH or REPLACE blocks. The
runtime needs to:
  - In SEARCH bodies: strip the LINENO and INDENT prefixes, keep the
    line# as a disambiguation hint for the fuzzy matcher.
  - In REPLACE bodies: the LINENO prefix is a leak (new content has no
    "current line number"); only the INDENT prefix is legitimate.

v9 design changes the format so line numbers are at the FRONT, not the
END. This solves the Q-NUMBERS silent-corruption bug — legitimate code
ending in integers (`MAX = 100`, `return 42`) was previously
indistinguishable from leaked trailing annotations. With prefixes at
the front, the trailing position is always pure code.

Detection of a leaked LINENO prefix in REPLACE:
  - `77|4|return result` — TWO `\\d+\\|` prefixes = line# + indent
    leaked. Strip the first.
  - `4|return result`    — ONE `\\d+\\|` prefix = indent only,
    legitimate REPLACE.
This 2-vs-1 distinction makes auto-strip safe.

Backwards-compat: the old `i<N>|<code> <LINENO>` format is still
recognized for any external callers / cached prompts that haven't
migrated. New code emits only the v9 format.
"""
import re

# v11 whitespace view: `LINENO:<real whitespace>content`. The line# is a
# `N:` gutter; everything after the FIRST `:` is the real line. Stripping
# the gutter also handles the older v10 `N:N|` form (drops `N:` → `N|`).
_WS_COLON_RE = re.compile(r'^\d+:')                   # LINE: gutter

# v10 format: line# uses a ':' separator, indent uses '|'.
_V10_FULL_PREFIX_RE = re.compile(r'^(\d+):(\d+\|)')   # LINE:INDENT|

# v9 format: front-positioned prefixes, both '|' (disambiguated by count)
_V9_FULL_PREFIX_RE = re.compile(r'^(\d+)\|(\d+)\|')   # LINE|INDENT|
_V9_INDENT_PREFIX_RE = re.compile(r'^(\d+)\|')        # INDENT|

# v8 legacy format (kept for back-compat parsing)
_V8_VIEW_PREFIX_RE = re.compile(r'^i\d+\|')
_V8_TRAILING_LINENO_RE = re.compile(r' \d+\s*$')


def strip_view_linenos(text: str) -> str:
    """Strip view-format line-number prefixes from SEARCH/REPLACE bodies.

    v9 rules (front-prefix format):
      - `LINE|INDENT|content` (two prefixes): strip the LINE prefix.
        Treats the line as a leaked annotation; the INDENT prefix is
        downstream-handled by the indent-expander.
      - `INDENT|content` (one prefix): leave as-is — this is legitimate
        REPLACE content; the indent-expander will convert `4|x` → `    x`.

    Legacy v8 rules (kept for back-compat):
      - `i<N>|<code> <LINENO>`: strip the trailing ` <LINENO>` when
        the line has the `i<N>|` view prefix.

    Args:
        text: raw multi-line text from a SEARCH or REPLACE block.

    Returns:
        Text with view-format line-number annotations stripped.
    """
    if not text:
        return text
    out_lines = []
    for line in text.splitlines(keepends=False):
        # v11 whitespace view: `LINENO:<real whitespace>content`. Strip the
        # `N:` gutter, leaving the line with its real indentation. Also covers
        # the v10 `N:N|` case (drops the `N:`, leaving `N|` for the indent-
        # expander). Anything after the FIRST `:` is kept verbatim.
        mws = _WS_COLON_RE.match(line)
        if mws:
            out_lines.append(line[mws.end():])
            continue
        # v9: strip leaked LINE prefix when both LINE|INDENT| are present
        m = _V9_FULL_PREFIX_RE.match(line)
        if m:
            # `77|4|return result` → `4|return result`
            line = line[m.end(1) + 1:]  # drop "77|"
            out_lines.append(line)
            continue
        # v9: strip leaked LINE prefix when both LINE|INDENT| are present
        m = _V9_FULL_PREFIX_RE.match(line)
        if m:
            # `77|4|return result` → `4|return result`
            line = line[m.end(1) + 1:]  # drop "77|"
            out_lines.append(line)
            continue
        # v8 legacy: trailing line# with i<N>| prefix
        if _V8_VIEW_PREFIX_RE.match(line):
            line = _V8_TRAILING_LINENO_RE.sub('', line)
        out_lines.append(line)
    result = '\n'.join(out_lines)
    if text.endswith('\n'):
        result += '\n'
    return result
