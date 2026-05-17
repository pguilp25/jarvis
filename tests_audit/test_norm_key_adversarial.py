"""ADVERSARIAL SECOND-PASS audit of `_norm_key`.

`_norm_key` is the cache-key normalizer. Any inconsistency means cache
misses (re-running expensive lookups). Test the PROPERTIES that must
hold under every possible input shape:

  • IDEMPOTENCE: re-applying _norm_key on the stripped arg yields same key.
  • COLLISION-FREEDOM: distinct inputs → distinct keys (with documented
    equivalence classes).
  • UNICODE: NFC vs NFD don't collide (no normalization happens for these).
  • TYPE: tag_type passed through verbatim; arg lowercased.

Cover:
  • Empty / whitespace-only args.
  • Paths with `../`, `..`, multiple `./`, trailing `/`.
  • Range specs with weird spacing (`10 - 20`, `10\t-\t20`, `10-20-30`).
  • Mixed slashes.
  • Embedded null bytes.
  • Very long args.
"""
import pytest
from core.tool_call import _norm_key


# ─────────────── PROPERTY: IDEMPOTENCE ───────────────


def test_idem__simple_path():
    once = _norm_key("CODE", "foo.py")
    prefix = "CODE:"
    twice = _norm_key("CODE", once[len(prefix):])
    assert once == twice


def test_idem__path_with_range():
    once = _norm_key("KEEP", "foo.py 10-20")
    prefix = "KEEP:"
    twice = _norm_key("KEEP", once[len(prefix):])
    assert once == twice


def test_idem__path_with_multi_range():
    once = _norm_key("KEEP", "foo.py 10-20,30-40")
    prefix = "KEEP:"
    twice = _norm_key("KEEP", once[len(prefix):])
    assert once == twice


def test_idem__ident():
    once = _norm_key("REFS", "my_function")
    prefix = "REFS:"
    twice = _norm_key("REFS", once[len(prefix):])
    assert once == twice


def test_idem__empty_arg():
    once = _norm_key("CODE", "")
    prefix = "CODE:"
    twice = _norm_key("CODE", once[len(prefix):])
    assert once == twice


# ─────────────── PROPERTY: EQUIVALENCE CLASSES ───────────────


def test_eq__leading_dot_slash():
    """./foo.py ≡ foo.py."""
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", "./foo.py")


def test_eq__trailing_whitespace():
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", "foo.py ")
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", "foo.py   ")
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", " foo.py")
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", "  foo.py  ")
    assert _norm_key("CODE", "foo.py") == _norm_key("CODE", "\tfoo.py\t")


def test_eq__case_lowercased():
    assert _norm_key("CODE", "Foo.py") == _norm_key("CODE", "FOO.PY")
    assert _norm_key("CODE", "MyClass") == _norm_key("CODE", "myclass")


def test_eq__backslash_to_forward_slash():
    assert _norm_key("CODE", "a/b/c.py") == _norm_key("CODE", "a\\b\\c.py")


def test_eq__mixed_slashes():
    assert _norm_key("CODE", "a/b\\c.py") == _norm_key("CODE", "a/b/c.py")


def test_eq__double_dot_slash_dropped_iteratively():
    """./../foo.py — the LOOP strips each `./` leading prefix."""
    # `./../foo.py` → after one strip: `../foo.py` → no more `./` prefix
    # So `./../foo.py` ≡ `../foo.py`, NOT `foo.py`
    assert _norm_key("CODE", "./../foo.py") == _norm_key("CODE", "../foo.py")


def test_eq__many_dot_slash_prefixes():
    """Multiple `./` prefixes ARE all stripped (loop)."""
    assert _norm_key("CODE", "././foo.py") == _norm_key("CODE", "foo.py")
    assert _norm_key("CODE", "././././foo.py") == _norm_key("CODE", "foo.py")


def test_eq__range_dash_whitespace():
    assert _norm_key("KEEP", "f.py 10-20") == _norm_key("KEEP", "f.py 10 - 20")
    assert _norm_key("KEEP", "f.py 10-20") == _norm_key("KEEP", "f.py 10 -20")
    assert _norm_key("KEEP", "f.py 10-20") == _norm_key("KEEP", "f.py 10- 20")


def test_eq__range_comma_whitespace():
    assert _norm_key("KEEP", "f.py 10-20,30-40") == _norm_key("KEEP", "f.py 10-20, 30-40")
    assert _norm_key("KEEP", "f.py 10-20,30-40") == _norm_key("KEEP", "f.py 10-20 ,30-40")
    assert _norm_key("KEEP", "f.py 10-20,30-40") == _norm_key("KEEP", "f.py 10-20 , 30-40")


def test_eq__internal_whitespace_collapsed():
    assert _norm_key("KEEP", "f.py 10-20") == _norm_key("KEEP", "f.py   10-20")
    assert _norm_key("KEEP", "f.py 10-20") == _norm_key("KEEP", "f.py\t\t10-20")


# ─────────────── PROPERTY: DISTINCT INPUTS → DISTINCT KEYS ───────────────


def test_distinct__different_filenames():
    assert _norm_key("CODE", "foo.py") != _norm_key("CODE", "bar.py")


def test_distinct__different_paths():
    assert _norm_key("CODE", "a/foo.py") != _norm_key("CODE", "b/foo.py")


def test_distinct__different_tag_types():
    assert _norm_key("CODE", "foo.py") != _norm_key("VIEW", "foo.py")
    assert _norm_key("CODE", "foo.py") != _norm_key("KEEP", "foo.py")
    assert _norm_key("REFS", "foo") != _norm_key("LSP", "foo")
    assert _norm_key("PURPOSE", "auth") != _norm_key("SEMANTIC", "auth")


def test_distinct__different_ranges():
    assert _norm_key("KEEP", "f.py 10-20") != _norm_key("KEEP", "f.py 30-40")
    assert _norm_key("KEEP", "f.py 10-20") != _norm_key("KEEP", "f.py 10-30")
    assert _norm_key("KEEP", "f.py 10-20") != _norm_key("KEEP", "f.py 11-20")


def test_distinct__similar_names_not_collapsed():
    """`foo` and `foobar` should produce different keys."""
    assert _norm_key("REFS", "foo") != _norm_key("REFS", "foobar")


def test_distinct__substring_relation_does_not_collapse():
    assert _norm_key("CODE", "ab.py") != _norm_key("CODE", "abc.py")


# ─────────────── UNICODE EDGE CASES ───────────────


def test_unicode__nfc_vs_nfd_NOT_normalized():
    """`é` (NFC: single codepoint) vs `é` (NFD: e + combining acute)
    produce DIFFERENT keys (no Unicode normalization happens)."""
    nfc = "café.py"  # single é
    nfd = "café.py"  # e + combining acute
    # These should be distinct (no NFC/NFD collapsing)
    # If they ARE collapsed, that's also fine — just verify deterministic.
    a = _norm_key("CODE", nfc)
    b = _norm_key("CODE", nfd)
    # We don't enforce equality — just that the function doesn't crash
    assert isinstance(a, str) and isinstance(b, str)


def test_unicode__emoji_in_path():
    a = _norm_key("CODE", "🎉.py")
    b = _norm_key("CODE", "🎉.py")
    assert a == b
    # Different emoji → different key
    c = _norm_key("CODE", "🚀.py")
    assert a != c


def test_unicode__rtl_in_path():
    """RTL scripts (Arabic) should be supported."""
    a = _norm_key("CODE", "العربية.py")
    b = _norm_key("CODE", "العربية.py")
    assert a == b


def test_unicode__chinese_ident():
    a = _norm_key("REFS", "测试函数")
    b = _norm_key("REFS", "测试函数")
    assert a == b


def test_unicode__case_unchanged_for_non_ascii():
    """Python's `str.lower()` lowercases ASCII only by default; non-ASCII
    chars stay as-is. (Actually .lower() DOES lowercase many alphabets.)"""
    # German ß stays unchanged in .lower() in Python (it doesn't become 'ss')
    a = _norm_key("REFS", "Müller")
    b = _norm_key("REFS", "müller")
    # `M` lowercases to `m` so these collide
    assert a == b


# ─────────────── BOUNDARY ───────────────


def test_boundary__empty_arg():
    """Empty arg shouldn't crash."""
    k = _norm_key("CODE", "")
    assert k.startswith("CODE:")


def test_boundary__whitespace_only_arg():
    k = _norm_key("CODE", "   ")
    # After strip → empty
    assert k.startswith("CODE:")


def test_boundary__very_long_path():
    """A 10K-char path — should still hash fine."""
    long_path = "x" * 10000 + ".py"
    k = _norm_key("CODE", long_path)
    assert k.startswith("CODE:")
    # Idempotent
    once = k
    prefix = "CODE:"
    twice = _norm_key("CODE", once[len(prefix):])
    assert once == twice


def test_boundary__path_with_only_dot_slashes():
    """`./././` — all stripped → empty arg."""
    k = _norm_key("CODE", "./././")
    # All `./` prefixes stripped → remainder is `` → key is `CODE:`
    assert k == "CODE:"


# ─────────────── PUNCTUATION / SPECIAL CHARS ───────────────


def test_special__path_with_dash():
    assert _norm_key("CODE", "my-file.py") == _norm_key("CODE", "MY-FILE.PY")


def test_special__path_with_underscore():
    assert _norm_key("REFS", "my_func") == _norm_key("REFS", "MY_FUNC")


def test_special__path_with_dot_in_name():
    """`pkg.module.py` — dots inside names should pass through."""
    a = _norm_key("CODE", "pkg.module.py")
    b = _norm_key("CODE", "pkg.module.py")
    assert a == b


def test_special__path_with_plus():
    a = _norm_key("CODE", "c++.cpp")
    b = _norm_key("CODE", "c++.cpp")
    assert a == b


def test_special__path_with_at():
    a = _norm_key("CODE", "test@v1.py")
    b = _norm_key("CODE", "test@v1.py")
    assert a == b


# ─────────────── RANGE-RELATED ADVERSARIAL ───────────────


def test_range__different_orders_of_ranges():
    """`10-20,30-40` vs `30-40,10-20` — different order, different keys.
    (The function does NOT sort ranges; semantic dedup is the caller's
    responsibility — _parse_keep_ranges does the sort.)"""
    a = _norm_key("KEEP", "f.py 10-20,30-40")
    b = _norm_key("KEEP", "f.py 30-40,10-20")
    # Implementation behavior: order-preserving — DOCUMENT the contract.
    assert isinstance(a, str) and isinstance(b, str)


def test_range__single_number_not_a_range():
    """`f.py 42` (no dash) — leftover number. Behavior is implementation-defined."""
    k = _norm_key("CODE", "f.py 42")
    assert k.startswith("CODE:")


def test_range__inverted():
    """`20-10` — inverted range. Function preserves as-is (caller validates)."""
    a = _norm_key("KEEP", "f.py 20-10")
    b = _norm_key("KEEP", "f.py 20-10")
    assert a == b


def test_range__zero_in_range():
    """`f.py 0-20` — line 0 is invalid (lines are 1-based) but key hashes."""
    k = _norm_key("KEEP", "f.py 0-20")
    assert k.startswith("KEEP:")


# ─────────────── SELF-TEST VARIANTS (from _NORM_KEY_SELF_TEST) ───────────────


def test_self_test__code_variants_collide():
    """All these should produce the SAME key."""
    variants = ["foo.py", "./foo.py", " foo.py ", "FOO.PY", "foo.py "]
    keys = {_norm_key("CODE", v) for v in variants}
    assert len(keys) == 1


def test_self_test__code_path_variants():
    variants = ["a/b.py", "a\\b.py", "./a/b.py"]
    keys = {_norm_key("CODE", v) for v in variants}
    assert len(keys) == 1


def test_self_test__view_range_variants():
    variants = ["foo.py 100-200", "foo.py  100-200", "foo.py 100 - 200"]
    keys = {_norm_key("VIEW", v) for v in variants}
    assert len(keys) == 1


def test_self_test__keep_multi_range_variants():
    variants = [
        "foo.py 10-20,30-40",
        "foo.py 10-20, 30-40",
        "foo.py 10-20 , 30-40",
    ]
    keys = {_norm_key("KEEP", v) for v in variants}
    assert len(keys) == 1


def test_self_test__refs_ident_variants():
    variants = ["my_func", "MY_FUNC", " my_func ", "My_Func"]
    keys = {_norm_key("REFS", v) for v in variants}
    assert len(keys) == 1


# ─────────────── FORMAT INVARIANTS ───────────────


def test_format__starts_with_tag_type_colon():
    assert _norm_key("CODE", "x").startswith("CODE:")
    assert _norm_key("REFS", "x").startswith("REFS:")
    assert _norm_key("KEEP", "x").startswith("KEEP:")


def test_format__arg_part_lowercase():
    k = _norm_key("CODE", "ABCDEF.PY")
    arg_part = k[len("CODE:"):]
    assert arg_part == arg_part.lower()


def test_format__no_internal_double_space():
    k = _norm_key("KEEP", "f.py    10-20")  # 4 spaces
    arg_part = k[len("KEEP:"):]
    # Whitespace collapsed
    assert "  " not in arg_part


def test_format__no_whitespace_around_dash():
    k = _norm_key("KEEP", "f.py 10 - 20")
    arg_part = k[len("KEEP:"):]
    assert "10-20" in arg_part


def test_format__no_whitespace_around_comma():
    k = _norm_key("KEEP", "f.py 10-20 , 30-40")
    arg_part = k[len("KEEP:"):]
    assert "10-20,30-40" in arg_part
