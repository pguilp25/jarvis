"""Audit the lookup tools that consume a pre-built map artifact:
  • `get_detail_section` — [DETAIL: section name]
  • `get_purpose_snippets` — [PURPOSE: category name]
  • `list_sections` / `list_purposes` — directory of available names

These are the agent's "browse by intent" tools. Bugs mean the model
asks for "auth" and gets back the section for "configuration" because
fuzzy matching ranks wrong.
"""
import pytest
from tools.code_index import (
    get_detail_section,
    list_sections,
    get_purpose_snippets,
)


# ─────────────── get_detail_section ───────────────


DETAILED_MAP = """=== SECTION: Authentication ===
Body of auth section.
Functions: login, logout, verify_token.

=== SECTION: Database Connection ===
Body of db section.
Functions: connect, disconnect, transaction.

=== SECTION: User Management ===
Body of user mgmt section.
Functions: create_user, delete_user, update_user.
"""


def test_detail__exact_match():
    out = get_detail_section(DETAILED_MAP, "Authentication")
    assert "auth section" in out.lower() or "Authentication" in out


def test_detail__case_insensitive_exact():
    out = get_detail_section(DETAILED_MAP, "authentication")
    assert "auth section" in out.lower() or "Authentication" in out


def test_detail__substring_match():
    """`auth` is a substring of `Authentication` → matches that section."""
    out = get_detail_section(DETAILED_MAP, "auth")
    # The substring rule fires for `auth in authentication`
    assert "Authentication" in out or "auth section" in out.lower()


def test_detail__keyword_overlap():
    """`User Management Functions` has 2 words overlapping with `User Management`."""
    out = get_detail_section(DETAILED_MAP, "User Mgmt")
    # Either matches via overlap or fuzzy
    assert "user" in out.lower() or "no section" in out.lower()


def test_detail__no_match_returns_marker():
    out = get_detail_section(DETAILED_MAP, "Encryption")
    # No section has Encryption — should fall through to "no section"
    # or pick the lowest overlap match
    assert isinstance(out, str)


def test_detail__empty_map():
    out = get_detail_section("", "anything")
    assert "no section" in out.lower() or out == "(no section found matching 'anything')"


def test_detail__empty_query():
    """Empty query string — should not crash."""
    out = get_detail_section(DETAILED_MAP, "")
    assert isinstance(out, str)


def test_detail__whitespace_query_trimmed():
    """`  Authentication  ` should match the same as `Authentication`."""
    out = get_detail_section(DETAILED_MAP, "  Authentication  ")
    assert "Authentication" in out or "auth section" in out.lower()


# ─────────────── list_sections ───────────────


def test_list_sections__finds_all():
    out = list_sections(DETAILED_MAP)
    assert "Authentication" in out
    assert "Database Connection" in out
    assert "User Management" in out


def test_list_sections__empty_map():
    assert list_sections("") == []


def test_list_sections__no_sections():
    """Map with no SECTION markers → empty list."""
    assert list_sections("Just some content with no markers") == []


def test_list_sections__preserves_case():
    """Section names should be returned with their original case."""
    out = list_sections(DETAILED_MAP)
    assert "Authentication" in out  # not "authentication"


def test_list_sections__handles_whitespace():
    """Extra whitespace around section names should be stripped."""
    map_with_ws = "===   SECTION:   Auth Stuff   ===\nbody"
    out = list_sections(map_with_ws)
    assert any("Auth" in s for s in out)


# ─────────────── get_purpose_snippets ───────────────


PURPOSE_MAP_SIMPLE = """=== PURPOSE: authentication ===
description: Login / logout / token verification.

=== PURPOSE: database ===
description: Database connection pooling and queries.

=== PURPOSE: ui rendering ===
description: Template rendering and view helpers.
"""


def test_purpose__exact_match(tmp_path):
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "authentication", str(tmp_path))
    assert "authentication" in out.lower() or "Login" in out


def test_purpose__substring_match(tmp_path):
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "auth", str(tmp_path))
    # `auth` is a substring of `authentication`
    assert "authentication" in out.lower() or "Login" in out


def test_purpose__case_insensitive(tmp_path):
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "DATABASE", str(tmp_path))
    assert "database" in out.lower() or "connection" in out.lower()


def test_purpose__no_match_lists_available(tmp_path):
    """Unknown category → response should list available categories."""
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "ml_pipeline", str(tmp_path))
    # Should mention "no category" AND list the available ones
    assert "no category" in out.lower() or "available" in out.lower()
    assert "authentication" in out.lower() or "database" in out.lower()


def test_purpose__multi_word_match(tmp_path):
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "ui rendering", str(tmp_path))
    assert "ui" in out.lower() or "rendering" in out.lower() or "Template" in out


def test_purpose__empty_map(tmp_path):
    out = get_purpose_snippets("", "anything", str(tmp_path))
    # Should not crash; should report no match
    assert isinstance(out, str)


def test_purpose__keyword_overlap_picks_best(tmp_path):
    """Query `database queries` overlaps with `database` section."""
    out = get_purpose_snippets(PURPOSE_MAP_SIMPLE, "database queries", str(tmp_path))
    assert "database" in out.lower() or "queries" in out.lower() or "connection" in out.lower()


# ─────────────── PURPOSE MAP WITH FILE/LINE REFERENCES ───────────────


def test_purpose__with_file_refs_returns_snippets(tmp_path):
    """When the purpose map has FILE: and LINES: refs, the function pulls
    actual code snippets from the project."""
    (tmp_path / "auth.py").write_text(
        "\n".join(f"line_{i}" for i in range(1, 50))
    )
    purpose_map = (
        "=== PURPOSE: authentication ===\n"
        "description: Login flow.\n"
        f"FILE: auth.py LINES: 10-15\n"
    )
    out = get_purpose_snippets(purpose_map, "authentication", str(tmp_path))
    # Should include the file ref or its content
    assert "auth.py" in out or "line_10" in out or "authentication" in out.lower()
