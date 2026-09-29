"""Tier-1 (no model) tests for the Markdown -> ANSI renderer (src/chad/markdown.py).

`render_markdown` is display-only: it renders one completed assistant message with rich
into a StringIO so the TUI can append styled prose to its scrollback without touching
the live terminal. These pin the contract the TUI depends on: styling is applied, the
answer survives, headings are flush left rather than rich's centered h1, and no line
carries rich's full-width trailing padding.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad import markdown  # noqa: E402

_ANSI = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;[^\x1b]*\x1b\\")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def test_render_applies_markdown_styling():
    out = markdown.render_markdown("# Title\n\nsome **bold** and `code`\n", 80)
    assert "\x1b[1m" in out            # bold heading / strong
    assert "\x1b[36m" in out           # cyan inline code
    assert "Title" in _plain(out) and "bold" in _plain(out) and "code" in _plain(out)


def test_render_headings_are_flush_left():
    # rich centers h1 by default; the renderer must override it so a heading lines up
    # with the prose and tool lines instead of padding the line into a banner.
    out = markdown.render_markdown("# Heading\n\nbody\n", 80)
    first = _plain(out).splitlines()[0]
    assert first.lstrip() == first and first.startswith("Heading"), repr(first)


def test_render_strips_width_padding():
    # rich pads every line to the Console width, sometimes inside a styled span; the
    # transcript must carry no trailing whitespace (it would land in copy/paste).
    out = markdown.render_markdown("hello **world**\n\n> quote\n\n- item\n", 60)
    for line in _plain(out).splitlines():
        assert line == line.rstrip(" \t"), repr(line)


def test_render_plain_text_roundtrips():
    out = markdown.render_markdown("just a plain line\n\nand another\n", 80)
    assert [ln for ln in _plain(out).splitlines() if ln.strip()] == [
        "just a plain line", "and another"]


def test_render_wraps_to_width():
    out = markdown.render_markdown("word " * 40, 40)
    assert len(_plain(out).splitlines()) > 1


def test_render_empty_is_empty():
    assert markdown.render_markdown("", 80) == ""


def test_render_falls_back_to_raw_on_failure():
    # The display path must never drop the answer: a failing renderer yields the raw
    # text. `_render_or_raw` takes the renderer as an argument (not a module patched out
    # from under the import graph), so a local raising callable is the whole test.
    def boom(_text: str) -> str:
        raise RuntimeError("no rich")

    assert markdown._render_or_raw("# Heading\n\nbody\n", boom) == "# Heading\n\nbody\n"


def test_render_normalizes_mismatched_table_delimiter():
    # A classic small-model slip: 3 header cells, 2 delimiter cells. GFM rejects the
    # whole table and it prints as raw pipes; the renderer pads the delimiter to match,
    # so the table draws. (Stored text is untouched — see _normalize_tables.)
    out = markdown.render_markdown("| A | B | C |\n|---|---|\n| 1 | 2 | 3 |\n", 80)
    assert "|---|" not in _plain(out)     # raw delimiter gone -> parsed as a table
    assert "─" in out                      # rich drew the header rule
    assert all(c in _plain(out) for c in ("A", "B", "C", "1", "2", "3"))


def test_normalize_tables_pads_and_trims_to_header():
    assert markdown._normalize_tables("| A | B | C |\n|---|---|\n") == (
        "| A | B | C |\n|---|---|---|\n")
    assert markdown._normalize_tables("| A |\n|---|---|\n") == "| A |\n|---|\n"
    good = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    assert markdown._normalize_tables(good) == good


def test_normalize_tables_ignores_code_fences_and_setext():
    fenced = "```\n| A | B | C |\n|---|---|\n```\n"
    assert markdown._normalize_tables(fenced) == fenced
    # `text` followed by `---` is a setext heading, not a one-cell table.
    setext = "Some | text\n---\n"
    assert markdown._normalize_tables(setext) == setext
