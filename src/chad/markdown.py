"""Assistant Markdown -> ANSI for the TUI transcript.

The model writes Markdown; without a parser the terminal shows the raw `**bold**`,
`#` headings and backtick fences. `render_markdown` renders one completed assistant
message with rich into an in-memory Console (a StringIO, never the live terminal) and
returns the captured ANSI string. The TUI appends that string to its scrollback buffer
on the worker thread, so its single-writer contract (`tui._pending`, drained by the UI
refresher) is preserved and rich can never fight the pinned input/status region.

Display-only: no model, no agent state. With rendering off (`CHAD_NO_MARKDOWN` or the
`/markdown` TUI toggle) the caller keeps the pre-existing raw-token path byte-for-byte.
"""

import functools
import io
import re
from collections.abc import Callable
from typing import ClassVar

# rich's style names for its Markdown renderable, mapped onto chad's understated
# palette (bold headings, cyan code, dim scaffolding) instead of rich's default theme.
# Plain strings, so importing this module costs no rich import; the Theme is built at
# render time, once per assistant message, on the same lazy-import path.
_THEME = {
    "markdown.h1": "bold",
    "markdown.h1.border": "dim",
    "markdown.h2": "bold",
    "markdown.h3": "bold",
    "markdown.h4": "bold",
    "markdown.h5": "bold",
    "markdown.h6": "bold",
    "markdown.strong": "bold",
    "markdown.em": "italic",
    "markdown.code": "cyan",
    "markdown.code_block": "dim",
    "markdown.block_quote": "dim",
    "markdown.item.bullet": "dim",
    "markdown.link": "underline cyan",
    "markdown.link_url": "dim",
    "markdown.hr": "dim",
    "markdown.table.header": "bold",
    "markdown.table.border": "dim",
}

# rich pads every rendered line to the Console width, sometimes *inside* a styled span
# (a block quote or code block pads its own trailing spaces). Strip that padding —
# allowing SGR and OSC-8 hyperlink sequences to trail the spaces — so the transcript
# carries no full-width trailing whitespace, which would otherwise land in copy/paste.
_TRAIL_PAD = re.compile(r"[ \t]+((?:\x1b\[[0-9;]*m|\x1b\]8;[^\x1b]*\x1b\\|\x1b\])*)$")


@functools.lru_cache(maxsize=1)
def _markdown_class() -> type:
    """The rich `Markdown` subclass chad renders with, built once on first use.

    rich centers `h1` (Heading.LEVEL_ALIGN), which in a scrolling transcript pads the
    line with leading spaces and reads as a banner; chad's headings line up with the
    prose and tool lines, so every level is flush left. Built lazily so importing this
    module never imports rich."""
    from rich.console import JustifyMethod
    from rich.markdown import Heading, Markdown

    class _LeftHeading(Heading):
        LEVEL_ALIGN: ClassVar[dict[str, JustifyMethod]] = {
            "h1": "left",
            "h2": "left",
            "h3": "left",
            "h4": "left",
            "h5": "left",
            "h6": "left",
        }

    class _TranscriptMarkdown(Markdown):
        elements = {**Markdown.elements, "heading_open": _LeftHeading}

    return _TranscriptMarkdown


def _strip_padding(rendered: str) -> str:
    """Drop rich's trailing width-padding from every line (see _TRAIL_PAD)."""
    lines = []
    for line in rendered.split("\n"):
        while True:
            trimmed = _TRAIL_PAD.sub(r"\1", line)
            if trimmed == line:
                break
            line = trimmed
        lines.append(line)
    return "\n".join(lines).rstrip("\n")


def _render_or_raw(text: str, render: Callable[[str], str]) -> str:
    """Run `render`; on any failure return `text` unchanged — the display path must
    never drop or corrupt the answer. A parameter, not a module lookup, so a test can
    pass a failing renderer without patching rich out from under the import graph."""
    try:
        return render(text)
    except Exception:  # noqa: BLE001 - display path; fall back to the raw text
        return text


def _rich_render(text: str, width: int) -> str:
    """Render with rich; may raise (a missing rich, a bad message) — the caller falls
    back to the raw text."""
    from rich.console import Console
    from rich.theme import Theme

    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=max(20, width),
        color_system="standard",
        theme=Theme(_THEME),
    )
    # "ansi_dark" keeps fenced code on the terminal's own 16 colours, like the rest of
    # the transcript, rather than a truecolour pygments theme.
    console.print(_markdown_class()(text, code_theme="ansi_dark"))
    return _strip_padding(buf.getvalue())


def render_markdown(text: str, width: int) -> str:
    """Render a full assistant message to ANSI at `width` columns, without a trailing
    newline. Returns `text` unchanged if rich is unavailable or rendering raises."""
    if not text:
        return ""
    return _render_or_raw(text, lambda t: _rich_render(t, width))
