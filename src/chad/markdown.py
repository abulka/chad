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
from dataclasses import dataclass
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


_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_DELIM_CELL_RE = re.compile(r"^:?-+:?$")


def _split_row(row: str) -> list[str]:
    """Cells of a GFM table row: split on unescaped pipes, dropping the optional
    leading/trailing pipe. A `\\|` is cell content, not a separator."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, i = [], [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
        elif s[i] == "|":
            cells.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(s[i])
            i += 1
    cells.append("".join(cur))
    return cells


def _is_table_separator(row: str) -> bool:
    """A GFM delimiter row: pipe-delimited cells that are all hyphens (colons for
    alignment). Requiring a pipe avoids mistaking a setext heading (`text` / `---`) for
    a one-cell table."""
    if "|" not in row:
        return False
    cells = _split_row(row)
    return bool(cells) and all(_DELIM_CELL_RE.match(c.strip()) for c in cells)


def _normalize_tables(text: str) -> str:
    """Pad/trim a GFM delimiter row to match its header's column count.

    Small models routinely emit a header with N cells but a delimiter row with N-1;
    GFM then rejects the *whole* table and the terminal prints raw pipes. Rebuilding the
    delimiter — and nothing else — makes the table render. Display-only: the stored
    message, `/copy` and `/export` keep the model's original text, which real renderers
    (VS Code, GitHub) will still refuse."""
    lines = text.split("\n")
    out: list[str] = []
    fence = None  # (fence char, run length) while inside a fenced block
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_LINE_RE.match(line)
        if m:
            run, rest = m.group(1), m.group(2)
            if fence is None:
                fence = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not rest.strip():
                # A closer needs the same character, at least the opener's length, and
                # nothing but whitespace after. Anything else (a nested ` ```python `, a
                # shorter run) is block content, so a table-like line inside it is left
                # alone.
                fence = None
            out.append(line)
            i += 1
            continue
        if (fence is None and "|" in line and i + 1 < len(lines)
                and _is_table_separator(lines[i + 1])):
            cols = len(_split_row(line))
            if cols and len(_split_row(lines[i + 1])) != cols:
                out.append(line)
                out.append("|" + "---|" * cols)
                i += 2
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


@dataclass
class _Fence:
    line: int              # index into the split lines
    char: str              # fence character (backtick or tilde)
    length: int            # run length as written
    indent: str            # leading spaces, preserved when the run is rewritten
    info: str              # trailing info string (empty for a closer)
    widest_child: int = 0  # widest same-char fence nested inside this opener


def _normalize_fences(text: str) -> str:
    """Widen an outer fence so same-character nesting parses.

    CommonMark cannot nest a ``` block inside another ```: the first bare ``` closes the
    outer one, so a model that writes ```markdown holding ```python gets a short block and
    the rest of the message drifts out of it. The intent is unambiguous (an info-bearing
    inner fence), and the standard representation is a longer outer fence. Pair fences
    with a stack — a fence with an info string opens, a bare fence of the same character
    at least as long closes, anything else is content — then widen each opener that holds
    same-character children to one more than its widest child, and its closer to match.
    Unclosed openers are closed too, so a dangling block can't swallow what follows."""
    lines = text.split("\n")
    out = list(lines)
    stack: list[_Fence] = []

    def widen(top: _Fence, closer_idx: int | None, closer_indent: str) -> None:
        eff = max(top.length, top.widest_child + 1) if top.widest_child else top.length
        out[top.line] = top.indent + top.char * eff + top.info
        if closer_idx is None:
            out.append(top.char * eff)  # synthesize the closer for an unclosed opener
        else:
            out[closer_idx] = closer_indent + top.char * eff
        if stack:  # report this block's width to its parent before it is popped
            stack[-1].widest_child = max(stack[-1].widest_child, eff)

    for i, line in enumerate(lines):
        m = _FENCE_LINE_RE.match(line)
        if not m:
            continue
        run, rest = m.group(1), m.group(2)
        char, n, info = run[0], len(run), rest.strip()
        indent = line[:m.start(1)]
        if stack:
            top = stack[-1]
            if char == top.char and info == "" and n >= top.length:
                stack.pop()
                widen(top, i, indent)
            elif char == top.char and info != "":
                stack.append(_Fence(i, char, n, indent, rest))
            # a different char, or a shorter bare fence, is block content: ignore
        else:
            stack.append(_Fence(i, char, n, indent, rest))
    while stack and out and out[-1] == "":
        out.pop()  # don't strand a synthesized closer below a trailing blank line
    while stack:  # unclosed openers: close innermost first
        widen(stack.pop(), None, "")
    return "\n".join(out)


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
    return _render_or_raw(_normalize_tables(_normalize_fences(text)),
                          lambda t: _rich_render(t, width))
