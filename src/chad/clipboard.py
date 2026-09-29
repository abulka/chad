"""System clipboard write for chad's `/copy`.

chad is a local, Apple-Silicon-only agent, so the clipboard is macOS `pbcopy`: one
subprocess, no dependency, and no OSC 52 escape interleaved with the transcript the TUI
prints into normal scrollback. `write` is a seam so tests can pin the success/failure
contract without touching the user's pasteboard (and without `mock.patch`, which
anti-slop bans).
"""

import subprocess
from collections.abc import Callable


def _pbcopy(text: str) -> bool:
    """Write `text` to the macOS pasteboard. False when `pbcopy` is missing or fails."""
    try:
        proc = subprocess.run(["pbcopy"], input=text, text=True, check=False)
    except OSError:
        return False
    return proc.returncode == 0


def copy(text: str, write: Callable[[str], bool] = _pbcopy) -> tuple[bool, str]:
    """Put `text` on the clipboard. Returns `(ok, reason)`, with `reason` "" on success.

    Empty or whitespace-only text is refused before spawning `pbcopy`: a `/copy` with
    nothing to copy should say so, not silently clobber the pasteboard with a blank."""
    if not text or not text.strip():
        return False, "nothing to copy"
    try:
        ok = write(text)
    except OSError as e:
        return False, f"clipboard unavailable: {e}"
    if not ok:
        return False, "clipboard write failed (is pbcopy available?)"
    return True, ""
