"""Unit tests for `clipboard.copy` — the `/copy` clipboard writer.

The writer is injected so these never touch the real pasteboard (and never `pbcopy`).
The contract they pin: empty text is refused before any write, a writer failure comes
back as a reason rather than an exception, and success reports "". No model, no I/O.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad import clipboard  # noqa: E402


def test_copy_writes_and_reports_success():
    seen = []
    ok, reason = clipboard.copy("hello", write=lambda t: seen.append(t) or True)
    assert ok is True
    assert reason == ""
    assert seen == ["hello"]


def test_copy_refuses_empty_without_writing():
    calls = []
    for text in ("", "   ", "\n\t"):
        ok, reason = clipboard.copy(text, write=lambda t: calls.append(t) or True)
        assert ok is False
        assert reason == "nothing to copy"
    assert calls == []  # pbcopy is never spawned for an empty payload


def test_copy_reports_writer_failure():
    ok, reason = clipboard.copy("hi", write=lambda _t: False)
    assert ok is False
    assert "pbcopy" in reason


def test_copy_reports_oserror_as_a_reason():
    def boom(_t):
        raise OSError("no such tool")

    ok, reason = clipboard.copy("hi", write=boom)
    assert ok is False
    assert "clipboard unavailable" in reason
