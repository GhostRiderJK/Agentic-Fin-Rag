"""The CLI must survive a Windows console that cannot encode its own output."""

from __future__ import annotations

import io
import sys

from app.cli import configure_console


def test_configure_console_makes_a_cp1252_stream_accept_checkmarks(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    cp1252 = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", cp1252)

    configure_console()

    cp1252.write("✓ database initialized")
    cp1252.flush()
    assert "✓".encode("utf-8") in buffer.getvalue()


def test_configure_console_leaves_a_stream_without_reconfigure_alone(monkeypatch):
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)
    configure_console()
    plain.write("ok")
    assert plain.getvalue() == "ok"
