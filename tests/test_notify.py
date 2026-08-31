"""notify() is best-effort: it must never raise and must no-op cleanly when
nothing is configured."""

from __future__ import annotations

from cryptoyolo import notify as notify_mod
from cryptoyolo.config import Config


def test_disabled_is_a_silent_noop():
    cfg = Config()
    cfg.notify.enabled = False
    assert notify_mod.notify("t", "m", cfg) is False


def test_no_channels_configured_never_raises(monkeypatch):
    monkeypatch.setattr(notify_mod.platform, "system", lambda: "Linux")
    cfg = Config()
    cfg.notify.ntfy_url = ""
    cfg.notify.webhook_url = ""
    assert notify_mod.notify("title", "body\nsecond line", cfg) is False


def test_macos_banner_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(notify_mod.platform, "system", lambda: "Darwin")

    def _boom(*a, **k):
        raise OSError("osascript missing")
    monkeypatch.setattr(notify_mod.subprocess, "run", _boom)
    cfg = Config()
    assert notify_mod.notify("t", "m", cfg) is False        # caught, not raised


def test_macos_banner_success_reports_delivered(monkeypatch):
    monkeypatch.setattr(notify_mod.platform, "system", lambda: "Darwin")
    calls = []
    monkeypatch.setattr(notify_mod.subprocess, "run",
                        lambda *a, **k: calls.append(a) or None)
    cfg = Config()
    assert notify_mod.notify("t", "m", cfg) is True
    assert calls and "osascript" in calls[0][0]
