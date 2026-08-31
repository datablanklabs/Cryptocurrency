"""Best-effort out-of-band alerts for the things you want to know about without
tailing a log: an order rejected, an exit trigger firing, the market regime
going risk_off, an equity drawdown, the price feed falling all the way through
to the yfinance backstop.

Three delivery paths, all optional and all best-effort — `notify()` never raises
and never blocks more than a few seconds:

  * ntfy topic       CONFIG.notify.ntfy_url    or env CRYPTO_YOLO_NTFY_URL
  * generic webhook  CONFIG.notify.webhook_url or env CRYPTO_YOLO_ALERT_WEBHOOK
                     (POSTs {"text": ...} — Slack / Discord compatible)
  * macOS banner     Notification Center, when run in a GUI login session

With nothing configured it degrades to a single log line and returns False.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import urllib.request

from .config import CONFIG, Config
from .logsetup import get_logger

_log = get_logger("notify")
_TIMEOUT = 5


def _post(url: str, data: bytes, headers: dict[str, str]) -> None:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    urllib.request.urlopen(req, timeout=_TIMEOUT).close()  # noqa: S310 - user-configured URL


def notify(title: str, message: str, cfg: Config = CONFIG, *,
           tag: str = "info") -> bool:
    """Fire an alert through every configured channel.

    Returns True if at least one channel accepted it. Never raises.
    """
    nc = getattr(cfg, "notify", None)
    if nc is None or not nc.enabled:
        return False

    first_line = message.splitlines()[0] if message else ""
    delivered = False

    ntfy_url = nc.ntfy_url or os.environ.get("CRYPTO_YOLO_NTFY_URL", "")
    if ntfy_url:
        try:
            _post(ntfy_url, message.encode("utf-8"),
                  {"Title": title, "Tags": tag, "Priority": "default"})
            delivered = True
        except Exception as exc:  # noqa: BLE001 - best effort
            _log.warning("ntfy alert failed: %s", exc)

    hook = nc.webhook_url or os.environ.get("CRYPTO_YOLO_ALERT_WEBHOOK", "")
    if hook:
        try:
            _post(hook,
                  json.dumps({"text": f"*{title}*\n{message}"}).encode("utf-8"),
                  {"Content-Type": "application/json"})
            delivered = True
        except Exception as exc:  # noqa: BLE001
            _log.warning("webhook alert failed: %s", exc)

    if nc.macos_banner and platform.system() == "Darwin":
        try:
            t = title.replace('"', "'")
            m = first_line.replace('"', "'")[:220]
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{m}" with title "{t}"'],
                capture_output=True, timeout=_TIMEOUT, check=False,
            )
            delivered = True
        except Exception as exc:  # noqa: BLE001
            _log.warning("macOS banner failed: %s", exc)

    _log.info("ALERT [%s] %s — %s (delivered=%s)",
              tag, title, first_line, delivered)
    return delivered
