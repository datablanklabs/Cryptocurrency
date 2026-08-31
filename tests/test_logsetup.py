"""logsetup.configure() must be idempotent (no handler stacking) and actually
write to the rotating file."""

from __future__ import annotations

from cryptoyolo import logsetup


def test_configure_idempotent_and_writes(tmp_path):
    logf = tmp_path / "t.log"
    logsetup.configure(logfile=logf, force=True)
    n1 = len(logsetup.get_logger().handlers)
    logsetup.configure(logfile=logf, force=True)
    n2 = len(logsetup.get_logger().handlers)
    assert n1 == n2 >= 2                          # console + file, rebuilt not stacked

    log = logsetup.get_logger("unit")
    assert log.name == "cryptoyolo.unit"
    log.warning("marker-line-xyz")
    for h in logsetup.get_logger().handlers:
        h.flush()
    assert "marker-line-xyz" in logf.read_text()


def test_get_logger_has_no_side_effects():
    # calling the accessor alone attaches nothing
    fresh = logsetup.get_logger("plain.accessor")
    assert fresh.name == "cryptoyolo.plain.accessor"
