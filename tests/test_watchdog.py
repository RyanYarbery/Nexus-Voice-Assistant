"""A dead microphone or a frozen loop must end in a restart, not in silence."""

import asyncio
import os
import socket
import tempfile

import harness as h
from harness import ns


def test_microphone_that_stops_delivering_audio_is_reported_and_restarts():
    async def body():
        task = asyncio.create_task(ns["microphone_watchdog"]())
        for _ in range(10):  # audio arriving: nothing happens
            h.mic_callback_block()
            await asyncio.sleep(0.03)
        assert h.exits == []
        await asyncio.sleep(0.6)  # the stream goes quiet
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    h.exits.clear()
    with h.setting(MIC_STALL_SECONDS=0.2):
        asyncio.run(body())
    assert len(h.exits) == 1
    assert ns["status"]["problem"] == "microphone"
    ns["set_status"]("ok")
    ns["clear_mic_queue"]()


def test_systemd_is_told_ready_and_pinged():
    path = os.path.join(tempfile.mkdtemp(), "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    server.settimeout(1)
    os.environ["NOTIFY_SOCKET"] = path
    try:
        assert ns["sd_notify"]("READY=1") is True
        assert server.recv(100) == b"READY=1"
    finally:
        os.environ.pop("NOTIFY_SOCKET")
        server.close()
    assert ns["sd_notify"]("WATCHDOG=1") is False  # no socket: quietly does nothing
