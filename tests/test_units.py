"""Pure helpers: resampling, formatting, env flags, queue polling, timers."""
import asyncio
import json
import os
import queue
from datetime import datetime, timedelta

import numpy as np

from harness import ns


def test_resample_int16():
    r = ns["resample_int16"]
    t = np.arange(16000) / 16000
    sig = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    out = r(sig, 16000, 24000)
    assert out.dtype == np.int16 and abs(len(out) - 24000) <= 1
    assert abs(len(r(out, 24000, 48000)) - 48000) <= 2
    ref = r(out, 24000, 16000)
    assert abs(len(ref) - 16000) <= 2
    assert 0.95 < np.abs(ref.astype(float)).max() / 10000 < 1.05
    assert len(r(np.array([], dtype=np.int16), 16000, 24000)) == 0
    assert r(sig, 16000, 16000) is sig
    loud = np.full(1600, 32767, dtype=np.int16)
    assert r(loud, 16000, 24000).max() <= 32767


def test_stream_resampler_chunked_equals_whole():
    R = ns["StreamResampler"]
    t = np.arange(24000) / 24000
    sig = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    for new_rate in (48000, 16000):
        r = R(24000, new_rate)
        chunked = np.concatenate([r.process(sig[i : i + 960]) for i in range(0, len(sig), 960)])
        whole = R(24000, new_rate).process(sig)
        assert abs(len(chunked) - new_rate) <= 2 and np.array_equal(chunked, whole)
        tail = chunked[2000:]
        assert 0.97 < np.abs(tail.astype(float)).max() / 12000 < 1.03
        step = np.abs(np.diff(tail.astype(float))).max()
        assert step < 12000 * 2 * np.pi * 440 / new_rate * 1.2  # no clicks


def test_format_duration_and_env_flag():
    f = ns["format_duration"]
    assert f(0) == "0 seconds" and f(61) == "1 minute 1 second" and f(3600) == "1 hour"
    os.environ["X_FLAG"] = "yes"
    assert ns["env_flag"]("X_FLAG") is True
    os.environ["X_FLAG"] = "0"
    assert ns["env_flag"]("X_FLAG") is False
    assert ns["env_flag"]("X_MISSING") is False


def test_queue_get_bounded_blocking():
    async def body():
        q = queue.Queue()
        task = asyncio.create_task(ns["queue_get"](q))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.6)  # the worker thread holds the queue for up to its 0.5 s poll
        q.put("x")
        assert await asyncio.wait_for(ns["queue_get"](q), 2) == "x"

    asyncio.run(body())


def test_timer_persistence_and_restore():
    fired = []

    async def fake_play(timer_id, label):
        fired.append((timer_id, label))

    async def body():
        real_play = ns["play_timer_alarm"]
        ns["play_timer_alarm"] = fake_play
        try:
            ns["timers"].clear()
            ns["register_timer"](1, "pasta", 600, 600)
            data = json.loads(ns["TIMERS_FILE"].read_text())
            assert data[0]["timer_id"] == 1 and data[0]["label"] == "pasta"
            due = datetime.fromisoformat(data[0]["due_at"])
            assert 590 < (due - datetime.now().astimezone()).total_seconds() <= 600
            for tm in ns["timers"].values():
                tm["task"].cancel()
            ns["timers"].clear()
            now = datetime.now().astimezone()
            ns["TIMERS_FILE"].write_text(
                json.dumps(
                    [
                        {"timer_id": 3, "label": "future", "duration_seconds": 100,
                         "due_at": (now + timedelta(seconds=100)).isoformat()},
                        {"timer_id": 5, "label": "late", "duration_seconds": 100,
                         "due_at": (now - timedelta(seconds=30)).isoformat()},
                        {"timer_id": 7, "label": "stale", "duration_seconds": 100,
                         "due_at": (now - timedelta(hours=2)).isoformat()},
                    ]
                )
            )
            await ns["restore_timers"]()
            assert set(ns["timers"]) == {3, 5}
            assert ns["next_timer_id"] == 8
            await asyncio.sleep(0.2)  # the late timer fires immediately
            assert fired == [(5, "late")]
            assert set(ns["timers"]) == {3}
            saved = json.loads(ns["TIMERS_FILE"].read_text())
            assert [e["timer_id"] for e in saved] == [3]
            for tm in ns["timers"].values():
                tm["task"].cancel()
            ns["timers"].clear()
        finally:
            ns["play_timer_alarm"] = real_play

    asyncio.run(body())
