"""Beeps, and alarms or timers that came due while Nexus was off."""

import asyncio
import json
from datetime import datetime, timedelta

import harness as h
from harness import A, ns, scenario, script, wait_for


def test_beeps_ring_until_the_wake_phrase_silences_them():
    async def body(ws, c):
        ring = asyncio.create_task(h.play_beeps(ring_seconds=30))
        await wait_for(h.speaking, what="ringing")
        await h.feed_mic(4)
        assert not ring.done()
        script["wake"] = 0.5
        await h.feed_mic(1)
        script["wake"] = 0.0
        assert await asyncio.wait_for(ring, 2) is True
        assert c.state == A.WAIT_WAKE  # the phrase silenced the beeps; it did not start a request

    scenario(body)


def test_beeps_stop_on_their_own_after_the_ring_time():
    async def body(ws, c):
        assert await asyncio.wait_for(h.play_beeps(ring_seconds=0.3), 3) is False
        assert len(h.played) >= 2

    scenario(body)


def test_alarm_due_while_off_rings_if_recent_and_is_reported_if_not():
    rung = []

    async def fake_sound(alarm_id, label):
        rung.append(label)

    async def body():
        now = datetime.now().astimezone()
        ns["ALARMS_FILE"].write_text(
            json.dumps(
                [
                    {
                        "alarm_id": 1,
                        "label": "recent",
                        "target_datetime": (now - timedelta(seconds=90)).isoformat(),
                    },
                    {
                        "alarm_id": 2,
                        "label": "old",
                        "target_datetime": (now - timedelta(hours=3)).isoformat(),
                    },
                    {
                        "alarm_id": 3,
                        "label": "future",
                        "target_datetime": (now + timedelta(hours=3)).isoformat(),
                    },
                ]
            )
        )
        real = ns["play_alarm_sound"]
        ns["play_alarm_sound"] = fake_sound
        ns["alarms"].clear()
        ns["missed_events"].clear()
        try:
            await ns["restore_alarms"]()
            await asyncio.sleep(0.2)
            assert rung == ["recent"]
            assert set(ns["alarms"]) == {3}
            assert len(ns["missed_events"]) == 1 and "old" in ns["missed_events"][0]
        finally:
            ns["play_alarm_sound"] = real
            for alarm in ns["alarms"].values():
                alarm["task"].cancel()
            ns["alarms"].clear()
            ns["missed_events"].clear()

    asyncio.run(body())


def test_a_missed_alarm_is_mentioned_with_the_next_reply():
    async def body(ws, c):
        ns["missed_events"].append("the alarm 'pills' set for 8:00 AM")
        await h.to_wait_response(ws, c)
        notes = [m for m in ws.sent if m["type"] == "conversation.item.create"]
        assert len(notes) == 1 and "pills" in notes[0]["item"]["content"][0]["text"]
        assert ws.types().index("conversation.item.create") < ws.types().index("response.create")
        assert ns["missed_events"] == []

    scenario(body)
