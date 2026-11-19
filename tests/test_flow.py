"""The conversation state machine, end to end, with no hardware."""

import asyncio
import threading
import time
from collections import deque

import numpy as np

import harness as h
from harness import A, b64, feed_mic, ns, scenario, script, wait_for, written


def test_reference_output_callback_drains_and_pads():
    RO = ns["ReferenceOutput"]
    ro = RO.__new__(RO)
    ro._lock = threading.Lock()
    ro._chunks = deque()
    ro._offset = 0
    ro.write(np.arange(1, 1001, dtype=np.int16))
    ro.write(np.arange(1001, 1201, dtype=np.int16))
    out = np.zeros((800, 2), np.int16)
    ro._callback(out, 800, None, None)
    assert out[0, 0] == 1 and out[799, 1] == 800
    out = np.zeros((800, 2), np.int16)
    ro._callback(out, 800, None, None)
    assert out[0, 0] == 801 and out[399, 0] == 1200 and out[400:].sum() == 0


def test_wake_stream_verify_play_followup_then_wake():
    async def body(ws, c):
        await feed_mic(2)
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE after wake")
        await feed_mic(3)
        assert ws.types()[:2] == ["input_audio_buffer.clear", "input_audio_buffer.append"], ws.types()[:3]
        assert c.verification_pending
        ws.push({"type": "input_audio_buffer.speech_started"})
        await asyncio.sleep(0.05)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
        assert not c.verification_pending and "response.create" in ws.types()
        ws.push({"type": "response.created", "response": {"id": "r1"}})
        for _ in range(3):
            ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(0.1)})
        ws.push({"type": "response.done", "response": {"id": "r1", "status": "completed", "output": []}})
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up")
        total48 = sum(len(a) for a in written["audible"])
        assert abs(total48 - 14400) <= 48, total48
        assert written["aborted"] == 0 and written["stopped"] >= 1
        assert c.pending_input_clear and not h.speaking()
        await feed_mic(60)  # a follow-up window with no speech closes on its own
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")

    scenario(body)


def test_barge_in_cancels_reply_and_listens():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r2"}})
        for _ in range(4):  # 2 s of audio, reply still being generated
            ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(0.5)})
        await wait_for(h.speaking, what="speaking")
        await asyncio.sleep(0.15)
        h.fire_barge()
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "interrupt", what="interrupt")
        assert "response.cancel" in ws.types(), ws.types()
        assert c.cancelled_response_id == "r2" and written["aborted"] >= 1
        total48 = sum(len(a) for a in written["audible"])
        assert total48 < 96000, total48  # nowhere near the full 2 s
        # late audio and the cancelled reply's end are ignored
        ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(0.5)})
        ws.push({"type": "response.done", "response": {"id": "r2", "status": "cancelled", "output": []}})
        await asyncio.sleep(0.1)
        assert c.player is None and c.cancelled_response_id is None and c.state == A.ACTIVE

    scenario(body)


def test_rejected_speaker_at_speech_end():
    async def body(ws, c):
        script["speaker"] = 0.1
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
        ws.push({"type": "input_audio_buffer.speech_started"})
        await asyncio.sleep(0.02)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="rejected -> wake")
        assert "response.create" not in ws.types()
        assert ws.types().count("input_audio_buffer.clear") >= 1
        assert c.wake_cooldown_until > time.monotonic()

    scenario(body)


def test_rejected_speaker_mid_stream():
    async def body(ws, c):
        script["speaker"] = 0.1
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
        await feed_mic(16)  # ~1.3 s with pre-roll, no speech yet: must not judge
        assert c.state == A.ACTIVE, "judged before the server reported speech"
        ws.push({"type": "input_audio_buffer.speech_started"})
        await asyncio.sleep(0.02)
        await feed_mic(13)  # ~1.0 s of speech
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="mid-stream rejection")

    scenario(body)


def test_strict_verification_with_buffered_playback():
    async def body(ws, c):
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.VERIFY_WAKE, what="VERIFY_WAKE")
        assert ws.sent == []  # nothing leaves the machine before the speaker is verified
        await feed_mic(20)  # 1.6 s
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE after strict verify")
        assert ws.types()[:2] == ["input_audio_buffer.clear", "input_audio_buffer.append"]
        ws.push({"type": "input_audio_buffer.speech_started"})
        await asyncio.sleep(0.02)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
        ws.push({"type": "response.created", "response": {"id": "r3"}})
        for _ in range(3):
            ws.push({"type": "response.output_audio.delta", "response_id": "r3", "delta": b64(0.1)})
        assert written["audible"] == []  # buffered: nothing plays before the reply is complete
        ws.push({"type": "response.done", "response": {"id": "r3", "status": "completed", "output": []}})
        await wait_for(lambda: c.state == A.FOLLOWUP, what="strict FOLLOWUP state")
        total48 = sum(len(a) for a in written["audible"])
        assert abs(total48 - 14400) <= 48, total48
        await feed_mic(30)  # 2.4 s of verified follow-up speech
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up")

    with h.setting(SPEAKER_VERIFY_MODE="strict", STREAMING_PLAYBACK=False):
        scenario(body)


def test_wake_phrase_heard_by_the_microphone_loop_interrupts_playback():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r2"}})
        for _ in range(4):
            ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(0.5)})
        await wait_for(h.speaking, what="speaking")
        await feed_mic(5)  # quiet blocks re-arm the detector after the first wake
        assert h.speaking()
        script["wake"] = 0.5
        await feed_mic(2)
        script["wake"] = 0.0
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "interrupt", what="interrupt")
        assert "response.cancel" in ws.types()

    scenario(body)


def test_one_spoken_wake_phrase_cannot_fire_twice():
    async def body(ws, c):
        await feed_mic(3)
        script["wake"] = 0.5  # the phrase stays in the model's memory for a while
        await feed_mic(1)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
        c.return_to_wake()
        await feed_mic(6)
        assert c.state == A.WAIT_WAKE, "the same phrase woke Nexus again"
        script["wake"] = 0.0
        await feed_mic(5)
        script["wake"] = 0.5  # a new phrase
        await feed_mic(1)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE again")

    scenario(body)
