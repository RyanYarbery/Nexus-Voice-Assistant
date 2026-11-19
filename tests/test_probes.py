"""Failure paths found in the 2026-11-18 review (probes A to G).

Each test describes behaviour the assistant must have. All seven failed on
the code as reviewed (commit 952e9a2).
"""

import asyncio

import harness as h
from harness import A, b64, feed_mic, ns, scenario, script, wait_for


def test_a_spoken_lead_in_before_a_tool_call_does_not_jam_the_speaker():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r1"}})
        ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(0.2)})
        ws.push(
            {
                "type": "response.done",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "output": [
                        {"type": "function_call", "name": "list_timers", "call_id": "c1", "arguments": "{}"}
                    ],
                },
            }
        )
        # The lead-in finishes playing and lets go of the speaker.
        await wait_for(lambda: c.player is None and not h.speaking(), timeout=2.0, what="lead-in to finish")
        assert not h.speaker_lock_held()
        # No follow-up window yet: the real answer is still to come.
        assert c.state == A.WAIT_RESPONSE, c.state
        ws.push({"type": "response.created", "response": {"id": "r2"}})
        ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(1.0)})
        seen = set()
        for _ in range(30):
            seen.add(c.state)
            await asyncio.sleep(0.01)
        assert A.ACTIVE not in seen, "follow-up window opened while the answer was still playing"
        ws.push({"type": "response.done", "response": {"id": "r2", "status": "completed", "output": []}})
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up")

    scenario(body)


def test_b_error_while_waiting_for_a_reply_returns_to_wake():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "error", "error": {"message": "Rate limit reached.", "code": "rate_limit_exceeded"}})
        await feed_mic(20)
        await wait_for(lambda: c.state == A.WAIT_WAKE, timeout=2.0, what="return to wake after error")

    scenario(body)


def test_b_no_reply_at_all_times_out():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        await feed_mic(80)  # 0.8 s of waiting with a 0.5 s limit
        await wait_for(lambda: c.state == A.WAIT_WAKE, timeout=2.0, what="reply timeout")

    with h.setting(REPLY_TIMEOUT_SECONDS=0.5):
        scenario(body)


def test_c_speech_that_never_stops_is_cut_off():
    async def body(ws, c):
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
        ws.push({"type": "input_audio_buffer.speech_started"})
        n0 = ws.types().count("input_audio_buffer.append")
        await feed_mic(250)  # 2.5 s of wall time with a 1 s limit
        assert c.state == A.WAIT_WAKE, c.state
        assert ws.types().count("input_audio_buffer.append") - n0 < 150

    with h.setting(REQUEST_MAX_SECONDS=1.0):
        scenario(body)


def test_d_failing_speaker_cancels_the_reply_once():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        orig = h.FakeStream.write

        def bad(self, data):
            if self.samplerate != 16000:
                raise h.PortAudioError("device busy")
            return orig(self, data)

        made = {"n": 0}
        RP = ns["ResponsePlayer"]

        class Counting(RP):
            def __init__(self, *a, **k):
                made["n"] += 1
                super().__init__(*a, **k)

        h.FakeStream.write = bad
        ns["ResponsePlayer"] = Counting
        try:
            ws.push({"type": "response.created", "response": {"id": "r1"}})
            for _ in range(8):
                ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(0.3)})
                await asyncio.sleep(0.25)
            ws.push({"type": "response.done", "response": {"id": "r1", "status": "completed", "output": []}})
            await asyncio.sleep(0.3)
        finally:
            h.FakeStream.write = orig
            ns["ResponsePlayer"] = RP
        assert made["n"] == 1, f"{made['n']} players for one reply"
        assert "response.cancel" in ws.types()
        assert c.state == A.WAIT_WAKE

    scenario(body)


def test_e_beeps_wait_for_the_request_to_end():
    async def body(ws, c):
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
        ws.push({"type": "input_audio_buffer.speech_started"})
        await feed_mic(5)
        n0 = ws.types().count("input_audio_buffer.append")
        beeps = asyncio.create_task(h.play_beeps(8))
        await asyncio.sleep(0.02)
        for _ in range(20):
            h.mic_callback_block()
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        forwarded = ws.types().count("input_audio_buffer.append") - n0
        assert forwarded >= 18, f"only {forwarded} of 20 blocks reached the request"
        assert not beeps.done(), "beeps rang during the request"
        c.return_to_wake()
        await asyncio.wait_for(beeps, 5)  # they ring once the turn is over

    scenario(body)


async def _reply_then_followup(ws, c):
    await h.to_wait_response(ws, c)
    ws.push({"type": "response.created", "response": {"id": "r1"}})
    ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(0.2)})
    ws.push({"type": "response.done", "response": {"id": "r1", "status": "completed", "output": []}})
    await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up")


def test_f_one_rejected_sound_does_not_close_the_followup_window():
    async def body(ws, c):
        await _reply_then_followup(ws, c)
        script["speaker"] = 0.1  # an unknown voice, or the echo tail
        await feed_mic(2)
        ws.push({"type": "input_audio_buffer.speech_started"})
        await feed_mic(3)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await feed_mic(10)
        assert c.state == A.ACTIVE and c.active_origin == "followup", "window closed by one rejected sound"
        assert "response.create" not in ws.types()[-12:]
        script["speaker"] = 0.6  # now the real follow-up
        creates = ws.types().count("response.create")
        ws.push({"type": "input_audio_buffer.speech_started"})
        await feed_mic(3)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="follow-up accepted")
        assert ws.types().count("response.create") == creates + 1

    with h.setting(FOLLOWUP_SECONDS=3.0, FOLLOWUP_GUARD_SECONDS=0.0, FOLLOWUP_MIN_SPEECH_SECONDS=0.0):
        scenario(body)


def test_f_blip_right_after_playback_is_ignored():
    async def body(ws, c):
        await _reply_then_followup(ws, c)
        appends = ws.types().count("input_audio_buffer.append")
        await feed_mic(5)  # inside the guard: the echo tail must not be uploaded
        assert ws.types().count("input_audio_buffer.append") == appends
        await asyncio.sleep(0.3)
        ws.push({"type": "input_audio_buffer.speech_started"})  # a 30 ms blip
        await feed_mic(3)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await feed_mic(5)
        assert c.state == A.ACTIVE and c.active_origin == "followup"
        assert ws.types().count("response.create") == 1

    with h.setting(FOLLOWUP_SECONDS=3.0, FOLLOWUP_GUARD_SECONDS=0.3, FOLLOWUP_MIN_SPEECH_SECONDS=0.4):
        scenario(body)


def test_g_request_that_ends_before_the_cancel_is_confirmed_still_gets_a_reply():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r2"}})
        for _ in range(4):
            ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(0.5)})
        await wait_for(h.speaking, what="speaking")
        await asyncio.sleep(0.15)
        h.fire_barge()
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "interrupt", what="interrupt")
        ws.push({"type": "input_audio_buffer.speech_started"})
        await feed_mic(5)
        ws.push({"type": "input_audio_buffer.speech_stopped"})
        await asyncio.sleep(0.1)
        ws.push({"type": "response.done", "response": {"id": "r2", "status": "cancelled", "output": []}})
        await wait_for(lambda: c.state == A.WAIT_RESPONSE, timeout=2.0, what="second request accepted")
        assert ws.types().count("response.create") == 2

    scenario(body)


def test_a_reply_whose_end_never_arrives_lets_go_of_the_speaker():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r1"}})
        ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(0.2)})
        await wait_for(h.speaking, what="speaking")
        await wait_for(lambda: not h.speaking(), timeout=3.0, what="playback to give up")
        assert not h.speaker_lock_held()

    with h.setting(PLAYER_STALL_SECONDS=0.3):
        scenario(body)


def test_a_interrupted_lead_in_drops_the_answer_that_follows():
    async def body(ws, c):
        await h.to_wait_response(ws, c)
        ws.push({"type": "response.created", "response": {"id": "r1"}})
        ws.push({"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(2.0)})
        await wait_for(h.speaking, what="speaking")
        ws.push(
            {
                "type": "response.done",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "output": [
                        {"type": "function_call", "name": "list_timers", "call_id": "c1", "arguments": "{}"}
                    ],
                },
            }
        )
        await asyncio.sleep(0.1)
        h.fire_barge()
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "interrupt", what="interrupt")
        ws.push({"type": "response.created", "response": {"id": "r2"}})
        ws.push({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64(0.5)})
        await asyncio.sleep(0.2)
        assert "response.cancel" in ws.types()
        assert not h.speaking() and c.player is None, "the answer played over the new request"

    scenario(body)


def test_d_speaker_failing_three_replies_in_a_row_is_reported_and_restarts():
    async def body(ws, c):
        orig = h.FakeStream.write

        def bad(self, data):
            if self.samplerate != 16000:
                raise h.PortAudioError("device busy")
            return orig(self, data)

        h.FakeStream.write = bad
        try:
            for n in range(3):
                assert h.exits == []
                await feed_mic(5)
                await h.to_wait_response(ws, c)
                ws.push({"type": "response.created", "response": {"id": f"r{n}"}})
                ws.push({"type": "response.output_audio.delta", "response_id": f"r{n}", "delta": b64(0.3)})
                await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")
                ws.push(
                    {
                        "type": "response.done",
                        "response": {"id": f"r{n}", "status": "cancelled", "output": []},
                    }
                )
                await asyncio.sleep(0.05)
        finally:
            h.FakeStream.write = orig
        assert len(h.exits) == 1
        assert ns["status"]["problem"] == "speaker"

    scenario(body)
    ns["set_status"]("ok")
