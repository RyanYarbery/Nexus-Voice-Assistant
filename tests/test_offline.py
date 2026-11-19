"""No connection to the voice service: keep listening, say why, tell the owner."""

import asyncio
import json

import harness as h
from harness import A, feed_mic, ns, scenario, script, wait_for


def test_failures_are_sorted_into_account_and_network_problems():
    kind = ns["classify_failure"]
    assert kind("received 1013 (try again later) insufficient_quota.credit_balance_exhausted") == "quota"
    assert kind("You have no credits remaining.") == "quota"
    assert kind("InvalidStatus: server rejected WebSocket connection: HTTP 401") == "auth"
    assert kind("RuntimeError: OPENAI_API_KEY is not set") == "auth"
    assert kind("The model `gpt-realtime` does not exist or you do not have access to it.") == "model"
    assert kind("TimeoutError: timed out during opening handshake") == "network"
    assert kind("gaierror: [Errno -3] Temporary failure in name resolution") == "network"


def test_wake_phrase_with_no_connection_plays_a_notice_and_keeps_listening():
    async def body(ws, c):
        c.link.detach()
        c.link.problem = "quota"
        await h.wake(ws, c)  # offline: the speaker is checked before anything is said
        await feed_mic(25)
        await wait_for(lambda: len(h.played) == 1, what="notice")
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")
        assert ws.sent == []  # nothing was sent anywhere
        await feed_mic(5)
        await h.wake(ws, c)  # the loop is still alive and hears the next one
        await feed_mic(25)
        await wait_for(lambda: len(h.played) == 2, what="second notice")

    scenario(body)


def test_unknown_voice_with_no_connection_gets_no_notice():
    async def body(ws, c):
        c.link.detach()
        script["speaker"] = 0.1
        await h.wake(ws, c)
        await feed_mic(25)
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")
        assert h.played == []

    scenario(body)


def test_notices_can_be_switched_off():
    async def body(ws, c):
        c.link.detach()
        await h.wake(ws, c)
        await feed_mic(25)
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")
        assert h.played == []

    with h.setting(OFFLINE_NOTICES=False):
        scenario(body)


def test_connection_dropping_mid_request_does_not_stop_the_microphone_loop():
    async def body(ws, c):
        await h.wake(ws, c)
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")

        async def broken(payload):
            raise ConnectionError("socket closed")

        ws.send = broken
        await feed_mic(3)
        await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")
        assert c.link.failed.is_set()  # the connection manager is told
        c.link.detach()
        await feed_mic(5)
        await h.wake(ws, c)  # still listening
        assert c.state == A.VERIFY_WAKE

    scenario(body)


def _run_manager(error_text, seconds):
    attempts = []

    async def failing_connection(api_key, controller):
        attempts.append(1)
        raise ConnectionError(error_text)

    async def body():
        controller = ns["NexusController"](None, None, None)
        real = ns["run_connection"]
        ns["run_connection"] = failing_connection
        task = asyncio.create_task(ns["connection_manager"]("key", controller))
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            ns["run_connection"] = real
        return controller

    with h.setting(RECONNECT_INITIAL_SECONDS=0.02, RECONNECT_MAX_SECONDS=0.05, ACCOUNT_RETRY_SECONDS=30):
        controller = asyncio.run(body())
    return attempts, controller


def test_out_of_credit_is_retried_slowly_and_reported():
    attempts, controller = _run_manager("received 1013 insufficient_quota.credit_balance_exhausted", 0.5)
    assert len(attempts) == 1, f"{len(attempts)} attempts in half a second"
    assert controller.link.problem == "quota"
    data = json.loads(ns["STATUS_FILE"].read_text())
    assert data["state"] == "problem" and data["problem"] == "quota"
    assert data["summary"].startswith("problem: out of credit")


def test_network_failures_are_retried_quickly_and_reported_after_a_few():
    attempts, controller = _run_manager("gaierror: Temporary failure in name resolution", 0.5)
    assert len(attempts) >= 4
    data = json.loads(ns["STATUS_FILE"].read_text())
    assert data["state"] == "problem" and data["problem"] == "network"


def test_status_returns_to_ok_and_secret_settings_are_masked():
    ns["set_status"]("ok")
    data = json.loads(ns["STATUS_FILE"].read_text())
    assert data["state"] == "ok" and data["summary"] == "ok" and "updated" in data
    ns["SETTINGS"]["NEXUS_NTFY_URL"]["value"] = "https://ntfy.example/secret-topic"
    try:
        text = "\n".join(ns["describe_settings"]())
        assert "secret-topic" not in text and "NEXUS_NTFY_URL=(set)" in text
    finally:
        ns["SETTINGS"]["NEXUS_NTFY_URL"]["value"] = ""


def test_notice_audio_falls_back_to_tones_without_a_synthesiser():
    pcm = ns["notice_audio"]("quota", "The OpenAI account is out of credit.")
    assert pcm.dtype.name == "int16" and len(pcm) > 4800 and abs(int(pcm.max())) > 1000
