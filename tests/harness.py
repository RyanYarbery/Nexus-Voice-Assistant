"""Hardware-free test rig.

Loads the assistant with `sounddevice`, `torch`, `websockets`, `openwakeword`
and `speechbrain` replaced by stubs, then lets a test drive the state machine
with scripted microphone blocks and scripted Realtime events.

Tests talk to the rig through the names below (`scenario`, `feed_mic`,
`setting`, ...), never to the code under test's internals directly, so the rig
can change with the code while the tests stay the same.
"""

import asyncio
import base64
import contextlib
import json
import os
import pathlib
import sys
import tempfile
import time
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "echo.py"

_tmp = tempfile.mkdtemp()
os.environ["NEXUS_HOME"] = _tmp
os.environ["OPENAI_API_KEY"] = "test"
os.environ.pop("NOTIFY_SOCKET", None)

# ---- stub modules ---------------------------------------------------------
written = {"audible": [], "reference": [], "aborted": 0, "stopped": 0}

sd = types.ModuleType("sounddevice")


class PortAudioError(Exception):
    pass


sd.PortAudioError = PortAudioError


def _query_devices(i=None):
    devs = [
        {"name": "ReSpeaker XVF3800 USB", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "HDA Intel: ALC3234 Analog", "max_input_channels": 0, "max_output_channels": 2},
    ]
    return devs if i is None else devs[i]


sd.query_devices = _query_devices


class FakeStream:
    def __init__(self, samplerate, device, channels, dtype, blocksize, callback=None):
        self.samplerate = samplerate
        self.device = device
        self.channels = channels
        self.callback = callback
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        written["stopped"] += 1

    def abort(self):
        written["aborted"] += 1

    def close(self):
        pass

    def write(self, data):
        key = "reference" if self.samplerate == 16000 else "audible"
        written[key].append(np.asarray(data).copy())
        time.sleep(len(data) / self.samplerate * 0.3)  # paced, but faster than real time


sd.InputStream = FakeStream
sd.OutputStream = FakeStream
played = []


def _play(data, rate, **kwargs):
    played.append(np.asarray(data).copy())
    time.sleep(0.01)


sd.play = _play
sys.modules["sounddevice"] = sd

for _name in (
    "torch",
    "websockets",
    "openwakeword",
    "openwakeword.model",
    "speechbrain",
    "speechbrain.inference",
    "speechbrain.inference.speaker",
):
    sys.modules[_name] = types.ModuleType(_name)

sys.modules["torch"].load = None
sys.modules["openwakeword.model"].Model = type("Model", (), {})
sys.modules["speechbrain.inference.speaker"].SpeakerRecognition = type("SR", (), {})

ns = {"__name__": "echo_under_test", "__file__": str(SRC)}
exec(compile(SRC.read_text(), str(SRC), "exec"), ns)

# ---- scripted scoring -----------------------------------------------------
script = {"wake": 0.0, "speaker": 0.6}
ns["get_wake_score"] = lambda model, block: script["wake"]
ns["speaker_similarity"] = lambda rec, emb, audio: script["speaker"]

DEFAULTS = {
    "NOTICE_TTS": "off",  # tones only: no speech synthesiser needed
    "NOTICE_MIN_GAP_SECONDS": 0.0,
    "STATUS_FILE": pathlib.Path(_tmp) / "status.json",
    "FOLLOWUP_SECONDS": 0.6,
    "ACTIVE_SPEECH_TIMEOUT_SECONDS": 1.0,
    "SOFT_VERIFY_SECONDS": 1.0,
    "WAKE_RESUME_COOLDOWN_SECONDS": 0.0,  # scenarios re-wake immediately
    "PLAYBACK_PREBUFFER_SECONDS": 0.1,
}
ns.update(DEFAULTS)

exits = []  # reasons the code under test asked for a clean restart
ns["hard_exit"] = exits.append

A = ns["AssistantState"]


@contextlib.contextmanager
def setting(**values):
    """Temporarily change tuning values of the code under test."""
    missing = object()
    old = {k: ns.get(k, missing) for k in values}
    ns.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                ns.pop(k, None)
            else:
                ns[k] = v


class FakeWS:
    def __init__(self):
        self.sent = []
        self.events = asyncio.Queue()

    async def send(self, s):
        self.sent.append(json.loads(s))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.events.get()
        if item is None:
            raise StopAsyncIteration
        return json.dumps(item)

    def types(self):
        return [m["type"] for m in self.sent]

    def push(self, event):
        self.events.put_nowait(event)


def block():
    return (np.zeros(1280, np.int16), np.zeros(1280, np.int16))


async def feed_mic(n, gap=0.01):
    for _ in range(n):
        ns["mic_queue"].put(block())
        await asyncio.sleep(gap)


def mic_callback_block():
    """Deliver one block the way PortAudio does (through the callback)."""
    ns["microphone_callback"](np.zeros((1280, 2), np.int16), 1280, None, None)


async def wait_for(pred, timeout=5.0, what=""):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timeout waiting for " + what)
        await asyncio.sleep(0.02)


def pcm24(seconds):
    return np.zeros(int(24000 * seconds), np.int16).tobytes()


def b64(seconds):
    return base64.b64encode(pcm24(seconds)).decode()


def speaking():
    return ns["assistant_speaking"].is_set()


def fire_barge():
    """What the barge-in detector does when it hears the wake phrase."""
    ns["barge_in_event"].set()


def speaker_lock_held():
    return ns["playback_lock"]().locked()


def play_beeps(count=8):
    return ns["play_beeps"](880, 0.05, 0.05, count, 0.1)


async def _scenario(body):
    ns["open_reference_output"]()
    ws = FakeWS()
    controller = ns["NexusController"](None, None, None)
    controller.reset_for_connection()
    ns["clear_mic_queue"]()
    for k in written:
        written[k] = [] if isinstance(written[k], list) else 0
    played.clear()
    exits.clear()
    script["wake"] = 0.0
    script["speaker"] = 0.6
    controller.link.attach(ws)
    proc = asyncio.create_task(ns["process_microphone"](controller.link, (None, None), controller))
    recv = asyncio.create_task(ns["receive_events"](ws, controller))
    try:
        await body(ws, controller)
    finally:
        ws.events.put_nowait(None)
        proc.cancel()
        await asyncio.gather(proc, recv, return_exceptions=True)
        controller.reset_for_connection()
        ns["close_reference_output"]()


def scenario(body, timeout=40):
    """Run one scripted conversation against a fresh controller."""
    asyncio.run(asyncio.wait_for(_scenario(body), timeout))


# ---- common steps ---------------------------------------------------------
async def wake(ws, c):
    script["wake"] = 0.0
    await feed_mic(3)
    script["wake"] = 0.5
    await feed_mic(1)
    script["wake"] = 0.0
    await wait_for(lambda: c.state in (A.ACTIVE, A.VERIFY_WAKE), what="wake detected")


async def to_wait_response(ws, c, speaker=0.6):
    script["speaker"] = speaker
    await wake(ws, c)
    await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
    ws.push({"type": "input_audio_buffer.speech_started"})
    await asyncio.sleep(0.02)
    ws.push({"type": "input_audio_buffer.speech_stopped"})
    await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
