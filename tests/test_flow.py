"""Hardware-free integration test: load echo.py with stubbed audio/ML
modules and drive the soft-verification + streaming-playback state
machine with scripted microphone blocks and Realtime events."""
import asyncio, json, os, sys, time, types, pathlib, tempfile, threading
import numpy as np

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / "echo.py"
tmp = tempfile.mkdtemp()
os.environ["NEXUS_HOME"] = tmp
os.environ["OPENAI_API_KEY"] = "test"

# ---- stub modules -------------------------------------------------------
written = {"audible": [], "reference": [], "aborted": 0, "stopped": 0}
sd = types.ModuleType("sounddevice")
class PortAudioError(Exception): pass
sd.PortAudioError = PortAudioError
def query_devices(i=None):
    devs = [
        {"name": "ReSpeaker XVF3800 USB", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "HDA Intel: ALC3234 Analog", "max_input_channels": 0, "max_output_channels": 2},
    ]
    return devs if i is None else devs[i]
sd.query_devices = query_devices
class FakeStream:
    def __init__(self, samplerate, device, channels, dtype, blocksize, callback=None):
        self.samplerate = samplerate; self.device = device; self.channels = channels
        self.callback = callback; self.started = False
    def start(self): self.started = True
    def stop(self): written["stopped"] += 1
    def abort(self): written["aborted"] += 1
    def close(self): pass
    def write(self, data):
        key = "reference" if self.samplerate == 16000 else "audible"
        written[key].append(np.asarray(data).copy())
        time.sleep(len(data) / self.samplerate * 0.3)   # paced, but faster than real time
sd.InputStream = FakeStream; sd.OutputStream = FakeStream
sd.play = lambda *a, **k: time.sleep(0.01)
sys.modules["sounddevice"] = sd
for name in ("torch", "websockets", "openwakeword", "openwakeword.model",
             "speechbrain", "speechbrain.inference", "speechbrain.inference.speaker"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].load = None
sys.modules["openwakeword.model"].Model = type("Model", (), {})
sys.modules["speechbrain.inference.speaker"].SpeakerRecognition = type("SR", (), {})

ns = {"__name__": "echo_under_test", "__file__": str(SRC)}
exec(compile(SRC.read_text(), str(SRC), "exec"), ns)

# ---- scripted scoring ---------------------------------------------------
script = {"wake": 0.0, "speaker": 0.6}
ns["get_wake_score"] = lambda model, block: script["wake"]
ns["speaker_similarity"] = lambda rec, emb, audio: script["speaker"]
ns["FOLLOWUP_SECONDS"] = 0.6
ns["ACTIVE_SPEECH_TIMEOUT_SECONDS"] = 1.0
ns["SOFT_VERIFY_SECONDS"] = 1.0
ns["PLAYBACK_PREBUFFER_SECONDS"] = 0.1
assert ns["SPEAKER_VERIFY_MODE"] == "soft" and ns["STREAMING_PLAYBACK"]

class FakeWS:
    def __init__(self): self.sent = []; self.events = asyncio.Queue()
    async def send(self, s): self.sent.append(json.loads(s))
    def __aiter__(self): return self
    async def __anext__(self):
        item = await self.events.get()
        if item is None: raise StopAsyncIteration
        return json.dumps(item)
    def types(self): return [m["type"] for m in self.sent]

def block():
    return (np.zeros(1280, np.int16), np.zeros(1280, np.int16))

async def feed_mic(n, gap=0.01):
    for _ in range(n):
        ns["mic_queue"].put(block()); await asyncio.sleep(gap)

async def wait_for(pred, timeout=5.0, what=""):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout: raise AssertionError("timeout waiting for " + what)
        await asyncio.sleep(0.02)

def pcm24(seconds): return (np.zeros(int(24000 * seconds), np.int16)).tobytes()

async def scenario(name, body):
    ns["open_reference_output"]()
    ws = FakeWS()
    controller = ns["NexusController"](None, None, None)
    controller.reset_for_connection()
    ns["clear_mic_queue"](); ns["clear_barge_queue"]()
    for k in written: written[k] = [] if isinstance(written[k], list) else 0
    proc = asyncio.create_task(ns["process_microphone"](ws, (None, None), controller))
    recv = asyncio.create_task(ns["receive_events"](ws, controller))
    try:
        await body(ws, controller)
    finally:
        ws.events.put_nowait(None)
        proc.cancel(); 
        await asyncio.gather(proc, recv, return_exceptions=True)
        ns["close_reference_output"]()
    print(f"{name}: ok")

A = ns["AssistantState"]

async def happy(ws, c):
    script["wake"] = 0.0; script["speaker"] = 0.6
    await feed_mic(5)                       # pre-roll
    script["wake"] = 0.5
    await feed_mic(1); script["wake"] = 0.0
    await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE after wake")
    await feed_mic(3)
    assert ws.types()[:2] == ["input_audio_buffer.clear", "input_audio_buffer.append"], ws.types()[:3]
    assert c.verification_pending
    ws.events.put_nowait({"type": "input_audio_buffer.speech_started"})
    await asyncio.sleep(0.05)
    ws.events.put_nowait({"type": "input_audio_buffer.speech_stopped"})
    await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
    assert not c.verification_pending and "response.create" in ws.types()
    ws.events.put_nowait({"type": "response.created", "response": {"id": "r1"}})
    for _ in range(3):
        ws.events.put_nowait({"type": "response.output_audio.delta", "response_id": "r1",
                              "delta": __import__("base64").b64encode(pcm24(0.1)).decode()})
    ws.events.put_nowait({"type": "response.done", "response": {"id": "r1", "status": "completed", "output": []}})
    await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up ACTIVE")
    total48 = sum(len(a) for a in written["audible"])
    assert abs(total48 - 14400) <= 48, total48
    assert written["aborted"] == 0 and written["stopped"] >= 1
    assert c.pending_input_clear and not ns["assistant_speaking"].is_set()
    # follow-up with no speech closes after FOLLOWUP_SECONDS
    await feed_mic(60)
    await wait_for(lambda: c.state == A.WAIT_WAKE, what="back to wake")

async def barge(ws, c):
    script["wake"] = 0.0; script["speaker"] = 0.6
    await feed_mic(3); script["wake"] = 0.5; await feed_mic(1); script["wake"] = 0.0
    await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
    ws.events.put_nowait({"type": "input_audio_buffer.speech_started"}); await asyncio.sleep(0.02)
    ws.events.put_nowait({"type": "input_audio_buffer.speech_stopped"})
    await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
    ws.events.put_nowait({"type": "response.created", "response": {"id": "r2"}})
    b64 = __import__("base64").b64encode(pcm24(0.5)).decode()
    for _ in range(4):  # 2 s of audio, response still generating (no response.done yet)
        ws.events.put_nowait({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64})
    await wait_for(lambda: ns["assistant_speaking"].is_set(), what="speaking")
    await asyncio.sleep(0.15)
    ns["barge_in_event"].set()          # what monitor_barge_in does on "Hey Nexus"
    await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "interrupt", what="interrupt ACTIVE")
    assert "response.cancel" in ws.types(), ws.types()
    assert c.cancelled_response_id == "r2" and written["aborted"] >= 1
    total48 = sum(len(a) for a in written["audible"])
    assert total48 < 96000, total48           # nowhere near the full 2 s
    # late deltas and the cancelled response.done are ignored
    ws.events.put_nowait({"type": "response.output_audio.delta", "response_id": "r2", "delta": b64})
    ws.events.put_nowait({"type": "response.done", "response": {"id": "r2", "status": "cancelled", "output": []}})
    await asyncio.sleep(0.1)
    assert c.player is None and c.cancelled_response_id is None and c.state == A.ACTIVE

async def rejected(ws, c):
    script["wake"] = 0.0; script["speaker"] = 0.1
    await feed_mic(3); script["wake"] = 0.5; await feed_mic(1); script["wake"] = 0.0
    await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
    ws.events.put_nowait({"type": "input_audio_buffer.speech_started"}); await asyncio.sleep(0.02)
    ws.events.put_nowait({"type": "input_audio_buffer.speech_stopped"})
    await wait_for(lambda: c.state == A.WAIT_WAKE, what="rejected -> wake")
    assert "response.create" not in ws.types()
    assert ws.types().count("input_audio_buffer.clear") >= 1
    assert c.wake_cooldown_until > time.monotonic()

async def rejected_midstream(ws, c):
    # Speaker judged at SOFT_VERIFY_SECONDS of audio before speech stops.
    script["wake"] = 0.0; script["speaker"] = 0.1
    await feed_mic(3); script["wake"] = 0.5; await feed_mic(1); script["wake"] = 0.0
    await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE")
    await feed_mic(16)   # ~1.3 s with pre-roll
    await wait_for(lambda: c.state == A.WAIT_WAKE, what="mid-stream rejection")

async def strict_buffered(ws, c):
    # Original behaviour: hold audio until 1.2 s is verified; buffered playback.
    ns["SPEAKER_VERIFY_MODE"] = "strict"; ns["STREAMING_PLAYBACK"] = False
    try:
        script["wake"] = 0.0; script["speaker"] = 0.6
        await feed_mic(3); script["wake"] = 0.5; await feed_mic(1); script["wake"] = 0.0
        await wait_for(lambda: c.state == A.VERIFY_WAKE, what="VERIFY_WAKE")
        assert ws.sent == []                      # nothing leaves the machine yet
        await feed_mic(16)                        # 1.28 s
        await wait_for(lambda: c.state == A.ACTIVE, what="ACTIVE after strict verify")
        assert ws.types()[:2] == ["input_audio_buffer.clear", "input_audio_buffer.append"]
        ws.events.put_nowait({"type": "input_audio_buffer.speech_started"}); await asyncio.sleep(0.02)
        ws.events.put_nowait({"type": "input_audio_buffer.speech_stopped"})
        await wait_for(lambda: c.state == A.WAIT_RESPONSE, what="WAIT_RESPONSE")
        ws.events.put_nowait({"type": "response.created", "response": {"id": "r3"}})
        b64 = __import__("base64").b64encode(pcm24(0.1)).decode()
        for _ in range(3):
            ws.events.put_nowait({"type": "response.output_audio.delta", "response_id": "r3", "delta": b64})
        assert written["audible"] == []           # buffered: nothing played until response.done
        ws.events.put_nowait({"type": "response.done", "response": {"id": "r3", "status": "completed", "output": []}})
        await wait_for(lambda: c.state == A.FOLLOWUP, what="strict FOLLOWUP state")
        total48 = sum(len(a) for a in written["audible"]); assert abs(total48 - 14400) <= 48, total48
        await feed_mic(30)                        # 2.4 s verified follow-up speech
        await wait_for(lambda: c.state == A.ACTIVE and c.active_origin == "followup", what="follow-up ACTIVE")
    finally:
        ns["SPEAKER_VERIFY_MODE"] = "soft"; ns["STREAMING_PLAYBACK"] = True

async def main():
    # StreamResampler: chunked == whole-signal, correct rate, no boundary clicks
    R = ns["StreamResampler"]
    t = np.arange(24000) / 24000
    sig = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    for new_rate in (48000, 16000):
        r = R(24000, new_rate)
        chunked = np.concatenate([r.process(sig[i:i + 960]) for i in range(0, len(sig), 960)])
        whole = R(24000, new_rate).process(sig)
        assert abs(len(chunked) - new_rate) <= 2 and np.array_equal(chunked, whole), (new_rate, len(chunked))
        tail = chunked[2000:]
        assert 0.97 < np.abs(tail.astype(float)).max() / 12000 < 1.03
        step = np.abs(np.diff(tail.astype(float))).max()
        assert step < 12000 * 2 * np.pi * 440 / new_rate * 1.2, step   # no clicks
    print("StreamResampler: ok")

    # ReferenceOutput callback: drains queue, pads with silence
    RO = ns["ReferenceOutput"]; ro = RO.__new__(RO)
    ro._lock = threading.Lock(); ro._chunks = __import__("collections").deque(); ro._offset = 0
    ro.write(np.arange(1, 1001, dtype=np.int16)); ro.write(np.arange(1001, 1201, dtype=np.int16))
    out = np.zeros((800, 2), np.int16); ro._callback(out, 800, None, None)
    assert out[0, 0] == 1 and out[799, 1] == 800
    out = np.zeros((800, 2), np.int16); ro._callback(out, 800, None, None)
    assert out[0, 0] == 801 and out[399, 0] == 1200 and out[400:].sum() == 0
    print("ReferenceOutput: ok")

    await scenario("soft wake -> stream -> verify -> streaming playback -> follow-up -> wake", happy)
    await scenario("barge-in cancels server response and resumes listening", barge)
    await scenario("rejected speaker at speech end", rejected)
    await scenario("rejected speaker mid-stream", rejected_midstream)
    await scenario("strict verification with buffered playback (legacy mode)", strict_buffered)
    print("ALL FLOW TESTS OK")

asyncio.run(main())
