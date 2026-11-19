"""The engine: the conversation state machine.

It listens for the wake phrase, checks the speaker, streams the request to
the voice service, plays the reply, lets the wake phrase interrupt it, runs
tools, and keeps a follow-up window open. It knows nothing about any
particular microphone, wake-word library or voice service: those are parts
(nexus/parts.py), chosen by settings and held in the four names below.

    AUDIO     microphone and speaker     nexus.parts.AudioIO
    WAKE      wake-word detector         nexus.parts.WakeEngine
    VERIFIER  speaker check              nexus.parts.SpeakerVerifier
    BACKEND   voice service              nexus.parts.VoiceBackend

One asyncio loop runs everything. The microphone loop lives for the whole
process, with or without a connection to the voice service.
"""

import asyncio
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from . import alerts, config, parts, tools
from .alerts import (
    ACCOUNT_PROBLEMS,
    PROBLEM_LABELS,
    classify_failure,
    hard_exit,
    play_notice,
    sd_notify,
    send_alert,
    set_status,
    status_heartbeat,
)
from .config import CLIP_RATE, INPUT_BLOCK_FRAMES, INPUT_RATE
from .dsp import blocks_have_seconds, blocks_seconds, select_channel_blocks

env_str, env_float, env_int, env_flag = config.group("core")

# The parts in use. Set by run(); tests put fakes here.
AUDIO = None
WAKE = None
VERIFIER = None
BACKEND = None

WAKE_PHRASE = env_str(
    "NEXUS_WAKE_PHRASE",
    "Hey Nexus",
    "The wake phrase as words, for log lines and the prompt (the wake model decides what is heard).",
)
PERSONA = env_str(
    "NEXUS_PERSONA",
    "plain",
    "Personality: plain, sardonic, or any name with a file in nexus/personas/.",
)
INSTRUCTIONS_FILE = env_str(
    "NEXUS_INSTRUCTIONS_FILE",
    "",
    "A text file with your own personality prompt; replaces NEXUS_PERSONA.",
)

WAKE_THRESHOLD = env_float(
    "NEXUS_WAKE_THRESHOLD",
    0.10,
    "Wake-word score (0 to 1) at which Nexus wakes.",
)

SPEAKER_THRESHOLD = env_float(
    "NEXUS_SPEAKER_THRESHOLD",
    0.25,
    "Voice-match score below which the speaker is rejected.",
)

WAKE_PREROLL_SECONDS = env_float(
    "NEXUS_WAKE_PREROLL_SECONDS", 1.0, "Audio kept from before the wake word fired."
)

SPEAKER_VERIFY_SECONDS = env_float(
    "NEXUS_SPEAKER_VERIFY_SECONDS", 1.6, "Strict mode: audio gathered before the speaker is judged."
)

WAKE_REJECT_COOLDOWN_SECONDS = env_float(
    "NEXUS_WAKE_REJECT_COOLDOWN_SECONDS", 1.0, "Wake word ignored for this long after a rejected speaker."
)

# The wake models are not fed while a request is active, so their
# feature buffers still end with the last wake phrase when listening
# resumes. Keep scoring (to flush them) but ignore detections for this
# long after every return to wake, or the old phrase fires again.
WAKE_ALWAYS_SCORE = env_flag(
    "NEXUS_WAKE_ALWAYS_SCORE",
    True,
    "Feed the wake models every microphone block in every state (0 = only while waiting or speaking).",
)

# With WAKE_ALWAYS_SCORE the models hear the whole exchange, so the old
# phrase has left their memory by the time listening resumes and only a
# short guard is needed. Without it they still end with the last wake
# phrase, and detections are ignored for 2 s after every return to wake.
WAKE_RESUME_COOLDOWN_SECONDS = env_float(
    "NEXUS_WAKE_RESUME_COOLDOWN_SECONDS",
    0.3 if WAKE_ALWAYS_SCORE else 2.0,
    "Wake word ignored for this long after each exchange.",
)

# After a wake word is acted on, the next one is accepted only once the
# score has stayed below this fraction of the threshold for this many
# blocks (80 ms each). One spoken phrase can therefore never fire twice.
WAKE_REARM_FRACTION = 0.5

WAKE_REARM_BLOCKS = 3

FOLLOWUP_SECONDS = env_float(
    "NEXUS_FOLLOWUP_SECONDS", 5.0, "How long the microphone stays open for a follow-up after a reply."
)

FOLLOWUP_GUARD_SECONDS = env_float(
    "NEXUS_FOLLOWUP_GUARD_SECONDS",
    0.35,
    "Microphone ignored for this long after a reply ends, so the tail of Nexus's own voice is not heard.",
)

FOLLOWUP_KEEP_OPEN = env_flag(
    "NEXUS_FOLLOWUP_KEEP_OPEN",
    True,
    "Keep the follow-up window open after a short or rejected sound (0 = one sound closes it).",
)

FOLLOWUP_MIN_SPEECH_SECONDS = env_float(
    "NEXUS_FOLLOWUP_MIN_SPEECH_SECONDS",
    0.4,
    "Sounds shorter than this in the follow-up window are ignored.",
)

FOLLOWUP_VERIFY_SECONDS = 2.2

FOLLOWUP_OVERLAP_SECONDS = 0.6

# After speaker verification, microphone audio streams to OpenAI
# until server VAD reports speech. If no speech is ever detected,
# stop streaming and return to wake instead of uploading indefinitely.
ACTIVE_SPEECH_TIMEOUT_SECONDS = env_float(
    "NEXUS_ACTIVE_SPEECH_TIMEOUT_SECONDS", 10.0, "Give up on a request if no speech starts within this."
)

BARGE_WAKE_THRESHOLD = env_float(
    "NEXUS_BARGE_WAKE_THRESHOLD",
    0.30,
    "Wake-word score needed to interrupt Nexus while it is speaking.",
)

# Which microphone channel may trigger an interruption: ch0 (the
# echo-cancelled beam, the default), ch1, or best (the higher score).
BARGE_CHANNEL = env_str(
    "NEXUS_BARGE_CHANNEL", "ch0", "Channel that can interrupt playback: ch0, ch1 or best."
).lower()

if BARGE_CHANNEL not in ("ch0", "ch1", "best"):
    print(f"WARNING: unknown NEXUS_BARGE_CHANNEL {BARGE_CHANNEL!r}; using 'ch0'.")
    BARGE_CHANNEL = "ch0"

# Per-block score logging is useful while tuning thresholds but
# floods journald in normal operation. Enable with NEXUS_BARGE_DEBUG=1
# and NEXUS_WAKE_DEBUG=1.
BARGE_DEBUG = env_flag("NEXUS_BARGE_DEBUG", False, "Print barge-in peak scores twice a second.")

WAKE_DEBUG = env_flag("NEXUS_WAKE_DEBUG", False, "Print wake scores above 0.02.")

BARGE_DEBUG_INTERVAL_SECONDS = 0.50

STREAMING_PLAYBACK = env_flag(
    "NEXUS_STREAMING_PLAYBACK",
    True,
    "Start speaking as soon as audio arrives (0 = wait for the whole reply).",
)

PLAYBACK_PREBUFFER_SECONDS = env_float(
    "NEXUS_PLAYBACK_PREBUFFER_SECONDS", 0.3, "Audio buffered before the speaker starts."
)

SPEAKER_VERIFY_MODE = env_str(
    "NEXUS_SPEAKER_VERIFY_MODE", "soft", "Voice check: soft, strict or off."
).lower()

if SPEAKER_VERIFY_MODE not in ("soft", "strict", "off"):
    print(f"WARNING: unknown NEXUS_SPEAKER_VERIFY_MODE {SPEAKER_VERIFY_MODE!r}; using 'soft'.")
    SPEAKER_VERIFY_MODE = "soft"

# Soft mode judges the speaker on up to this much audio (wake-word
# pre-roll included), or on whatever has arrived when speech stops.
SOFT_VERIFY_SECONDS = env_float(
    "NEXUS_SOFT_VERIFY_SECONDS", 2.5, "Soft mode: most audio used to judge the speaker."
)

# The timed soft check waits until the server has reported speech and at
# least this much of it has arrived, so a pause after the wake phrase is
# not judged as if it were the request.
SOFT_VERIFY_SPEECH_SECONDS = env_float(
    "NEXUS_SOFT_VERIFY_SPEECH_SECONDS", 1.0, "Soft mode: speech needed before a timed check."
)

# Microphone audio kept while waiting for a reply so that speech
# captured right after a barge-in is not lost.
WAIT_BLOCKS_MAX = max(
    1,
    int(1.6 * INPUT_RATE / INPUT_BLOCK_FRAMES),
)

RECENT_BLOCKS_MAX = max(
    1,
    int(0.8 * INPUT_RATE / INPUT_BLOCK_FRAMES),
)

RECONNECT_INITIAL_SECONDS = 1

RECONNECT_MAX_SECONDS = env_int(
    "NEXUS_RECONNECT_MAX_SECONDS", 15, "Longest wait between reconnect attempts after a network failure."
)

REPLY_TIMEOUT_SECONDS = env_float(
    "NEXUS_REPLY_TIMEOUT_SECONDS",
    15,
    "Give up on a reply if no audio has arrived this long after asking (0 = wait for ever).",
)

REQUEST_MAX_SECONDS = env_float(
    "NEXUS_REQUEST_MAX_SECONDS",
    30,
    "Longest one spoken request may last before it is dropped (0 = no limit).",
)

PLAYER_STALL_SECONDS = env_float(
    "NEXUS_PLAYER_STALL_SECONDS",
    10,
    "Stop a reply's playback after this long with no audio arriving (0 = no limit).",
)

SPEAKER_FAIL_LIMIT = env_int(
    "NEXUS_SPEAKER_FAIL_LIMIT",
    3,
    "Failed replies in a row before the speaker is reported broken and Nexus restarts (0 = never).",
)

MIC_STALL_SECONDS = env_float(
    "NEXUS_MIC_STALL_SECONDS",
    10,
    "Restart if the microphone delivers no audio for this long (0 = never).",
)

ACCOUNT_RETRY_SECONDS = env_float(
    "NEXUS_ACCOUNT_RETRY_SECONDS",
    600,
    "Wait between attempts when the account, key or model is the problem.",
)

NETWORK_PROBLEM_AFTER_FAILURES = env_int(
    "NEXUS_NETWORK_PROBLEM_AFTER_FAILURES",
    3,
    "Failed connection attempts in a row before a network problem is reported.",
)

STATUS_OK_AFTER_SECONDS = env_float(
    "NEXUS_STATUS_OK_AFTER_SECONDS", 5, "A connection counts as healthy once it has lasted this long."
)

WATCHDOG_PING_SECONDS = 10

BEEP_HOLD_MAX_SECONDS = env_float(
    "NEXUS_BEEP_HOLD_MAX_SECONDS",
    60,
    "Beeps wait up to this long for a request in progress to end (0 = ring at once).",
)

MISSED_NOTES = env_flag(
    "NEXUS_MISSED_ALARM_NOTES",
    True,
    "Have Nexus mention, with its next reply, an alarm or timer missed while it was off.",
)

REQUEST_LOG = env_flag(
    "NEXUS_REQUEST_LOG", True, "Print one REQUEST line with scores, timings and usage per request."
)


# =========================================================
# SHARED STATE
# =========================================================

mic_queue = queue.Queue(maxsize=250)

assistant_speaking = threading.Event()

barge_in_event = threading.Event()

wake_preroll = deque(
    maxlen=max(
        1,
        int(WAKE_PREROLL_SECONDS * INPUT_RATE / INPUT_BLOCK_FRAMES),
    )
)

# Serialises every use of the speaker: response playback, timer
# beeps, and alarm beeps. Without it an alarm firing mid-response
# opened a second stream on the same device and cleared
# assistant_speaking while Nexus was still talking.
_playback_lock = None


def playback_lock():
    global _playback_lock

    if _playback_lock is None:
        _playback_lock = asyncio.Lock()

    return _playback_lock


async def queue_get(target_queue):
    # A worker thread blocked on an unbounded queue.get() outlives the
    # asyncio task that was cancelled around it, then swallows the
    # next block. Polling with a short timeout bounds that leak.
    while True:
        try:
            return await asyncio.to_thread(
                target_queue.get,
                True,
                0.5,
            )
        except queue.Empty:
            continue


def clear_mic_queue():
    while True:
        try:
            mic_queue.get_nowait()

        except queue.Empty:
            break


async def microphone_watchdog():
    """Exit when the microphone stream has died.

    A USB microphone that drops off the bus and comes back leaves a dead
    stream behind: the service looks healthy and hears nothing. A fresh
    process reopens the device.
    """
    if MIC_STALL_SECONDS <= 0:
        await asyncio.Event().wait()  # switched off: idle for ever

    mic_health["last_block"] = time.monotonic()

    while True:
        await asyncio.sleep(min(1.0, MIC_STALL_SECONDS / 2))

        silent_for = time.monotonic() - mic_health["last_block"]

        if silent_for >= MIC_STALL_SECONDS:
            set_status(
                "problem",
                "microphone",
                f"no audio from the microphone for {silent_for:.0f} s",
            )
            hard_exit("the microphone stream stopped")
            return


# =========================================================
# MICROPHONE
# =========================================================

# Time of the last microphone block, and of the last watchdog ping.
mic_health = {"last_block": time.monotonic(), "last_watchdog_ping": 0.0}


def on_microphone_block(ch0, ch1):
    """Called by the audio part for every 80 ms of microphone audio, from its own thread.

    One queue and one consumer for every state. The consumer scores each
    block for the wake phrase and decides from the state whether a hit
    means wake, barge-in or nothing.
    """
    mic_health["last_block"] = time.monotonic()

    stereo_block = (ch0, ch1)

    try:
        mic_queue.put_nowait(stereo_block)

    except queue.Full:
        try:
            mic_queue.get_nowait()

        except queue.Empty:
            pass

        try:
            mic_queue.put_nowait(stereo_block)

        except queue.Full:
            pass


# =========================================================
# BEEPS AND LOCAL SOUNDS
# =========================================================


async def wait_for_turn_to_end():
    """Beeps must not take the microphone away from a request in progress."""
    controller = active_controller

    if controller is None or BEEP_HOLD_MAX_SECONDS <= 0:
        return

    deadline = time.monotonic() + BEEP_HOLD_MAX_SECONDS

    while controller.state != AssistantState.WAIT_WAKE and time.monotonic() < deadline:
        await asyncio.sleep(0.1)


async def play_beeps(
    frequency,
    beep_seconds,
    gap_seconds,
    count,
    volume,
    ring_seconds=0.0,
):
    """Ring until the wake phrase is heard. Returns True if it was silenced that way.

    With ring_seconds the ring lasts that long at most; otherwise it is
    `count` beeps. Plugins reach this through nexus.ring().
    """
    t = np.arange(int(CLIP_RATE * beep_seconds), dtype=np.float32) / CLIP_RATE

    tone = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)

    await wait_for_turn_to_end()

    silenced = False

    async with playback_lock():
        # Hold assistant_speaking for the whole ring so the microphone
        # loop treats the wake phrase as "be quiet".
        barge_in_event.clear()
        assistant_speaking.set()

        started = time.monotonic()
        played = 0

        try:
            while True:
                if barge_in_event.is_set():
                    print("Beeps silenced by wake phrase.")
                    silenced = True
                    break

                if ring_seconds > 0:
                    if time.monotonic() - started >= ring_seconds:
                        break
                elif played >= count:
                    break

                await asyncio.to_thread(AUDIO.play_clip, tone, CLIP_RATE)

                played += 1

                await asyncio.sleep(gap_seconds)

        finally:
            assistant_speaking.clear()
            barge_in_event.clear()
            clear_mic_queue()

    return silenced


async def play_local_audio(pcm):
    """Play locally generated audio (a notice) through the speaker. `pcm` is at CLIP_RATE."""
    async with playback_lock():
        barge_in_event.clear()
        assistant_speaking.set()

        try:
            await asyncio.to_thread(AUDIO.play_clip, pcm, CLIP_RATE)
        finally:
            assistant_speaking.clear()
            barge_in_event.clear()
            clear_mic_queue()


# Notices (nexus/alerts.py) are played through the speaker in use.
alerts.play_audio = play_local_audio


# =========================================================
# MISSED ALARMS AND TIMERS
# =========================================================

# What came due while Nexus was off and was too late to ring. The owner
# is told by push message at once, and by Nexus itself with the next
# reply.
missed_events = []


def note_missed(description):
    missed_events.append(description)

    send_alert("Nexus missed an alarm", f"While Nexus was off it missed {description}.")


async def send_missed_note(session):
    """Before a reply is asked for: tell the model what was missed so it says so first."""
    if not missed_events or not MISSED_NOTES:
        return

    text = (
        "System note: while you were switched off you missed "
        + "; ".join(missed_events)
        + ". Tell the user this first, briefly, then answer their request."
    )

    missed_events.clear()

    await session.add_system_note(text)


class _PluginHost(tools.Host):
    """What plugins can ask the running assistant to do."""

    async def ring(self, frequency, beep_seconds, gap_seconds, count, volume, ring_seconds=0.0):
        return await play_beeps(
            frequency, beep_seconds, gap_seconds, count, volume, ring_seconds=ring_seconds
        )

    def alert(self, title, body):
        send_alert(title, body)

    def note_missed(self, description):
        note_missed(description)


tools.host = _PluginHost()


# =========================================================
# THE CONNECTION, WHICH MAY BE DOWN
# =========================================================


class Link:
    """The current session with the voice service, or none.

    The microphone loop lives for the whole process and talks to the
    service through this object, so it keeps listening for the wake
    phrase while the connection is down. Every call raises
    ConnectionError when there is no session.
    """

    def __init__(self):
        self.session = None
        self.failed = None
        self.failure = None

        # Why there is no connection: None, "starting", or a
        # classify_failure() kind.
        self.problem = "starting"
        self.failures = 0

        # Set from an error event so the close that follows is classified
        # correctly even when its own reason is generic.
        self.last_error_kind = None

    @property
    def online(self):
        return self.session is not None

    @property
    def output_rate(self):
        return getattr(self.session, "output_rate", 24000)

    def attach(self, session):
        self.failed = asyncio.Event()
        self.failure = None
        self.session = session

    def detach(self):
        self.session = None

    def fail(self, exc):
        """Called by the microphone loop when a send fails."""
        if self.failure is None:
            self.failure = exc

        if self.failed is not None:
            self.failed.set()

    def _session(self):
        session = self.session

        if session is None:
            raise ConnectionError("not connected to the voice service")

        return session

    async def send_audio(self, pcm):
        await self._session().send_audio(pcm)

    async def clear_input(self):
        await self._session().clear_input()

    async def request_reply(self):
        await self._session().request_reply()

    async def cancel_reply(self):
        await self._session().cancel_reply()

    async def send_tool_result(self, call_id, result):
        await self._session().send_tool_result(call_id, result)

    async def add_system_note(self, text):
        await self._session().add_system_note(text)


async def send_audio(session, pcm):
    if len(pcm) == 0:
        return

    await session.send_audio(pcm)


async def send_blocks(session, blocks):
    if not blocks:
        return

    await session.send_audio(np.concatenate(blocks))


# =========================================================
# PLAYBACK
# =========================================================


class ResponsePlayer:
    """Plays one reply through the speaker as it arrives.

    receive_events() feeds audio with feed() and calls finish() when the
    reply is complete. A worker thread hands 50 ms chunks to the audio
    part and stops within one chunk of a barge-in.
    """

    def __init__(self, response_id, controller):
        self.response_id = response_id
        self.controller = controller
        self.rate = controller.link.output_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.interrupted = False
        self.underruns = 0
        self.task = None

        # False for a spoken lead-in before a tool call: the real answer
        # is still to come, so no follow-up window opens after it.
        self.followup = True

    def feed(self, pcm_bytes):
        if pcm_bytes:
            self.queue.put(np.frombuffer(pcm_bytes, dtype=np.int16))

    def finish(self):
        self.queue.put(None)

    def abort(self):
        self.interrupted = True
        self.stop_event.set()
        self.queue.put(None)

    def _should_stop(self):
        return self.stop_event.is_set() or barge_in_event.is_set()

    async def run(self):
        async with playback_lock():
            barge_in_event.clear()
            clear_mic_queue()
            self.controller.wait_blocks.clear()
            assistant_speaking.set()

            try:
                await asyncio.to_thread(self._play)

            finally:
                assistant_speaking.clear()

                interrupted = self.interrupted or barge_in_event.is_set()

                # After a barge-in, keep what the user said next.
                if not interrupted:
                    clear_mic_queue()

        interrupted = self.interrupted or barge_in_event.is_set()

        barge_in_event.clear()

        if self.underruns:
            print(
                f"Playback underruns: {self.underruns}",
                flush=True,
            )

        return interrupted

    def _play(self):
        chunk_frames = int(self.rate * 0.05)
        prebuffer_frames = int(self.rate * PLAYBACK_PREBUFFER_SECONDS)

        pending = np.zeros(0, dtype=np.int16)
        done = False
        starved_since = None

        stream = None
        stage = "opening the speaker"

        try:
            stream = AUDIO.open_reply(self.rate)

            # Pre-buffer a little so a slow start to generation does
            # not underrun the device on the first words.
            while len(pending) < prebuffer_frames and not done and not self._should_stop():
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if item is None:
                    done = True
                else:
                    pending = np.concatenate([pending, item])

            if self._should_stop():
                self.interrupted = True
                return

            stage = "starting the speaker"
            stream.start()

            while not self._should_stop():
                # Top up from the network while less than one chunk
                # is ready. Give up quickly so the device stays fed.
                while len(pending) < chunk_frames and not done:
                    try:
                        item = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        break

                    if item is None:
                        done = True
                    else:
                        pending = np.concatenate([pending, item])

                if len(pending) == 0:
                    if done:
                        break

                    # Generation has fallen behind real time. Keep the
                    # device fed with silence, but not for ever: a reply
                    # whose end never arrives must let go of the speaker.
                    now = time.monotonic()

                    if starved_since is None:
                        starved_since = now
                    elif PLAYER_STALL_SECONDS > 0 and now - starved_since >= PLAYER_STALL_SECONDS:
                        print(f"No audio for {PLAYER_STALL_SECONDS:g} s; ending playback.", flush=True)
                        break

                    self.underruns += 1
                    chunk = np.zeros(chunk_frames, dtype=np.int16)
                else:
                    starved_since = None
                    n = min(chunk_frames, len(pending))
                    chunk = pending[:n]
                    pending = pending[n:]

                stage = "playing the reply"
                stream.write(chunk)

            if self._should_stop():
                self.interrupted = True

        except Exception as exc:
            print(
                f"Audio playback failure while {stage}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        finally:
            if stream is not None:
                stream.close(abort=self.interrupted)


class WakeGate:
    """Stops one spoken wake phrase from firing twice.

    A detection disarms the gate. It re-arms once the wake score has
    stayed low for a few blocks, which means the phrase has left the
    model's memory.
    """

    def __init__(self):
        self.armed = True
        self._quiet_blocks = 0

    def disarm(self):
        self.armed = False
        self._quiet_blocks = 0

    def observe(self, best_score):
        if self.armed:
            return

        if best_score < min(WAKE_THRESHOLD, BARGE_WAKE_THRESHOLD) * WAKE_REARM_FRACTION:
            self._quiet_blocks += 1

            if self._quiet_blocks >= WAKE_REARM_BLOCKS:
                self.armed = True
        else:
            self._quiet_blocks = 0


def handle_block_while_speaking(
    controller,
    block,
    score0,
    score1,
):
    """Barge-in: called for every microphone block while Nexus is audible."""
    controller.barge_peak0 = max(controller.barge_peak0, score0)
    controller.barge_peak1 = max(controller.barge_peak1, score1)
    controller.barge_debug_peak0 = max(controller.barge_debug_peak0, score0)
    controller.barge_debug_peak1 = max(controller.barge_debug_peak1, score1)

    now = time.monotonic()

    if BARGE_DEBUG and now - controller.barge_debug_printed >= BARGE_DEBUG_INTERVAL_SECONDS:
        print(
            f"BARGE DEBUG peak CH0(AEC)={controller.barge_debug_peak0:.3f} "
            f"CH1(residual)={controller.barge_debug_peak1:.3f}",
            flush=True,
        )

        controller.barge_debug_peak0 = 0.0
        controller.barge_debug_peak1 = 0.0
        controller.barge_debug_printed = now

    if barge_in_event.is_set():
        # Already interrupted; playback stops within one chunk. Keep
        # what is said meanwhile so the new request loses nothing.
        controller.wait_blocks.append(block)
        return

    barge_score = {"ch0": score0, "ch1": score1, "best": max(score0, score1)}[BARGE_CHANNEL]

    if controller.wake_gate.armed and barge_score >= BARGE_WAKE_THRESHOLD:
        print()
        print(f"*** {WAKE_PHRASE.upper()} INTERRUPT DETECTED ***")

        print(f"Interrupt wake CH0(AEC): {score0:.3f}   CH1(residual): {score1:.3f}")

        controller.wake_gate.disarm()

        barge_in_event.set()


async def open_followup(
    session,
    controller,
    interrupted,
):
    controller.start_followup(interrupted=interrupted)

    print()

    if interrupted:
        print("Playback stopped.")
        print(f"Listening for your new request for {FOLLOWUP_SECONDS:.0f}s.")
    else:
        print(f"Follow-up window open for {FOLLOWUP_SECONDS:.0f}s.")
        print("Listening briefly for a direct follow-up.")


async def run_player(
    session,
    controller,
    player,
):
    try:
        interrupted = await player.run()

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        # Local audio failures must not discard a healthy Realtime
        # session or trigger network reconnection.
        if controller.player is player:
            controller.player = None

        await note_playback_failure(session, controller, player.response_id, exc)
        return

    finally:
        if controller.player is player:
            controller.player = None

    if interrupted and controller.response_active:
        if controller.current_response_id == player.response_id and player.followup:
            # The server is still generating this reply. Stop it so no
            # more audio arrives for a response nobody is listening to.
            controller.cancelled_response_id = player.response_id

            try:
                await session.cancel_reply()
            except Exception:
                pass

        elif not player.followup:
            # A lead-in was interrupted; the answer that follows it must
            # not start playing over the new request.
            if controller.current_response_id != player.response_id:
                controller.cancelled_response_id = controller.current_response_id

                try:
                    await session.cancel_reply()
                except Exception:
                    pass
            else:
                controller.cancel_next_response = True

        # The cancel is on its way; a new request must not have to wait
        # for the server to confirm it.
        controller.response_active = False

    controller.playback_failures = 0

    if not player.followup and not interrupted:
        # Lead-in finished; keep waiting for the answer.
        return

    await open_followup(
        session,
        controller,
        interrupted,
    )


class AssistantState:
    WAIT_WAKE = "wait_wake"
    VERIFY_WAKE = "verify_wake"
    ACTIVE = "active"
    WAIT_RESPONSE = "wait_response"
    FOLLOWUP = "followup"


# The controller of the running assistant, for code (beeps, tools)
# that needs to know whether a request is in progress.
active_controller = None


def add_usage(total, usage):
    """Add the numbers in one reply's usage report into `total`, keeping its shape."""
    for key, value in (usage or {}).items():
        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            add_usage(total.setdefault(key, {}), value)

    return total


class NexusController:
    def __init__(self):
        global active_controller
        active_controller = self

        self.state = AssistantState.WAIT_WAKE

        # Stereo blocks awaiting speaker verification.
        self.verify_blocks = []

        # Strict mode: stereo blocks gathered during the follow-up
        # window before verification.
        self.followup_blocks = []

        # Mono blocks to send to OpenAI before live audio resumes
        # (wake-word pre-roll, or speech captured after a barge-in).
        self.pending_send_blocks = []

        # Clear the server's input buffer before the next send.
        self.pending_input_clear = False

        # Microphone audio captured while a reply is pending or
        # just after it was interrupted.
        self.wait_blocks = deque(maxlen=WAIT_BLOCKS_MAX)

        # Rolling window of recent ACTIVE audio, used to back-fill
        # the verification buffer when the server reports speech.
        self.recent_blocks = deque(maxlen=RECENT_BLOCKS_MAX)

        self.active_channel = 0
        self.active_origin = "wake"
        self.verification_pending = False
        self.followup_deadline = 0.0
        self.active_deadline = 0.0
        self.wake_cooldown_until = 0.0
        self.response_active = False
        self.speech_started_seen = False
        self.speech_started_at = 0.0
        self.speech_start_block_index = None

        # When the reply being waited for is given up on (0 = no limit running).
        self.reply_deadline = 0.0

        # Facts about the request in progress, printed as one line when
        # it ends. Every tuning question needs this record.
        self.request = None
        self.request_count = 0

        # Microphone blocks are dropped until this time (follow-up guard).
        self.mic_ignore_until = 0.0

        self.wake_gate = WakeGate()
        self.link = Link()
        self.was_speaking = False
        self.offline_wake = False

        # Highest wake scores heard while Nexus was audible: per reply
        # (always logged) and per debug interval.
        self.barge_peak0 = 0.0
        self.barge_peak1 = 0.0
        self.barge_debug_peak0 = 0.0
        self.barge_debug_peak1 = 0.0
        self.barge_debug_printed = 0.0

        # Streaming playback bookkeeping.
        self.player = None
        self.current_response_id = None
        self.cancelled_response_id = None
        self.cancel_next_response = False
        self.playback_failures = 0

    def begin_request(self, origin, **facts):
        self.end_request("superseded")
        self.request_count += 1
        self.request = {"n": self.request_count, "origin": origin, "t0": time.monotonic(), "usage": {}}
        self.request.update(facts)

    def note_request(self, **facts):
        if self.request is not None:
            self.request.update(facts)

    def mark_request(self, name):
        """Record the time of a step, in seconds since the request began."""
        if self.request is not None and name not in self.request:
            self.request[name] = round(time.monotonic() - self.request["t0"], 2)

    def end_request(self, outcome):
        request = self.request
        self.request = None

        if request is None:
            return

        request["seconds"] = round(time.monotonic() - request.pop("t0"), 2)
        request["outcome"] = outcome
        request["backend"] = getattr(BACKEND, "label", getattr(BACKEND, "name", None))

        if not request["usage"]:
            request.pop("usage")

        if REQUEST_LOG:
            print("REQUEST " + json.dumps(request, separators=(",", ":")), flush=True)

    def enter_active(
        self,
        channel,
        origin="wake",
        verification_pending=False,
        timeout=None,
    ):
        self.active_channel = channel
        self.active_origin = origin
        self.verification_pending = verification_pending
        self.speech_started_seen = False
        self.speech_start_block_index = None
        self.recent_blocks.clear()

        if timeout is None:
            timeout = ACTIVE_SPEECH_TIMEOUT_SECONDS

        self.active_deadline = time.monotonic() + timeout

        self.state = AssistantState.ACTIVE

    def start_followup(
        self,
        interrupted=False,
    ):
        self.verify_blocks = []
        self.pending_send_blocks = []
        self.speech_started_seen = False
        self.speech_start_block_index = None

        if SPEAKER_VERIFY_MODE == "strict":
            self.followup_blocks.clear()
            self.wait_blocks.clear()

            self.followup_deadline = time.monotonic() + FOLLOWUP_SECONDS

            self.state = AssistantState.FOLLOWUP
            return

        # Soft/off: open the microphone to OpenAI immediately, like a
        # smart speaker's follow-up mode. If playback was interrupted,
        # begin with the speech captured since the interruption.
        if interrupted and self.wait_blocks:
            blocks = list(self.wait_blocks)

            self.pending_send_blocks = select_channel_blocks(
                blocks,
                self.active_channel,
            )

            self.verify_blocks = [(block[0].copy(), block[1].copy()) for block in blocks]

        self.wait_blocks.clear()
        self.followup_blocks.clear()

        self.enter_active(
            self.active_channel,
            origin="interrupt" if interrupted else "followup",
            verification_pending=(SPEAKER_VERIFY_MODE == "soft"),
            timeout=FOLLOWUP_SECONDS,
        )

        self.begin_request("interrupt" if interrupted else "followup")

        if not interrupted and FOLLOWUP_KEEP_OPEN:
            # The room is still ringing with Nexus's own last word.
            self.mic_ignore_until = time.monotonic() + FOLLOWUP_GUARD_SECONDS

        self.pending_input_clear = True

    def _clear_buffers(
        self,
    ):
        self.verify_blocks = []
        self.followup_blocks.clear()
        self.pending_send_blocks = []
        self.pending_input_clear = False
        self.wait_blocks.clear()
        self.recent_blocks.clear()
        self.reply_deadline = 0.0
        self.verification_pending = False
        self.speech_started_seen = False
        self.speech_start_block_index = None

        wake_preroll.clear()
        clear_mic_queue()

    def reset_for_connection(
        self,
    ):
        player = self.player
        self.player = None

        if player is not None:
            player.abort()

            if player.task is not None:
                player.task.cancel()

        self.end_request("connection_reset")

        self._clear_buffers()

        self.response_active = False
        self.current_response_id = None
        self.cancelled_response_id = None
        self.cancel_next_response = False
        self.followup_deadline = 0.0
        self.active_deadline = 0.0
        self.wake_cooldown_until = 0.0
        self.mic_ignore_until = 0.0
        self.active_channel = 0

        self.state = AssistantState.WAIT_WAKE

    def return_to_wake(
        self,
        rejected=False,
        outcome=None,
    ):
        self.end_request(outcome or ("rejected" if rejected else "ended"))

        self._clear_buffers()

        cooldown = WAKE_RESUME_COOLDOWN_SECONDS

        if rejected:
            cooldown = max(
                cooldown,
                WAKE_REJECT_COOLDOWN_SECONDS,
            )

        self.wake_cooldown_until = time.monotonic() + cooldown

        self.state = AssistantState.WAIT_WAKE

        print()

        print(f'Waiting for "{WAKE_PHRASE}"...')


def start_reply_clock(controller):
    """Called whenever a reply has just been asked for."""
    if REPLY_TIMEOUT_SECONDS > 0:
        controller.reply_deadline = time.monotonic() + REPLY_TIMEOUT_SECONDS
    else:
        controller.reply_deadline = 0.0


_background_tasks = set()


def run_in_background(coroutine):
    task = asyncio.create_task(coroutine)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def give_up_on_reply(session, controller, reason):
    """The reply is not coming: stop waiting, say so, listen for the wake phrase again."""
    print(f"Giving up on the reply: {reason}.")

    if controller.response_active:
        if controller.current_response_id:
            controller.cancelled_response_id = controller.current_response_id

        try:
            await session.cancel_reply()
        except Exception:
            pass

    controller.response_active = False
    controller.return_to_wake(outcome="no_reply")

    run_in_background(play_notice("error"))


async def note_playback_failure(session, controller, response_id, exc):
    """The speaker could not play a reply.

    Cancel the reply so no more of it is generated (and paid for) or
    retried chunk by chunk, and count the failure: several in a row mean
    the speaker is gone, which a clean restart may cure and the owner
    must hear about.
    """
    print(f"Response playback failed; keeping the connection: {type(exc).__name__}: {exc}")

    controller.playback_failures += 1

    if response_id is not None:
        controller.cancelled_response_id = response_id

    if controller.response_active:
        controller.response_active = False

        try:
            await session.cancel_reply()
        except Exception:
            pass

    try:
        await session.clear_input()
    except Exception:
        pass

    controller.return_to_wake()

    if SPEAKER_FAIL_LIMIT > 0 and controller.playback_failures >= SPEAKER_FAIL_LIMIT:
        set_status(
            "problem",
            "speaker",
            f"{controller.playback_failures} replies in a row could not be played: {exc}",
        )
        hard_exit("the speaker output keeps failing")


async def score_speaker(
    controller,
    blocks,
):
    """Return (similarity_ch0, similarity_ch1) for stereo blocks."""
    audio0 = np.concatenate(select_channel_blocks(blocks, 0))
    audio1 = np.concatenate(select_channel_blocks(blocks, 1))

    return await asyncio.to_thread(VERIFIER.score, audio0, audio1)


def followup_can_stay_open(controller):
    return (
        FOLLOWUP_KEEP_OPEN
        and controller.state == AssistantState.ACTIVE
        and controller.active_origin == "followup"
        and time.monotonic() < controller.active_deadline
    )


async def reopen_followup(session, controller):
    """Forget a sound heard in the follow-up window and keep the window open."""
    try:
        await session.clear_input()
    except Exception:
        pass

    controller.verify_blocks = []
    controller.verification_pending = SPEAKER_VERIFY_MODE == "soft"
    controller.speech_started_seen = False
    controller.speech_start_block_index = None


async def reject_current_speaker(
    session,
    controller,
):
    """Soft mode: undo a request already in flight for a rejected speaker."""
    try:
        await session.clear_input()
    except Exception:
        pass

    if controller.response_active and controller.current_response_id:
        controller.cancelled_response_id = controller.current_response_id

        try:
            await session.cancel_reply()
        except Exception:
            pass

        controller.response_active = False

    if controller.player is not None:
        controller.player.abort()

    controller.return_to_wake(rejected=True)


async def verify_speaker(
    session,
    controller,
    context,
):
    """Soft mode: judge the speaker on the audio gathered so far.

    Returns True when the request may continue. On rejection the
    request is cancelled and the controller returns to wake.
    """
    blocks = controller.verify_blocks
    controller.verify_blocks = []
    controller.verification_pending = False

    if not blocks:
        return True

    similarity0, similarity1 = await score_speaker(
        controller,
        blocks,
    )

    winner = 0 if similarity0 >= similarity1 else 1
    best_similarity = max(similarity0, similarity1)

    controller.note_request(
        speaker_ch0=round(float(similarity0), 3), speaker_ch1=round(float(similarity1), 3)
    )

    print(
        f"{context} speaker CH0: {similarity0:.3f}   "
        f"CH1: {similarity1:.3f}   "
        f"winner: CH{winner}   "
        f"({blocks_seconds(blocks):.1f}s)"
    )

    if best_similarity < SPEAKER_THRESHOLD:
        print("Speaker rejected.")

        if followup_can_stay_open(controller) and not controller.response_active:
            # An unknown voice or the echo of Nexus's own: ignore it, but
            # the person it was talking to may still want to follow up.
            print("Follow-up window stays open.")
            await reopen_followup(session, controller)
            return False

        await reject_current_speaker(session, controller)
        return False

    if winner != controller.active_channel:
        print(f"Switching to microphone channel {winner}.")

    controller.active_channel = winner

    print("Speaker accepted.")

    return True


async def process_microphone(
    link,
    controller,
):
    """Consume microphone blocks for the life of the process.

    `link` is the connection to the voice service, which may be down. A
    failed send must never end this loop: the block is dropped, the
    connection is reported as failed, and listening continues.
    """
    while True:
        block = await queue_get(mic_queue)

        now = time.monotonic()

        if now - mic_health["last_watchdog_ping"] >= WATCHDOG_PING_SECONDS:
            # Fed from here, so a frozen loop or a dead microphone both
            # stop the pings and systemd restarts the service.
            mic_health["last_watchdog_ping"] = now
            sd_notify("WATCHDOG=1")

        try:
            await handle_microphone_block(
                link,
                controller,
                block,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            if link.online:
                print(f"Microphone step failed: {type(exc).__name__}: {exc}")
                link.fail(exc)

            if controller.state != AssistantState.WAIT_WAKE:
                controller.return_to_wake()


async def handle_microphone_block(
    session,
    controller,
    block,
):
    ch0, ch1 = block

    speaking = assistant_speaking.is_set()

    if speaking != controller.was_speaking:
        controller.was_speaking = speaking

        if not WAKE_ALWAYS_SCORE:
            # Old behaviour: the models are only fed while waiting
            # or speaking, so clear their score history on a change.
            WAKE.reset()

        if speaking:
            controller.barge_peak0 = 0.0
            controller.barge_peak1 = 0.0
            controller.barge_debug_peak0 = 0.0
            controller.barge_debug_peak1 = 0.0
            controller.barge_debug_printed = time.monotonic()
        else:
            print(
                f"Barge-in peak while speaking: CH0={controller.barge_peak0:.3f} "
                f"CH1={controller.barge_peak1:.3f} (threshold {BARGE_WAKE_THRESHOLD}, {BARGE_CHANNEL})"
            )

    # The wake models hear every block in every state, so they never
    # hold a stale phrase when listening resumes. What a hit means
    # is decided below from the state.
    score0 = 0.0
    score1 = 0.0

    if WAKE_ALWAYS_SCORE or speaking or controller.state == AssistantState.WAIT_WAKE:
        score0 = WAKE.score(0, ch0)

        # A one-channel microphone delivers the same audio twice.
        score1 = WAKE.score(1, ch1) if AUDIO.channels > 1 else score0

        controller.wake_gate.observe(max(score0, score1))

    if speaking:
        handle_block_while_speaking(
            controller,
            block,
            score0,
            score1,
        )
        return

    if time.monotonic() < controller.mic_ignore_until:
        return

    if controller.state == AssistantState.WAIT_WAKE:
        wake_preroll.append(block)

        if time.monotonic() < controller.wake_cooldown_until:
            return

        if not controller.wake_gate.armed:
            return

        best_score = max(
            score0,
            score1,
        )

        if WAKE_DEBUG and best_score >= 0.02:
            print(
                f"WAKE DEBUG CH0={score0:.3f} CH1={score1:.3f} best={best_score:.3f}",
                flush=True,
            )

        if best_score < WAKE_THRESHOLD:
            return

        wake_winner = 0 if score0 >= score1 else 1

        controller.wake_gate.disarm()

        controller.begin_request(
            "wake" if session.online else "wake_offline",
            wake_ch0=round(score0, 3),
            wake_ch1=round(score1, 3),
        )

        print()
        print(f"*** {WAKE_PHRASE.upper()} DETECTED ***")
        print(f"Wake CH0: {score0:.3f}   CH1: {score1:.3f}   winner: CH{wake_winner}")

        controller.verify_blocks = [(b[0].copy(), b[1].copy()) for b in wake_preroll]

        wake_preroll.clear()

        controller.offline_wake = not session.online

        if controller.offline_wake and SPEAKER_VERIFY_MODE == "off":
            controller.verify_blocks = []
            await play_notice(session.problem or "network")
            controller.return_to_wake()
            return

        if SPEAKER_VERIFY_MODE == "strict" or controller.offline_wake:
            # Offline, nothing is streamed: check the speaker first so
            # a stray sound does not set off the notice.
            print("Checking speaker on both channels...")
            controller.state = AssistantState.VERIFY_WAKE
            return

        # Soft/off: start streaming now. The pre-roll (which
        # contains the wake phrase) goes first so the server's VAD
        # sees the start of the utterance.
        controller.enter_active(
            wake_winner,
            origin="wake",
            verification_pending=(SPEAKER_VERIFY_MODE == "soft"),
        )

        controller.pending_send_blocks = select_channel_blocks(
            controller.verify_blocks,
            wake_winner,
        )
        controller.pending_input_clear = True

        if SPEAKER_VERIFY_MODE == "soft":
            print("Listening for request; verifying speaker in parallel...")
        else:
            controller.verify_blocks = []
            print("Listening for request...")

        return

    if controller.state == AssistantState.VERIFY_WAKE:
        controller.verify_blocks.append(block)

        if not blocks_have_seconds(
            controller.verify_blocks,
            SPEAKER_VERIFY_SECONDS,
        ):
            return

        similarity0, similarity1 = await score_speaker(
            controller,
            controller.verify_blocks,
        )

        winner = 0 if similarity0 >= similarity1 else 1
        best_similarity = max(
            similarity0,
            similarity1,
        )

        print(f"Speaker CH0: {similarity0:.3f}   CH1: {similarity1:.3f}   winner: CH{winner}")

        if best_similarity < SPEAKER_THRESHOLD:
            print("Speaker rejected.")
            controller.return_to_wake(rejected=True)
            return

        print("Speaker accepted.")

        if controller.offline_wake or not session.online:
            controller.offline_wake = False
            controller.verify_blocks = []
            await play_notice(session.problem or "network")
            controller.return_to_wake()
            return

        print(f"Using microphone channel {winner}.")
        print("Listening for request...")

        controller.enter_active(winner)

        await session.clear_input()
        await send_blocks(
            session,
            select_channel_blocks(
                controller.verify_blocks,
                winner,
            ),
        )

        controller.verify_blocks = []
        return

    if controller.state == AssistantState.ACTIVE:
        if not controller.speech_started_seen and time.monotonic() >= controller.active_deadline:
            if controller.active_origin == "wake":
                print("No speech detected; stopping microphone upload.")
            else:
                print("Follow-up window closed.")

            try:
                await session.clear_input()
            except Exception:
                pass

            controller.return_to_wake(outcome="no_speech")
            return

        if (
            REQUEST_MAX_SECONDS > 0
            and controller.speech_started_seen
            and time.monotonic() - controller.speech_started_at >= REQUEST_MAX_SECONDS
        ):
            # A television or steady noise can keep the server's speech
            # detector "on" for ever. Stop sending the room to it.
            print(f"Request ran past {REQUEST_MAX_SECONDS:g} s without a pause; dropping it.")

            try:
                await session.clear_input()
            except Exception:
                pass

            controller.return_to_wake(outcome="too_long")
            return

        if controller.pending_input_clear:
            controller.pending_input_clear = False
            await session.clear_input()

        if controller.pending_send_blocks:
            blocks = controller.pending_send_blocks
            controller.pending_send_blocks = []
            await send_blocks(
                session,
                blocks,
            )

        controller.recent_blocks.append(block)

        if controller.verification_pending:
            # Collect from the start for a wake-word activation
            # (the pre-roll holds the phrase); for follow-ups,
            # wait until the server reports speech so silence is
            # not what gets judged.
            if (
                controller.active_origin == "wake"
                or controller.speech_started_seen
                or controller.verify_blocks
            ):
                controller.verify_blocks.append(block)

            speech_index = controller.speech_start_block_index

            if (
                controller.speech_started_seen
                and speech_index is not None
                and blocks_have_seconds(
                    controller.verify_blocks,
                    SOFT_VERIFY_SECONDS,
                )
                and blocks_have_seconds(
                    controller.verify_blocks[speech_index:],
                    SOFT_VERIFY_SPEECH_SECONDS,
                )
            ):
                accepted = await verify_speaker(
                    session,
                    controller,
                    "Request",
                )

                if not accepted:
                    return

        await send_audio(
            session,
            block[controller.active_channel],
        )
        return

    if controller.state == AssistantState.WAIT_RESPONSE:
        if (
            controller.reply_deadline
            and controller.player is None
            and time.monotonic() >= controller.reply_deadline
        ):
            await give_up_on_reply(session, controller, "no audio arrived in time")
            return

        # Kept so speech right after a barge-in is not lost.
        controller.wait_blocks.append(block)
        return

    if controller.state == AssistantState.FOLLOWUP:
        # Strict mode only.
        if time.monotonic() >= controller.followup_deadline:
            controller.return_to_wake()
            wake_preroll.append(block)
            return

        controller.followup_blocks.append(block)

        if not blocks_have_seconds(
            controller.followup_blocks,
            FOLLOWUP_VERIFY_SECONDS,
        ):
            return

        similarity0, similarity1 = await score_speaker(
            controller,
            controller.followup_blocks,
        )

        winner = 0 if similarity0 >= similarity1 else 1
        best_similarity = max(
            similarity0,
            similarity1,
        )

        print(f"Follow-up CH0: {similarity0:.3f}   CH1: {similarity1:.3f}   winner: CH{winner}")

        if best_similarity >= SPEAKER_THRESHOLD:
            print("Follow-up speaker accepted.")
            print(f"Using microphone channel {winner}.")

            controller.enter_active(
                winner,
                origin="followup",
            )

            await session.clear_input()
            await send_blocks(
                session,
                select_channel_blocks(
                    controller.followup_blocks,
                    winner,
                ),
            )

            controller.followup_blocks.clear()
            print("Listening for follow-up...")
            return

        print("Background/unknown speaker ignored.")

        overlap_samples = int(FOLLOWUP_OVERLAP_SECONDS * INPUT_RATE)

        audio0 = np.concatenate(
            select_channel_blocks(
                controller.followup_blocks,
                0,
            )
        )
        audio1 = np.concatenate(
            select_channel_blocks(
                controller.followup_blocks,
                1,
            )
        )

        if len(audio0) > overlap_samples:
            controller.followup_blocks = [
                (
                    audio0[-overlap_samples:].copy(),
                    audio1[-overlap_samples:].copy(),
                )
            ]
        else:
            controller.followup_blocks.clear()


# =========================================================
# EVENTS FROM THE VOICE SERVICE
# =========================================================


def start_player(session, controller, response_id):
    player = ResponsePlayer(
        response_id,
        controller,
    )
    controller.player = player
    player.task = asyncio.create_task(
        run_player(
            session,
            controller,
            player,
        )
    )

    print("Speaking...")

    return player


async def receive_events(
    session,
    controller,
):
    """Act on what the voice service says, until it closes the connection.

    The events are described in nexus.parts.VoiceSession.
    """
    # Buffered playback (NEXUS_STREAMING_PLAYBACK=0) gathers the whole
    # reply here and plays it when it is complete.
    response_audio = bytearray()

    async for message in session.events():
        event_type = message.get("type")

        if event_type == "ready":
            print("Session configured.")

            controller.return_to_wake()

        elif event_type == "speech_started":
            if controller.state == AssistantState.ACTIVE:
                controller.speech_started_seen = True
                controller.speech_started_at = time.monotonic()

                # Soft follow-up: judge the speaker from just before
                # the server noticed speech, not from the silence
                # that preceded it.
                if controller.verification_pending and not controller.verify_blocks:
                    controller.verify_blocks = [(b[0].copy(), b[1].copy()) for b in controller.recent_blocks]

                if controller.verification_pending:
                    controller.speech_start_block_index = len(controller.verify_blocks)

                print()

                print("Speech detected...")

        elif event_type == "speech_stopped":
            if (
                controller.state == AssistantState.ACTIVE
                and controller.speech_started_seen
                and not controller.response_active
            ):
                if (
                    followup_can_stay_open(controller)
                    and time.monotonic() - controller.speech_started_at < FOLLOWUP_MIN_SPEECH_SECONDS
                ):
                    print("Short sound ignored; follow-up window stays open.")
                    await reopen_followup(session, controller)
                    continue

                controller.speech_started_seen = False

                print("You stopped speaking.")

                if controller.verification_pending:
                    accepted = await verify_speaker(
                        session,
                        controller,
                        "Request",
                    )

                    if not accepted:
                        continue

                controller.state = AssistantState.WAIT_RESPONSE
                controller.wait_blocks.clear()

                controller.response_active = True

                print("Thinking...")

                controller.mark_request("speech_end")

                await send_missed_note(session)

                await session.request_reply()

                start_reply_clock(controller)

        elif event_type == "reply_started":
            response_audio.clear()

            controller.current_response_id = message.get("id")

            if controller.cancel_next_response:
                controller.cancel_next_response = False
                controller.cancelled_response_id = controller.current_response_id
                controller.response_active = False

                await session.cancel_reply()

        elif event_type == "audio":
            pcm = message.get("pcm")

            if not pcm:
                continue

            controller.reply_deadline = 0.0
            controller.mark_request("first_audio")

            response_id = message.get("id")

            if response_id is not None and response_id == controller.cancelled_response_id:
                # Audio still draining from a cancelled reply.
                continue

            if not STREAMING_PLAYBACK:
                response_audio.extend(pcm)
                continue

            player = controller.player

            if player is None or player.response_id != response_id:
                if player is not None:
                    player.finish()

                player = start_player(session, controller, response_id)

            player.feed(pcm)

        elif event_type == "transcript":
            text = message.get("text", "")

            if text:
                print(
                    text,
                    end="",
                    flush=True,
                )

        elif event_type == "audio_done":
            print()

        elif event_type == "reply_done":
            response_id = message.get("id")

            if response_id is not None and response_id == controller.cancelled_response_id:
                # Cancelled after a barge-in or speaker rejection. Whoever
                # cancelled it has already moved the state on, and a newer
                # request may be running, so leave everything else alone.
                controller.cancelled_response_id = None

                print("Response status: cancelled")

                continue

            if controller.request is not None:
                add_usage(controller.request["usage"], message.get("usage"))

            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                controller.note_request(
                    tools=(controller.request or {}).get("tools", [])
                    + [call.get("name") for call in tool_calls]
                )

                response_audio.clear()

                player = controller.player

                if player is not None and player.response_id == response_id:
                    # The model said something ("one moment") before
                    # calling the tool. Let it finish and let go of the
                    # speaker; the follow-up window belongs to the answer.
                    player.followup = False
                    player.finish()

                for call in tool_calls:
                    name = call.get("name")
                    arguments = call.get("arguments") or {}

                    print(f"Tool call: {name} {tools.redact(name, arguments)}")

                    result = await tools.execute(
                        name,
                        arguments,
                    )

                    print(f"Tool result: {tools.redact(name, result)}")

                    await session.send_tool_result(
                        call.get("call_id"),
                        result,
                    )

                await session.request_reply()

                controller.response_active = True

                start_reply_clock(controller)

                continue

            controller.response_active = False

            controller.mark_request("reply_done")
            controller.end_request("answered")

            status = message.get("status")

            if status != "completed":
                print(
                    "Response status:",
                    status,
                )

            player = controller.player

            if STREAMING_PLAYBACK and player is not None and player.response_id == response_id:
                # run_player opens the follow-up window once the
                # remaining audio has drained.
                player.finish()
                continue

            if response_audio:
                # Buffered playback: the reply is complete, play it now.
                player = start_player(session, controller, response_id)
                player.feed(bytes(response_audio))
                player.finish()

                response_audio.clear()
                continue

            # No audio in this reply.
            await open_followup(
                session,
                controller,
                False,
            )

        elif event_type == "error":
            error_message = str(message.get("message", ""))
            error_code = str(message.get("code", ""))

            print()

            print(f"Voice service error: {error_message}")

            error_kind = classify_failure(f"{error_code} {error_message}")

            if error_kind in ACCOUNT_PROBLEMS:
                # The server usually closes the connection next; make sure
                # that close is treated as an account problem.
                controller.link.last_error_kind = error_kind

            if "response_cancel_not_active" in error_code or "no active response" in error_message.lower():
                # Our own cancel arrived after the reply had ended. Harmless.
                continue

            if controller.state == AssistantState.WAIT_RESPONSE and controller.player is None:
                # The request failed. Without this Nexus would wait for
                # audio that never comes until the session ended.
                controller.response_active = False
                await give_up_on_reply(session, controller, "the service reported an error")
                continue

            controller.response_active = False


# =========================================================
# THE PROMPT
# =========================================================


def load_persona():
    """The personality paragraph of the prompt."""
    if INSTRUCTIONS_FILE.strip():
        path = Path(INSTRUCTIONS_FILE).expanduser()
    else:
        path = Path(__file__).resolve().parent / "personas" / f"{PERSONA.strip().lower()}.txt"

    try:
        return " ".join(path.read_text(encoding="utf-8").split())
    except OSError as exc:
        print(f"WARNING: could not read the personality file {path} ({exc}); using the plain one.")

        fallback = Path(__file__).resolve().parent / "personas" / "plain.txt"

        return " ".join(fallback.read_text(encoding="utf-8").split())


def build_instructions():
    """The system prompt: personality, a few fixed rules, then what each plugin adds."""
    now = datetime.now().astimezone()

    current_time_text = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")

    rules = (
        f"The phrase '{WAKE_PHRASE}' is the wake phrase and is not part of the "
        "substantive request. "
        "If the audio is unclear, is only the wake phrase, or is not addressed "
        "to you, say you did not catch it in a few words or stay silent. Never "
        "guess at a request you did not hear. "
        f"This computer's current local date and time at session creation is "
        f"{current_time_text}. This value may become stale."
    )

    return " ".join(part for part in (load_persona(), rules, tools.plugin_instructions()) if part)


# =========================================================
# ONE CONNECTION
# =========================================================


async def run_connection(
    controller,
):
    link = controller.link

    async with BACKEND.connect() as session:
        controller.reset_for_connection()

        link.attach(session)

        receiver = None
        failure = None
        settled = None

        async def mark_healthy():
            # An account problem lets the session start and closes it a
            # moment later, so "connected" only counts once it has lasted.
            await asyncio.sleep(STATUS_OK_AFTER_SECONDS)

            if link.problem is not None:
                print("Connected to the voice service.")

            link.problem = None
            link.failures = 0
            set_status("ok")

        try:
            await session.configure(
                build_instructions(),
                tools.schemas(),
            )

            receiver = asyncio.create_task(
                receive_events(
                    session,
                    controller,
                )
            )

            failure = asyncio.create_task(link.failed.wait())
            settled = asyncio.create_task(mark_healthy())

            # FIRST_COMPLETED: when the service closes the connection
            # cleanly, receive_events() returns without raising.
            done, pending = await asyncio.wait(
                [receiver, failure],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receiver in done and receiver.exception() is not None:
                raise receiver.exception()

            if failure in done and link.failure is not None:
                raise link.failure

        finally:
            link.detach()

            tasks = [task for task in (receiver, failure, settled) if task is not None]

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            controller.reset_for_connection()


# =========================================================
# RECONNECT MANAGER
# =========================================================


async def connection_manager(
    controller,
):
    link = controller.link

    delay = RECONNECT_INITIAL_SECONDS

    consecutive_handshake_failures = 0
    max_handshake_failures = 3

    while True:
        connection_started = time.monotonic()
        link.last_error_kind = None

        try:
            await run_connection(
                controller,
            )

            failure_text = "connection closed by the server"
            consecutive_handshake_failures = 0

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            failure_text = f"{type(exc).__name__}: {exc}"

            opening_handshake_timeout = (
                isinstance(exc, TimeoutError) and "opening handshake" in str(exc).lower()
            )

            if opening_handshake_timeout:
                consecutive_handshake_failures += 1

                if consecutive_handshake_failures >= max_handshake_failures:
                    hard_exit("repeated handshake failures with the voice service")
            else:
                consecutive_handshake_failures = 0

        connection_elapsed = time.monotonic() - connection_started

        kind = link.last_error_kind or classify_failure(failure_text)

        if connection_elapsed >= 30:
            delay = RECONNECT_INITIAL_SECONDS
            consecutive_handshake_failures = 0

        if connection_elapsed >= 30 and kind == "network":
            # A session that ran and then ended (the hourly session
            # limit, or a brief drop) is not a problem yet.
            print(f"Voice session ended ({failure_text}); reconnecting.")
            wait = RECONNECT_INITIAL_SECONDS

        else:
            first = link.problem != kind

            link.problem = kind
            link.failures += 1

            if kind in ACCOUNT_PROBLEMS:
                wait = ACCOUNT_RETRY_SECONDS
            else:
                wait = delay
                delay = min(
                    delay * 2,
                    RECONNECT_MAX_SECONDS,
                )

            if first:
                # One line when the state changes, not five per attempt.
                print()
                print(f"Connection to the voice service lost: {failure_text}")
                print(
                    f"Cause: {PROBLEM_LABELS[kind]}. Retrying every {wait:g} s"
                    + ("." if kind in ACCOUNT_PROBLEMS else f", up to every {RECONNECT_MAX_SECONDS} s.")
                )

            if kind in ACCOUNT_PROBLEMS or link.failures >= NETWORK_PROBLEM_AFTER_FAILURES:
                set_status(
                    "problem",
                    kind,
                    failure_text,
                )

        await asyncio.sleep(wait)


# =========================================================
# START
# =========================================================


def build_parts():
    """Create the four parts named by the settings. Nothing is opened yet."""
    global AUDIO, WAKE, VERIFIER, BACKEND

    AUDIO = parts.create("audio")
    WAKE = parts.create("wake")
    VERIFIER = parts.create("speaker_id")
    BACKEND = parts.create("backend")


async def run():
    """Run the assistant until it is stopped."""
    global SPEAKER_VERIFY_MODE

    print(f"Nexus {config.code_version()}")

    # Plugins first: a plugin may register a hardware part by name.
    tools.load_all()

    build_parts()

    print(
        f"Parts: audio={AUDIO.name} wake={WAKE.name} speaker_id={VERIFIER.name} "
        f"backend={getattr(BACKEND, 'label', BACKEND.name)}"
    )

    for plugin_name, origin, tool_names in tools.LOADED:
        print(f"Plugin {plugin_name} ({origin}): {', '.join(tool_names) or 'no tools'}")

    for plugin_name, origin, error in tools.FAILED:
        print(f"Plugin {plugin_name} ({origin}) was skipped: {error}")

    if not VERIFIER.enabled and SPEAKER_VERIFY_MODE != "off":
        # No speaker check is installed, so there is nothing to wait for.
        SPEAKER_VERIFY_MODE = "off"

    print("Settings in effect (* = changed from the default):")

    for line in config.describe_settings():
        print(line)

    print()

    set_status("starting")

    await asyncio.to_thread(WAKE.load)

    if VERIFIER.enabled:
        await asyncio.to_thread(VERIFIER.load)

    controller = NexusController()

    await tools.start_plugins()

    await asyncio.to_thread(AUDIO.start, on_microphone_block)

    for line in AUDIO.describe():
        print(line)

    # The microphone loop runs for the life of the process, with or
    # without a connection, so the wake phrase is always heard.
    tasks = [
        asyncio.create_task(
            process_microphone(
                controller.link,
                controller,
            )
        ),
        asyncio.create_task(status_heartbeat()),
        asyncio.create_task(microphone_watchdog()),
        asyncio.create_task(connection_manager(controller)),
    ]

    print()
    print(f'Waiting for "{WAKE_PHRASE}"...')

    sd_notify("READY=1")

    try:
        # None of these should ever finish; if one does, stop so the
        # service manager restarts the whole process.
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            if task.exception() is not None:
                raise task.exception()

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        await asyncio.to_thread(AUDIO.stop)

        tools.stop_plugins()
