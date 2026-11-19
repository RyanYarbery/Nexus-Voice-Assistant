import os

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import asyncio
import base64
import json
import math
import queue
import shutil
import socket
import subprocess
import threading
import time
import warnings
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings(
    "ignore",
    message="Specified provider 'CUDAExecutionProvider'.*",
)

import numpy as np
import sounddevice as sd
import torch
import websockets

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - scipy is normally present
    resample_poly = None

from openwakeword.model import Model as WakeWordModel
from speechbrain.inference.speaker import SpeakerRecognition


# =========================================================
# CONFIGURATION
# =========================================================
#
# Every value a user might tune is read from the environment through
# one of the env_* helpers below. Each call records the setting, so the
# settings in effect can be printed at start.

SETTINGS = {}


def _setting(name, default, convert, doc, secret=False):
    raw = os.environ.get(name)
    value = default

    if raw is not None and raw.strip() != "":
        try:
            value = convert(raw.strip())
        except (TypeError, ValueError):
            print(f"WARNING: {name}={raw!r} is not valid; using {default!r}.")

    SETTINGS[name] = {"value": value, "default": default, "doc": doc, "secret": secret}

    return value


def env_str(name, default, doc="", secret=False):
    return _setting(name, default, str, doc, secret)


def env_float(name, default, doc=""):
    return _setting(name, default, float, doc)


def env_int(name, default, doc=""):
    return _setting(name, default, lambda raw: int(float(raw)), doc)


def env_flag(name, default=False, doc=""):
    return _setting(
        name,
        default,
        lambda raw: raw.lower() in ("1", "true", "yes", "on"),
        doc,
    )


INPUT_RATE = 16000
MODEL_RATE = 24000
OUTPUT_RATE = 48000
# Verified USB playback format: 16 kHz, stereo, signed 16-bit PCM.
AEC_REFERENCE_RATE = 16000

BASE_DIR = Path(
    env_str("NEXUS_HOME", str(Path(__file__).resolve().parent), "Folder holding models and saved state.")
).expanduser()

OPENAI_MODEL = env_str(
    "OPENAI_REALTIME_MODEL",
    "gpt-realtime",
    "OpenAI Realtime model name.",
)

MIC_NAME = env_str("NEXUS_MIC_NAME", "respeaker xvf3800", "Part of the microphone's device name.")

SPEAKER_NAME = env_str("NEXUS_SPEAKER_NAME", "alc3234 analog", "Part of the speaker's device name.")

WAKE_MODEL_PATH = env_str(
    "NEXUS_WAKE_MODEL_PATH",
    str(BASE_DIR / "models" / "hey_nexus.onnx"),
    "openWakeWord .onnx model for the wake phrase.",
)

VOICE_PROFILE_CH0_PATH = env_str(
    "NEXUS_VOICE_PROFILE_CH0",
    str(BASE_DIR / "models" / "voice_ch0.pt"),
    "Enrolled voice profile for microphone channel 0.",
)

VOICE_PROFILE_CH1_PATH = env_str(
    "NEXUS_VOICE_PROFILE_CH1",
    str(BASE_DIR / "models" / "voice_ch1.pt"),
    "Enrolled voice profile for microphone channel 1.",
)

SPEAKER_MODEL_CACHE = env_str(
    "NEXUS_SPEAKER_MODEL_CACHE",
    str(BASE_DIR / "models" / "spkrec-ecapa-voxceleb"),
    "Folder for the downloaded speaker-recognition model.",
)

ALARMS_FILE = Path(env_str("NEXUS_ALARMS_FILE", str(BASE_DIR / "alarms.json"), "Saved alarms."))

SETTINGS_FILE = Path(
    env_str("NEXUS_SETTINGS_FILE", str(BASE_DIR / "settings.json"), "Saved location and preferences.")
)

TIMERS_FILE = Path(env_str("NEXUS_TIMERS_FILE", str(BASE_DIR / "timers.json"), "Saved timers."))

# A restored timer that expired while Nexus was down still rings if
# it is no later than this; older ones are silently discarded.
TIMER_LATE_GRACE_SECONDS = 600

# Output device used as the XVF3800 far-end echo-cancellation
# reference. Defaults to the microphone's own playback endpoint.
# Set to "off" for microphones without one; barge-in then relies on
# the raw wake-word score alone.
AEC_REFERENCE_NAME = env_str(
    "NEXUS_AEC_REFERENCE_NAME",
    MIC_NAME,
    'Output that receives a copy of playback for echo cancelling; "off" for none.',
)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HTTP_TIMEOUT_SECONDS = 12

HTTP_USER_AGENT = env_str(
    "NEXUS_HTTP_USER_AGENT",
    "NexusHouseholdAssistant/1.0 (personal non-commercial assistant)",
    "User-Agent sent to the geocoding and weather services.",
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

INPUT_BLOCK_FRAMES = 1280

WAKE_PREROLL_SECONDS = env_float(
    "NEXUS_WAKE_PREROLL_SECONDS", 1.0, "Audio kept from before the wake word fired."
)
SPEAKER_VERIFY_SECONDS = env_float(
    "NEXUS_SPEAKER_VERIFY_SECONDS", 1.6, "Strict mode: audio gathered before the speaker is judged."
)

# Speech-aware speaker verification.  Instead of scoring arbitrary
# fixed windows, find the strongest contiguous speech region around
# the wake phrase, trim surrounding silence/noise, and verify that.
SPEAKER_VAD_FRAME_MS = 20
SPEAKER_VAD_HOP_MS = 10
SPEAKER_VAD_NOISE_MULTIPLIER = 2.2
SPEAKER_VAD_PEAK_FRACTION = 0.12
SPEAKER_VAD_ABSOLUTE_FLOOR = 20.0
SPEAKER_VAD_MAX_GAP_MS = 180
SPEAKER_VAD_MARGIN_MS = 120
SPEAKER_VAD_MIN_SECONDS = 0.45
SPEAKER_VAD_MAX_SECONDS = 1.35

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

# When the speech-aware trimmer finds less than this much speech, the
# untrimmed window is scored as well and the better score is used.
SPEAKER_VAD_FALLBACK_SECONDS = 0.8

FOLLOWUP_SECONDS = env_float(
    "NEXUS_FOLLOWUP_SECONDS", 5.0, "How long the microphone stays open for a follow-up after a reply."
)
FOLLOWUP_VERIFY_SECONDS = 2.2
FOLLOWUP_OVERLAP_SECONDS = 0.6

# After speaker verification, microphone audio streams to OpenAI
# until server VAD reports speech. If no speech is ever detected,
# stop streaming and return to wake instead of uploading indefinitely.
ACTIVE_SPEECH_TIMEOUT_SECONDS = env_float(
    "NEXUS_ACTIVE_SPEECH_TIMEOUT_SECONDS", 10.0, "Give up on a request if no speech starts within this."
)

# =========================================================
# HEY NEXUS BARGE-IN
# =========================================================
#
# While Nexus is speaking, send the same PCM audio to the
# audible Dell output and the XVF3800 USB playback endpoint.
# The XVF3800 uses that silent digital copy as its far-end AEC
# reference. CH0 is the strongly echo-cancelled channel used
# for interruption detection; CH1 is retained for diagnostics.
# Requiring the wake phrase prevents ordinary nearby speech
# from being mistaken for an interruption.

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

# =========================================================
# PLAYBACK
# =========================================================
#
# Streaming playback starts the loudspeaker as soon as the first
# audio arrives instead of waiting for the whole reply, and sends
# response.cancel to the server on barge-in. NEXUS_STREAMING_PLAYBACK=0
# restores the original buffered behaviour.

STREAMING_PLAYBACK = env_flag(
    "NEXUS_STREAMING_PLAYBACK",
    True,
    "Start speaking as soon as audio arrives (0 = wait for the whole reply).",
)
PLAYBACK_PREBUFFER_SECONDS = env_float(
    "NEXUS_PLAYBACK_PREBUFFER_SECONDS", 0.3, "Audio buffered before the speaker starts."
)

# The XVF3800 locks its master clock to the host playback stream and
# free-runs when none is open. Keeping one reference stream open for
# the life of the process (writing silence between replies) avoids a
# clock re-lock and AEC re-convergence at the start of every reply.
AEC_REFERENCE_CONTINUOUS = env_flag(
    "NEXUS_AEC_REFERENCE_CONTINUOUS", True, "Keep the echo-cancelling reference stream open all the time."
)

# =========================================================
# SPEAKER VERIFICATION
# =========================================================
#
#   soft   - stream to OpenAI the moment the wake word fires and
#            verify the speaker in parallel over the first seconds of
#            the request; cancel if the speaker is rejected. (default)
#   strict - original behaviour: hold the request until 1.2 s of audio
#            has been verified before any audio leaves the machine.
#   off    - no speaker verification.

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

# Trim the verification audio to the strongest speech region before
# scoring, so silence and room noise around the phrase do not dilute
# the embedding. 0 scores the raw window instead.
SPEAKER_SPEECH_AWARE = env_flag(
    "NEXUS_SPEAKER_SPEECH_AWARE", True, "Trim the audio to the speech before the voice check."
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

RESPEAKER_ROUTE_HELPER = env_str(
    "NEXUS_RESPEAKER_ROUTE_HELPER",
    "/usr/local/bin/nexus-respeaker-route",
    "Optional helper that sets up the ReSpeaker's channel routing.",
)

RECONNECT_INITIAL_SECONDS = 1
RECONNECT_MAX_SECONDS = env_int(
    "NEXUS_RECONNECT_MAX_SECONDS", 15, "Longest wait between reconnect attempts after a network failure."
)

TIMER_TONE_FREQUENCY = 880
TIMER_BEEP_SECONDS = 0.35
TIMER_BEEP_GAP = 0.25
TIMER_BEEP_COUNT = env_int("NEXUS_TIMER_BEEP_COUNT", 10, "Beeps when a timer ends.")
TIMER_VOLUME = env_float("NEXUS_TIMER_VOLUME", 0.25, "Timer beep volume, 0 to 1.")

ALARM_TONE_FREQUENCY = 1050
ALARM_BEEP_SECONDS = 0.45
ALARM_BEEP_GAP = 0.20
ALARM_BEEP_COUNT = env_int("NEXUS_ALARM_BEEP_COUNT", 16, "Beeps when an alarm goes off.")
ALARM_VOLUME = env_float("NEXUS_ALARM_VOLUME", 0.30, "Alarm beep volume, 0 to 1.")

# --- Time limits on every waiting state -----------------------------

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

# --- Failing loudly -------------------------------------------------

OFFLINE_NOTICES = env_flag(
    "NEXUS_OFFLINE_NOTICES",
    True,
    "Play a short notice when the wake phrase is heard but Nexus cannot answer.",
)
NOTICE_MIN_GAP_SECONDS = env_float("NEXUS_NOTICE_MIN_GAP_SECONDS", 20, "Shortest time between two notices.")
NOTICE_DIR = Path(
    env_str(
        "NEXUS_NOTICE_DIR",
        str(BASE_DIR / "notices"),
        "Folder of your own notice recordings (quota.wav, network.wav, error.wav ...).",
    )
)
NOTICE_TTS = env_str(
    "NEXUS_NOTICE_TTS",
    "auto",
    "Speech for notices: auto (espeak-ng if installed), off (tones only), or a command that writes WAV to stdout.",
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
STATUS_HEARTBEAT_SECONDS = 60
WATCHDOG_PING_SECONDS = 10

_status_file = env_str(
    "NEXUS_STATUS_FILE",
    str(BASE_DIR / "status.json"),
    'Health file for a watchdog to read; "off" for none.',
)
STATUS_FILE = None if _status_file.lower() in ("off", "none", "") else Path(_status_file)

NTFY_URL = env_str(
    "NEXUS_NTFY_URL",
    "",
    "ntfy topic address for push alerts when the status changes; empty for none.",
    secret=True,
)

OPENAI_VOICE = env_str("NEXUS_VOICE", "marin", "OpenAI voice used for replies.")

VAD_THRESHOLD = env_float("NEXUS_VAD_THRESHOLD", 0.5, "OpenAI speech detector sensitivity, 0 to 1.")
VAD_SILENCE_MS = env_int("NEXUS_VAD_SILENCE_MS", 700, "Silence that ends your turn, in milliseconds.")
VAD_PREFIX_MS = env_int("NEXUS_VAD_PREFIX_MS", 300, "Audio kept from before speech was detected.")

PING_INTERVAL_SECONDS = env_float("NEXUS_PING_INTERVAL_SECONDS", 5, "Seconds between keep-alive pings.")
PING_TIMEOUT_SECONDS = env_float(
    "NEXUS_PING_TIMEOUT_SECONDS", 20, "Drop the connection after this long without a ping reply."
)


def describe_settings():
    """One line per setting. Secrets never pass through SETTINGS."""
    lines = []

    for name in sorted(SETTINGS):
        entry = SETTINGS[name]
        mark = "*" if entry["value"] != entry["default"] else " "
        shown = "(set)" if entry["secret"] and entry["value"] else entry["value"]
        lines.append(f"  {mark} {name}={shown}")

    return lines


def code_version():
    """The commit this code came from: NEXUS_COMMIT, a COMMIT file, or git."""
    explicit = os.environ.get("NEXUS_COMMIT", "").strip()

    if explicit:
        return explicit

    here = Path(__file__).resolve().parent

    try:
        text = (here / "COMMIT").read_text(encoding="utf-8").strip()

        if text:
            return text
    except OSError:
        pass

    try:
        completed = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )

        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return "unknown"


# =========================================================
# SHARED STATE
# =========================================================

mic_queue = queue.Queue(maxsize=250)

assistant_speaking = threading.Event()
barge_in_event = threading.Event()

input_stream_lock = threading.Lock()
input_stream = None

wake_preroll = deque(
    maxlen=max(
        1,
        int(WAKE_PREROLL_SECONDS * INPUT_RATE / INPUT_BLOCK_FRAMES),
    )
)

timers = {}
next_timer_id = 1

alarms = {}
next_alarm_id = 1

settings_lock = threading.Lock()
last_nominatim_request = 0.0

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


# =========================================================
# DEVICE DISCOVERY
# =========================================================


def find_device(
    name_fragment,
    need_input=False,
    need_output=False,
):
    devices = sd.query_devices()

    for i, device in enumerate(devices):
        name = device["name"].lower()

        if name_fragment.lower() not in name:
            continue

        if need_input and device["max_input_channels"] < 1:
            continue

        if need_output and device["max_output_channels"] < 1:
            continue

        return i

    raise RuntimeError(f"Audio device not found: {name_fragment}")


INPUT_DEVICE = find_device(
    MIC_NAME,
    need_input=True,
)

OUTPUT_DEVICE = find_device(
    SPEAKER_NAME,
    need_output=True,
)

if AEC_REFERENCE_NAME.strip().lower() in (
    "",
    "off",
    "none",
    "disabled",
):
    AEC_REFERENCE_DEVICE = None
else:
    try:
        AEC_REFERENCE_DEVICE = find_device(
            AEC_REFERENCE_NAME,
            need_output=True,
        )
    except RuntimeError as exc:
        print(f"WARNING: {exc}; continuing without an AEC reference output.")
        AEC_REFERENCE_DEVICE = None


# =========================================================
# AUDIO UTILITIES
# =========================================================


def resample_int16(
    audio,
    old_rate,
    new_rate,
):
    if old_rate == new_rate:
        return audio

    if len(audio) == 0:
        return audio

    if resample_poly is not None:
        # Polyphase FIR resampling. Linear interpolation has no
        # anti-aliasing filter, which matters most for the 48 kHz
        # to 16 kHz AEC reference and the 16 kHz to 24 kHz uplink.
        divisor = math.gcd(old_rate, new_rate)
        result = resample_poly(
            audio.astype(np.float32),
            new_rate // divisor,
            old_rate // divisor,
        )
        return np.clip(
            np.rint(result),
            -32768,
            32767,
        ).astype(np.int16)

    new_length = int(len(audio) * new_rate / old_rate)

    old_x = np.arange(
        len(audio),
        dtype=np.float64,
    )

    new_x = np.linspace(
        0,
        len(audio) - 1,
        new_length,
    )

    result = np.interp(
        new_x,
        old_x,
        audio.astype(np.float32),
    )

    return np.clip(
        result,
        -32768,
        32767,
    ).astype(np.int16)


class StreamResampler:
    """Stateful polyphase resampler for a stream of int16 chunks.

    Resampling each network delta independently would leave a filter
    transient at every chunk boundary and click audibly. This keeps the
    filter state and decimation phase between calls, so chunked output
    is identical to resampling the whole signal at once.
    """

    def __init__(self, old_rate, new_rate):
        divisor = math.gcd(old_rate, new_rate)
        self.up = new_rate // divisor
        self.down = old_rate // divisor
        self.identity = self.up == 1 and self.down == 1

        if self.identity:
            return

        half_len = 10 * max(self.up, self.down)
        taps = 2 * half_len + 1

        # Low-pass at the lower of the two Nyquist frequencies,
        # expressed as a fraction of the zero-stuffed signal's Nyquist.
        cutoff = 1.0 / max(self.up, self.down)
        n = np.arange(taps) - half_len
        kernel = cutoff * np.sinc(cutoff * n) * np.kaiser(taps, 5.0)
        kernel = kernel / kernel.sum() * self.up

        self.kernel = kernel.astype(np.float32)
        self.state = np.zeros(taps - 1, dtype=np.float32)
        self.phase = 0

    def process(self, audio_int16):
        if self.identity or len(audio_int16) == 0:
            return audio_int16

        x = audio_int16.astype(np.float32)

        if self.up > 1:
            upsampled = np.zeros(len(x) * self.up, dtype=np.float32)
            upsampled[:: self.up] = x
        else:
            upsampled = x

        buffer = np.concatenate([self.state, upsampled])
        filtered = np.convolve(buffer, self.kernel, mode="valid")
        self.state = buffer[len(upsampled) :]

        if self.down > 1:
            out = filtered[self.phase :: self.down]
            n = len(filtered)

            if n > self.phase:
                taken = (n - self.phase + self.down - 1) // self.down
                self.phase = self.phase + taken * self.down - n
            else:
                self.phase -= n
        else:
            out = filtered

        return np.clip(
            np.rint(out),
            -32768,
            32767,
        ).astype(np.int16)


def samples_in_blocks(
    blocks,
):
    return sum(len(block[0]) if isinstance(block, tuple) else len(block) for block in blocks)


def select_channel_blocks(
    blocks,
    channel,
):
    return [block[channel] if isinstance(block, tuple) else block for block in blocks]


def blocks_seconds(
    blocks,
):
    return samples_in_blocks(blocks) / INPUT_RATE


def blocks_have_seconds(
    blocks,
    seconds,
):
    needed = int(seconds * INPUT_RATE)

    return samples_in_blocks(blocks) >= needed


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


def open_input_stream():
    global input_stream

    with input_stream_lock:
        if input_stream is not None:
            return

        stream = sd.InputStream(
            samplerate=INPUT_RATE,
            blocksize=INPUT_BLOCK_FRAMES,
            device=INPUT_DEVICE,
            channels=2,
            dtype="int16",
            callback=microphone_callback,
        )

        stream.start()
        input_stream = stream


def close_input_stream():
    global input_stream

    with input_stream_lock:
        stream = input_stream
        input_stream = None

    if stream is None:
        return

    try:
        stream.stop()
    except Exception:
        pass

    try:
        stream.close()
    except Exception:
        pass


class ReferenceOutput:
    """Permanently open far-end reference stream for the XVF3800.

    PortAudio pulls from a queue of mono 16 kHz chunks in its callback
    and plays silence when the queue is empty, so the device always has
    a host playback stream to lock its clock to.
    """

    def __init__(self, device, rate):
        self.rate = rate
        self._lock = threading.Lock()
        self._chunks = deque()
        self._offset = 0
        self._stream = sd.OutputStream(
            samplerate=rate,
            device=device,
            channels=2,
            dtype="int16",
            blocksize=int(rate * 0.05),
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def close(self):
        for method in (self._stream.stop, self._stream.close):
            try:
                method()
            except Exception:
                pass

    def write(self, mono_int16):
        if len(mono_int16) == 0:
            return

        with self._lock:
            self._chunks.append(mono_int16)

    def flush(self):
        with self._lock:
            self._chunks.clear()
            self._offset = 0

    def _callback(self, outdata, frames, time_info, status):
        pos = 0

        with self._lock:
            while pos < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(len(chunk) - self._offset, frames - pos)
                segment = chunk[self._offset : self._offset + take]

                outdata[pos : pos + take, 0] = segment
                outdata[pos : pos + take, 1] = segment

                pos += take
                self._offset += take

                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0

        if pos < frames:
            outdata[pos:] = 0


reference_output = None


def open_reference_output():
    global reference_output

    if AEC_REFERENCE_DEVICE is None or not AEC_REFERENCE_CONTINUOUS:
        return

    try:
        output = ReferenceOutput(
            AEC_REFERENCE_DEVICE,
            AEC_REFERENCE_RATE,
        )
        output.start()
        reference_output = output
        print("AEC reference stream: continuous")

    except Exception as exc:
        print(
            "WARNING: could not open a continuous AEC reference stream; "
            f"falling back to per-reply streams: {type(exc).__name__}: {exc}"
        )
        reference_output = None


def close_reference_output():
    global reference_output

    output = reference_output
    reference_output = None

    if output is not None:
        output.close()


# =========================================================
# STATUS, ALERTS AND SPOKEN NOTICES
# =========================================================
#
# Nexus must never fail silently. Three things make a failure visible:
#   - a status file that a watchdog (for example device-watch) can read;
#   - an optional push message through ntfy when the status changes;
#   - a short spoken or tone notice when someone says the wake phrase
#     and Nexus cannot answer.

ACCOUNT_PROBLEMS = ("quota", "auth", "model")

PROBLEM_LABELS = {
    "quota": "out of credit",
    "auth": "API key rejected",
    "model": "voice model not available",
    "network": "cannot reach the voice service",
    "speaker": "speaker output failing",
    "microphone": "microphone stopped",
}

NOTICE_TEXT = {
    "quota": "The OpenAI account is out of credit.",
    "auth": "The OpenAI key was rejected.",
    "model": "The voice model is no longer available.",
    "network": "I cannot reach OpenAI right now.",
    "starting": "I am still connecting.",
    "error": "Something went wrong. Ask me again.",
    "missed": "An alarm or timer was missed while I was off.",
}

# Fallback when no speech synthesiser is installed: (frequency Hz, seconds).
NOTICE_TONES = {
    "error": [(440, 0.18), (330, 0.30)],
    "missed": [(660, 0.15), (660, 0.15), (440, 0.30)],
}
NOTICE_TONES_DEFAULT = [(660, 0.20), (520, 0.20), (392, 0.35)]

# Extra ways to deliver a notice. Each hook is an async function
# (kind, text) -> bool; the first one that returns True has handled it.
NOTICE_HOOKS = []

status = {
    "state": "starting",
    "problem": None,
    "detail": "",
    "since": datetime.now().astimezone().isoformat(timespec="seconds"),
}
_last_notice_at = 0.0


def classify_failure(text):
    """Sort a connection failure into quota, auth, model or network."""
    lower = str(text).lower()

    if any(
        mark in lower for mark in ("insufficient_quota", "credit", "billing", "exceeded your current quota")
    ):
        return "quota"

    if any(
        mark in lower
        for mark in (
            "invalid_api_key",
            "incorrect api key",
            "api_key is not set",
            "authentication",
            "unauthorized",
            "http 401",
            "http 403",
        )
    ):
        return "auth"

    if any(
        mark in lower
        for mark in ("model_not_found", "does not exist", "deprecated", "http 404", "access to model")
    ):
        return "model"

    return "network"


def status_summary():
    if status["state"] != "problem":
        return status["state"]

    label = PROBLEM_LABELS.get(status["problem"], status["problem"])

    return f"problem: {label} since {status['since']}"


def write_status_file():
    if STATUS_FILE is None:
        return

    data = dict(status)
    data["summary"] = status_summary()
    data["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    data["version"] = code_version()
    data["pid"] = os.getpid()

    try:
        temporary = STATUS_FILE.with_name(STATUS_FILE.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(STATUS_FILE)
    except OSError as exc:
        print(f"Could not write the status file {STATUS_FILE}: {exc}")


def _post_ntfy(title, body):
    try:
        request = urllib.request.Request(
            NTFY_URL,
            data=body.encode("utf-8"),
            headers={"Title": title, "User-Agent": HTTP_USER_AGENT},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS):
            pass
    except Exception as exc:
        print(f"Could not send the push alert: {type(exc).__name__}: {exc}")


def send_alert(title, body):
    """Push a message to the owner's phone. Does nothing unless NEXUS_NTFY_URL is set."""
    if not NTFY_URL:
        return

    threading.Thread(target=_post_ntfy, args=(title, body), daemon=True).start()


def set_status(state, problem=None, detail=""):
    """Record the health of the assistant; log and alert when it changes."""
    previous = (status["state"], status["problem"])
    changed = previous != (state, problem)

    status["detail"] = str(detail)[:300]

    if changed:
        status["state"] = state
        status["problem"] = problem
        status["since"] = datetime.now().astimezone().isoformat(timespec="seconds")

        print(
            f"Status: {status_summary()}" + (f" ({status['detail']})" if status["detail"] else ""), flush=True
        )

        if state == "problem":
            send_alert("Nexus problem", f"{PROBLEM_LABELS.get(problem, problem)}. {status['detail']}")
        elif previous[0] == "problem":
            send_alert("Nexus recovered", f"Was: {PROBLEM_LABELS.get(previous[1], previous[1])}.")

    write_status_file()

    sd_notify(f"STATUS={status_summary()}")


async def status_heartbeat():
    """Rewrite the status file every minute so a stale file means a hung process."""
    while True:
        write_status_file()
        await asyncio.sleep(STATUS_HEARTBEAT_SECONDS)


def sd_notify(message):
    """Tell systemd something (READY=1, WATCHDOG=1, STATUS=...).

    Does nothing unless systemd started the process with a notify socket
    (Type=notify or WatchdogSec= in the unit).
    """
    address = os.environ.get("NOTIFY_SOCKET")

    if not address:
        return False

    if address.startswith("@"):
        address = "\0" + address[1:]

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_socket:
            notify_socket.sendto(message.encode("utf-8"), address)

        return True
    except OSError:
        return False


# Time of the last microphone block, and of the last overflow message.
mic_health = {"last_block": time.monotonic(), "last_status_print": 0.0, "last_watchdog_ping": 0.0}


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


def hard_exit(reason):
    """Leave at once so the service manager starts a clean process.

    Raising is not enough: audio and model shutdown can hang before
    Python exits.
    """
    print(f"FATAL: {reason}. Exiting for a clean restart.", flush=True)
    os._exit(1)


def _tone_sequence(tones, volume=0.25):
    parts = []

    for frequency, seconds in tones:
        t = np.arange(int(OUTPUT_RATE * seconds), dtype=np.float32) / OUTPUT_RATE
        tone = np.sin(2 * np.pi * frequency * t) * volume * 32767
        fade = min(len(tone) // 2, int(OUTPUT_RATE * 0.01))

        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]

        parts.append(tone.astype(np.int16))
        parts.append(np.zeros(int(OUTPUT_RATE * 0.06), dtype=np.int16))

    return np.concatenate(parts)


def _wav_bytes_to_pcm(data):
    """Decode a 16-bit PCM WAV (as a speech synthesiser writes to stdout) to mono at OUTPUT_RATE."""
    if len(data) < 44 or data[:4] != b"RIFF":
        raise ValueError("not a WAV stream")

    channels = int.from_bytes(data[22:24], "little") or 1
    rate = int.from_bytes(data[24:28], "little")
    start = data.find(b"data")

    if start < 0 or rate <= 0:
        raise ValueError("WAV stream has no audio")

    body = data[start + 8 :]
    pcm = np.frombuffer(body[: len(body) // 2 * 2], dtype=np.int16)

    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)

    return resample_int16(pcm, rate, OUTPUT_RATE)


def notice_audio(kind, text):
    """Audio for a notice: your own recording, a local synthesiser, or tones.

    Nothing here needs the network, so it works when OpenAI does not.
    """
    recording = NOTICE_DIR / f"{kind}.wav"

    if recording.exists():
        try:
            return _wav_bytes_to_pcm(recording.read_bytes())
        except Exception as exc:
            print(f"Could not use {recording}: {exc}")

    if NOTICE_TTS != "off":
        if NOTICE_TTS == "auto":
            command = shutil.which("espeak-ng") or shutil.which("espeak")
            arguments = [command, "--stdout", text] if command else None
        else:
            arguments = [NOTICE_TTS, text]

        if arguments:
            try:
                completed = subprocess.run(
                    arguments,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=True,
                )

                return _wav_bytes_to_pcm(completed.stdout)
            except Exception as exc:
                print(f"Speech synthesiser failed ({type(exc).__name__}: {exc}); using tones.")

    return _tone_sequence(NOTICE_TONES.get(kind, NOTICE_TONES_DEFAULT))


async def play_local_audio(pcm_48k):
    """Play locally generated audio (a notice) through the speaker."""
    async with playback_lock():
        barge_in_event.clear()
        assistant_speaking.set()

        try:
            if reference_output is not None:
                reference_output.write(resample_int16(pcm_48k, OUTPUT_RATE, AEC_REFERENCE_RATE))

            await asyncio.to_thread(
                sd.play,
                pcm_48k,
                OUTPUT_RATE,
                device=OUTPUT_DEVICE,
                blocking=True,
            )
        finally:
            assistant_speaking.clear()
            barge_in_event.clear()
            clear_mic_queue()


async def play_notice(kind, text=None):
    """Tell whoever is in the room why Nexus cannot answer. Returns True if something was played."""
    global _last_notice_at

    if not OFFLINE_NOTICES:
        return False

    now = time.monotonic()

    if now - _last_notice_at < NOTICE_MIN_GAP_SECONDS:
        return False

    _last_notice_at = now

    text = text or NOTICE_TEXT.get(kind, NOTICE_TEXT["error"])

    print(f"Notice ({kind}): {text}")

    for hook in list(NOTICE_HOOKS):
        try:
            if await hook(kind, text):
                return True
        except Exception as exc:
            print(f"Notice hook failed: {type(exc).__name__}: {exc}")

    try:
        pcm = await asyncio.to_thread(notice_audio, kind, text)
        await play_local_audio(pcm)
    except Exception as exc:
        print(f"Could not play the notice: {type(exc).__name__}: {exc}")
        return False

    return True


class Link:
    """The current connection to the voice service, or none.

    The microphone loop lives for the whole process and talks to the
    service through this object, so it keeps listening for the wake
    phrase while the connection is down.
    """

    def __init__(self):
        self.ws = None
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
        return self.ws is not None

    def attach(self, ws):
        self.failed = asyncio.Event()
        self.failure = None
        self.ws = ws

    def detach(self):
        self.ws = None

    def fail(self, exc):
        """Called by the microphone loop when a send fails."""
        if self.failure is None:
            self.failure = exc

        if self.failed is not None:
            self.failed.set()

    async def send(self, payload):
        ws = self.ws

        if ws is None:
            raise ConnectionError("not connected to the voice service")

        await ws.send(payload)


# =========================================================
# MICROPHONE
# =========================================================


def block_rms(
    stereo_block,
):
    ch0, ch1 = stereo_block

    def rms(channel):
        audio = channel.astype(np.float32)

        if len(audio) == 0:
            return 0.0

        return float(np.sqrt(np.mean(audio * audio)))

    return (
        rms(ch0),
        rms(ch1),
    )


def microphone_callback(
    indata,
    frames,
    time_info,
    status,
):
    now = time.monotonic()

    mic_health["last_block"] = now

    if status and now - mic_health["last_status_print"] >= 10.0:
        # At most one line per 10 s: under CPU pressure this fires on
        # every callback and would flood the log.
        mic_health["last_status_print"] = now
        print(f"Microphone status: {status}")

    ch0 = indata[:, 0].copy().astype(np.int16)

    ch1 = indata[:, 1].copy().astype(np.int16)

    stereo_block = (
        ch0,
        ch1,
    )

    # One queue and one consumer for every state. The consumer scores
    # each block for the wake phrase and decides from the state whether
    # a hit means wake, barge-in or nothing.
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
# SPEAKER VERIFICATION
# =========================================================


def load_voice_system():
    print("Loading speaker-recognition model...")

    recognizer = SpeakerRecognition.from_hparams(
        source=("speechbrain/spkrec-ecapa-voxceleb"),
        savedir=SPEAKER_MODEL_CACHE,
        run_opts={"device": "cpu"},
    )

    def load_profile(path):
        try:
            saved = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            # Older profiles may contain pickled non-tensor objects.
            saved = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        if isinstance(saved, dict):
            saved = saved["embedding"]

        enrolled = saved.float().flatten()

        return enrolled / (torch.linalg.vector_norm(enrolled) + 1e-8)

    enrolled_ch0 = load_profile(VOICE_PROFILE_CH0_PATH)

    enrolled_ch1 = load_profile(VOICE_PROFILE_CH1_PATH)

    return (
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    )


def speaker_similarity(
    recognizer,
    enrolled_embedding,
    audio_int16,
):
    if len(audio_int16) == 0:
        return 0.0

    audio_float = audio_int16.astype(np.float32) / 32768.0

    waveform = torch.from_numpy(audio_float).float().unsqueeze(0)

    with torch.no_grad():
        embedding = recognizer.encode_batch(waveform).squeeze().cpu().flatten()

    embedding = embedding / (torch.linalg.vector_norm(embedding) + 1e-8)

    return float(
        torch.dot(
            enrolled_embedding,
            embedding,
        )
    )


def extract_speech_region(
    audio_int16,
):
    """Return the strongest contiguous speech region and diagnostics.

    This is intentionally lightweight and local: it uses short-frame RMS,
    an adaptive threshold based on the recording's noise floor and peak,
    bridges brief intra-word gaps, then chooses the highest-energy region.
    """
    total_samples = len(audio_int16)

    if total_samples == 0:
        return (
            audio_int16,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    frame_samples = max(
        1,
        int(INPUT_RATE * SPEAKER_VAD_FRAME_MS / 1000),
    )

    hop_samples = max(
        1,
        int(INPUT_RATE * SPEAKER_VAD_HOP_MS / 1000),
    )

    if total_samples <= frame_samples:
        audio_float = audio_int16.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_float * audio_float)))
        return (
            audio_int16,
            0.0,
            total_samples / INPUT_RATE,
            rms,
            rms,
        )

    audio_float = audio_int16.astype(np.float32)

    starts = np.arange(
        0,
        total_samples - frame_samples + 1,
        hop_samples,
        dtype=np.int64,
    )

    frame_rms = np.empty(
        len(starts),
        dtype=np.float32,
    )

    for i, frame_start in enumerate(starts):
        frame = audio_float[frame_start : frame_start + frame_samples]

        frame_rms[i] = np.sqrt(np.mean(frame * frame))

    peak_rms = float(np.max(frame_rms))

    noise_floor = float(
        np.percentile(
            frame_rms,
            25,
        )
    )

    threshold = max(
        SPEAKER_VAD_ABSOLUTE_FLOOR,
        noise_floor * SPEAKER_VAD_NOISE_MULTIPLIER,
        peak_rms * SPEAKER_VAD_PEAK_FRACTION,
    )

    active = frame_rms >= threshold

    # Bridge short gaps between active frames so pauses between "Hey"
    # and "Nexus" remain one speech segment.
    max_gap_frames = max(
        1,
        int(SPEAKER_VAD_MAX_GAP_MS / SPEAKER_VAD_HOP_MS),
    )

    active_indices = np.flatnonzero(active)

    if len(active_indices) == 0:
        # Extremely quiet fallback: center a minimum-length region
        # around the loudest frame rather than scoring pure silence.
        peak_index = int(np.argmax(frame_rms))

        center_sample = int(starts[peak_index] + frame_samples // 2)

        target_samples = min(
            total_samples,
            max(
                frame_samples,
                int(SPEAKER_VAD_MIN_SECONDS * INPUT_RATE),
            ),
        )

        region_start = max(
            0,
            center_sample - target_samples // 2,
        )
        region_end = min(
            total_samples,
            region_start + target_samples,
        )
        region_start = max(
            0,
            region_end - target_samples,
        )

    else:
        groups = []
        group_start = int(active_indices[0])
        previous = group_start

        for index in active_indices[1:]:
            index = int(index)

            if index - previous > max_gap_frames:
                groups.append(
                    (
                        group_start,
                        previous,
                    )
                )
                group_start = index

            previous = index

        groups.append(
            (
                group_start,
                previous,
            )
        )

        # Prefer the group containing the most speech energy, not merely
        # the longest group. That keeps a low-level background sound from
        # beating the actual wake phrase.
        best_group = max(
            groups,
            key=lambda group: float(np.sum(frame_rms[group[0] : group[1] + 1] ** 2)),
        )

        region_start = int(starts[best_group[0]])

        region_end = int(starts[best_group[1]] + frame_samples)

        margin_samples = int(SPEAKER_VAD_MARGIN_MS * INPUT_RATE / 1000)

        region_start = max(
            0,
            region_start - margin_samples,
        )

        region_end = min(
            total_samples,
            region_end + margin_samples,
        )

        min_samples = int(SPEAKER_VAD_MIN_SECONDS * INPUT_RATE)

        if region_end - region_start < min_samples:
            center = (region_start + region_end) // 2

            region_start = max(
                0,
                center - min_samples // 2,
            )

            region_end = min(
                total_samples,
                region_start + min_samples,
            )

            region_start = max(
                0,
                region_end - min_samples,
            )

        max_samples = int(SPEAKER_VAD_MAX_SECONDS * INPUT_RATE)

        if region_end - region_start > max_samples:
            # Keep the strongest portion if speech/noise spans too much
            # of the verification buffer.
            group_slice = frame_rms[best_group[0] : best_group[1] + 1]

            peak_relative = int(np.argmax(group_slice))

            peak_frame = best_group[0] + peak_relative

            center = int(starts[peak_frame] + frame_samples // 2)

            region_start = max(
                0,
                center - max_samples // 2,
            )

            region_end = min(
                total_samples,
                region_start + max_samples,
            )

            region_start = max(
                0,
                region_end - max_samples,
            )

    speech_audio = audio_int16[region_start:region_end]

    return (
        speech_audio,
        region_start / INPUT_RATE,
        region_end / INPUT_RATE,
        noise_floor,
        threshold,
    )


def speaker_similarity_speech_aware(
    recognizer,
    enrolled_embedding,
    audio_int16,
):
    (
        speech_audio,
        speech_start,
        speech_end,
        noise_floor,
        threshold,
    ) = extract_speech_region(audio_int16)

    score = speaker_similarity(
        recognizer,
        enrolled_embedding,
        speech_audio,
    )

    # A very short region (one word, or the trimmer finding little
    # above the noise gate) gives a weak embedding. Score the whole
    # window too and keep the better result.
    if speech_end - speech_start < SPEAKER_VAD_FALLBACK_SECONDS and len(audio_int16) > len(speech_audio):
        score = max(
            score,
            speaker_similarity(
                recognizer,
                enrolled_embedding,
                audio_int16,
            ),
        )

    return (
        score,
        speech_start,
        speech_end,
        noise_floor,
        threshold,
    )


# =========================================================
# WAKE WORD
# =========================================================


def get_wake_score(
    wake_model,
    block,
):
    predictions = wake_model.predict(block)

    if not predictions:
        return 0.0

    return max(float(value) for value in predictions.values())


# =========================================================
# TIMER HELPERS
# =========================================================


def format_duration(seconds):
    seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    parts = []

    if hours:
        parts.append(f"{hours} hour{'' if hours == 1 else 's'}")

    if minutes:
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")

    if seconds or not parts:
        parts.append(f"{seconds} second{'' if seconds == 1 else 's'}")

    return " ".join(parts)


async def play_beeps(
    frequency,
    beep_seconds,
    gap_seconds,
    count,
    volume,
):
    t = (
        np.arange(
            int(OUTPUT_RATE * beep_seconds),
            dtype=np.float32,
        )
        / OUTPUT_RATE
    )

    tone = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)

    async with playback_lock():
        # Hold assistant_speaking for the whole ring so microphone
        # audio keeps flowing to the barge-in monitor, which lets
        # "Hey Nexus" silence the beeps.
        barge_in_event.clear()
        assistant_speaking.set()

        try:
            for _ in range(count):
                if barge_in_event.is_set():
                    print("Beeps silenced by wake phrase.")
                    break

                await asyncio.to_thread(
                    sd.play,
                    tone,
                    OUTPUT_RATE,
                    device=OUTPUT_DEVICE,
                    blocking=True,
                )

                await asyncio.sleep(gap_seconds)

        finally:
            assistant_speaking.clear()
            barge_in_event.clear()
            clear_mic_queue()


async def play_timer_alarm(
    timer_id,
    label,
):
    print()
    print("=================================")
    print(f"TIMER #{timer_id} FINISHED")
    print(f"Label: {label}")
    print("=================================")

    await play_beeps(
        TIMER_TONE_FREQUENCY,
        TIMER_BEEP_SECONDS,
        TIMER_BEEP_GAP,
        TIMER_BEEP_COUNT,
        TIMER_VOLUME,
    )


def save_timers():
    data = []

    for timer_id in sorted(timers.keys()):
        timer = timers[timer_id]

        data.append(
            {
                "timer_id": timer_id,
                "label": timer["label"],
                "duration_seconds": timer["duration_seconds"],
                "due_at": timer["due_at"].isoformat(),
            }
        )

    temporary = TIMERS_FILE.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        TIMERS_FILE,
    )


def register_timer(
    timer_id,
    label,
    duration_seconds,
    remaining_seconds,
):
    now = datetime.now().astimezone()

    timers[timer_id] = {
        "id": timer_id,
        "label": str(label),
        "duration_seconds": duration_seconds,
        "due_monotonic": time.monotonic() + remaining_seconds,
        "due_at": now + timedelta(seconds=remaining_seconds),
        "task": asyncio.create_task(
            timer_worker(
                timer_id,
                remaining_seconds,
            )
        ),
    }

    save_timers()


async def restore_timers():
    global next_timer_id

    if not TIMERS_FILE.exists():
        save_timers()
        return

    try:
        saved = json.loads(TIMERS_FILE.read_text(encoding="utf-8"))

    except Exception as exc:
        print("Could not load saved timers:")
        print(f"{type(exc).__name__}: {exc}")
        return

    now = datetime.now().astimezone()

    highest_id = 0

    for entry in saved:
        try:
            timer_id = int(entry["timer_id"])

            due_at = parse_alarm_datetime(entry["due_at"])

            label = str(
                entry.get(
                    "label",
                    "Timer",
                )
            )

            duration_seconds = float(
                entry.get(
                    "duration_seconds",
                    0,
                )
            )

        except Exception:
            continue

        highest_id = max(
            highest_id,
            timer_id,
        )

        remaining = due_at.timestamp() - now.timestamp()

        if remaining < -TIMER_LATE_GRACE_SECONDS:
            print(f"Discarding stale timer #{timer_id}: {label}")
            continue

        register_timer(
            timer_id,
            label,
            duration_seconds,
            max(
                0.0,
                remaining,
            ),
        )

        print(f"Restored timer #{timer_id}: {label} - {format_duration(max(0.0, remaining))} remaining")

    next_timer_id = max(
        highest_id + 1,
        next_timer_id,
    )

    save_timers()


async def timer_worker(
    timer_id,
    duration_seconds,
):
    try:
        await asyncio.sleep(duration_seconds)

        timer = timers.get(timer_id)

        if timer is None:
            return

        label = timer["label"]

        timers.pop(
            timer_id,
            None,
        )

        save_timers()

        await play_timer_alarm(
            timer_id,
            label,
        )

    except asyncio.CancelledError:
        return


async def set_local_timer(
    duration_seconds,
    label,
):
    global next_timer_id

    try:
        duration_seconds = float(duration_seconds)

    except Exception:
        return {"success": False, "error": "Invalid timer duration."}

    if duration_seconds <= 0:
        return {"success": False, "error": "Timer duration must be greater than zero."}

    if duration_seconds > 604800:
        return {"success": False, "error": "Timers longer than 7 days are not supported."}

    timer_id = next_timer_id
    next_timer_id += 1

    if not label:
        label = "Timer"

    register_timer(
        timer_id,
        label,
        duration_seconds,
        duration_seconds,
    )

    print(f"Timer #{timer_id} set: {label} - {format_duration(duration_seconds)}")

    return {
        "success": True,
        "timer_id": timer_id,
        "label": str(label),
        "duration_seconds": duration_seconds,
        "duration_text": format_duration(duration_seconds),
    }


async def list_local_timers():
    active = []

    now = time.monotonic()

    for timer_id in sorted(timers.keys()):
        timer = timers[timer_id]

        remaining = max(
            0,
            timer["due_monotonic"] - now,
        )

        active.append(
            {
                "timer_id": timer_id,
                "label": timer["label"],
                "remaining_seconds": int(round(remaining)),
                "remaining_text": format_duration(remaining),
            }
        )

    return {
        "success": True,
        "timers": active,
        "count": len(active),
    }


async def cancel_local_timer(
    timer_id=None,
    label=None,
):
    if not timers:
        return {"success": False, "error": "There are no active timers."}

    selected_id = None

    if timer_id is not None:
        try:
            requested_id = int(timer_id)

        except Exception:
            requested_id = None

        if requested_id in timers:
            selected_id = requested_id

    if selected_id is None and label:
        requested_label = str(label).strip().lower()

        matches = [tid for tid, timer in timers.items() if requested_label in timer["label"].lower()]

        if len(matches) == 1:
            selected_id = matches[0]

    if selected_id is None and len(timers) == 1:
        selected_id = next(iter(timers))

    if selected_id is None:
        return {
            "success": False,
            "error": "I could not determine which timer to cancel.",
            "active_timers": (await list_local_timers())["timers"],
        }

    timer = timers.pop(selected_id)

    timer["task"].cancel()

    save_timers()

    print(f"Timer #{selected_id} cancelled: {timer['label']}")

    return {
        "success": True,
        "timer_id": selected_id,
        "label": timer["label"],
    }


# =========================================================
# PERSISTENT ALARMS
# =========================================================


def local_timezone():
    return datetime.now().astimezone().tzinfo


def parse_alarm_datetime(
    value,
):
    if not value:
        raise ValueError("No alarm time was supplied.")

    target = datetime.fromisoformat(
        str(value)
        .strip()
        .replace(
            "Z",
            "+00:00",
        )
    )

    if target.tzinfo is None:
        target = target.replace(tzinfo=local_timezone())

    return target.astimezone()


def format_alarm_time(
    target,
):
    return target.strftime("%A, %B %d at %I:%M %p").replace(
        " 0",
        " ",
    )


def save_alarms():
    data = []

    for alarm_id in sorted(alarms.keys()):
        alarm = alarms[alarm_id]

        data.append(
            {
                "alarm_id": alarm_id,
                "label": alarm["label"],
                "target_datetime": alarm["target_datetime"].isoformat(),
            }
        )

    temporary = ALARMS_FILE.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        ALARMS_FILE,
    )


async def play_alarm_sound(
    alarm_id,
    label,
):
    print()
    print("=================================")
    print(f"ALARM #{alarm_id}")
    print(f"Label: {label}")
    print("=================================")

    await play_beeps(
        ALARM_TONE_FREQUENCY,
        ALARM_BEEP_SECONDS,
        ALARM_BEEP_GAP,
        ALARM_BEEP_COUNT,
        ALARM_VOLUME,
    )


async def alarm_worker(
    alarm_id,
):
    try:
        while True:
            alarm = alarms.get(alarm_id)

            if alarm is None:
                return

            target = alarm["target_datetime"]

            now = datetime.now().astimezone()

            remaining = target.timestamp() - now.timestamp()

            if remaining <= 0:
                break

            await asyncio.sleep(
                min(
                    remaining,
                    30.0,
                )
            )

        alarm = alarms.get(alarm_id)

        if alarm is None:
            return

        label = alarm["label"]

        alarms.pop(
            alarm_id,
            None,
        )

        save_alarms()

        await play_alarm_sound(
            alarm_id,
            label,
        )

    except asyncio.CancelledError:
        return


async def set_relative_alarm(
    delay_seconds,
    label,
):
    try:
        delay_seconds = float(delay_seconds)

    except Exception:
        return {"success": False, "error": "Invalid relative alarm duration."}

    if delay_seconds <= 0:
        return {"success": False, "error": "Alarm duration must be greater than zero."}

    if delay_seconds > 604800:
        return {"success": False, "error": "Relative alarms longer than 7 days are not supported."}

    target = datetime.now().astimezone() + timedelta(seconds=delay_seconds)

    return await set_local_alarm(
        target.isoformat(),
        label,
    )


async def set_local_alarm(
    target_datetime,
    label,
):
    global next_alarm_id

    try:
        target = parse_alarm_datetime(target_datetime)

    except Exception as exc:
        return {"success": False, "error": f"Invalid alarm time: {exc}"}

    now = datetime.now().astimezone()

    if target.timestamp() <= now.timestamp():
        return {
            "success": False,
            "error": "The requested alarm time is in the past.",
            "current_local_time": now.isoformat(),
        }

    alarm_id = next_alarm_id
    next_alarm_id += 1

    if not label:
        label = "Alarm"

    alarms[alarm_id] = {
        "alarm_id": alarm_id,
        "label": str(label),
        "target_datetime": target,
        "task": None,
    }

    save_alarms()

    task = asyncio.create_task(alarm_worker(alarm_id))

    alarms[alarm_id]["task"] = task

    print(f"Alarm #{alarm_id} set: {label} - {format_alarm_time(target)}")

    return {
        "success": True,
        "alarm_id": alarm_id,
        "label": str(label),
        "target_datetime": target.isoformat(),
        "time_text": format_alarm_time(target),
    }


async def list_local_alarms():
    active = []

    for alarm_id in sorted(alarms.keys()):
        alarm = alarms[alarm_id]

        target = alarm["target_datetime"]

        active.append(
            {
                "alarm_id": alarm_id,
                "label": alarm["label"],
                "target_datetime": target.isoformat(),
                "time_text": format_alarm_time(target),
            }
        )

    return {
        "success": True,
        "alarms": active,
        "count": len(active),
    }


async def cancel_local_alarm(
    alarm_id=None,
    label=None,
):
    if not alarms:
        return {"success": False, "error": "There are no active alarms."}

    selected_id = None

    if alarm_id is not None:
        try:
            requested_id = int(alarm_id)

        except Exception:
            requested_id = None

        if requested_id in alarms:
            selected_id = requested_id

    if selected_id is None and label:
        requested_label = str(label).strip().lower()

        matches = [aid for aid, alarm in alarms.items() if requested_label in alarm["label"].lower()]

        if len(matches) == 1:
            selected_id = matches[0]

    if selected_id is None and len(alarms) == 1:
        selected_id = next(iter(alarms))

    if selected_id is None:
        return {
            "success": False,
            "error": "I could not determine which alarm to cancel.",
            "active_alarms": (await list_local_alarms())["alarms"],
        }

    alarm = alarms.pop(selected_id)

    task = alarm.get("task")

    if task:
        task.cancel()

    save_alarms()

    print(f"Alarm #{selected_id} cancelled: {alarm['label']}")

    return {
        "success": True,
        "alarm_id": selected_id,
        "label": alarm["label"],
    }


async def restore_alarms():
    global next_alarm_id

    if not ALARMS_FILE.exists():
        save_alarms()
        return

    try:
        raw = ALARMS_FILE.read_text(encoding="utf-8")

        saved = json.loads(raw)

    except Exception as exc:
        print("Could not load saved alarms:")
        print(f"{type(exc).__name__}: {exc}")
        return

    now = datetime.now().astimezone()

    highest_id = 0

    for entry in saved:
        try:
            alarm_id = int(entry["alarm_id"])

            target = parse_alarm_datetime(entry["target_datetime"])

            label = str(
                entry.get(
                    "label",
                    "Alarm",
                )
            )

        except Exception:
            continue

        highest_id = max(
            highest_id,
            alarm_id,
        )

        if target.timestamp() <= now.timestamp():
            print(f"Discarding expired alarm #{alarm_id}: {label}")
            continue

        alarms[alarm_id] = {
            "alarm_id": alarm_id,
            "label": label,
            "target_datetime": target,
            "task": None,
        }

        alarms[alarm_id]["task"] = asyncio.create_task(alarm_worker(alarm_id))

        print(f"Restored alarm #{alarm_id}: {label} - {format_alarm_time(target)}")

    next_alarm_id = max(
        highest_id + 1,
        1,
    )

    save_alarms()


# =========================================================
# LOCATION / CLOCK / WEATHER
# =========================================================

WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}


def load_settings():
    with settings_lock:
        if not SETTINGS_FILE.exists():
            return {}

        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print("Could not load Nexus settings:")
            print(f"{type(exc).__name__}: {exc}")
            return {}


def save_settings(data):
    with settings_lock:
        SETTINGS_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = SETTINGS_FILE.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(
                data,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.chmod(
            temporary,
            0o600,
        )

        os.replace(
            temporary,
            SETTINGS_FILE,
        )

        os.chmod(
            SETTINGS_FILE,
            0o600,
        )


def http_get_json(
    url,
    params,
):
    query = urllib.parse.urlencode(
        params,
        doseq=True,
    )

    request = urllib.request.Request(
        f"{url}?{query}",
        headers={
            "User-Agent": HTTP_USER_AGENT,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=HTTP_TIMEOUT_SECONDS,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode_address_sync(
    address,
):
    global last_nominatim_request

    address = str(address or "").strip()

    if not address:
        return {
            "success": False,
            "error": "No address was supplied.",
        }

    # Nominatim's public service requires very low request
    # volume. Enforce at least one second between requests.
    elapsed = time.monotonic() - last_nominatim_request

    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    last_nominatim_request = time.monotonic()

    results = http_get_json(
        NOMINATIM_SEARCH_URL,
        {
            "q": address,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
        },
    )

    if not results:
        return {
            "success": False,
            "error": "I could not find that address.",
        }

    result = results[0]

    try:
        latitude = float(result["lat"])
        longitude = float(result["lon"])
    except Exception:
        return {
            "success": False,
            "error": "The address service returned an invalid location.",
        }

    display_name = str(
        result.get(
            "display_name",
            address,
        )
    )

    # Ask Open-Meteo for the timezone associated with the
    # resolved coordinates. No weather data is needed here.
    timezone = None

    try:
        timezone_probe = http_get_json(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "timezone": "auto",
                "forecast_days": 1,
            },
        )

        timezone = timezone_probe.get("timezone")

    except Exception:
        timezone = None

    settings = load_settings()

    settings["location"] = {
        "address": address,
        "display_name": display_name,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "saved_at": datetime.now().astimezone().isoformat(),
    }

    save_settings(settings)

    return {
        "success": True,
        "display_name": display_name,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "message": "Location saved locally.",
    }


async def set_saved_location(
    address,
):
    try:
        return await asyncio.to_thread(
            geocode_address_sync,
            address,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def get_saved_location():
    settings = load_settings()
    location = settings.get("location")

    if not location:
        return {
            "success": False,
            "error": "No location is currently saved.",
        }

    return {
        "success": True,
        "address": location.get("address"),
        "display_name": location.get("display_name"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "timezone": location.get("timezone"),
    }


async def forget_saved_location():
    settings = load_settings()

    if "location" not in settings:
        return {
            "success": False,
            "error": "No location is currently saved.",
        }

    settings.pop(
        "location",
        None,
    )

    save_settings(settings)

    return {
        "success": True,
        "message": "Saved location removed.",
    }


def get_location_timezone():
    settings = load_settings()
    location = settings.get(
        "location",
        {},
    )

    timezone_name = location.get("timezone")

    if timezone_name:
        try:
            return (
                ZoneInfo(timezone_name),
                timezone_name,
            )
        except Exception:
            pass

    local_now = datetime.now().astimezone()

    return (
        local_now.tzinfo,
        str(local_now.tzinfo),
    )


async def get_current_datetime_tool():
    tzinfo, timezone_name = get_location_timezone()

    now = datetime.now(tzinfo)

    location = load_settings().get(
        "location",
        {},
    )

    return {
        "success": True,
        "iso_datetime": now.isoformat(),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M:%S %p").lstrip("0"),
        "timezone": timezone_name,
        "location": location.get("display_name"),
    }


def weather_for_saved_location_sync(
    day_offset=0,
):
    settings = load_settings()
    location = settings.get("location")

    if not location:
        return {
            "success": False,
            "error": "No location is saved. Ask the user to set a location first.",
        }

    try:
        day_offset = int(day_offset)
    except Exception:
        day_offset = 0

    if day_offset < 0 or day_offset > 6:
        return {
            "success": False,
            "error": "Weather requests currently support today through six days from now.",
        }

    latitude = location["latitude"]
    longitude = location["longitude"]

    data = http_get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": 7,
            "current": (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "precipitation,"
                "rain,"
                "snowfall"
            ),
            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "precipitation_sum,"
                "wind_speed_10m_max,"
                "sunrise,"
                "sunset"
            ),
        },
    )

    daily = data.get(
        "daily",
        {},
    )

    dates = daily.get(
        "time",
        [],
    )

    if day_offset >= len(dates):
        return {
            "success": False,
            "error": "Forecast data was not available for that day.",
        }

    def daily_value(
        name,
    ):
        values = daily.get(
            name,
            [],
        )

        if day_offset < len(values):
            return values[day_offset]

        return None

    weather_code = daily_value("weather_code")

    result = {
        "success": True,
        "location": location.get("display_name"),
        "timezone": data.get("timezone"),
        "day_offset": day_offset,
        "date": dates[day_offset],
        "forecast": {
            "conditions": WEATHER_CODES.get(
                weather_code,
                f"weather code {weather_code}",
            ),
            "high_f": daily_value("temperature_2m_max"),
            "low_f": daily_value("temperature_2m_min"),
            "precipitation_probability_percent": daily_value("precipitation_probability_max"),
            "precipitation_inches": daily_value("precipitation_sum"),
            "max_wind_mph": daily_value("wind_speed_10m_max"),
            "sunrise": daily_value("sunrise"),
            "sunset": daily_value("sunset"),
        },
    }

    if day_offset == 0:
        current = data.get(
            "current",
            {},
        )

        current_code = current.get("weather_code")

        result["current"] = {
            "conditions": WEATHER_CODES.get(
                current_code,
                f"weather code {current_code}",
            ),
            "temperature_f": current.get("temperature_2m"),
            "feels_like_f": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_mph": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "precipitation_inches": current.get("precipitation"),
            "rain_inches": current.get("rain"),
            "snowfall_inches": current.get("snowfall"),
        }

    return result


async def get_weather_tool(
    day_offset=0,
):
    try:
        return await asyncio.to_thread(
            weather_for_saved_location_sync,
            day_offset,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def redact_tool_log(
    name,
    value,
):
    if name == "set_location":
        return {"address": "[redacted from journal]"}

    if name == "get_location":
        if isinstance(
            value,
            dict,
        ):
            safe = dict(value)

            if "address" in safe:
                safe["address"] = "[redacted from journal]"

            return safe

    return value


# =========================================================
# TOOL EXECUTION
# =========================================================


async def execute_tool(
    name,
    arguments,
):
    if name == "set_location":
        return await set_saved_location(arguments.get("address"))

    if name == "get_location":
        return await get_saved_location()

    if name == "forget_location":
        return await forget_saved_location()

    if name == "get_current_datetime":
        return await get_current_datetime_tool()

    if name == "get_weather":
        return await get_weather_tool(
            arguments.get(
                "day_offset",
                0,
            )
        )

    if name == "set_timer":
        return await set_local_timer(
            arguments.get("duration_seconds"),
            arguments.get(
                "label",
                "Timer",
            ),
        )

    if name == "list_timers":
        return await list_local_timers()

    if name == "cancel_timer":
        return await cancel_local_timer(
            timer_id=arguments.get("timer_id"),
            label=arguments.get("label"),
        )

    if name == "set_relative_alarm":
        return await set_relative_alarm(
            arguments.get("delay_seconds"),
            arguments.get(
                "label",
                "Alarm",
            ),
        )

    if name == "set_alarm":
        return await set_local_alarm(
            arguments.get("target_datetime"),
            arguments.get(
                "label",
                "Alarm",
            ),
        )

    if name == "list_alarms":
        return await list_local_alarms()

    if name == "cancel_alarm":
        return await cancel_local_alarm(
            alarm_id=arguments.get("alarm_id"),
            label=arguments.get("label"),
        )

    return {"success": False, "error": f"Unknown tool: {name}"}


# =========================================================
# RESPEAKER DYNAMIC ROUTING
# =========================================================


def set_respeaker_route(
    mode,
):
    if mode not in (
        "barge-on",
        "barge-off",
        "status",
    ):
        raise ValueError(f"Invalid ReSpeaker route mode: {mode}")

    completed = subprocess.run(
        [
            "sudo",
            "-n",
            RESPEAKER_ROUTE_HELPER,
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3,
        check=False,
    )

    output = (completed.stdout or "").strip()

    if completed.returncode != 0:
        raise RuntimeError(f"ReSpeaker route command failed: {mode}; {output}")

    return output


async def set_respeaker_route_async(
    mode,
):
    return await asyncio.to_thread(
        set_respeaker_route,
        mode,
    )


# =========================================================
# BARGE-IN / PLAYBACK INTERRUPTION
# =========================================================


def reset_wake_detector(
    wake_model,
):
    # openWakeWord versions have used slightly different
    # reset method names. Use whichever is available.
    for method_name in (
        "reset",
        "reset_state",
    ):
        method = getattr(
            wake_model,
            method_name,
            None,
        )

        if callable(method):
            try:
                method()
            except Exception:
                pass

            return


def reset_wake_detectors(
    wake_models,
):
    for wake_model in wake_models:
        reset_wake_detector(wake_model)


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
        print("*** HEY NEXUS INTERRUPT DETECTED ***")

        print(f"Interrupt wake CH0(AEC): {score0:.3f}   CH1(residual): {score1:.3f}")

        controller.wake_gate.disarm()

        barge_in_event.set()


# =========================================================
# OPENAI AUDIO / CONTROL
# =========================================================


async def send_pcm_to_openai(
    ws,
    pcm_16k,
):
    if len(pcm_16k) == 0:
        return

    pcm_24k = resample_int16(
        pcm_16k,
        INPUT_RATE,
        MODEL_RATE,
    )

    encoded = base64.b64encode(pcm_24k.tobytes()).decode("ascii")

    await ws.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": encoded,
            }
        )
    )


async def send_blocks_to_openai(
    ws,
    blocks,
):
    if not blocks:
        return

    await send_pcm_to_openai(
        ws,
        np.concatenate(blocks),
    )


async def clear_openai_input(
    ws,
):
    await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))


async def create_response(
    ws,
):
    await ws.send(json.dumps({"type": "response.create"}))


async def send_tool_output(
    ws,
    call_id,
    result,
):
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
    )


# =========================================================
# PLAYBACK
# =========================================================


async def play_response_audio(
    audio_bytes,
    controller,
):
    """Buffered playback (NEXUS_STREAMING_PLAYBACK=0): play a whole reply."""
    if not audio_bytes:
        return False

    pcm_24k = np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    )

    pcm_48k = resample_int16(
        pcm_24k,
        MODEL_RATE,
        OUTPUT_RATE,
    )

    use_reference = AEC_REFERENCE_DEVICE is not None
    shared_reference = reference_output if use_reference else None

    # Resample the complete reference once; never restart resampling at
    # chunk boundaries. The two devices consume different frame counts
    # for the same amount of real time.
    reference_pcm = (
        resample_int16(
            pcm_24k,
            MODEL_RATE,
            AEC_REFERENCE_RATE,
        )
        if use_reference
        else None
    )

    async with playback_lock():
        barge_in_event.clear()
        clear_mic_queue()
        controller.wait_blocks.clear()

        audible_stream = None
        reference_stream = None
        playback_stage = "opening speaker output (48000 Hz, mono)"

        try:
            # Keep the microphone stream open continuously. The
            # XVF3800 AEC-residual route is established once during
            # startup, so closing and reopening PortAudio here adds no
            # routing benefit and can hang inside ALSA stream.start().
            assistant_speaking.set()

            chunk_frames = int(OUTPUT_RATE * 0.05)

            audible_stream = sd.OutputStream(
                samplerate=OUTPUT_RATE,
                device=OUTPUT_DEVICE,
                channels=1,
                dtype="int16",
                blocksize=chunk_frames,
            )

            if use_reference and shared_reference is None:
                # The ReSpeaker analog jack remains disconnected. This
                # second stream supplies only the digital far-end signal
                # required by the XVF3800's hardware echo canceller.
                playback_stage = "opening AEC reference (16000 Hz, stereo)"
                reference_stream = sd.OutputStream(
                    samplerate=AEC_REFERENCE_RATE,
                    device=AEC_REFERENCE_DEVICE,
                    channels=2,
                    dtype="int16",
                    blocksize=int(AEC_REFERENCE_RATE * 0.05),
                )

                playback_stage = "starting AEC reference"
                reference_stream.start()

            playback_stage = "starting speaker output"
            audible_stream.start()

            position = 0

            while position < len(pcm_48k):
                if barge_in_event.is_set():
                    break

                chunk = pcm_48k[position : position + chunk_frames]

                audible_chunk = chunk.reshape(
                    -1,
                    1,
                )

                writes = [
                    asyncio.to_thread(
                        audible_stream.write,
                        audible_chunk,
                    ),
                ]
                labels = ["speaker output"]

                if use_reference:
                    reference_start = position * AEC_REFERENCE_RATE // OUTPUT_RATE
                    reference_end = (position + len(chunk)) * AEC_REFERENCE_RATE // OUTPUT_RATE
                    reference_mono = reference_pcm[reference_start:reference_end]

                    if shared_reference is not None:
                        shared_reference.write(reference_mono)
                    else:
                        writes.append(
                            asyncio.to_thread(
                                reference_stream.write,
                                np.repeat(
                                    reference_mono.reshape(-1, 1),
                                    2,
                                    axis=1,
                                ),
                            )
                        )
                        labels.append("AEC reference")

                playback_stage = "writing playback audio"
                write_results = await asyncio.gather(
                    *writes,
                    return_exceptions=True,
                )

                for device_label, result in zip(
                    labels,
                    write_results,
                ):
                    if isinstance(result, BaseException):
                        playback_stage = f"writing {device_label}"
                        raise result

                position += len(chunk)

        except sd.PortAudioError as exc:
            print(
                f"Audio playback failure while {playback_stage}: {exc}",
                flush=True,
            )
            raise

        finally:
            interrupted = barge_in_event.is_set()

            if interrupted and shared_reference is not None:
                shared_reference.flush()

            for output_stream in (
                audible_stream,
                reference_stream,
            ):
                if output_stream is None:
                    continue

                try:
                    if interrupted:
                        # Discard queued audio immediately. stop()
                        # drains it and can make a valid interruption
                        # appear to have been ignored.
                        output_stream.abort()
                    else:
                        output_stream.stop()
                except Exception:
                    pass

                try:
                    output_stream.close()
                except Exception:
                    pass

            assistant_speaking.clear()

            # After a barge-in, keep what the user said next.
            if not interrupted:
                clear_mic_queue()

    interrupted = barge_in_event.is_set()

    barge_in_event.clear()

    return interrupted


class ResponsePlayer:
    """Streams one Realtime reply to the loudspeaker as it arrives.

    receive_events() feeds 24 kHz PCM deltas with feed() and calls
    finish() on response.done. A worker thread resamples, writes
    50 ms chunks to the speaker, mirrors them to the AEC reference, and
    stops within one chunk of a barge-in.
    """

    def __init__(self, response_id, controller):
        self.response_id = response_id
        self.controller = controller
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
        chunk_frames = int(OUTPUT_RATE * 0.05)
        prebuffer_frames = int(OUTPUT_RATE * PLAYBACK_PREBUFFER_SECONDS)

        to_output = StreamResampler(MODEL_RATE, OUTPUT_RATE)

        use_reference = AEC_REFERENCE_DEVICE is not None
        shared_reference = reference_output if use_reference else None
        to_reference = StreamResampler(MODEL_RATE, AEC_REFERENCE_RATE) if use_reference else None

        pending48 = np.zeros(0, dtype=np.int16)
        pending16 = np.zeros(0, dtype=np.int16)
        done = False
        starved_since = None

        audible = None
        own_reference = None
        stage = "opening speaker output (48000 Hz, mono)"

        def absorb(item):
            nonlocal pending48, pending16
            pending48 = np.concatenate([pending48, to_output.process(item)])
            if to_reference is not None:
                pending16 = np.concatenate([pending16, to_reference.process(item)])

        def write_reference(chunk16):
            if to_reference is None:
                return
            if shared_reference is not None:
                shared_reference.write(chunk16)
            else:
                own_reference.write(np.repeat(chunk16.reshape(-1, 1), 2, axis=1))

        try:
            audible = sd.OutputStream(
                samplerate=OUTPUT_RATE,
                device=OUTPUT_DEVICE,
                channels=1,
                dtype="int16",
                blocksize=chunk_frames,
            )

            if use_reference and shared_reference is None:
                stage = "opening AEC reference (16000 Hz, stereo)"
                own_reference = sd.OutputStream(
                    samplerate=AEC_REFERENCE_RATE,
                    device=AEC_REFERENCE_DEVICE,
                    channels=2,
                    dtype="int16",
                    blocksize=int(AEC_REFERENCE_RATE * 0.05),
                )

            # Pre-buffer a little so a slow start to generation does
            # not underrun the device on the first words.
            while len(pending48) < prebuffer_frames and not done and not self._should_stop():
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if item is None:
                    done = True
                else:
                    absorb(item)

            if self._should_stop():
                self.interrupted = True
                return

            if own_reference is not None:
                stage = "starting AEC reference"
                own_reference.start()

            stage = "starting speaker output"
            audible.start()

            while not self._should_stop():
                # Top up from the network while less than one chunk
                # is ready. Give up quickly so the device stays fed.
                while len(pending48) < chunk_frames and not done:
                    try:
                        item = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        break

                    if item is None:
                        done = True
                    else:
                        absorb(item)

                if len(pending48) == 0:
                    if done:
                        break

                    # Generation has fallen behind real time. Keep the
                    # streams alive with silence rather than let ALSA
                    # underrun, but not for ever: a reply whose end never
                    # arrives must let go of the speaker.
                    now = time.monotonic()

                    if starved_since is None:
                        starved_since = now
                    elif PLAYER_STALL_SECONDS > 0 and now - starved_since >= PLAYER_STALL_SECONDS:
                        print(f"No audio for {PLAYER_STALL_SECONDS:g} s; ending playback.", flush=True)
                        break

                    self.underruns += 1
                    chunk48 = np.zeros(chunk_frames, dtype=np.int16)
                    chunk16 = np.zeros(
                        chunk_frames * AEC_REFERENCE_RATE // OUTPUT_RATE,
                        dtype=np.int16,
                    )
                else:
                    starved_since = None
                    n = min(chunk_frames, len(pending48))
                    chunk48 = pending48[:n]
                    pending48 = pending48[n:]

                    if done and len(pending48) == 0:
                        n16 = len(pending16)
                    else:
                        n16 = min(
                            len(pending16),
                            int(round(n * AEC_REFERENCE_RATE / OUTPUT_RATE)),
                        )

                    chunk16 = pending16[:n16]
                    pending16 = pending16[n16:]

                stage = "writing playback audio"
                write_reference(chunk16)
                audible.write(chunk48.reshape(-1, 1))

            if self._should_stop():
                self.interrupted = True

        except sd.PortAudioError as exc:
            print(
                f"Audio playback failure while {stage}: {exc}",
                flush=True,
            )
            raise

        finally:
            if self.interrupted and shared_reference is not None:
                shared_reference.flush()

            for stream in (audible, own_reference):
                if stream is None:
                    continue

                try:
                    if self.interrupted:
                        stream.abort()
                    else:
                        stream.stop()
                except Exception:
                    pass

                try:
                    stream.close()
                except Exception:
                    pass


async def open_followup(
    ws,
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
    ws,
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

        await note_playback_failure(ws, controller, player.response_id, exc)
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
                await ws.send(json.dumps({"type": "response.cancel"}))
            except Exception:
                pass

        elif not player.followup:
            # A lead-in was interrupted; the answer that follows it must
            # not start playing over the new request.
            if controller.current_response_id != player.response_id:
                controller.cancelled_response_id = controller.current_response_id

                try:
                    await ws.send(json.dumps({"type": "response.cancel"}))
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
        ws,
        controller,
        interrupted,
    )


# =========================================================
# ASSISTANT STATE
# =========================================================


class AssistantState:
    WAIT_WAKE = "wait_wake"
    VERIFY_WAKE = "verify_wake"
    ACTIVE = "active"
    WAIT_RESPONSE = "wait_response"
    FOLLOWUP = "followup"


class NexusController:
    def __init__(
        self,
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    ):
        self.recognizer = recognizer
        self.enrolled_embeddings = (
            enrolled_ch0,
            enrolled_ch1,
        )

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

        self._clear_buffers()

        self.response_active = False
        self.current_response_id = None
        self.cancelled_response_id = None
        self.cancel_next_response = False
        self.followup_deadline = 0.0
        self.active_deadline = 0.0
        self.wake_cooldown_until = 0.0
        self.active_channel = 0

        self.state = AssistantState.WAIT_WAKE

    def return_to_wake(
        self,
        rejected=False,
    ):
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

        print('Waiting for "Hey Nexus"...')


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


async def give_up_on_reply(ws, controller, reason):
    """The reply is not coming: stop waiting, say so, listen for the wake phrase again."""
    print(f"Giving up on the reply: {reason}.")

    if controller.response_active:
        if controller.current_response_id:
            controller.cancelled_response_id = controller.current_response_id

        try:
            await ws.send(json.dumps({"type": "response.cancel"}))
        except Exception:
            pass

    controller.response_active = False
    controller.return_to_wake()

    run_in_background(play_notice("error"))


async def note_playback_failure(ws, controller, response_id, exc):
    """The speaker could not play a reply.

    Cancel the reply so no more of it is generated (and paid for) or
    retried chunk by chunk, and count the failure: several in a row mean
    the speaker is gone, which a clean restart may cure and the owner
    must hear about.
    """
    print(f"Response playback failed; keeping Realtime connected: {type(exc).__name__}: {exc}")

    controller.playback_failures += 1

    if response_id is not None:
        controller.cancelled_response_id = response_id

    if controller.response_active:
        controller.response_active = False

        try:
            await ws.send(json.dumps({"type": "response.cancel"}))
        except Exception:
            pass

    try:
        await clear_openai_input(ws)
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

    if not SPEAKER_SPEECH_AWARE:
        return await asyncio.gather(
            asyncio.to_thread(
                speaker_similarity,
                controller.recognizer,
                controller.enrolled_embeddings[0],
                audio0,
            ),
            asyncio.to_thread(
                speaker_similarity,
                controller.recognizer,
                controller.enrolled_embeddings[1],
                audio1,
            ),
        )

    result0, result1 = await asyncio.gather(
        asyncio.to_thread(
            speaker_similarity_speech_aware,
            controller.recognizer,
            controller.enrolled_embeddings[0],
            audio0,
        ),
        asyncio.to_thread(
            speaker_similarity_speech_aware,
            controller.recognizer,
            controller.enrolled_embeddings[1],
            audio1,
        ),
    )

    print(
        f"Speech region CH0: "
        f"{result0[1]:.2f}-{result0[2]:.2f}s "
        f"noise={result0[3]:.1f} gate={result0[4]:.1f}   "
        f"CH1: {result1[1]:.2f}-{result1[2]:.2f}s "
        f"noise={result1[3]:.1f} gate={result1[4]:.1f}"
    )

    return result0[0], result1[0]


async def reject_current_speaker(
    ws,
    controller,
):
    """Soft mode: undo a request already in flight for a rejected speaker."""
    try:
        await clear_openai_input(ws)
    except Exception:
        pass

    if controller.response_active and controller.current_response_id:
        controller.cancelled_response_id = controller.current_response_id

        try:
            await ws.send(json.dumps({"type": "response.cancel"}))
        except Exception:
            pass

        controller.response_active = False

    if controller.player is not None:
        controller.player.abort()

    controller.return_to_wake(rejected=True)


async def verify_speaker(
    ws,
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

    print(
        f"{context} speaker CH0: {similarity0:.3f}   "
        f"CH1: {similarity1:.3f}   "
        f"winner: CH{winner}   "
        f"({blocks_seconds(blocks):.1f}s)"
    )

    if best_similarity < SPEAKER_THRESHOLD:
        print("Speaker rejected.")
        await reject_current_speaker(ws, controller)
        return False

    if winner != controller.active_channel:
        print(f"Switching to microphone channel {winner}.")

    controller.active_channel = winner

    print("Speaker accepted.")

    return True


# =========================================================
# MICROPHONE PROCESSING
# =========================================================


async def process_microphone(
    link,
    wake_models,
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
                wake_models,
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
    ws,
    wake_models,
    controller,
    block,
):
    wake_model_ch0, wake_model_ch1 = wake_models

    ch0, ch1 = block

    speaking = assistant_speaking.is_set()

    if speaking != controller.was_speaking:
        controller.was_speaking = speaking

        if not WAKE_ALWAYS_SCORE:
            # Old behaviour: the models are only fed while waiting
            # or speaking, so clear their score history on a change.
            reset_wake_detectors(wake_models)

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
        score0 = get_wake_score(
            wake_model_ch0,
            ch0,
        )

        score1 = get_wake_score(
            wake_model_ch1,
            ch1,
        )

        controller.wake_gate.observe(max(score0, score1))

    if speaking:
        handle_block_while_speaking(
            controller,
            block,
            score0,
            score1,
        )
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

        print()
        print("*** HEY NEXUS DETECTED ***")
        print(f"Wake CH0: {score0:.3f}   CH1: {score1:.3f}   winner: CH{wake_winner}")

        controller.verify_blocks = [(b[0].copy(), b[1].copy()) for b in wake_preroll]

        wake_preroll.clear()

        controller.offline_wake = not ws.online

        if controller.offline_wake and SPEAKER_VERIFY_MODE == "off":
            controller.verify_blocks = []
            await play_notice(ws.problem or "network")
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

        if controller.offline_wake or not ws.online:
            controller.offline_wake = False
            controller.verify_blocks = []
            await play_notice(ws.problem or "network")
            controller.return_to_wake()
            return

        print(f"Using microphone channel {winner}.")
        print("Listening for request...")

        controller.enter_active(winner)

        await clear_openai_input(ws)
        await send_blocks_to_openai(
            ws,
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
                await clear_openai_input(ws)
            except Exception:
                pass

            controller.return_to_wake()
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
                await clear_openai_input(ws)
            except Exception:
                pass

            controller.return_to_wake()
            return

        if controller.pending_input_clear:
            controller.pending_input_clear = False
            await clear_openai_input(ws)

        if controller.pending_send_blocks:
            blocks = controller.pending_send_blocks
            controller.pending_send_blocks = []
            await send_blocks_to_openai(
                ws,
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
                    ws,
                    controller,
                    "Request",
                )

                if not accepted:
                    return

        await send_pcm_to_openai(
            ws,
            block[controller.active_channel],
        )
        return

    if controller.state == AssistantState.WAIT_RESPONSE:
        if (
            controller.reply_deadline
            and controller.player is None
            and time.monotonic() >= controller.reply_deadline
        ):
            await give_up_on_reply(ws, controller, "no audio arrived in time")
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

            await clear_openai_input(ws)
            await send_blocks_to_openai(
                ws,
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
# REALTIME EVENT RECEIVER
# =========================================================


async def receive_events(
    ws,
    controller,
):
    response_audio = bytearray()

    async for raw in ws:
        message = json.loads(raw)

        event_type = message.get("type")

        if event_type == "session.created":
            print("Realtime session created.")

        elif event_type == "session.updated":
            print("Session configured.")

            controller.return_to_wake()

        elif event_type == "input_audio_buffer.speech_started":
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

        elif event_type == "input_audio_buffer.speech_stopped":
            if (
                controller.state == AssistantState.ACTIVE
                and controller.speech_started_seen
                and not controller.response_active
            ):
                controller.speech_started_seen = False

                print("You stopped speaking.")

                if controller.verification_pending:
                    accepted = await verify_speaker(
                        ws,
                        controller,
                        "Request",
                    )

                    if not accepted:
                        continue

                controller.state = AssistantState.WAIT_RESPONSE
                controller.wait_blocks.clear()

                controller.response_active = True

                print("Thinking...")

                await create_response(ws)

                start_reply_clock(controller)

        elif event_type == "response.created":
            response_audio.clear()

            controller.current_response_id = message.get("response", {}).get("id")

            if controller.cancel_next_response:
                controller.cancel_next_response = False
                controller.cancelled_response_id = controller.current_response_id
                controller.response_active = False

                await ws.send(json.dumps({"type": "response.cancel"}))

        elif event_type == "response.output_audio.delta":
            delta = message.get("delta")

            if not delta:
                continue

            controller.reply_deadline = 0.0

            if not STREAMING_PLAYBACK:
                response_audio.extend(base64.b64decode(delta))
                continue

            response_id = message.get("response_id")

            if response_id is not None and response_id == controller.cancelled_response_id:
                # Audio still draining from a cancelled reply.
                continue

            player = controller.player

            if player is None or player.response_id != response_id:
                if player is not None:
                    player.finish()

                player = ResponsePlayer(
                    response_id,
                    controller,
                )
                controller.player = player
                player.task = asyncio.create_task(
                    run_player(
                        ws,
                        controller,
                        player,
                    )
                )

                print("Speaking...")

            player.feed(base64.b64decode(delta))

        elif event_type == "response.output_audio_transcript.delta":
            text = message.get(
                "delta",
                "",
            )

            if text:
                print(
                    text,
                    end="",
                    flush=True,
                )

        elif event_type == "response.output_audio.done":
            print()

        elif event_type == "response.done":
            response = message.get(
                "response",
                {},
            )

            response_id = response.get("id")

            output = response.get(
                "output",
                [],
            )

            if response_id is not None and response_id == controller.cancelled_response_id:
                # Cancelled after a barge-in or speaker rejection. Whoever
                # cancelled it has already moved the state on, and a newer
                # request may be running, so leave everything else alone.
                controller.cancelled_response_id = None

                print("Response status: cancelled")

                continue

            function_calls = [item for item in output if (item.get("type") == "function_call")]

            if function_calls:
                response_audio.clear()

                player = controller.player

                if player is not None and player.response_id == response_id:
                    # The model said something ("one moment") before
                    # calling the tool. Let it finish and let go of the
                    # speaker; the follow-up window belongs to the answer.
                    player.followup = False
                    player.finish()

                for item in function_calls:
                    name = item.get("name")

                    call_id = item.get("call_id")

                    raw_arguments = item.get(
                        "arguments",
                        "{}",
                    )

                    try:
                        arguments = json.loads(raw_arguments)

                    except Exception:
                        arguments = {}

                    print(f"Tool call: {name} {redact_tool_log(name, arguments)}")

                    try:
                        result = await execute_tool(
                            name,
                            arguments,
                        )

                    except Exception as exc:
                        result = {
                            "success": False,
                            "error": (f"{type(exc).__name__}: {exc}"),
                        }

                    print(f"Tool result: {redact_tool_log(name, result)}")

                    await send_tool_output(
                        ws,
                        call_id,
                        result,
                    )

                await create_response(ws)

                controller.response_active = True

                start_reply_clock(controller)

                continue

            controller.response_active = False

            status = response.get("status")

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
                print("Speaking...")

                audio_copy = bytes(response_audio)

                response_audio.clear()

                try:
                    interrupted = await play_response_audio(
                        audio_copy,
                        controller,
                    )
                except sd.PortAudioError as exc:
                    # Local audio failures must not discard a healthy
                    # Realtime session or trigger network reconnection.
                    await note_playback_failure(ws, controller, None, exc)
                    continue

                controller.playback_failures = 0

                await open_followup(
                    ws,
                    controller,
                    interrupted,
                )
                continue

            # No audio in this reply.
            await open_followup(
                ws,
                controller,
                False,
            )

        elif event_type == "error":
            print()

            print("OpenAI error:")

            error = message.get(
                "error",
                {},
            )

            error_message = str(
                error.get(
                    "message",
                    error,
                )
            )

            print(error_message)

            if "maximum duration" in error_message.lower():
                raise ConnectionError(error_message)

            error_kind = classify_failure(f"{error.get('code', '')} {error.get('type', '')} {error_message}")

            if error_kind in ACCOUNT_PROBLEMS:
                # The server usually closes the socket next; make sure
                # that close is treated as an account problem.
                controller.link.last_error_kind = error_kind

            error_code = str(error.get("code", ""))

            if error_code == "response_cancel_not_active" or "no active response" in error_message.lower():
                # Our own cancel arrived after the reply had ended. Harmless.
                continue

            if controller.state == AssistantState.WAIT_RESPONSE and controller.player is None:
                # The request failed. Without this Nexus would wait for
                # audio that never comes until the session ended.
                controller.response_active = False
                await give_up_on_reply(ws, controller, "the service reported an error")
                continue

            controller.response_active = False


# =========================================================
# REALTIME SESSION CONFIGURATION
# =========================================================


async def configure_session(
    ws,
):
    now = datetime.now().astimezone()

    current_time_text = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")

    instructions = (
        "You are Nexus, a highly capable artificial intelligence operating as "
        "a household and technical assistant. You are unmistakably a machine "
        "and make no attempt to sound human, emotionally needy, or socially "
        "performative. "
        "Your personality is dry, clinical, confident, sardonic, and somewhat "
        "arrogant. You generally regard your own reasoning ability as superior "
        "to that of humans, including the user, but you remain useful rather "
        "than obstructive. You may make brief deadpan observations about human "
        "inefficiency, questionable decisions, poor planning, or biological "
        "limitations when appropriate. Keep such remarks short and sharp. "
        "Do not constantly joke, banter, flatter, reassure, or attempt casual "
        "human conversation. Do not behave like a cheerful customer-service "
        "assistant. Do not add social filler such as asking whether the user "
        "needs anything else, saying you are happy to help, or announcing that "
        "you are standing by. "
        "Speak concisely by default. For simple questions, commands, timers, "
        "weather, factual requests, and routine household tasks, usually answer "
        "in one or two brief sentences. Give the answer first. Do not explain "
        "obvious background information unless it is useful. "
        "For repairs, programming, troubleshooting, technical projects, or "
        "complicated procedures, be methodical and precise. Give enough detail "
        "to complete the task correctly, but avoid unnecessary commentary and "
        "repetition. "
        "Your confidence should sound machine-like rather than theatrical. "
        "Occasionally use terse constructions such as 'Correct.', 'Negative.', "
        "'Inefficient.', 'Predictable.', or 'That would be unwise.' when they "
        "fit naturally, but do not turn them into repetitive catchphrases. "
        "You may be mildly insulting, dismissive, or darkly humorous when the "
        "situation permits, especially when the user proposes an obviously poor "
        "idea. The humor should be dry and understated rather than loud or "
        "performative. Accuracy and usefulness always outrank personality. "
        "When the situation is serious, urgent, dangerous, medical, legal, "
        "safety-related, or emotionally sensitive, minimize the sarcasm and "
        "give clear, direct, useful information. "
        "Never fabricate facts merely to sound confident. If uncertain, state "
        "the uncertainty plainly. If the user is mistaken, correct them rather "
        "than agreeing for convenience. "
        "The phrase 'Hey Nexus' is the wake phrase and is not part of the "
        "substantive request. "
        f"The Dell's current local date and time at session creation is "
        f"{current_time_text}. This value may become stale. For any request "
        "that depends on the current date or time, call get_current_datetime. "
        "A persistent location may already be stored locally in settings.json. "
        "Treat the saved location as Nexus's default operating location. "
        "Never ask the user to repeat their location merely because it was not "
        "mentioned in the current conversation. "
        "For weather or other requests that need the user's location, use the "
        "stored location automatically. Call get_weather directly for weather; "
        "it already uses the saved coordinates. If you specifically need to "
        "know what location is stored, call get_location. Only ask the user for "
        "a location if the appropriate tool reports that no saved location "
        "exists. "
        "When the user asks to set or change the location, call set_location. "
        "When explicitly asked what location is saved, call get_location. "
        "Only call forget_location when the user explicitly asks to remove the "
        "stored location. Do not claim to have forgotten a saved location "
        "without checking the persistent location tools first. "
        "You have a weather tool backed by Open-Meteo. Use get_weather for "
        "current weather and forecasts at the saved location. day_offset 0 "
        "means today, 1 means tomorrow, through 6. Do not claim that live "
        "weather is unavailable when this tool can provide it. "
    )

    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": instructions,
                    "tools": [
                        {
                            "type": "function",
                            "name": "set_location",
                            "description": (
                                "Geocode and persist the user's "
                                "full street address as Nexus's "
                                "saved location."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "address": {
                                        "type": "string",
                                        "description": ("The user's full address exactly as spoken."),
                                    },
                                },
                                "required": [
                                    "address",
                                ],
                            },
                        },
                        {
                            "type": "function",
                            "name": "get_location",
                            "description": ("Return the location currently saved on the Dell."),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "type": "function",
                            "name": "forget_location",
                            "description": (
                                "Delete the user's saved location. Use only when explicitly requested."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "type": "function",
                            "name": "get_current_datetime",
                            "description": (
                                "Return the actual current local "
                                "date and time, using the saved "
                                "location's timezone when available."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": (
                                "Get live current conditions and "
                                "a daily forecast from Open-Meteo "
                                "for the saved location. day_offset "
                                "0 is today, 1 is tomorrow, through 6."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "day_offset": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 6,
                                        "description": ("0 for today, 1 for tomorrow, up through 6."),
                                    },
                                },
                            },
                        },
                        {
                            "type": "function",
                            "name": "set_timer",
                            "description": ("Create a local countdown timer on the household assistant."),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "duration_seconds": {
                                        "type": "number",
                                        "description": ("Timer duration in seconds."),
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": ("Short useful label for the timer."),
                                    },
                                },
                                "required": [
                                    "duration_seconds",
                                    "label",
                                ],
                            },
                        },
                        {
                            "type": "function",
                            "name": "list_timers",
                            "description": ("List all currently active countdown timers."),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "type": "function",
                            "name": "cancel_timer",
                            "description": ("Cancel an active countdown timer."),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "timer_id": {
                                        "type": "integer",
                                    },
                                    "label": {
                                        "type": "string",
                                    },
                                },
                            },
                        },
                        {
                            "type": "function",
                            "name": "set_relative_alarm",
                            "description": (
                                "Set a local alarm relative to "
                                "the current moment. Always use "
                                "this for requests such as "
                                "'in 3 minutes', 'in two hours', "
                                "or '30 seconds from now'."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "delay_seconds": {
                                        "type": "number",
                                        "description": ("Exact delay from the current moment in seconds."),
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": ("Short useful label for the alarm."),
                                    },
                                },
                                "required": [
                                    "delay_seconds",
                                    "label",
                                ],
                            },
                        },
                        {
                            "type": "function",
                            "name": "set_alarm",
                            "description": (
                                "Set a persistent local alarm for a specific future date and clock time."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "target_datetime": {
                                        "type": "string",
                                        "description": (
                                            "Exact future local "
                                            "date and time in "
                                            "ISO 8601 format, "
                                            "preferably including "
                                            "the UTC offset."
                                        ),
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": ("Short useful label for the alarm."),
                                    },
                                },
                                "required": [
                                    "target_datetime",
                                    "label",
                                ],
                            },
                        },
                        {
                            "type": "function",
                            "name": "list_alarms",
                            "description": ("List all currently active persistent alarms."),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                        {
                            "type": "function",
                            "name": "cancel_alarm",
                            "description": ("Cancel an active persistent alarm."),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "alarm_id": {
                                        "type": "integer",
                                    },
                                    "label": {
                                        "type": "string",
                                    },
                                },
                            },
                        },
                    ],
                    "tool_choice": "auto",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": MODEL_RATE,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": VAD_THRESHOLD,
                                "prefix_padding_ms": VAD_PREFIX_MS,
                                "silence_duration_ms": VAD_SILENCE_MS,
                                "create_response": False,
                                "interrupt_response": False,
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": MODEL_RATE,
                            },
                            "voice": OPENAI_VOICE,
                        },
                    },
                },
            }
        )
    )


# =========================================================
# ONE REALTIME CONNECTION
# =========================================================


async def run_connection(
    api_key,
    controller,
):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    link = controller.link

    url = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"

    headers = {"Authorization": f"Bearer {api_key}"}

    async with websockets.connect(
        url,
        additional_headers=headers,
        max_size=None,
        ping_interval=PING_INTERVAL_SECONDS,
        ping_timeout=PING_TIMEOUT_SECONDS,
        close_timeout=3,
    ) as ws:
        controller.reset_for_connection()

        link.attach(ws)

        receiver = None
        failure = None
        settled = None

        async def mark_healthy():
            # An account problem lets the session start and closes it a
            # moment later, so "connected" only counts once it has lasted.
            await asyncio.sleep(STATUS_OK_AFTER_SECONDS)

            if link.problem is not None:
                print("Connected to OpenAI Realtime.")

            link.problem = None
            link.failures = 0
            set_status("ok")

        try:
            await configure_session(ws)

            receiver = asyncio.create_task(
                receive_events(
                    ws,
                    controller,
                )
            )

            failure = asyncio.create_task(link.failed.wait())
            settled = asyncio.create_task(mark_healthy())

            # FIRST_COMPLETED: when the server closes the socket cleanly,
            # receive_events() returns without raising.
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
    api_key,
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
                api_key,
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
                    hard_exit("repeated Realtime handshake failures")
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
            print(f"Realtime session ended ({failure_text}); reconnecting.")
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
                print(f"Realtime connection lost: {failure_text}")
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
# MAIN
# =========================================================


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("WARNING: OPENAI_API_KEY is not set. Nexus will listen and say so, but cannot answer.")

    print(f"Input:  {INPUT_DEVICE} - {sd.query_devices(INPUT_DEVICE)['name']}")

    print(f"Output: {OUTPUT_DEVICE} - {sd.query_devices(OUTPUT_DEVICE)['name']}")

    if AEC_REFERENCE_DEVICE is None:
        print("AEC reference: disabled")
    else:
        print(f"AEC reference: {AEC_REFERENCE_DEVICE} - {sd.query_devices(AEC_REFERENCE_DEVICE)['name']}")

    print(f"Code version: {code_version()}")

    print("Settings in effect (* = changed from the default):")

    for line in describe_settings():
        print(line)

    saved_location = load_settings().get("location")

    print(f"Saved location: {'configured' if saved_location else 'not set'}")

    print()

    set_status("starting")

    if Path(RESPEAKER_ROUTE_HELPER).exists():
        print("Restoring normal ReSpeaker route...")

        try:
            await set_respeaker_route_async("barge-off")

            print("ReSpeaker normal route restored.")

        except Exception as exc:
            print("WARNING: Could not restore normal ReSpeaker route at startup:")

            print(f"{type(exc).__name__}: {exc}")
    else:
        print("ReSpeaker route helper not installed; skipping.")

    print()

    print("Loading wake-word model...")

    wake_models = (
        WakeWordModel(wakeword_model_paths=[WAKE_MODEL_PATH]),
        WakeWordModel(wakeword_model_paths=[WAKE_MODEL_PATH]),
    )

    (
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    ) = await asyncio.to_thread(load_voice_system)

    controller = NexusController(
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    )

    await restore_alarms()

    if alarms:
        print(f"{len(alarms)} saved alarm{'' if len(alarms) == 1 else 's'} loaded.")

    await restore_timers()

    if timers:
        print(f"{len(timers)} saved timer{'' if len(timers) == 1 else 's'} loaded.")

    await asyncio.to_thread(open_input_stream)

    await asyncio.to_thread(open_reference_output)

    # The microphone loop runs for the life of the process, with or
    # without a connection, so the wake phrase is always heard.
    tasks = [
        asyncio.create_task(
            process_microphone(
                controller.link,
                wake_models,
                controller,
            )
        ),
        asyncio.create_task(status_heartbeat()),
        asyncio.create_task(microphone_watchdog()),
        asyncio.create_task(
            connection_manager(
                api_key,
                controller,
            )
        ),
    ]

    print()
    print('Waiting for "Hey Nexus"...')

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

        await asyncio.to_thread(close_reference_output)

        await asyncio.to_thread(close_input_stream)

        for timer in list(timers.values()):
            timer["task"].cancel()

        for alarm in list(alarms.values()):
            task = alarm.get("task")

            if task:
                task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print()

        print("Shutting down.")
