import os

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import asyncio
import base64
import json
import queue
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

from openwakeword.model import Model as WakeWordModel
from speechbrain.inference.speaker import SpeakerRecognition


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_RATE = 16000
MODEL_RATE = 24000
OUTPUT_RATE = 48000

BASE_DIR = Path(
    os.environ.get(
        "NEXUS_HOME",
        str(Path(__file__).resolve().parent),
    )
).expanduser()

OPENAI_MODEL = os.environ.get(
    "OPENAI_REALTIME_MODEL",
    "gpt-realtime",
)

MIC_NAME = os.environ.get(
    "NEXUS_MIC_NAME",
    "respeaker xvf3800",
)

SPEAKER_NAME = os.environ.get(
    "NEXUS_SPEAKER_NAME",
    "alc3234 analog",
)

WAKE_MODEL_PATH = os.environ.get(
    "NEXUS_WAKE_MODEL_PATH",
    str(BASE_DIR / "models" / "hey_nexus.onnx"),
)

VOICE_PROFILE_CH0_PATH = os.environ.get(
    "NEXUS_VOICE_PROFILE_CH0",
    str(BASE_DIR / "models" / "voice_ch0.pt"),
)

VOICE_PROFILE_CH1_PATH = os.environ.get(
    "NEXUS_VOICE_PROFILE_CH1",
    str(BASE_DIR / "models" / "voice_ch1.pt"),
)

SPEAKER_MODEL_CACHE = os.environ.get(
    "NEXUS_SPEAKER_MODEL_CACHE",
    str(BASE_DIR / "models" / "spkrec-ecapa-voxceleb"),
)

ALARMS_FILE = Path(
    os.environ.get(
        "NEXUS_ALARMS_FILE",
        str(BASE_DIR / "alarms.json"),
    )
)

SETTINGS_FILE = Path(
    os.environ.get(
        "NEXUS_SETTINGS_FILE",
        str(BASE_DIR / "settings.json"),
    )
)

NOMINATIM_SEARCH_URL = (
    "https://nominatim.openstreetmap.org/search"
)

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

HTTP_TIMEOUT_SECONDS = 12

HTTP_USER_AGENT = os.environ.get(
    "NEXUS_HTTP_USER_AGENT",
    "NexusHouseholdAssistant/1.0 "
    "(personal non-commercial assistant)",
)

WAKE_THRESHOLD = 0.10
SPEAKER_THRESHOLD = 0.25

INPUT_BLOCK_FRAMES = 1280

WAKE_PREROLL_SECONDS = 0.8
SPEAKER_VERIFY_SECONDS = 1.2

WAKE_REJECT_COOLDOWN_SECONDS = 1.0

FOLLOWUP_SECONDS = 5.0
FOLLOWUP_VERIFY_SECONDS = 2.2
FOLLOWUP_OVERLAP_SECONDS = 0.6

# =========================================================
# CONVERSATIONAL BARGE-IN
# =========================================================
#
# While Nexus is speaking, monitor the XVF3800 AEC residual
# for near-end speech. Once enough speech is present, verify
# that it matches the enrolled speaker. If it does, stop
# Nexus and preserve the speech already captured so the
# beginning of the interruption is not lost.

BARGE_VERIFY_SECONDS = 0.55

# Retain a little audio before the point where speech becomes
# strong enough to trigger verification.
BARGE_PREROLL_SECONDS = 0.40

# Minimum RMS required before we treat the AEC residual as
# possible near-end speech. This prevents Marin's tiny AEC
# residue from constantly invoking speaker recognition.
BARGE_RMS_THRESHOLD = 40.0

# Speaker-recognition threshold used only while Nexus talks.
# Slightly lower than the normal 0.30 threshold because the
# speech is passing through AEC while Marin is playing.
BARGE_SPEAKER_THRESHOLD = 0.24

BARGE_DEBUG = True
BARGE_DEBUG_INTERVAL_SECONDS = 0.50

RESPEAKER_ROUTE_HELPER = os.environ.get(
    "NEXUS_RESPEAKER_ROUTE_HELPER",
    "/usr/local/bin/nexus-respeaker-route",
)

RECONNECT_INITIAL_SECONDS = 1
RECONNECT_MAX_SECONDS = 15

TIMER_TONE_FREQUENCY = 880
TIMER_BEEP_SECONDS = 0.35
TIMER_BEEP_GAP = 0.25
TIMER_BEEP_COUNT = 10
TIMER_VOLUME = 0.25

ALARM_TONE_FREQUENCY = 1050
ALARM_BEEP_SECONDS = 0.45
ALARM_BEEP_GAP = 0.20
ALARM_BEEP_COUNT = 16
ALARM_VOLUME = 0.30


# =========================================================
# SHARED STATE
# =========================================================

mic_queue = queue.Queue(
    maxsize=250
)

barge_queue = queue.Queue(
    maxsize=100
)

assistant_speaking = threading.Event()
barge_in_event = threading.Event()

input_stream_lock = threading.Lock()
input_stream = None

wake_preroll = deque(
    maxlen=max(
        1,
        int(
            WAKE_PREROLL_SECONDS
            * INPUT_RATE
            / INPUT_BLOCK_FRAMES
        ),
    )
)

timers = {}
next_timer_id = 1

alarms = {}
next_alarm_id = 1

settings_lock = threading.Lock()
last_nominatim_request = 0.0


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

        if (
            need_input
            and device["max_input_channels"] < 1
        ):
            continue

        if (
            need_output
            and device["max_output_channels"] < 1
        ):
            continue

        return i

    raise RuntimeError(
        f"Audio device not found: {name_fragment}"
    )


INPUT_DEVICE = find_device(
    MIC_NAME,
    need_input=True,
)

OUTPUT_DEVICE = find_device(
    SPEAKER_NAME,
    need_output=True,
)


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

    new_length = int(
        len(audio)
        * new_rate
        / old_rate
    )

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


def samples_in_blocks(
    blocks,
):
    return sum(
        len(block[0])
        if isinstance(block, tuple)
        else len(block)
        for block in blocks
    )


def select_channel_blocks(
    blocks,
    channel,
):
    return [
        block[channel]
        if isinstance(block, tuple)
        else block
        for block in blocks
    ]


def blocks_have_seconds(
    blocks,
    seconds,
):
    needed = int(
        seconds * INPUT_RATE
    )

    return (
        samples_in_blocks(blocks)
        >= needed
    )


def clear_mic_queue():
    while True:
        try:
            mic_queue.get_nowait()

        except queue.Empty:
            break


def clear_barge_queue():
    while True:
        try:
            barge_queue.get_nowait()

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


async def reopen_input_stream_after_route(
    route_mode,
):
    await asyncio.to_thread(
        close_input_stream
    )

    await set_respeaker_route_async(
        route_mode
    )

    await asyncio.sleep(
        0.15
    )

    clear_mic_queue()
    clear_barge_queue()

    await asyncio.to_thread(
        open_input_stream
    )

    await asyncio.sleep(
        0.10
    )


# =========================================================
# MICROPHONE
# =========================================================

def block_rms(
    stereo_block,
):
    ch0, ch1 = stereo_block

    def rms(channel):
        audio = channel.astype(
            np.float32
        )

        if len(audio) == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    audio * audio
                )
            )
        )

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
    if status:
        print(
            f"Microphone status: {status}"
        )

    ch0 = (
        indata[:, 0]
        .copy()
        .astype(np.int16)
    )

    ch1 = (
        indata[:, 1]
        .copy()
        .astype(np.int16)
    )

    stereo_block = (
        ch0,
        ch1,
    )

    target_queue = (
        barge_queue
        if assistant_speaking.is_set()
        else mic_queue
    )

    try:
        target_queue.put_nowait(
            stereo_block
        )

    except queue.Full:
        try:
            target_queue.get_nowait()

        except queue.Empty:
            pass

        try:
            target_queue.put_nowait(
                stereo_block
            )

        except queue.Full:
            pass


# =========================================================
# SPEAKER VERIFICATION
# =========================================================

def load_voice_system():
    print(
        "Loading speaker-recognition model..."
    )

    recognizer = (
        SpeakerRecognition.from_hparams(
            source=(
                "speechbrain/"
                "spkrec-ecapa-voxceleb"
            ),
            savedir=SPEAKER_MODEL_CACHE,
            run_opts={
                "device": "cpu"
            },
        )
    )

    def load_profile(path):
        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(saved, dict):
            saved = saved["embedding"]

        enrolled = (
            saved
            .float()
            .flatten()
        )

        return enrolled / (
            torch.linalg.vector_norm(
                enrolled
            )
            + 1e-8
        )

    enrolled_ch0 = load_profile(
        VOICE_PROFILE_CH0_PATH
    )

    enrolled_ch1 = load_profile(
        VOICE_PROFILE_CH1_PATH
    )

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

    audio_float = (
        audio_int16.astype(
            np.float32
        )
        / 32768.0
    )

    waveform = (
        torch.from_numpy(
            audio_float
        )
        .float()
        .unsqueeze(0)
    )

    with torch.no_grad():
        embedding = (
            recognizer.encode_batch(
                waveform
            )
            .squeeze()
            .cpu()
            .flatten()
        )

    embedding = embedding / (
        torch.linalg.vector_norm(
            embedding
        )
        + 1e-8
    )

    return float(
        torch.dot(
            enrolled_embedding,
            embedding,
        )
    )


# =========================================================
# WAKE WORD
# =========================================================

def get_wake_score(
    wake_model,
    block,
):
    predictions = wake_model.predict(
        block
    )

    if not predictions:
        return 0.0

    return max(
        float(value)
        for value
        in predictions.values()
    )


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
        parts.append(
            f"{hours} hour"
            f"{'' if hours == 1 else 's'}"
        )

    if minutes:
        parts.append(
            f"{minutes} minute"
            f"{'' if minutes == 1 else 's'}"
        )

    if seconds or not parts:
        parts.append(
            f"{seconds} second"
            f"{'' if seconds == 1 else 's'}"
        )

    return " ".join(parts)


async def play_timer_alarm(
    timer_id,
    label,
):
    print()
    print(
        "================================="
    )
    print(
        f"TIMER #{timer_id} FINISHED"
    )
    print(
        f"Label: {label}"
    )
    print(
        "================================="
    )

    t = np.arange(
        int(
            OUTPUT_RATE
            * TIMER_BEEP_SECONDS
        ),
        dtype=np.float32,
    ) / OUTPUT_RATE

    tone = (
        np.sin(
            2
            * np.pi
            * TIMER_TONE_FREQUENCY
            * t
        )
        * TIMER_VOLUME
    )

    tone = (
        tone
        * 32767
    ).astype(np.int16)

    for _ in range(
        TIMER_BEEP_COUNT
    ):
        assistant_speaking.set()

        try:
            await asyncio.to_thread(
                sd.play,
                tone,
                OUTPUT_RATE,
                device=OUTPUT_DEVICE,
                blocking=True,
            )

        finally:
            assistant_speaking.clear()

        await asyncio.sleep(
            TIMER_BEEP_GAP
        )

    clear_mic_queue()


async def timer_worker(
    timer_id,
    duration_seconds,
):
    try:
        await asyncio.sleep(
            duration_seconds
        )

        timer = timers.get(
            timer_id
        )

        if timer is None:
            return

        label = timer["label"]

        timers.pop(
            timer_id,
            None,
        )

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
        duration_seconds = float(
            duration_seconds
        )

    except Exception:
        return {
            "success": False,
            "error":
                "Invalid timer duration."
        }

    if duration_seconds <= 0:
        return {
            "success": False,
            "error":
                "Timer duration must be greater than zero."
        }

    if duration_seconds > 604800:
        return {
            "success": False,
            "error":
                "Timers longer than 7 days are not supported."
        }

    timer_id = next_timer_id
    next_timer_id += 1

    if not label:
        label = "Timer"

    due_monotonic = (
        time.monotonic()
        + duration_seconds
    )

    task = asyncio.create_task(
        timer_worker(
            timer_id,
            duration_seconds,
        )
    )

    timers[timer_id] = {
        "id":
            timer_id,

        "label":
            str(label),

        "duration_seconds":
            duration_seconds,

        "due_monotonic":
            due_monotonic,

        "task":
            task,
    }

    print(
        f"Timer #{timer_id} set: "
        f"{label} - "
        f"{format_duration(duration_seconds)}"
    )

    return {
        "success": True,
        "timer_id": timer_id,
        "label": str(label),
        "duration_seconds":
            duration_seconds,
        "duration_text":
            format_duration(
                duration_seconds
            ),
    }


async def list_local_timers():
    active = []

    now = time.monotonic()

    for timer_id in sorted(
        timers.keys()
    ):
        timer = timers[
            timer_id
        ]

        remaining = max(
            0,
            timer["due_monotonic"]
            - now,
        )

        active.append({
            "timer_id":
                timer_id,

            "label":
                timer["label"],

            "remaining_seconds":
                int(round(remaining)),

            "remaining_text":
                format_duration(
                    remaining
                ),
        })

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
        return {
            "success": False,
            "error":
                "There are no active timers."
        }

    selected_id = None

    if timer_id is not None:
        try:
            requested_id = int(
                timer_id
            )

        except Exception:
            requested_id = None

        if requested_id in timers:
            selected_id = requested_id

    if (
        selected_id is None
        and label
    ):
        requested_label = (
            str(label)
            .strip()
            .lower()
        )

        matches = [
            tid
            for tid, timer
            in timers.items()
            if requested_label
            in timer["label"].lower()
        ]

        if len(matches) == 1:
            selected_id = matches[0]

    if (
        selected_id is None
        and len(timers) == 1
    ):
        selected_id = next(
            iter(timers)
        )

    if selected_id is None:
        return {
            "success": False,
            "error":
                "I could not determine which timer to cancel.",
            "active_timers":
                (
                    await list_local_timers()
                )["timers"],
        }

    timer = timers.pop(
        selected_id
    )

    timer["task"].cancel()

    print(
        f"Timer #{selected_id} cancelled: "
        f"{timer['label']}"
    )

    return {
        "success": True,
        "timer_id": selected_id,
        "label": timer["label"],
    }


# =========================================================
# PERSISTENT ALARMS
# =========================================================

def local_timezone():
    return (
        datetime.now()
        .astimezone()
        .tzinfo
    )


def parse_alarm_datetime(
    value,
):
    if not value:
        raise ValueError(
            "No alarm time was supplied."
        )

    target = datetime.fromisoformat(
        str(value)
        .strip()
        .replace(
            "Z",
            "+00:00",
        )
    )

    if target.tzinfo is None:
        target = target.replace(
            tzinfo=local_timezone()
        )

    return target.astimezone()


def format_alarm_time(
    target,
):
    return target.strftime(
        "%A, %B %d at %I:%M %p"
    ).replace(
        " 0",
        " ",
    )


def save_alarms():
    data = []

    for alarm_id in sorted(
        alarms.keys()
    ):
        alarm = alarms[
            alarm_id
        ]

        data.append({
            "alarm_id":
                alarm_id,

            "label":
                alarm["label"],

            "target_datetime":
                alarm[
                    "target_datetime"
                ].isoformat(),
        })

    temporary = (
        ALARMS_FILE.with_suffix(
            ".json.tmp"
        )
    )

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
    print(
        "================================="
    )
    print(
        f"ALARM #{alarm_id}"
    )
    print(
        f"Label: {label}"
    )
    print(
        "================================="
    )

    t = np.arange(
        int(
            OUTPUT_RATE
            * ALARM_BEEP_SECONDS
        ),
        dtype=np.float32,
    ) / OUTPUT_RATE

    tone = (
        np.sin(
            2
            * np.pi
            * ALARM_TONE_FREQUENCY
            * t
        )
        * ALARM_VOLUME
    )

    tone = (
        tone
        * 32767
    ).astype(np.int16)

    for _ in range(
        ALARM_BEEP_COUNT
    ):
        assistant_speaking.set()

        try:
            await asyncio.to_thread(
                sd.play,
                tone,
                OUTPUT_RATE,
                device=OUTPUT_DEVICE,
                blocking=True,
            )

        finally:
            assistant_speaking.clear()

        await asyncio.sleep(
            ALARM_BEEP_GAP
        )

    clear_mic_queue()


async def alarm_worker(
    alarm_id,
):
    try:
        while True:
            alarm = alarms.get(
                alarm_id
            )

            if alarm is None:
                return

            target = alarm[
                "target_datetime"
            ]

            now = (
                datetime.now()
                .astimezone()
            )

            remaining = (
                target.timestamp()
                - now.timestamp()
            )

            if remaining <= 0:
                break

            await asyncio.sleep(
                min(
                    remaining,
                    30.0,
                )
            )

        alarm = alarms.get(
            alarm_id
        )

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
        delay_seconds = float(
            delay_seconds
        )

    except Exception:
        return {
            "success": False,
            "error":
                "Invalid relative alarm duration."
        }

    if delay_seconds <= 0:
        return {
            "success": False,
            "error":
                "Alarm duration must be greater than zero."
        }

    if delay_seconds > 604800:
        return {
            "success": False,
            "error":
                "Relative alarms longer than 7 days are not supported."
        }

    target = (
        datetime.now()
        .astimezone()
        + timedelta(
            seconds=delay_seconds
        )
    )

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
        target = parse_alarm_datetime(
            target_datetime
        )

    except Exception as exc:
        return {
            "success": False,
            "error":
                f"Invalid alarm time: {exc}"
        }

    now = (
        datetime.now()
        .astimezone()
    )

    if (
        target.timestamp()
        <= now.timestamp()
    ):
        return {
            "success": False,
            "error":
                "The requested alarm time is in the past.",
            "current_local_time":
                now.isoformat(),
        }

    alarm_id = next_alarm_id
    next_alarm_id += 1

    if not label:
        label = "Alarm"

    alarms[alarm_id] = {
        "alarm_id":
            alarm_id,

        "label":
            str(label),

        "target_datetime":
            target,

        "task":
            None,
    }

    save_alarms()

    task = asyncio.create_task(
        alarm_worker(
            alarm_id
        )
    )

    alarms[alarm_id][
        "task"
    ] = task

    print(
        f"Alarm #{alarm_id} set: "
        f"{label} - "
        f"{format_alarm_time(target)}"
    )

    return {
        "success": True,
        "alarm_id":
            alarm_id,
        "label":
            str(label),
        "target_datetime":
            target.isoformat(),
        "time_text":
            format_alarm_time(
                target
            ),
    }


async def list_local_alarms():
    active = []

    for alarm_id in sorted(
        alarms.keys()
    ):
        alarm = alarms[
            alarm_id
        ]

        target = alarm[
            "target_datetime"
        ]

        active.append({
            "alarm_id":
                alarm_id,

            "label":
                alarm["label"],

            "target_datetime":
                target.isoformat(),

            "time_text":
                format_alarm_time(
                    target
                ),
        })

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
        return {
            "success": False,
            "error":
                "There are no active alarms."
        }

    selected_id = None

    if alarm_id is not None:
        try:
            requested_id = int(
                alarm_id
            )

        except Exception:
            requested_id = None

        if requested_id in alarms:
            selected_id = requested_id

    if (
        selected_id is None
        and label
    ):
        requested_label = (
            str(label)
            .strip()
            .lower()
        )

        matches = [
            aid
            for aid, alarm
            in alarms.items()
            if requested_label
            in alarm["label"].lower()
        ]

        if len(matches) == 1:
            selected_id = matches[0]

    if (
        selected_id is None
        and len(alarms) == 1
    ):
        selected_id = next(
            iter(alarms)
        )

    if selected_id is None:
        return {
            "success": False,
            "error":
                "I could not determine which alarm to cancel.",
            "active_alarms":
                (
                    await list_local_alarms()
                )["alarms"],
        }

    alarm = alarms.pop(
        selected_id
    )

    task = alarm.get(
        "task"
    )

    if task:
        task.cancel()

    save_alarms()

    print(
        f"Alarm #{selected_id} cancelled: "
        f"{alarm['label']}"
    )

    return {
        "success": True,
        "alarm_id":
            selected_id,
        "label":
            alarm["label"],
    }


async def restore_alarms():
    global next_alarm_id

    if not ALARMS_FILE.exists():
        save_alarms()
        return

    try:
        raw = ALARMS_FILE.read_text(
            encoding="utf-8"
        )

        saved = json.loads(
            raw
        )

    except Exception as exc:
        print(
            "Could not load saved alarms:"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return

    now = (
        datetime.now()
        .astimezone()
    )

    highest_id = 0

    for entry in saved:
        try:
            alarm_id = int(
                entry["alarm_id"]
            )

            target = (
                parse_alarm_datetime(
                    entry[
                        "target_datetime"
                    ]
                )
            )

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

        if (
            target.timestamp()
            <= now.timestamp()
        ):
            print(
                "Discarding expired alarm "
                f"#{alarm_id}: {label}"
            )
            continue

        alarms[alarm_id] = {
            "alarm_id":
                alarm_id,

            "label":
                label,

            "target_datetime":
                target,

            "task":
                None,
        }

        alarms[alarm_id][
            "task"
        ] = asyncio.create_task(
            alarm_worker(
                alarm_id
            )
        )

        print(
            f"Restored alarm #{alarm_id}: "
            f"{label} - "
            f"{format_alarm_time(target)}"
        )

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
            return json.loads(
                SETTINGS_FILE.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            print(
                "Could not load Nexus settings:"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )
            return {}


def save_settings(data):
    with settings_lock:
        SETTINGS_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = SETTINGS_FILE.with_suffix(
            ".json.tmp"
        )

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
            "User-Agent":
                HTTP_USER_AGENT,
            "Accept":
                "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=HTTP_TIMEOUT_SECONDS,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def geocode_address_sync(
    address,
):
    global last_nominatim_request

    address = str(
        address or ""
    ).strip()

    if not address:
        return {
            "success": False,
            "error":
                "No address was supplied.",
        }

    # Nominatim's public service requires very low request
    # volume. Enforce at least one second between requests.
    elapsed = (
        time.monotonic()
        - last_nominatim_request
    )

    if elapsed < 1.1:
        time.sleep(
            1.1 - elapsed
        )

    last_nominatim_request = (
        time.monotonic()
    )

    results = http_get_json(
        NOMINATIM_SEARCH_URL,
        {
            "q":
                address,
            "format":
                "jsonv2",
            "addressdetails":
                1,
            "limit":
                1,
        },
    )

    if not results:
        return {
            "success": False,
            "error":
                "I could not find that address.",
        }

    result = results[0]

    try:
        latitude = float(
            result["lat"]
        )
        longitude = float(
            result["lon"]
        )
    except Exception:
        return {
            "success": False,
            "error":
                "The address service returned an invalid location.",
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
                "latitude":
                    latitude,
                "longitude":
                    longitude,
                "timezone":
                    "auto",
                "forecast_days":
                    1,
            },
        )

        timezone = timezone_probe.get(
            "timezone"
        )

    except Exception:
        timezone = None

    settings = load_settings()

    settings["location"] = {
        "address":
            address,
        "display_name":
            display_name,
        "latitude":
            latitude,
        "longitude":
            longitude,
        "timezone":
            timezone,
        "saved_at":
            datetime.now()
            .astimezone()
            .isoformat(),
    }

    save_settings(
        settings
    )

    return {
        "success": True,
        "display_name":
            display_name,
        "latitude":
            latitude,
        "longitude":
            longitude,
        "timezone":
            timezone,
        "message":
            "Location saved locally.",
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
            "error":
                f"{type(exc).__name__}: {exc}",
        }


async def get_saved_location():
    settings = load_settings()
    location = settings.get(
        "location"
    )

    if not location:
        return {
            "success": False,
            "error":
                "No location is currently saved.",
        }

    return {
        "success": True,
        "address":
            location.get(
                "address"
            ),
        "display_name":
            location.get(
                "display_name"
            ),
        "latitude":
            location.get(
                "latitude"
            ),
        "longitude":
            location.get(
                "longitude"
            ),
        "timezone":
            location.get(
                "timezone"
            ),
    }


async def forget_saved_location():
    settings = load_settings()

    if "location" not in settings:
        return {
            "success": False,
            "error":
                "No location is currently saved.",
        }

    settings.pop(
        "location",
        None,
    )

    save_settings(
        settings
    )

    return {
        "success": True,
        "message":
            "Saved location removed.",
    }


def get_location_timezone():
    settings = load_settings()
    location = settings.get(
        "location",
        {},
    )

    timezone_name = location.get(
        "timezone"
    )

    if timezone_name:
        try:
            return (
                ZoneInfo(
                    timezone_name
                ),
                timezone_name,
            )
        except Exception:
            pass

    local_now = (
        datetime.now()
        .astimezone()
    )

    return (
        local_now.tzinfo,
        str(
            local_now.tzinfo
        ),
    )


async def get_current_datetime_tool():
    tzinfo, timezone_name = (
        get_location_timezone()
    )

    now = datetime.now(
        tzinfo
    )

    location = (
        load_settings()
        .get(
            "location",
            {},
        )
    )

    return {
        "success": True,
        "iso_datetime":
            now.isoformat(),
        "date":
            now.strftime(
                "%A, %B %d, %Y"
            ),
        "time":
            now.strftime(
                "%I:%M:%S %p"
            ).lstrip("0"),
        "timezone":
            timezone_name,
        "location":
            location.get(
                "display_name"
            ),
    }


def weather_for_saved_location_sync(
    day_offset=0,
):
    settings = load_settings()
    location = settings.get(
        "location"
    )

    if not location:
        return {
            "success": False,
            "error":
                "No location is saved. Ask the user to set a location first.",
        }

    try:
        day_offset = int(
            day_offset
        )
    except Exception:
        day_offset = 0

    if day_offset < 0 or day_offset > 6:
        return {
            "success": False,
            "error":
                "Weather requests currently support today through six days from now.",
        }

    latitude = location[
        "latitude"
    ]
    longitude = location[
        "longitude"
    ]

    data = http_get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude":
                latitude,
            "longitude":
                longitude,
            "timezone":
                "auto",
            "temperature_unit":
                "fahrenheit",
            "wind_speed_unit":
                "mph",
            "precipitation_unit":
                "inch",
            "forecast_days":
                7,
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

    if day_offset >= len(
        dates
    ):
        return {
            "success": False,
            "error":
                "Forecast data was not available for that day.",
        }

    def daily_value(
        name,
    ):
        values = daily.get(
            name,
            [],
        )

        if day_offset < len(
            values
        ):
            return values[
                day_offset
            ]

        return None

    weather_code = daily_value(
        "weather_code"
    )

    result = {
        "success": True,
        "location":
            location.get(
                "display_name"
            ),
        "timezone":
            data.get(
                "timezone"
            ),
        "day_offset":
            day_offset,
        "date":
            dates[
                day_offset
            ],
        "forecast": {
            "conditions":
                WEATHER_CODES.get(
                    weather_code,
                    f"weather code {weather_code}",
                ),
            "high_f":
                daily_value(
                    "temperature_2m_max"
                ),
            "low_f":
                daily_value(
                    "temperature_2m_min"
                ),
            "precipitation_probability_percent":
                daily_value(
                    "precipitation_probability_max"
                ),
            "precipitation_inches":
                daily_value(
                    "precipitation_sum"
                ),
            "max_wind_mph":
                daily_value(
                    "wind_speed_10m_max"
                ),
            "sunrise":
                daily_value(
                    "sunrise"
                ),
            "sunset":
                daily_value(
                    "sunset"
                ),
        },
    }

    if day_offset == 0:
        current = data.get(
            "current",
            {},
        )

        current_code = current.get(
            "weather_code"
        )

        result["current"] = {
            "conditions":
                WEATHER_CODES.get(
                    current_code,
                    f"weather code {current_code}",
                ),
            "temperature_f":
                current.get(
                    "temperature_2m"
                ),
            "feels_like_f":
                current.get(
                    "apparent_temperature"
                ),
            "humidity_percent":
                current.get(
                    "relative_humidity_2m"
                ),
            "wind_mph":
                current.get(
                    "wind_speed_10m"
                ),
            "wind_direction_degrees":
                current.get(
                    "wind_direction_10m"
                ),
            "precipitation_inches":
                current.get(
                    "precipitation"
                ),
            "rain_inches":
                current.get(
                    "rain"
                ),
            "snowfall_inches":
                current.get(
                    "snowfall"
                ),
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
            "error":
                f"{type(exc).__name__}: {exc}",
        }


def redact_tool_log(
    name,
    value,
):
    if name == "set_location":
        return {
            "address":
                "[redacted from journal]"
        }

    if name == "get_location":
        if isinstance(
            value,
            dict,
        ):
            safe = dict(
                value
            )

            if "address" in safe:
                safe[
                    "address"
                ] = "[redacted from journal]"

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
        return await set_saved_location(
            arguments.get(
                "address"
            )
        )

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
            arguments.get(
                "duration_seconds"
            ),
            arguments.get(
                "label",
                "Timer",
            ),
        )

    if name == "list_timers":
        return await list_local_timers()

    if name == "cancel_timer":
        return await cancel_local_timer(
            timer_id=arguments.get(
                "timer_id"
            ),
            label=arguments.get(
                "label"
            ),
        )

    if name == "set_relative_alarm":
        return await set_relative_alarm(
            arguments.get(
                "delay_seconds"
            ),
            arguments.get(
                "label",
                "Alarm",
            ),
        )

    if name == "set_alarm":
        return await set_local_alarm(
            arguments.get(
                "target_datetime"
            ),
            arguments.get(
                "label",
                "Alarm",
            ),
        )

    if name == "list_alarms":
        return await list_local_alarms()

    if name == "cancel_alarm":
        return await cancel_local_alarm(
            alarm_id=arguments.get(
                "alarm_id"
            ),
            label=arguments.get(
                "label"
            ),
        )

    return {
        "success": False,
        "error":
            f"Unknown tool: {name}"
    }



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
        raise ValueError(
            f"Invalid ReSpeaker route mode: {mode}"
        )

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

    output = (
        completed.stdout
        or ""
    ).strip()

    if completed.returncode != 0:
        raise RuntimeError(
            "ReSpeaker route command failed: "
            f"{mode}; {output}"
        )

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

        if callable(
            method
        ):
            try:
                method()
            except Exception:
                pass

            return


def reset_wake_detectors(
    wake_models,
):
    for wake_model in wake_models:
        reset_wake_detector(
            wake_model
        )


async def monitor_barge_in(
    controller,
):
    loop = asyncio.get_running_loop()

    preroll = deque()
    speech_blocks = []

    preroll_samples = int(
        BARGE_PREROLL_SECONDS
        * INPUT_RATE
    )

    last_print = time.monotonic()
    peak_rms0 = 0.0
    peak_rms1 = 0.0

    while True:
        block = await loop.run_in_executor(
            None,
            barge_queue.get,
        )

        if not assistant_speaking.is_set():
            preroll.clear()
            speech_blocks.clear()
            peak_rms0 = 0.0
            peak_rms1 = 0.0
            last_print = time.monotonic()
            continue

        rms0, rms1 = block_rms(
            block
        )

        peak_rms0 = max(
            peak_rms0,
            rms0,
        )

        peak_rms1 = max(
            peak_rms1,
            rms1,
        )

        now = time.monotonic()

        if (
            BARGE_DEBUG
            and
            now - last_print
            >= BARGE_DEBUG_INTERVAL_SECONDS
        ):
            print(
                "BARGE VOICE DEBUG peak "
                f"CH0={peak_rms0:.1f} "
                f"CH1={peak_rms1:.1f}",
                flush=True,
            )

            peak_rms0 = 0.0
            peak_rms1 = 0.0
            last_print = now

        # Always retain a short rolling window so we can keep
        # the beginning of the user's sentence.
        preroll.append(
            (
                block[0].copy(),
                block[1].copy(),
            )
        )

        while (
            preroll
            and
            samples_in_blocks(preroll)
            > preroll_samples
        ):
            preroll.popleft()

        # We are not currently evaluating a possible interruption.
        # Wait until the AEC residual contains enough energy to
        # plausibly be near-end speech.
        if not speech_blocks:
            if (
                max(rms0, rms1)
                < BARGE_RMS_THRESHOLD
            ):
                continue

            speech_blocks = list(
                preroll
            )

        else:
            speech_blocks.append(
                (
                    block[0].copy(),
                    block[1].copy(),
                )
            )

        if not blocks_have_seconds(
            speech_blocks,
            BARGE_VERIFY_SECONDS,
        ):
            continue

        audio0 = np.concatenate(
            select_channel_blocks(
                speech_blocks,
                0,
            )
        )

        audio1 = np.concatenate(
            select_channel_blocks(
                speech_blocks,
                1,
            )
        )

        similarity0, similarity1 = (
            await asyncio.gather(
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
        )

        winner = (
            0
            if similarity0 >= similarity1
            else 1
        )

        best_similarity = max(
            similarity0,
            similarity1,
        )

        print(
            "Barge speaker check "
            f"CH0={similarity0:.3f} "
            f"CH1={similarity1:.3f} "
            f"winner=CH{winner}",
            flush=True,
        )

        if (
            assistant_speaking.is_set()
            and
            best_similarity
            >= BARGE_SPEAKER_THRESHOLD
        ):
            print()
            print(
                "*** CONVERSATIONAL INTERRUPTION DETECTED ***",
                flush=True,
            )

            print(
                f"Interrupting for enrolled speaker "
                f"on CH{winner}.",
                flush=True,
            )

            # Save everything already spoken during the
            # interruption. The follow-up logic will consume this
            # instead of making the user repeat the beginning.
            controller.interrupt_blocks = [
                (
                    saved_block[0].copy(),
                    saved_block[1].copy(),
                )
                for saved_block
                in speech_blocks
            ]

            controller.active_channel = winner

            barge_in_event.set()

            speech_blocks.clear()
            preroll.clear()

            # Playback may take a few milliseconds to notice the
            # interruption. Keep collecting anything the user says
            # during that transition instead of throwing it away.
            while assistant_speaking.is_set():
                try:
                    extra_block = (
                        barge_queue.get_nowait()
                    )

                    controller.interrupt_blocks.append(
                        (
                            extra_block[0].copy(),
                            extra_block[1].copy(),
                        )
                    )

                except queue.Empty:
                    await asyncio.sleep(
                        0.01
                    )

            # Grab anything that reached the barge queue immediately
            # before playback shut down.
            while True:
                try:
                    extra_block = (
                        barge_queue.get_nowait()
                    )

                    controller.interrupt_blocks.append(
                        (
                            extra_block[0].copy(),
                            extra_block[1].copy(),
                        )
                    )

                except queue.Empty:
                    break

            continue

        # Candidate audio was not the enrolled speaker.
        # Go back to monitoring without interrupting Nexus.
        speech_blocks.clear()


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

    encoded = base64.b64encode(
        pcm_24k.tobytes()
    ).decode("ascii")

    await ws.send(
        json.dumps({
            "type":
                "input_audio_buffer.append",
            "audio":
                encoded,
        })
    )


async def send_blocks_to_openai(
    ws,
    blocks,
):
    if not blocks:
        return

    await send_pcm_to_openai(
        ws,
        np.concatenate(
            blocks
        ),
    )


async def clear_openai_input(
    ws,
):
    await ws.send(
        json.dumps({
            "type":
                "input_audio_buffer.clear"
        })
    )


async def create_response(
    ws,
):
    await ws.send(
        json.dumps({
            "type":
                "response.create"
        })
    )


async def send_tool_output(
    ws,
    call_id,
    result,
):
    await ws.send(
        json.dumps({
            "type":
                "conversation.item.create",

            "item": {
                "type":
                    "function_call_output",

                "call_id":
                    call_id,

                "output":
                    json.dumps(
                        result
                    ),
            },
        })
    )


# =========================================================
# PLAYBACK
# =========================================================

async def play_response_audio(
    audio_bytes,
):
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

    barge_in_event.clear()
    clear_barge_queue()
    clear_mic_queue()

    stream = None
    route_changed = False

    try:
        # Close and reopen the ALSA input stream around the
        # XVF3800 route change so CH1 actually reflects the
        # amplified-microphone path while Nexus is speaking.
        await reopen_input_stream_after_route(
            "barge-on"
        )

        route_changed = True

        assistant_speaking.set()

        chunk_frames = int(
            OUTPUT_RATE * 0.05
        )

        stream = sd.OutputStream(
            samplerate=OUTPUT_RATE,
            device=OUTPUT_DEVICE,
            channels=1,
            dtype="int16",
            blocksize=chunk_frames,
        )

        stream.start()

        position = 0

        while position < len(
            pcm_48k
        ):
            if barge_in_event.is_set():
                break

            chunk = pcm_48k[
                position:
                position + chunk_frames
            ]

            await asyncio.to_thread(
                stream.write,
                chunk.reshape(
                    -1,
                    1,
                ),
            )

            position += len(
                chunk
            )

    finally:
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

        assistant_speaking.clear()

        # Always restore normal CH1 routing and reopen the
        # input stream, including after interruption/errors.
        if route_changed:
            try:
                await reopen_input_stream_after_route(
                    "barge-off"
                )
            except Exception as exc:
                print(
                    "WARNING: Could not restore "
                    "normal ReSpeaker route/input stream:"
                )
                print(
                    f"{type(exc).__name__}: {exc}"
                )

        clear_mic_queue()
        clear_barge_queue()

    interrupted = (
        barge_in_event.is_set()
    )

    barge_in_event.clear()

    return interrupted


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

        self.state = (
            AssistantState.WAIT_WAKE
        )

        self.verify_blocks = []
        self.followup_blocks = []

        # Audio captured while the enrolled speaker interrupts
        # Nexus during playback.
        self.interrupt_blocks = []

        self.active_channel = 0
        self.followup_deadline = 0.0
        self.wake_cooldown_until = 0.0
        self.response_active = False
        self.speech_started_seen = False


    def start_followup(
        self,
    ):
        self.verify_blocks.clear()

        # If playback was interrupted, begin the follow-up
        # with the speech that was already captured while
        # Nexus was still talking.
        if self.interrupt_blocks:
            self.followup_blocks = [
                (
                    block[0].copy(),
                    block[1].copy(),
                )
                for block
                in self.interrupt_blocks
            ]

            self.interrupt_blocks.clear()

            print(
                "Preserved interruption audio for follow-up.",
                flush=True,
            )
        else:
            self.followup_blocks.clear()

        self.speech_started_seen = False

        self.followup_deadline = (
            time.monotonic()
            + FOLLOWUP_SECONDS
        )

        self.state = (
            AssistantState.FOLLOWUP
        )

    def reset_for_connection(
        self,
    ):
        self.verify_blocks.clear()
        self.followup_blocks.clear()
        self.interrupt_blocks.clear()

        self.response_active = False
        self.speech_started_seen = False
        self.followup_deadline = 0.0
        self.wake_cooldown_until = 0.0
        self.active_channel = 0

        wake_preroll.clear()
        clear_mic_queue()

        self.state = (
            AssistantState.WAIT_WAKE
        )


    def return_to_wake(
        self,
        rejected=False,
    ):
        self.verify_blocks.clear()
        self.followup_blocks.clear()
        self.interrupt_blocks.clear()

        wake_preroll.clear()
        clear_mic_queue()
        self.speech_started_seen = False

        if rejected:
            self.wake_cooldown_until = (
                time.monotonic()
                + WAKE_REJECT_COOLDOWN_SECONDS
            )

        self.state = (
            AssistantState.WAIT_WAKE
        )

        print()

        print(
            'Waiting for "Hey Nexus"...'
        )


# =========================================================
# MICROPHONE PROCESSING
# =========================================================

async def process_microphone(
    ws,
    wake_models,
    controller,
):
    loop = asyncio.get_running_loop()
    wake_model_ch0, wake_model_ch1 = wake_models

    while True:
        block = await loop.run_in_executor(
            None,
            mic_queue.get,
        )

        if assistant_speaking.is_set():
            continue

        ch0, ch1 = block

        if (
            controller.state
            == AssistantState.WAIT_WAKE
        ):
            wake_preroll.append(
                block
            )

            if (
                time.monotonic()
                < controller.wake_cooldown_until
            ):
                continue

            score0 = get_wake_score(
                wake_model_ch0,
                ch0,
            )

            score1 = get_wake_score(
                wake_model_ch1,
                ch1,
            )

            best_score = max(
                score0,
                score1,
            )

            if best_score >= 0.02:
                print(
                    f"WAKE DEBUG CH0={score0:.3f} "
                    f"CH1={score1:.3f} "
                    f"best={best_score:.3f}",
                    flush=True,
                )

            if best_score < WAKE_THRESHOLD:
                continue

            wake_winner = (
                0 if score0 >= score1 else 1
            )

            print()
            print(
                "*** HEY NEXUS DETECTED ***"
            )
            print(
                f"Wake CH0: {score0:.3f}   "
                f"CH1: {score1:.3f}   "
                f"winner: CH{wake_winner}"
            )
            print(
                "Checking speaker on both channels..."
            )

            controller.verify_blocks = [
                (b[0].copy(), b[1].copy())
                for b in wake_preroll
            ]

            wake_preroll.clear()
            controller.state = (
                AssistantState.VERIFY_WAKE
            )
            continue

        if (
            controller.state
            == AssistantState.VERIFY_WAKE
        ):
            controller.verify_blocks.append(
                block
            )

            if not blocks_have_seconds(
                controller.verify_blocks,
                SPEAKER_VERIFY_SECONDS,
            ):
                continue

            audio0 = np.concatenate(
                select_channel_blocks(
                    controller.verify_blocks,
                    0,
                )
            )
            audio1 = np.concatenate(
                select_channel_blocks(
                    controller.verify_blocks,
                    1,
                )
            )

            similarity0, similarity1 = (
                await asyncio.gather(
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
            )

            winner = (
                0 if similarity0 >= similarity1 else 1
            )
            best_similarity = max(
                similarity0,
                similarity1,
            )

            print(
                f"Speaker CH0: {similarity0:.3f}   "
                f"CH1: {similarity1:.3f}   "
                f"winner: CH{winner}"
            )

            if best_similarity < SPEAKER_THRESHOLD:
                print(
                    "Speaker rejected."
                )
                controller.return_to_wake(
                    rejected=True
                )
                continue

            controller.active_channel = winner
            print(
                "Speaker accepted."
            )
            print(
                f"Using microphone channel {winner}."
            )
            print(
                "Listening for request..."
            )

            controller.state = (
                AssistantState.ACTIVE
            )

            await clear_openai_input(
                ws
            )
            await send_blocks_to_openai(
                ws,
                select_channel_blocks(
                    controller.verify_blocks,
                    winner,
                ),
            )

            controller.verify_blocks.clear()
            continue

        if (
            controller.state
            == AssistantState.ACTIVE
        ):
            await send_pcm_to_openai(
                ws,
                block[controller.active_channel],
            )
            continue

        if (
            controller.state
            == AssistantState.WAIT_RESPONSE
        ):
            continue

        if (
            controller.state
            == AssistantState.FOLLOWUP
        ):
            if (
                time.monotonic()
                >= controller.followup_deadline
            ):
                controller.return_to_wake()
                wake_preroll.append(
                    block
                )
                continue

            controller.followup_blocks.append(
                block
            )

            if not blocks_have_seconds(
                controller.followup_blocks,
                FOLLOWUP_VERIFY_SECONDS,
            ):
                continue

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

            similarity0, similarity1 = (
                await asyncio.gather(
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
            )

            winner = (
                0 if similarity0 >= similarity1 else 1
            )
            best_similarity = max(
                similarity0,
                similarity1,
            )

            print(
                f"Follow-up CH0: {similarity0:.3f}   "
                f"CH1: {similarity1:.3f}   "
                f"winner: CH{winner}"
            )

            if best_similarity >= SPEAKER_THRESHOLD:
                controller.active_channel = winner
                print(
                    "Follow-up speaker accepted."
                )
                print(
                    f"Using microphone channel {winner}."
                )

                controller.state = (
                    AssistantState.ACTIVE
                )
                controller.speech_started_seen = False

                await clear_openai_input(
                    ws
                )
                await send_blocks_to_openai(
                    ws,
                    select_channel_blocks(
                        controller.followup_blocks,
                        winner,
                    ),
                )

                controller.followup_blocks.clear()
                print(
                    "Listening for follow-up..."
                )
                continue

            print(
                "Background/unknown speaker ignored."
            )

            overlap_samples = int(
                FOLLOWUP_OVERLAP_SECONDS
                * INPUT_RATE
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
        message = json.loads(
            raw
        )

        event_type = message.get(
            "type"
        )

        if event_type == "session.created":
            print(
                "Realtime session created."
            )

        elif event_type == "session.updated":
            print(
                "Session configured."
            )

            controller.return_to_wake()

        elif (
            event_type
            == "input_audio_buffer.speech_started"
        ):
            if (
                controller.state
                == AssistantState.ACTIVE
            ):
                controller.speech_started_seen = True

                print()

                print(
                    "Speech detected..."
                )

        elif (
            event_type
            == "input_audio_buffer.speech_stopped"
        ):
            if (
                controller.state
                == AssistantState.ACTIVE
                and
                controller.speech_started_seen
                and
                not controller.response_active
            ):
                controller.speech_started_seen = False

                print(
                    "You stopped speaking."
                )

                controller.state = (
                    AssistantState.WAIT_RESPONSE
                )

                controller.response_active = True

                print(
                    "Thinking..."
                )

                await create_response(
                    ws
                )

        elif event_type == "response.created":
            response_audio.clear()

        elif (
            event_type
            == "response.output_audio.delta"
        ):
            delta = message.get(
                "delta"
            )

            if delta:
                response_audio.extend(
                    base64.b64decode(
                        delta
                    )
                )

        elif (
            event_type
            == "response.output_audio_transcript.delta"
        ):
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

        elif (
            event_type
            == "response.output_audio.done"
        ):
            print()

        elif event_type == "response.done":
            response = message.get(
                "response",
                {},
            )

            output = response.get(
                "output",
                [],
            )

            function_calls = [
                item
                for item in output
                if (
                    item.get("type")
                    == "function_call"
                )
            ]

            if function_calls:
                response_audio.clear()

                for item in function_calls:
                    name = item.get(
                        "name"
                    )

                    call_id = item.get(
                        "call_id"
                    )

                    raw_arguments = item.get(
                        "arguments",
                        "{}",
                    )

                    try:
                        arguments = json.loads(
                            raw_arguments
                        )

                    except Exception:
                        arguments = {}

                    print(
                        f"Tool call: {name} "
                        f"{redact_tool_log(name, arguments)}"
                    )

                    try:
                        result = (
                            await execute_tool(
                                name,
                                arguments,
                            )
                        )

                    except Exception as exc:
                        result = {
                            "success": False,
                            "error":
                                (
                                    f"{type(exc).__name__}: "
                                    f"{exc}"
                                ),
                        }

                    print(
                        "Tool result: "
                        f"{redact_tool_log(name, result)}"
                    )

                    await send_tool_output(
                        ws,
                        call_id,
                        result,
                    )

                await create_response(
                    ws
                )

                controller.response_active = True

                continue

            controller.response_active = False

            status = response.get(
                "status"
            )

            if status != "completed":
                print(
                    "Response status:",
                    status,
                )

            if response_audio:
                print(
                    "Speaking..."
                )

                audio_copy = bytes(
                    response_audio
                )

                response_audio.clear()

                interrupted = (
                    await play_response_audio(
                        audio_copy
                    )
                )

                if interrupted:
                    try:
                        await clear_openai_input(
                            ws
                        )

                    except Exception:
                        pass

                    controller.start_followup()

                    print()
                    print(
                        "Playback stopped."
                    )

                    print(
                        "Listening for your new request "
                        f"for {FOLLOWUP_SECONDS:.0f}s."
                    )

                    continue

            try:
                await clear_openai_input(
                    ws
                )

            except Exception:
                pass

            controller.start_followup()

            print()

            print(
                "Follow-up window open "
                f"for {FOLLOWUP_SECONDS:.0f}s."
            )

            print(
                "Listening briefly for "
                "a direct follow-up."
            )

        elif event_type == "error":
            print()

            print(
                "OpenAI error:"
            )

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

            print(
                error_message
            )

            if (
                "maximum duration"
                in error_message.lower()
            ):
                raise ConnectionError(
                    error_message
                )

            controller.response_active = False


# =========================================================
# REALTIME SESSION CONFIGURATION
# =========================================================

async def configure_session(
    ws,
):
    now = (
        datetime.now()
        .astimezone()
    )

    current_time_text = now.strftime(
        "%A, %B %d, %Y at "
        "%I:%M:%S %p %Z"
    )

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
        json.dumps({
            "type":
                "session.update",

            "session": {
                "type":
                    "realtime",

                "instructions":
                    instructions,

                "tools": [
                    {
                        "type":
                            "function",

                        "name":
                            "set_location",

                        "description":
                            (
                                "Geocode and persist the user's "
                                "full street address as Nexus's "
                                "saved location."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "address": {
                                    "type":
                                        "string",

                                    "description":
                                        (
                                            "The user's full address "
                                            "exactly as spoken."
                                        ),
                                },
                            },

                            "required": [
                                "address",
                            ],
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "get_location",

                        "description":
                            (
                                "Return the location currently "
                                "saved on the Dell."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {},
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "forget_location",

                        "description":
                            (
                                "Delete the user's saved location. "
                                "Use only when explicitly requested."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {},
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "get_current_datetime",

                        "description":
                            (
                                "Return the actual current local "
                                "date and time, using the saved "
                                "location's timezone when available."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {},
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "get_weather",

                        "description":
                            (
                                "Get live current conditions and "
                                "a daily forecast from Open-Meteo "
                                "for the saved location. day_offset "
                                "0 is today, 1 is tomorrow, through 6."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "day_offset": {
                                    "type":
                                        "integer",

                                    "minimum":
                                        0,

                                    "maximum":
                                        6,

                                    "description":
                                        (
                                            "0 for today, 1 for "
                                            "tomorrow, up through 6."
                                        ),
                                },
                            },
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "set_timer",

                        "description":
                            (
                                "Create a local countdown timer "
                                "on the household assistant."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "duration_seconds": {
                                    "type":
                                        "number",

                                    "description":
                                        (
                                            "Timer duration in "
                                            "seconds."
                                        ),
                                },

                                "label": {
                                    "type":
                                        "string",

                                    "description":
                                        (
                                            "Short useful label "
                                            "for the timer."
                                        ),
                                },
                            },

                            "required": [
                                "duration_seconds",
                                "label",
                            ],
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "list_timers",

                        "description":
                            (
                                "List all currently active "
                                "countdown timers."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {},
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "cancel_timer",

                        "description":
                            (
                                "Cancel an active countdown timer."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "timer_id": {
                                    "type":
                                        "integer",
                                },

                                "label": {
                                    "type":
                                        "string",
                                },
                            },
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "set_relative_alarm",

                        "description":
                            (
                                "Set a local alarm relative to "
                                "the current moment. Always use "
                                "this for requests such as "
                                "'in 3 minutes', 'in two hours', "
                                "or '30 seconds from now'."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "delay_seconds": {
                                    "type":
                                        "number",

                                    "description":
                                        (
                                            "Exact delay from "
                                            "the current moment "
                                            "in seconds."
                                        ),
                                },

                                "label": {
                                    "type":
                                        "string",

                                    "description":
                                        (
                                            "Short useful label "
                                            "for the alarm."
                                        ),
                                },
                            },

                            "required": [
                                "delay_seconds",
                                "label",
                            ],
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "set_alarm",

                        "description":
                            (
                                "Set a persistent local alarm "
                                "for a specific future date and "
                                "clock time."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "target_datetime": {
                                    "type":
                                        "string",

                                    "description":
                                        (
                                            "Exact future local "
                                            "date and time in "
                                            "ISO 8601 format, "
                                            "preferably including "
                                            "the UTC offset."
                                        ),
                                },

                                "label": {
                                    "type":
                                        "string",

                                    "description":
                                        (
                                            "Short useful label "
                                            "for the alarm."
                                        ),
                                },
                            },

                            "required": [
                                "target_datetime",
                                "label",
                            ],
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "list_alarms",

                        "description":
                            (
                                "List all currently active "
                                "persistent alarms."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {},
                        },
                    },

                    {
                        "type":
                            "function",

                        "name":
                            "cancel_alarm",

                        "description":
                            (
                                "Cancel an active persistent "
                                "alarm."
                            ),

                        "parameters": {
                            "type":
                                "object",

                            "properties": {
                                "alarm_id": {
                                    "type":
                                        "integer",
                                },

                                "label": {
                                    "type":
                                        "string",
                                },
                            },
                        },
                    },
                ],

                "tool_choice":
                    "auto",

                "output_modalities": [
                    "audio"
                ],

                "audio": {
                    "input": {
                        "format": {
                            "type":
                                "audio/pcm",

                            "rate":
                                MODEL_RATE,
                        },

                        "turn_detection": {
                            "type":
                                "server_vad",

                            "threshold":
                                0.5,

                            "prefix_padding_ms":
                                300,

                            "silence_duration_ms":
                                700,

                            "create_response":
                                False,

                            "interrupt_response":
                                False,
                        },
                    },

                    "output": {
                        "format": {
                            "type":
                                "audio/pcm",

                            "rate":
                                MODEL_RATE,
                        },

                        "voice":
                            "marin",
                    },
                },
            },
        })
    )


# =========================================================
# ONE REALTIME CONNECTION
# =========================================================

async def realtime_session_rotation_monitor(
    controller,
):
    # Do not deliberately close a healthy connection. OpenAI's
    # maximum-duration error is handled by receive_events(), which
    # raises ConnectionError and lets connection_manager reconnect.
    await asyncio.Future()


async def run_connection(
    api_key,
    wake_models,
    barge_wake_models,
    controller,
):
    url = (
        "wss://api.openai.com/v1/realtime"
        f"?model={OPENAI_MODEL}"
    )

    headers = {
        "Authorization":
            f"Bearer {api_key}"
    }

    async with websockets.connect(
        url,
        additional_headers=headers,
        max_size=None,
        ping_interval=5,
        ping_timeout=5,
        close_timeout=3,
    ) as ws:

        print(
            "Connected to OpenAI Realtime."
        )

        controller.reset_for_connection()

        await configure_session(
            ws
        )

        processor = asyncio.create_task(
            process_microphone(
                ws,
                wake_models,
                controller,
            )
        )

        receiver = asyncio.create_task(
            receive_events(
                ws,
                controller,
            )
        )

        barge_monitor = asyncio.create_task(
            monitor_barge_in(
                controller
            )
        )

        rotation_monitor = asyncio.create_task(
            realtime_session_rotation_monitor(
                controller,
            )
        )

        try:
            done, pending = (
                await asyncio.wait(
                    [
                        processor,
                        receiver,
                        barge_monitor,
                        rotation_monitor,
                    ],
                    return_when=(
                        asyncio.FIRST_EXCEPTION
                    ),
                )
            )

            for task in done:
                exc = task.exception()

                if exc is not None:
                    raise exc

        finally:
            processor.cancel()
            receiver.cancel()
            barge_monitor.cancel()
            rotation_monitor.cancel()

            await asyncio.gather(
                processor,
                receiver,
                barge_monitor,
                rotation_monitor,
                return_exceptions=True,
            )

            controller.reset_for_connection()


# =========================================================
# RECONNECT MANAGER
# =========================================================

async def connection_manager(
    api_key,
    wake_models,
    barge_wake_models,
    controller,
):
    delay = (
        RECONNECT_INITIAL_SECONDS
    )

    consecutive_handshake_failures = 0
    max_handshake_failures = 3

    while True:
        connection_started = None

        try:
            connection_started = (
                time.monotonic()
            )

            await run_connection(
                api_key,
                wake_models,
                barge_wake_models,
                controller,
            )

            print(
                "Realtime connection closed."
            )

            consecutive_handshake_failures = 0

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            connection_elapsed = (
                time.monotonic()
                - connection_started
                if connection_started is not None
                else 0.0
            )

            print()

            print(
                "Realtime connection lost:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            opening_handshake_timeout = (
                isinstance(exc, TimeoutError)
                and
                "opening handshake"
                in str(exc).lower()
            )

            if opening_handshake_timeout:
                consecutive_handshake_failures += 1

                print(
                    "Realtime opening-handshake failure "
                    f"{consecutive_handshake_failures}/"
                    f"{max_handshake_failures}."
                )

                if (
                    consecutive_handshake_failures
                    >= max_handshake_failures
                ):
                    print(
                        "Repeated Realtime handshake failures; "
                        "forcing a clean systemd restart.",
                        flush=True,
                    )

                    # Raising an exception is insufficient here because
                    # audio/model shutdown can hang before Python exits.
                    # Exit immediately; nexus.service uses Restart=always
                    # and will replace the entire process group.
                    os._exit(1)

            else:
                consecutive_handshake_failures = 0

            if connection_elapsed >= 30:
                delay = (
                    RECONNECT_INITIAL_SECONDS
                )

                consecutive_handshake_failures = 0

        controller.reset_for_connection()

        print(
            f"Reconnecting in {delay} second"
            f"{'' if delay == 1 else 's'}..."
        )

        await asyncio.sleep(
            delay
        )

        print(
            "Attempting Realtime connection..."
        )

        delay = min(
            delay * 2,
            RECONNECT_MAX_SECONDS,
        )


# =========================================================
# MAIN
# =========================================================

async def main():
    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set"
        )

    print(
        f"Input:  {INPUT_DEVICE} - "
        f"{sd.query_devices(INPUT_DEVICE)['name']}"
    )

    print(
        f"Output: {OUTPUT_DEVICE} - "
        f"{sd.query_devices(OUTPUT_DEVICE)['name']}"
    )

    print(
        f"Wake model: "
        f"{WAKE_MODEL_PATH}"
    )

    print(
        f"Wake threshold: "
        f"{WAKE_THRESHOLD}"
    )

    print(
        f"Speaker threshold: "
        f"{SPEAKER_THRESHOLD}"
    )

    print(
        "Microphone mode: "
        "XVF3800 adaptive dual-channel"
    )

    print(
        "Reconnect: enabled"
    )

    print(
        "Local timers: enabled"
    )

    print(
        "Persistent alarms: enabled"
    )

    print(
        "Relative alarms: enabled"
    )

    print(
        f"Alarm file: {ALARMS_FILE}"
    )

    print(
        f"Settings file: {SETTINGS_FILE}"
    )

    saved_location = (
        load_settings()
        .get(
            "location"
        )
    )

    if saved_location:
        print(
            "Saved location: configured"
        )
    else:
        print(
            "Saved location: not set"
        )

    print(
        "Weather: Open-Meteo enabled"
    )

    print(
        "Current clock tool: enabled"
    )

    print(
        "Barge-in: stream-reopen raw-mic Hey Nexus interruption enabled"
    )

    print(
        "Barge debug: CH1 raw-mic wake-score peaks enabled"
    )

    print(
        f"Follow-up window: "
        f"{FOLLOWUP_SECONDS:.0f}s"
    )

    print()

    print(
        "Restoring normal ReSpeaker route..."
    )

    try:
        route_status = (
            await set_respeaker_route_async(
                "barge-off"
            )
        )

        print(
            "ReSpeaker normal route restored."
        )

    except Exception as exc:
        print(
            "WARNING: Could not restore "
            "normal ReSpeaker route at startup:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

    print()

    print(
        "Loading wake-word model..."
    )

    wake_models = (
        WakeWordModel(
            wakeword_model_paths=[
                WAKE_MODEL_PATH
            ]
        ),
        WakeWordModel(
            wakeword_model_paths=[
                WAKE_MODEL_PATH
            ]
        ),
    )

    print(
        "Loading interruption wake-word models..."
    )

    barge_wake_models = (
        WakeWordModel(
            wakeword_model_paths=[
                WAKE_MODEL_PATH
            ]
        ),
        WakeWordModel(
            wakeword_model_paths=[
                WAKE_MODEL_PATH
            ]
        ),
    )

    (
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    ) = await asyncio.to_thread(
        load_voice_system
    )

    controller = NexusController(
        recognizer,
        enrolled_ch0,
        enrolled_ch1,
    )

    await restore_alarms()

    if alarms:
        print(
            f"{len(alarms)} saved alarm"
            f"{'' if len(alarms) == 1 else 's'} loaded."
        )

    await asyncio.to_thread(
        open_input_stream
    )

    try:
        await connection_manager(
            api_key,
            wake_models,
            barge_wake_models,
            controller,
        )

    finally:
        await asyncio.to_thread(
            close_input_stream
        )

        for timer in list(
            timers.values()
        ):
            timer["task"].cancel()

        for alarm in list(
            alarms.values()
        ):
            task = alarm.get(
                "task"
            )

            if task:
                task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print()

        print(
            "Shutting down."
        )
