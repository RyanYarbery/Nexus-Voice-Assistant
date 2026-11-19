"""Failing loudly: the status file, push alerts, and spoken notices.

Nexus must never fail silently. Three things make a failure visible:

  - a status file (NEXUS_STATUS_FILE) that a watchdog can read;
  - an optional push message through ntfy when the status changes;
  - a short spoken or tone notice when someone says the wake phrase and
    Nexus cannot answer.

None of it needs the voice service, so it works when the service does not.
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .config import BASE_DIR, CLIP_RATE, HTTP_TIMEOUT_SECONDS, HTTP_USER_AGENT
from .dsp import resample_int16

env_str, env_float, env_int, env_flag = config.group("alerts")

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

STATUS_HEARTBEAT_SECONDS = 60

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

# How a notice is played: an async function taking mono int16 audio at
# CLIP_RATE. The engine sets this to its own speaker path.
play_audio = None

ACCOUNT_PROBLEMS = ("quota", "auth", "model")

PROBLEM_LABELS = {
    "quota": "out of credit",
    "auth": "API key missing or rejected",
    "model": "voice model not available",
    "network": "cannot reach the voice service",
    "speaker": "speaker output failing",
    "microphone": "microphone stopped",
}

NOTICE_TEXT = {
    "quota": "The voice service account is out of credit.",
    "auth": "The voice service key was rejected.",
    "model": "The voice model is no longer available.",
    "network": "I cannot reach the voice service right now.",
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
    data["version"] = config.code_version()
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
        t = np.arange(int(CLIP_RATE * seconds), dtype=np.float32) / CLIP_RATE
        tone = np.sin(2 * np.pi * frequency * t) * volume * 32767
        fade = min(len(tone) // 2, int(CLIP_RATE * 0.01))

        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]

        parts.append(tone.astype(np.int16))
        parts.append(np.zeros(int(CLIP_RATE * 0.06), dtype=np.int16))

    return np.concatenate(parts)


def _wav_bytes_to_pcm(data):
    """Decode a 16-bit PCM WAV (as a speech synthesiser writes to stdout) to mono at CLIP_RATE."""
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

    return resample_int16(pcm, rate, CLIP_RATE)


def notice_audio(kind, text):
    """Audio for a notice: your own recording, a local synthesiser, or tones.

    Nothing here needs the network, so it works when the voice service does not.
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
        if play_audio is None:
            return False

        pcm = await asyncio.to_thread(notice_audio, kind, text)
        await play_audio(pcm)
    except Exception as exc:
        print(f"Could not play the notice: {type(exc).__name__}: {exc}")
        return False

    return True
