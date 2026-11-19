"""Countdown timers that survive a restart.

"Set a timer for ten minutes." A finished timer beeps until the wake phrase
silences it. Timers are saved to a file, so a restart or a power cut does not
lose them: one that ended while Nexus was off rings at start if it is recent,
and is reported as missed if it is not.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from ... import config, tools
from ..timeutil import LATE_GRACE_SECONDS, format_alarm_time, format_duration, parse_alarm_datetime

env_str, env_float, env_int, env_flag = config.group("plugin: timers")

TIMERS_FILE = Path(env_str("NEXUS_TIMERS_FILE", str(config.BASE_DIR / "timers.json"), "Saved timers."))

TIMER_TONE_FREQUENCY = 880
TIMER_BEEP_SECONDS = 0.35
TIMER_BEEP_GAP = 0.25
TIMER_BEEP_COUNT = env_int("NEXUS_TIMER_BEEP_COUNT", 10, "Beeps when a timer ends.")

TIMER_VOLUME = env_float("NEXUS_TIMER_VOLUME", 0.25, "Timer beep volume, 0 to 1.")

TIMER_RING_SECONDS = env_float(
    "NEXUS_TIMER_RING_SECONDS",
    60,
    "A finished timer rings until the wake phrase or this long (0 = a fixed number of beeps).",
)

timers = {}
next_timer_id = 1

# The assistant, as this plugin sees it. register() replaces it.
_nexus = tools.PluginAPI("timers")


async def play_timer_alarm(
    timer_id,
    label,
):
    print()
    print("=================================")
    print(f"TIMER #{timer_id} FINISHED")
    print(f"Label: {label}")
    print("=================================")

    silenced = await _nexus.ring(
        TIMER_TONE_FREQUENCY,
        TIMER_BEEP_SECONDS,
        TIMER_BEEP_GAP,
        TIMER_BEEP_COUNT,
        TIMER_VOLUME,
        ring_seconds=TIMER_RING_SECONDS,
    )

    if not silenced and TIMER_RING_SECONDS > 0:
        _nexus.alert("Nexus timer not dismissed", f"Timer '{label}' rang and nobody answered.")


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

        if remaining < -LATE_GRACE_SECONDS:
            print(f"Discarding stale timer #{timer_id}: {label}")
            _nexus.note_missed(f"the timer '{label}', which ended at {format_alarm_time(due_at)}")
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


def stop():
    for timer in list(timers.values()):
        timer["task"].cancel()


async def _set_timer(arguments):
    return await set_local_timer(arguments.get("duration_seconds"), arguments.get("label", "Timer"))


async def _list_timers(arguments):
    return await list_local_timers()


async def _cancel_timer(arguments):
    return await cancel_local_timer(timer_id=arguments.get("timer_id"), label=arguments.get("label"))


def register(nexus):
    global _nexus
    _nexus = nexus

    nexus.tool(
        name="set_timer",
        description="Create a local countdown timer on the household assistant.",
        parameters={
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "number", "description": "Timer duration in seconds."},
                "label": {"type": "string", "description": "Short useful label for the timer."},
            },
            "required": ["duration_seconds", "label"],
        },
        handler=_set_timer,
    )
    nexus.tool(
        name="list_timers",
        description="List all currently active countdown timers.",
        parameters={"type": "object", "properties": {}},
        handler=_list_timers,
    )
    nexus.tool(
        name="cancel_timer",
        description="Cancel an active countdown timer.",
        parameters={
            "type": "object",
            "properties": {"timer_id": {"type": "integer"}, "label": {"type": "string"}},
        },
        handler=_cancel_timer,
    )
    nexus.on_start(restore_timers)
    nexus.on_stop(stop)
