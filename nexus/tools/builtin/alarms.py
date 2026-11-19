"""Alarms at a time of day, saved across restarts.

"Wake me at seven." An alarm beeps until the wake phrase silences it. One
that came due while Nexus was off rings at start if it is recent, and is
reported as missed if it is not.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from ... import config, tools
from ..timeutil import LATE_GRACE_SECONDS, format_alarm_time, parse_alarm_datetime

env_str, env_float, env_int, env_flag = config.group("plugin: alarms")

ALARMS_FILE = Path(env_str("NEXUS_ALARMS_FILE", str(config.BASE_DIR / "alarms.json"), "Saved alarms."))

ALARM_TONE_FREQUENCY = 1050
ALARM_BEEP_SECONDS = 0.45
ALARM_BEEP_GAP = 0.20
ALARM_BEEP_COUNT = env_int("NEXUS_ALARM_BEEP_COUNT", 16, "Beeps when an alarm goes off.")

ALARM_VOLUME = env_float("NEXUS_ALARM_VOLUME", 0.30, "Alarm beep volume, 0 to 1.")

ALARM_RING_SECONDS = env_float(
    "NEXUS_ALARM_RING_SECONDS",
    180,
    "An alarm rings until the wake phrase or this long (0 = a fixed number of beeps).",
)

alarms = {}
next_alarm_id = 1

# The assistant, as this plugin sees it. register() replaces it.
_nexus = tools.PluginAPI("alarms")


async def play_alarm_sound(
    alarm_id,
    label,
):
    print()
    print("=================================")
    print(f"ALARM #{alarm_id}")
    print(f"Label: {label}")
    print("=================================")

    silenced = await _nexus.ring(
        ALARM_TONE_FREQUENCY,
        ALARM_BEEP_SECONDS,
        ALARM_BEEP_GAP,
        ALARM_BEEP_COUNT,
        ALARM_VOLUME,
        ring_seconds=ALARM_RING_SECONDS,
    )

    if not silenced and ALARM_RING_SECONDS > 0:
        _nexus.alert("Nexus alarm not dismissed", f"Alarm '{label}' rang and nobody answered.")


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

        if target.timestamp() < now.timestamp() - LATE_GRACE_SECONDS:
            print(f"Discarding expired alarm #{alarm_id}: {label}")
            _nexus.note_missed(f"the alarm '{label}' set for {format_alarm_time(target)}")
            continue

        if target.timestamp() <= now.timestamp():
            # Came due while Nexus was off or starting, recently enough
            # to still matter: the worker rings it at once.
            print(f"Alarm #{alarm_id} came due while Nexus was off; ringing it now.")

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


def stop():
    for alarm in list(alarms.values()):
        task = alarm.get("task")

        if task:
            task.cancel()


async def _set_relative_alarm(arguments):
    return await set_relative_alarm(arguments.get("delay_seconds"), arguments.get("label", "Alarm"))


async def _set_alarm(arguments):
    return await set_local_alarm(arguments.get("target_datetime"), arguments.get("label", "Alarm"))


async def _list_alarms(arguments):
    return await list_local_alarms()


async def _cancel_alarm(arguments):
    return await cancel_local_alarm(alarm_id=arguments.get("alarm_id"), label=arguments.get("label"))


def register(nexus):
    global _nexus
    _nexus = nexus

    nexus.tool(
        name="set_relative_alarm",
        description="Set a local alarm relative to the current moment. Always use this for requests such as 'in 3 minutes', 'in two hours', or '30 seconds from now'.",
        parameters={
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "number",
                    "description": "Exact delay from the current moment in seconds.",
                },
                "label": {"type": "string", "description": "Short useful label for the alarm."},
            },
            "required": ["delay_seconds", "label"],
        },
        handler=_set_relative_alarm,
    )
    nexus.tool(
        name="set_alarm",
        description="Set a persistent local alarm for a specific future date and clock time.",
        parameters={
            "type": "object",
            "properties": {
                "target_datetime": {
                    "type": "string",
                    "description": "Exact future local date and time "
                    "in ISO 8601 format, preferably "
                    "including the UTC offset.",
                },
                "label": {"type": "string", "description": "Short useful label for the alarm."},
            },
            "required": ["target_datetime", "label"],
        },
        handler=_set_alarm,
    )
    nexus.tool(
        name="list_alarms",
        description="List all currently active persistent alarms.",
        parameters={"type": "object", "properties": {}},
        handler=_list_alarms,
    )
    nexus.tool(
        name="cancel_alarm",
        description="Cancel an active persistent alarm.",
        parameters={
            "type": "object",
            "properties": {"alarm_id": {"type": "integer"}, "label": {"type": "string"}},
        },
        handler=_cancel_alarm,
    )
    nexus.on_start(restore_alarms)
    nexus.on_stop(stop)
