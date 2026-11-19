"""Times and durations shared by the timer and alarm plugins."""

from datetime import datetime

from .. import config

env_str, env_float, env_int, env_flag = config.group("plugin: timers and alarms")

LATE_GRACE_SECONDS = env_float(
    "NEXUS_LATE_GRACE_SECONDS",
    600,
    "A timer or alarm that came due while Nexus was off still rings if it is no later than this.",
)


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
