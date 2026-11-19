"""Settings.

Every value a user might tune is read from the environment through one of
the helpers here. Each call records the setting (name, value, default, a
one-line description and the part it belongs to), which is what lets Nexus
print the settings in effect at start, lets `python -m nexus settings` list
them, and keeps docs/SETTINGS.md honest.

A module gets its helpers with:

    env_str, env_float, env_int, env_flag = config.group("audio: my-mic")

Nothing here touches hardware or loads a model.
"""

import os
import subprocess
from pathlib import Path

# name -> entry, for the parts loaded in this process.
SETTINGS = {}

# Every entry ever registered, including two parts that give the same
# name different defaults. Used to write docs/SETTINGS.md.
ALL_SETTINGS = []


def _setting(group_name, name, default, convert, doc, secret=False):
    raw = os.environ.get(name)
    value = default

    if raw is not None and raw.strip() != "":
        try:
            value = convert(raw.strip())
        except (TypeError, ValueError):
            print(f"WARNING: {name}={raw!r} is not valid; using {default!r}.")

    entry = {
        "name": name,
        "group": group_name,
        "value": value,
        "default": default,
        "doc": doc,
        "secret": secret,
    }

    SETTINGS[name] = entry

    if not any(e["name"] == name and e["group"] == group_name for e in ALL_SETTINGS):
        ALL_SETTINGS.append(entry)

    return value


def _to_flag(raw):
    return raw.lower() in ("1", "true", "yes", "on")


def group(group_name):
    """Return (env_str, env_float, env_int, env_flag) that record settings under `group_name`."""

    def env_str(name, default, doc="", secret=False):
        return _setting(group_name, name, default, str, doc, secret)

    def env_float(name, default, doc=""):
        return _setting(group_name, name, default, float, doc)

    def env_int(name, default, doc=""):
        return _setting(group_name, name, default, lambda raw: int(float(raw)), doc)

    def env_flag(name, default=False, doc=""):
        return _setting(group_name, name, default, _to_flag, doc)

    return env_str, env_float, env_int, env_flag


env_str, env_float, env_int, env_flag = group("core")

# The engine works on 16 kHz, 16-bit microphone audio in blocks of 80 ms.
# Wake-word models are built for exactly this, so it is not a setting.
INPUT_RATE = 16000
INPUT_BLOCK_FRAMES = 1280

# Sample rate of locally generated sounds (beeps, tones, notices).
CLIP_RATE = 48000

BASE_DIR = Path(
    env_str(
        "NEXUS_HOME",
        str(Path(__file__).resolve().parent.parent),
        "Folder holding models, plugins and saved state.",
    )
).expanduser()

HTTP_TIMEOUT_SECONDS = 12

HTTP_USER_AGENT = env_str(
    "NEXUS_HTTP_USER_AGENT",
    "NexusHouseholdAssistant/1.0 (personal non-commercial assistant)",
    "User-Agent sent to web services (geocoding, weather, alerts).",
)


def describe_settings():
    """One line per setting in effect. Secret values are never shown."""
    lines = []

    for name in sorted(SETTINGS, key=lambda n: (SETTINGS[n]["group"], n)):
        entry = SETTINGS[name]
        mark = "*" if entry["value"] != entry["default"] else " "
        shown = "(set)" if entry["secret"] and entry["value"] else entry["value"]
        lines.append(f"  {mark} {name}={shown}   [{entry['group']}]")

    return lines


def settings_markdown():
    """The settings table for docs/SETTINGS.md, built from what the code registers."""
    home = str(BASE_DIR)
    lines = []

    groups = []

    for entry in ALL_SETTINGS:
        if entry["group"] not in groups:
            groups.append(entry["group"])

    for group_name in groups:
        lines.append(f"## {group_name}")
        lines.append("")
        lines.append("| Setting | Default | What it does |")
        lines.append("|---|---|---|")

        for entry in sorted((e for e in ALL_SETTINGS if e["group"] == group_name), key=lambda e: e["name"]):
            if entry["name"] == "NEXUS_HOME":
                default = "the folder Nexus is installed in"
            elif isinstance(entry["default"], bool):
                default = "`1`" if entry["default"] else "`0`"
            elif entry["default"] == "":
                default = "(empty)"
            else:
                default = "`" + str(entry["default"]).replace(home, "$NEXUS_HOME") + "`"

            lines.append(f"| `{entry['name']}` | {default} | {entry['doc']} |")

        lines.append("")

    return "\n".join(lines)


def code_version():
    """The commit this code came from: NEXUS_COMMIT, a COMMIT file, or git."""
    explicit = os.environ.get("NEXUS_COMMIT", "").strip()

    if explicit:
        return explicit

    here = Path(__file__).resolve().parent.parent

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
