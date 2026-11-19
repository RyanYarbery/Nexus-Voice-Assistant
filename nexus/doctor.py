"""`python -m nexus doctor`: check the setup and say in plain words what is wrong.

It opens no audio stream, loads no model, and never prints a secret.
"""

import importlib
import os
import shutil
import sys

from . import config, parts

EVERY_MODULE = (
    "nexus.engine",
    "nexus.alerts",
    "nexus.tools",
    "nexus.tools.timeutil",
    "nexus.tools.builtin.timers",
    "nexus.tools.builtin.alarms",
    "nexus.tools.builtin.location",
    "nexus.tools.builtin.clock",
    "nexus.tools.builtin.weather",
    "nexus.audio.sounddevice_io",
    "nexus.audio.xvf3800",
    "nexus.audio.fake",
    "nexus.wake.openwakeword_engine",
    "nexus.wake.fake",
    "nexus.speaker_id.none",
    "nexus.speaker_id.ecapa",
    "nexus.speaker_id.fake",
    "nexus.backends.openai_realtime",
    "nexus.backends.fake",
)


def import_everything():
    """Import every built-in module so every setting is registered. Loads no model, opens no device."""
    for name in EVERY_MODULE:
        importlib.import_module(name)


def run_doctor():
    problems = []
    notes = []

    def ok(text):
        print(f"OK       {text}")

    def problem(text):
        problems.append(text)
        print(f"PROBLEM  {text}")

    def note(text):
        notes.append(text)
        print(f"NOTE     {text}")

    print(f"Nexus {config.code_version()} doctor")
    print(f"Home folder: {config.BASE_DIR}")
    print()

    if sys.version_info < (3, 9):
        problem(f"Python {sys.version.split()[0]} is too old; Nexus needs 3.9 or newer.")
    else:
        ok(f"Python {sys.version.split()[0]}")

    try:
        import numpy

        ok(f"numpy {numpy.__version__}")
    except ImportError as exc:
        problem(f"numpy is not installed ({exc}). pip install numpy")

    try:
        import scipy  # noqa: F401

        ok("scipy (better resampling)")
    except ImportError:
        note("scipy is not installed; audio is resampled with a simpler method. pip install scipy")

    # ---- plugins (first: one may register a hardware part) -----------------
    print()

    try:
        from . import tools

        tools.clear()
        tools.load_all()

        for plugin_name, origin, tool_names in tools.LOADED:
            ok(f"plugin {plugin_name} ({origin}): {', '.join(tool_names) or 'no tools'}")

        for plugin_name, origin, error in tools.FAILED:
            problem(f"plugin {plugin_name} ({origin}) does not load and is skipped: {error}")
    except Exception as exc:
        problem(f"the plugin system could not be loaded: {type(exc).__name__}: {exc}")

    # ---- the four parts ----------------------------------------------------
    print()

    labels = {
        "audio": "microphone and speaker",
        "wake": "wake word",
        "speaker_id": "voice check",
        "backend": "voice service",
    }

    for kind, label in labels.items():
        chosen = parts.chosen(kind)

        try:
            part = parts.create(kind)
        except parts.PartError as exc:
            problem(str(exc))
            continue
        except Exception as exc:
            problem(
                f"{label} ({parts.SETTING_FOR[kind]}={chosen}) could not be created: {type(exc).__name__}: {exc}"
            )
            continue

        try:
            found = list(part.check() or [])
        except Exception as exc:
            found = [f"its self-check failed: {type(exc).__name__}: {exc}"]

        if found:
            for text in found:
                problem(f"{label} ({chosen}): {text}")
        else:
            ok(f"{label}: {chosen}")

        if kind == "speaker_id" and not part.enabled:
            note(
                "no voice check: anyone who says the wake phrase is answered (NEXUS_SPEAKER_ID=ecapa adds one)."
            )

    # ---- alerts ------------------------------------------------------------
    print()

    from . import alerts

    if alerts.STATUS_FILE is None:
        note("the status file is switched off (NEXUS_STATUS_FILE=off); a watchdog cannot see inside Nexus.")
    elif os.access(alerts.STATUS_FILE.parent, os.W_OK):
        ok(f"status file: {alerts.STATUS_FILE}")
    else:
        problem(f"the status file's folder is not writable: {alerts.STATUS_FILE.parent}")

    if alerts.NTFY_URL:
        ok("push alerts: an ntfy address is set")
    else:
        note("no push alerts: set NEXUS_NTFY_URL to hear about problems on your phone.")

    if not alerts.OFFLINE_NOTICES:
        note("spoken notices are switched off (NEXUS_OFFLINE_NOTICES=0).")
    elif alerts.NOTICE_TTS == "auto" and not (shutil.which("espeak-ng") or shutil.which("espeak")):
        note("espeak-ng is not installed, so notices are tones, not words. apt install espeak-ng")
    else:
        ok("spoken notices")

    # ---- settings nobody reads --------------------------------------------
    try:
        import_everything()
    except Exception as exc:
        note(f"could not import every module to check setting names: {type(exc).__name__}: {exc}")
    else:
        known = {entry["name"] for entry in config.ALL_SETTINGS} | {"NEXUS_COMMIT"}

        for name in sorted(os.environ):
            if name.startswith("NEXUS_") and name not in known:
                note(f"{name} is set but nothing reads it. A typo? See docs/SETTINGS.md.")

    print()

    if problems:
        print(f"{len(problems)} problem(s) to fix before Nexus will work properly.")
        return 1

    print("No problems found." + (f" {len(notes)} note(s) above." if notes else ""))
    return 0
