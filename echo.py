"""Launcher kept for installs that start Nexus as `python echo.py`.

The assistant itself lives in the nexus/ package. New installs should use
`python -m nexus`, which defaults to plain hardware: any sound card and no
voice check.

This file starts the original reference build instead: ReSpeaker XVF3800
array, Dell analogue output, ECAPA voice check and the sardonic persona. It
only fills in settings that the environment does not already set, so an
existing env file keeps working unchanged.
"""

import os
import sys
from pathlib import Path

LEGACY_DEFAULTS = {
    "NEXUS_HOME": str(Path(__file__).resolve().parent),
    "NEXUS_AUDIO": "xvf3800",
    "NEXUS_WAKE": "openwakeword",
    "NEXUS_SPEAKER_ID": "ecapa",
    "NEXUS_BACKEND": "openai",
    "NEXUS_PERSONA": "sardonic",
    "NEXUS_MIC_NAME": "respeaker xvf3800",
    "NEXUS_SPEAKER_NAME": "alc3234 analog",
}

for _name, _value in LEGACY_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

# NEXUS_SPEAKER_VERIFY_MODE=off used to be how the voice check was turned off.
if os.environ.get("NEXUS_SPEAKER_VERIFY_MODE", "").strip().lower() == "off":
    os.environ["NEXUS_SPEAKER_ID"] = "none"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexus.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
