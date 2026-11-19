"""Example plugin: read a status page on your own network and answer from it.

"What's the battery at?" This one reads a JSON document from a URL you set
(a solar monitor, a weather station, a NAS: anything that serves JSON) and
hands the model the fields you list. It shows three things a real plugin
usually needs: its own settings, slow work kept off the event loop, and a
failure reported in words the model can pass on.

Copy it into plugins/ and set, in your env file:

    NEXUS_SOLAR_URL=http://192.168.1.50:5000/api/latest
    NEXUS_SOLAR_FIELDS=battery_soc,pv_watts,load_watts

    python -m nexus check-plugin examples/solar_plugin.py
"""

import asyncio
import json
import urllib.request

SOLAR_URL = ""
SOLAR_FIELDS = ""


def fetch(url, wanted):
    with urllib.request.urlopen(url, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))

    if not isinstance(data, dict):
        return {"success": False, "error": "The status page did not return a JSON object."}

    if wanted:
        data = {name: data[name] for name in wanted if name in data}

    return {"success": True, "readings": data}


async def get_solar_status(arguments):
    if not SOLAR_URL:
        return {"success": False, "error": "No status address is set. Set NEXUS_SOLAR_URL."}

    wanted = [name.strip() for name in SOLAR_FIELDS.split(",") if name.strip()]

    try:
        # urllib blocks, so run it in a worker thread: the assistant must
        # keep hearing the wake phrase while it waits.
        return await asyncio.to_thread(fetch, SOLAR_URL, wanted)
    except Exception as exc:
        return {"success": False, "error": f"The status page could not be read: {exc}"}


def register(nexus):
    global SOLAR_URL, SOLAR_FIELDS

    env_str, env_float, env_int, env_flag = nexus.settings()

    SOLAR_URL = env_str("NEXUS_SOLAR_URL", "", "Address of a JSON status page on your network.")
    SOLAR_FIELDS = env_str(
        "NEXUS_SOLAR_FIELDS", "", "Fields to read from it, comma separated; empty for all."
    )

    nexus.tool(
        name="get_solar_status",
        description=(
            "Read the home solar system's live figures (battery charge, panel power, load). "
            "Use for questions about the battery, solar production or power use."
        ),
        parameters={"type": "object", "properties": {}},
        handler=get_solar_status,
    )

    nexus.instructions(
        "For questions about the home battery or solar power, call get_solar_status and answer "
        "from its readings in one short sentence."
    )
