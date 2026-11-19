"""A saved home location, used by the clock and weather plugins.

"Set my location to ..." geocodes the address with OpenStreetMap's Nominatim
and saves it to settings.json (readable only by the owner). The address is
never written to the log.
"""

import asyncio
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ... import config
from ...config import HTTP_TIMEOUT_SECONDS, HTTP_USER_AGENT

env_str, env_float, env_int, env_flag = config.group("plugin: location")

SETTINGS_FILE = Path(
    env_str("NEXUS_SETTINGS_FILE", str(config.BASE_DIR / "settings.json"), "Saved location and preferences.")
)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

settings_lock = threading.Lock()
last_nominatim_request = 0.0


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


def _redact_address(value):
    if isinstance(value, dict) and "address" in value:
        safe = dict(value)
        safe["address"] = "[redacted from the log]"
        return safe

    return value


async def _set_location(arguments):
    return await set_saved_location(arguments.get("address"))


async def _get_location(arguments):
    return await get_saved_location()


async def _forget_location(arguments):
    return await forget_saved_location()


def register(nexus):
    nexus.tool(
        name="set_location",
        description="Geocode and persist the user's full street address as Nexus's saved location.",
        parameters={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "The user's full address exactly as spoken."}
            },
            "required": ["address"],
        },
        handler=_set_location,
        redact=_redact_address,
    )
    nexus.tool(
        name="get_location",
        description="Return the location currently saved on the Dell.",
        parameters={"type": "object", "properties": {}},
        handler=_get_location,
        redact=_redact_address,
    )
    nexus.tool(
        name="forget_location",
        description="Delete the user's saved location. Use only when explicitly requested.",
        parameters={"type": "object", "properties": {}},
        handler=_forget_location,
    )
    nexus.instructions(
        "A persistent location may already be stored locally in settings.json. Treat "
        "the saved location as Nexus's default operating location. Never ask the user "
        "to repeat their location merely because it was not mentioned in the current "
        "conversation. For weather or other requests that need the user's location, use "
        "the stored location automatically. Call get_weather directly for weather; it "
        "already uses the saved coordinates. If you specifically need to know what "
        "location is stored, call get_location. Only ask the user for a location if the "
        "appropriate tool reports that no saved location exists. When the user asks to "
        "set or change the location, call set_location. When explicitly asked what "
        "location is saved, call get_location. Only call forget_location when the user "
        "explicitly asks to remove the stored location. Do not claim to have forgotten "
        "a saved location without checking the persistent location tools first. "
    )
