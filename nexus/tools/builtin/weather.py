"""Weather for the saved location, from Open-Meteo (no account or key needed)."""

import asyncio

from .location import OPEN_METEO_FORECAST_URL, http_get_json, load_settings

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


def weather_for_saved_location_sync(
    day_offset=0,
):
    settings = load_settings()
    location = settings.get("location")

    if not location:
        return {
            "success": False,
            "error": "No location is saved. Ask the user to set a location first.",
        }

    try:
        day_offset = int(day_offset)
    except Exception:
        day_offset = 0

    if day_offset < 0 or day_offset > 6:
        return {
            "success": False,
            "error": "Weather requests currently support today through six days from now.",
        }

    latitude = location["latitude"]
    longitude = location["longitude"]

    data = http_get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": 7,
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

    if day_offset >= len(dates):
        return {
            "success": False,
            "error": "Forecast data was not available for that day.",
        }

    def daily_value(
        name,
    ):
        values = daily.get(
            name,
            [],
        )

        if day_offset < len(values):
            return values[day_offset]

        return None

    weather_code = daily_value("weather_code")

    result = {
        "success": True,
        "location": location.get("display_name"),
        "timezone": data.get("timezone"),
        "day_offset": day_offset,
        "date": dates[day_offset],
        "forecast": {
            "conditions": WEATHER_CODES.get(
                weather_code,
                f"weather code {weather_code}",
            ),
            "high_f": daily_value("temperature_2m_max"),
            "low_f": daily_value("temperature_2m_min"),
            "precipitation_probability_percent": daily_value("precipitation_probability_max"),
            "precipitation_inches": daily_value("precipitation_sum"),
            "max_wind_mph": daily_value("wind_speed_10m_max"),
            "sunrise": daily_value("sunrise"),
            "sunset": daily_value("sunset"),
        },
    }

    if day_offset == 0:
        current = data.get(
            "current",
            {},
        )

        current_code = current.get("weather_code")

        result["current"] = {
            "conditions": WEATHER_CODES.get(
                current_code,
                f"weather code {current_code}",
            ),
            "temperature_f": current.get("temperature_2m"),
            "feels_like_f": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_mph": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "precipitation_inches": current.get("precipitation"),
            "rain_inches": current.get("rain"),
            "snowfall_inches": current.get("snowfall"),
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
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _get_weather(arguments):
    return await get_weather_tool(arguments.get("day_offset", 0))


def register(nexus):
    nexus.tool(
        name="get_weather",
        description="Get live current conditions and a daily forecast from Open-Meteo for the saved location. day_offset 0 is today, 1 is tomorrow, through 6.",
        parameters={
            "type": "object",
            "properties": {
                "day_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 6,
                    "description": "0 for today, 1 for tomorrow, up through 6.",
                }
            },
        },
        handler=_get_weather,
    )
    nexus.instructions(
        "You have a weather tool backed by Open-Meteo. Use get_weather for current "
        "weather and forecasts at the saved location. day_offset 0 means today, 1 means "
        "tomorrow, through 6. Do not claim that live weather is unavailable when this "
        "tool can provide it. "
    )
