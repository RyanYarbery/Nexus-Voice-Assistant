"""The current date and time, in the saved location's time zone if there is one."""

from datetime import datetime

from .location import get_location_timezone, load_settings


async def get_current_datetime_tool():
    tzinfo, timezone_name = get_location_timezone()

    now = datetime.now(tzinfo)

    location = load_settings().get(
        "location",
        {},
    )

    return {
        "success": True,
        "iso_datetime": now.isoformat(),
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M:%S %p").lstrip("0"),
        "timezone": timezone_name,
        "location": location.get("display_name"),
    }


async def _get_current_datetime(arguments):
    return await get_current_datetime_tool()


def register(nexus):
    nexus.tool(
        name="get_current_datetime",
        description="Return the actual current local date and time, using the saved location's timezone when available.",
        parameters={"type": "object", "properties": {}},
        handler=_get_current_datetime,
    )
    nexus.instructions("For any request that depends on the current date or time, call get_current_datetime.")
