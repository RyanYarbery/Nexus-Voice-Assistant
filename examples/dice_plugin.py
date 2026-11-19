"""Example plugin: roll dice.

The smallest useful plugin. Copy it into your plugins/ folder and say
"roll two dice".

    python -m nexus check-plugin examples/dice_plugin.py
"""

import random


async def roll_dice(arguments):
    count = max(1, min(int(arguments.get("count", 1)), 20))
    sides = max(2, min(int(arguments.get("sides", 6)), 1000))
    numbers = [random.randint(1, sides) for _ in range(count)]

    return {"numbers": numbers, "total": sum(numbers)}


def register(nexus):
    nexus.tool(
        name="roll_dice",
        description="Roll dice and return each number and the total. Use when the user asks for a dice roll.",
        parameters={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "How many dice, 1 to 20. Default 1."},
                "sides": {"type": "integer", "description": "Sides per die. Default 6."},
            },
        },
        handler=roll_dice,
    )
