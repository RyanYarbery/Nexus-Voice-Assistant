# Your plugins

Put a Python file here and Nexus loads it at start. One file is one feature:

```python
async def roll(arguments):
    import random
    return {"numbers": [random.randint(1, 6) for _ in range(arguments.get("count", 1))]}

def register(nexus):
    nexus.tool(
        name="roll_dice",
        description="Roll six-sided dice and return the numbers.",
        parameters={"type": "object", "properties": {"count": {"type": "integer"}}},
        handler=roll,
    )
```

Check a file before trusting it: `python -m nexus check-plugin plugins/my_plugin.py`.
A plugin that fails is logged and skipped; it cannot stop Nexus from starting.

Files here are not committed (see `.gitignore`). The full guide is
[docs/ADDING-FEATURES.md](../docs/ADDING-FEATURES.md); working examples are in
[examples/](../examples).
