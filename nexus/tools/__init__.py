"""Features, as plugins.

A feature is something the assistant can do when asked: set a timer, read
the weather, report your solar battery. Each one is a *tool* the voice model
may call, registered by a plugin. A plugin is a Python file with one
function:

    def register(nexus):
        nexus.tool(
            name="roll_dice",
            description="Roll dice and return the numbers.",
            parameters={"type": "object", "properties": {"count": {"type": "integer"}}},
            handler=roll_dice,              # async def roll_dice(arguments) -> dict
        )

Plugins come from three places, loaded in this order:

  1. built-ins named in NEXUS_PLUGINS (nexus/tools/builtin/);
  2. every *.py file in NEXUS_PLUGIN_DIR (default: plugins/ in NEXUS_HOME);
  3. installed packages that declare a "nexus.plugins" entry point.

A plugin that fails to load or register is logged and skipped. It can never
stop the assistant from starting. docs/ADDING-FEATURES.md has the full guide.
"""

import asyncio
import importlib
import importlib.util
import inspect
import re
import traceback
from pathlib import Path

from .. import config

env_str, env_float, env_int, env_flag = config.group("plugins")

BUILTIN_PLUGINS = ("timers", "alarms", "location", "clock", "weather")

PLUGINS = env_str(
    "NEXUS_PLUGINS",
    ",".join(BUILTIN_PLUGINS),
    'Built-in features to load, comma separated; "none" for none.',
)

PLUGIN_DIR = Path(
    env_str(
        "NEXUS_PLUGIN_DIR",
        str(config.BASE_DIR / "plugins"),
        "Folder of your own plugin files; every *.py in it is loaded.",
    )
).expanduser()

TOOL_TIMEOUT_SECONDS = env_float(
    "NEXUS_TOOL_TIMEOUT_SECONDS",
    30,
    "A tool that runs longer than this is stopped and reported as failed.",
)

NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class Tool:
    def __init__(self, name, description, parameters, handler, plugin, redact=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.plugin = plugin
        self.redact = redact


class Host:
    """What the running assistant offers to plugins. The engine fills these in."""

    async def ring(self, frequency, beep_seconds, gap_seconds, count, volume, ring_seconds=0.0):
        """Play beeps until the wake phrase silences them. Returns True if silenced."""
        return False

    def alert(self, title, body):
        """Push a message to the owner's phone, if alerts are set up."""

    def note_missed(self, description):
        """Record something that came due while Nexus was off."""


host = Host()

# What plugins registered.
TOOLS = {}
INSTRUCTIONS = []
START_HOOKS = []
STOP_HOOKS = []

# For the start-up log and `doctor`.
LOADED = []
FAILED = []


class PluginAPI:
    """The object a plugin's register() receives."""

    def __init__(self, plugin_name):
        self.plugin_name = plugin_name
        self.data_dir = config.BASE_DIR

    def tool(self, name, description, parameters, handler, redact=None):
        """Offer a tool to the voice model.

        name         letters, digits, _ and -; unique across all plugins.
        description  one or two sentences telling the model when to call it.
        parameters   a JSON Schema object describing the arguments.
        handler      async def handler(arguments: dict) -> dict. Return
                     something JSON-able; {"success": False, "error": "..."}
                     for a failure the model should explain. An exception is
                     caught and reported the same way.
        redact       optional def redact(value) -> value, applied to the
                     arguments and the result before they are logged, for
                     tools that handle private data such as an address.
        """
        if not NAME_PATTERN.match(str(name)):
            raise ValueError(f"Tool name {name!r} must be 1 to 64 letters, digits, _ or -.")

        if name in TOOLS:
            raise ValueError(f"Tool {name!r} is already registered by plugin {TOOLS[name].plugin!r}.")

        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Tool {name!r} needs a description.")

        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f'Tool {name!r}: parameters must be a JSON Schema with "type": "object".')

        if not inspect.iscoroutinefunction(handler):
            raise ValueError(f"Tool {name!r}: handler must be an async function taking one dict.")

        TOOLS[name] = Tool(name, description.strip(), parameters, handler, self.plugin_name, redact)

    def instructions(self, text):
        """Add a paragraph to the system prompt: when and how to use your tools."""
        INSTRUCTIONS.append((self.plugin_name, " ".join(str(text).split())))

    def on_start(self, function):
        """Run `async def function()` once at start, inside the event loop (restore saved state)."""
        START_HOOKS.append((self.plugin_name, function))

    def on_stop(self, function):
        """Run `def function()` at shutdown (cancel tasks, close files)."""
        STOP_HOOKS.append((self.plugin_name, function))

    def settings(self):
        """Return (env_str, env_float, env_int, env_flag) for this plugin's own settings."""
        return config.group(f"plugin: {self.plugin_name}")

    def part(self, kind, name, target):
        """Register a hardware part (audio, wake, speaker_id, backend) under a short name."""
        from .. import parts

        parts.register(kind, name, target)

    async def ring(
        self, frequency=880, beep_seconds=0.35, gap_seconds=0.25, count=10, volume=0.25, ring_seconds=0.0
    ):
        """Play beeps through the speaker until the wake phrase silences them."""
        return await host.ring(frequency, beep_seconds, gap_seconds, count, volume, ring_seconds=ring_seconds)

    def alert(self, title, body):
        """Push a message to the owner's phone (does nothing unless alerts are set up)."""
        host.alert(title, body)

    def note_missed(self, description):
        """Report something that came due while Nexus was off."""
        host.note_missed(description)


def _register(plugin_name, register_function, origin):
    before = set(TOOLS)
    marks = (len(INSTRUCTIONS), len(START_HOOKS), len(STOP_HOOKS))

    try:
        register_function(PluginAPI(plugin_name))
    except Exception as exc:
        # Undo whatever the plugin managed to register before it failed.
        for name in set(TOOLS) - before:
            del TOOLS[name]

        del INSTRUCTIONS[marks[0] :]
        del START_HOOKS[marks[1] :]
        del STOP_HOOKS[marks[2] :]

        _failed(plugin_name, origin, exc)
        return False

    LOADED.append((plugin_name, origin, sorted(set(TOOLS) - before)))
    return True


def _failed(plugin_name, origin, exc):
    FAILED.append((plugin_name, origin, f"{type(exc).__name__}: {exc}"))
    print(f"Plugin {plugin_name!r} ({origin}) skipped: {type(exc).__name__}: {exc}")
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).rstrip())


def load_builtin(names=None):
    if names is None:
        raw = PLUGINS.strip().lower()
        names = [] if raw in ("none", "off", "") else [n.strip() for n in raw.split(",") if n.strip()]

    for name in names:
        try:
            module = importlib.import_module(f"{__name__}.builtin.{name}")
        except Exception as exc:
            _failed(name, "built-in", exc)
            continue

        _register(name, module.register, "built-in")


def load_file(path):
    """Load one plugin file. Returns True if it registered."""
    path = Path(path)
    name = path.stem

    try:
        spec = importlib.util.spec_from_file_location(f"nexus_plugin_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register_function = getattr(module, "register", None)

        if not callable(register_function):
            raise TypeError("the file has no register(nexus) function")
    except Exception as exc:
        _failed(name, str(path), exc)
        return False

    return _register(name, register_function, str(path))


def load_folder(folder=None):
    folder = Path(folder or PLUGIN_DIR)

    if not folder.is_dir():
        return

    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue

        load_file(path)


def load_entry_points():
    try:
        from importlib.metadata import entry_points

        try:
            found = entry_points(group="nexus.plugins")
        except TypeError:  # Python 3.9
            found = entry_points().get("nexus.plugins", [])
    except Exception:
        return

    for entry in found:
        try:
            register_function = entry.load()
        except Exception as exc:
            _failed(entry.name, "entry point", exc)
            continue

        _register(entry.name, register_function, "entry point")


def load_all():
    """Load every plugin. Never raises."""
    load_builtin()
    load_folder()
    load_entry_points()


def clear():
    """Forget every registered plugin (for tests)."""
    TOOLS.clear()
    del INSTRUCTIONS[:], START_HOOKS[:], STOP_HOOKS[:], LOADED[:], FAILED[:]


def schemas():
    """The tool list in the shape voice services expect (JSON Schema function tools)."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in TOOLS.values()
    ]


def plugin_instructions():
    return " ".join(text for _, text in INSTRUCTIONS)


def redact(name, value):
    """What may be written to the log for this tool's arguments or result."""
    tool = TOOLS.get(name)

    if tool is None or tool.redact is None:
        return value

    try:
        return tool.redact(value)
    except Exception:
        return "[redacted]"


async def execute(name, arguments):
    """Run a tool. Always returns a dict; never raises."""
    tool = TOOLS.get(name)

    if tool is None:
        return {"success": False, "error": f"Unknown tool: {name}"}

    if not isinstance(arguments, dict):
        arguments = {}

    try:
        result = await asyncio.wait_for(tool.handler(arguments), TOOL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return {"success": False, "error": f"{name} took longer than {TOOL_TIMEOUT_SECONDS:g} seconds."}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(result, dict):
        result = {"result": result}

    return result


async def start_plugins():
    for plugin_name, function in list(START_HOOKS):
        try:
            await function()
        except Exception as exc:
            print(f"Plugin {plugin_name!r} failed to start: {type(exc).__name__}: {exc}")


def stop_plugins():
    for plugin_name, function in list(STOP_HOOKS):
        try:
            function()
        except Exception as exc:
            print(f"Plugin {plugin_name!r} failed to stop cleanly: {type(exc).__name__}: {exc}")
