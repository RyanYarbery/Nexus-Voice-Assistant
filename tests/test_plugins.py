"""Plugins: loading, a broken one being skipped, tools never raising."""

import asyncio
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import harness  # noqa: F401  (stubs sounddevice before nexus is imported)
from nexus import parts, tools

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"

BUILT_IN_TOOLS = {
    "set_timer", "list_timers", "cancel_timer",
    "set_relative_alarm", "set_alarm", "list_alarms", "cancel_alarm",
    "set_location", "get_location", "forget_location",
    "get_current_datetime", "get_weather",
}  # fmt: skip


@pytest.fixture
def fresh_registry():
    """Each test starts with only the built-in plugins and leaves the registry as it found it."""
    saved = (dict(tools.TOOLS), list(tools.INSTRUCTIONS), list(tools.START_HOOKS), list(tools.STOP_HOOKS))
    saved_logs = (list(tools.LOADED), list(tools.FAILED))
    del tools.FAILED[:]
    yield
    tools.clear()
    tools.TOOLS.update(saved[0])
    tools.INSTRUCTIONS.extend(saved[1])
    tools.START_HOOKS.extend(saved[2])
    tools.STOP_HOOKS.extend(saved[3])
    tools.LOADED.extend(saved_logs[0])
    tools.FAILED.extend(saved_logs[1])


def test_the_built_in_features_are_plugins(fresh_registry):
    assert BUILT_IN_TOOLS <= set(tools.TOOLS)
    names = [schema["name"] for schema in tools.schemas()]
    assert len(names) == len(set(names))
    for schema in tools.schemas():
        assert schema["type"] == "function" and schema["parameters"]["type"] == "object"
    assert "get_weather" in tools.plugin_instructions()


def test_a_folder_of_plugins_is_loaded_and_a_broken_one_is_skipped(fresh_registry, tmp_path):
    (tmp_path / "dice.py").write_text((EXAMPLES / "dice_plugin.py").read_text())
    (tmp_path / "broken_import.py").write_text("import a_module_that_does_not_exist\n")
    (tmp_path / "broken_register.py").write_text(
        "async def h(a):\n    return {}\n\n"
        "def register(nexus):\n"
        "    nexus.tool('half_made', 'A tool registered just before the plugin fails.',"
        " {'type': 'object', 'properties': {}}, h)\n"
        "    raise RuntimeError('boom')\n"
    )
    (tmp_path / "no_register.py").write_text("X = 1\n")
    (tmp_path / "_ignored.py").write_text("raise SystemExit('files starting with _ are not loaded')\n")

    tools.load_folder(tmp_path)

    assert "roll_dice" in tools.TOOLS
    assert "half_made" not in tools.TOOLS, "a failed plugin left a tool behind"
    assert BUILT_IN_TOOLS <= set(tools.TOOLS), "a broken plugin damaged the others"
    failed = {name for name, origin, error in tools.FAILED}
    assert failed == {"broken_import", "broken_register", "no_register"}


def test_a_plugin_cannot_take_another_plugins_tool_name(fresh_registry, tmp_path):
    (tmp_path / "clash.py").write_text(
        "async def h(a):\n    return {}\n\n"
        "def register(nexus):\n"
        "    nexus.tool('set_timer', 'Tries to replace the built-in timer tool.',"
        " {'type': 'object', 'properties': {}}, h)\n"
    )
    original = tools.TOOLS["set_timer"]
    tools.load_folder(tmp_path)
    assert tools.TOOLS["set_timer"] is original
    assert "already registered" in tools.FAILED[-1][2]


def test_running_a_tool_never_raises(fresh_registry):
    async def explode(arguments):
        raise ValueError("bad input")

    async def slow(arguments):
        await asyncio.sleep(5)

    async def plain(arguments):
        return 42

    api = tools.PluginAPI("test")
    schema = {"type": "object", "properties": {}}
    api.tool("explode", "Always fails, to test error handling.", schema, explode)
    api.tool("slow", "Never finishes, to test the time limit.", schema, slow)
    api.tool("plain", "Returns a bare number, to test wrapping.", schema, plain)

    async def body():
        assert (await tools.execute("no_such_tool", {}))["success"] is False
        assert "ValueError: bad input" in (await tools.execute("explode", {}))["error"]
        assert (await tools.execute("plain", None)) == {"result": 42}
        old = tools.TOOL_TIMEOUT_SECONDS
        tools.TOOL_TIMEOUT_SECONDS = 0.1
        try:
            assert "longer than" in (await tools.execute("slow", {}))["error"]
        finally:
            tools.TOOL_TIMEOUT_SECONDS = old

    asyncio.run(body())


def test_a_tool_must_be_well_formed(fresh_registry):
    api = tools.PluginAPI("test")

    async def handler(arguments):
        return {}

    def not_async(arguments):
        return {}

    schema = {"type": "object", "properties": {}}
    with pytest.raises(ValueError):
        api.tool("bad name!", "A name with a space and a mark in it.", schema, handler)
    with pytest.raises(ValueError):
        api.tool("no_description", "", schema, handler)
    with pytest.raises(ValueError):
        api.tool("bad_schema", "The parameters are not an object schema.", {"type": "string"}, handler)
    with pytest.raises(ValueError):
        api.tool("not_async", "The handler is a plain function.", schema, not_async)


def test_private_tool_data_is_kept_out_of_the_log(fresh_registry):
    shown = tools.redact("set_location", {"address": "1 Private Road"})
    assert "Private Road" not in json.dumps(shown)
    assert tools.redact("set_timer", {"label": "pasta"}) == {"label": "pasta"}


def test_the_dice_example_works(fresh_registry):
    assert tools.load_file(EXAMPLES / "dice_plugin.py")
    result = asyncio.run(tools.execute("roll_dice", {"count": 3, "sides": 6}))
    assert len(result["numbers"]) == 3 and result["total"] == sum(result["numbers"])


def test_the_status_page_example_reads_a_page_and_reports_a_dead_one(fresh_registry, monkeypatch):
    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"battery_soc": 87, "pv_watts": 1200, "internal": "x"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("NEXUS_SOLAR_URL", f"http://127.0.0.1:{server.server_port}/api/latest")
        monkeypatch.setenv("NEXUS_SOLAR_FIELDS", "battery_soc,pv_watts")
        assert tools.load_file(EXAMPLES / "solar_plugin.py")
        result = asyncio.run(tools.execute("get_solar_status", {}))
        assert result == {"success": True, "readings": {"battery_soc": 87, "pv_watts": 1200}}
    finally:
        server.shutdown()
    result = asyncio.run(tools.execute("get_solar_status", {}))  # the page is gone now
    assert result["success"] is False and "could not be read" in result["error"]


def test_a_part_is_chosen_by_name_or_by_module_and_class(tmp_path, monkeypatch):
    assert parts.load_class("audio", "fake").__name__ == "FakeAudio"
    assert parts.load_class("wake", "nexus.wake.fake:FakeWakeEngine").__name__ == "FakeWakeEngine"

    with pytest.raises(parts.PartError) as error:
        parts.load_class("audio", "no-such-audio")
    assert "generic" in str(error.value) and "NEXUS_AUDIO" in str(error.value)

    with pytest.raises(parts.PartError):
        parts.load_class("wake", "nexus.audio.fake:FakeAudio")  # a real class of the wrong kind

    (tmp_path / "my_wake_part.py").write_text((EXAMPLES / "template_wake.py").read_text())
    monkeypatch.syspath_prepend(str(tmp_path))
    assert parts.load_class("wake", "my_wake_part:MyWakeEngine").name == "my-wake"

    parts.register("wake", "mine", "my_wake_part:MyWakeEngine")
    try:
        assert parts.create("wake", "mine").name == "my-wake"
    finally:
        del parts.BUILT_IN["wake"]["mine"]
