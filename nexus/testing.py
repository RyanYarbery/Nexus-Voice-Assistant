"""Checks for parts and plugins you write yourself.

Each check takes your class (or plugin file) and returns a list of problems
in plain words; an empty list means it follows the contract the engine
relies on. Use them from a test:

    from nexus.testing import check_audio
    from my_mic import MyAudio

    def test_my_audio():
        assert check_audio(MyAudio) == []

or from the command line for a plugin:  python -m nexus check-plugin my_plugin.py

check_audio and check_wake use your real hardware and models, so run them on
the machine the part is for.
"""

import inspect
import json
import threading
import time

import numpy as np

from . import config, parts, tools


def _silence(frames=config.INPUT_BLOCK_FRAMES):
    return np.zeros(frames, dtype=np.int16)


def check_audio(factory, listen_seconds=2.0):
    """Check a nexus.parts.AudioIO: capture, a reply stream, a clip, and a clean stop."""
    problems = []

    try:
        audio = factory()
    except Exception as exc:
        return [f"could not be created: {type(exc).__name__}: {exc}"]

    if not isinstance(audio, parts.AudioIO):
        return ["is not a subclass of nexus.parts.AudioIO"]

    if audio.channels not in (1, 2):
        problems.append(f"channels must be 1 or 2, not {audio.channels!r}")

    blocks = []
    arrived = threading.Event()

    def on_block(ch0, ch1):
        blocks.append((ch0, ch1))

        if len(blocks) >= 3:
            arrived.set()

    try:
        audio.start(on_block)
    except Exception as exc:
        return problems + [f"start() failed: {type(exc).__name__}: {exc}"]

    try:
        if not arrived.wait(listen_seconds):
            problems.append(
                f"start() delivered {len(blocks)} microphone block(s) in {listen_seconds:g} s; "
                "expected one every 80 ms"
            )

        for ch0, ch1 in blocks[:3]:
            for label, channel in (("ch0", ch0), ("ch1", ch1)):
                if not isinstance(channel, np.ndarray) or channel.dtype != np.int16:
                    problems.append(f"{label} must be an int16 numpy array, got {type(channel).__name__}")
                    break

                if channel.shape != (config.INPUT_BLOCK_FRAMES,):
                    problems.append(
                        f"{label} must have {config.INPUT_BLOCK_FRAMES} samples "
                        f"({config.INPUT_RATE} Hz, 80 ms), got shape {channel.shape}"
                    )
                    break

        if len(blocks) >= 3:
            time.sleep(0.5)
            rate = (len(blocks) - 3) / 0.5

            if not 6 <= rate <= 20:
                problems.append(f"blocks arrive at about {rate:.0f} per second; expected about 12.5")

        try:
            lines = audio.describe()

            if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                problems.append("describe() must return a list of strings")
        except Exception as exc:
            problems.append(f"describe() failed: {type(exc).__name__}: {exc}")

        rate = 24000
        chunk = np.zeros(int(rate * 0.05), dtype=np.int16)

        try:
            reply = audio.open_reply(rate)
            reply.start()
            started = time.monotonic()

            for _ in range(10):
                reply.write(chunk)

            took = time.monotonic() - started
            reply.close(abort=False)

            if took > 2.0:
                problems.append(
                    f"writing 0.5 s of reply audio took {took:.1f} s; write() must not lag playback"
                )

            reply = audio.open_reply(rate)
            reply.start()
            reply.write(chunk)
            reply.close(abort=True)
        except Exception as exc:
            problems.append(f"playing a reply failed: {type(exc).__name__}: {exc}")

        try:
            audio.play_clip(np.zeros(int(config.CLIP_RATE * 0.1), dtype=np.int16), config.CLIP_RATE)
        except Exception as exc:
            problems.append(f"play_clip() failed: {type(exc).__name__}: {exc}")

    finally:
        try:
            audio.stop()
            audio.stop()
        except Exception as exc:
            problems.append(f"stop() must be safe to call twice: {type(exc).__name__}: {exc}")

    count = len(blocks)
    time.sleep(0.3)

    if len(blocks) > count + 1:
        problems.append("microphone blocks kept arriving after stop()")

    return problems


def check_wake(factory, max_milliseconds=40.0):
    """Check a nexus.parts.WakeEngine: loads, scores both channels in range, fast enough."""
    problems = []

    try:
        wake = factory()
    except Exception as exc:
        return [f"could not be created: {type(exc).__name__}: {exc}"]

    if not isinstance(wake, parts.WakeEngine):
        return ["is not a subclass of nexus.parts.WakeEngine"]

    try:
        wake.load()
    except Exception as exc:
        return [f"load() failed: {type(exc).__name__}: {exc}"]

    noise = (np.random.default_rng(0).standard_normal(config.INPUT_BLOCK_FRAMES) * 300).astype(np.int16)
    started = time.monotonic()
    calls = 0

    try:
        for block in (_silence(), noise) * 10:
            for channel in (0, 1):
                score = wake.score(channel, block)
                calls += 1

                if not isinstance(score, float):
                    problems.append(f"score() must return a float, got {type(score).__name__}")
                    return problems

                if not 0.0 <= score <= 1.0:
                    problems.append(f"score() returned {score}; it must be between 0 and 1")
                    return problems
    except Exception as exc:
        return problems + [f"score() failed: {type(exc).__name__}: {exc}"]

    per_call = (time.monotonic() - started) / calls * 1000

    if per_call > max_milliseconds:
        problems.append(f"score() takes {per_call:.0f} ms per block; two channels must fit well inside 80 ms")

    try:
        wake.reset()
    except Exception as exc:
        problems.append(f"reset() failed: {type(exc).__name__}: {exc}")

    return problems


def check_verifier(factory):
    """Check a nexus.parts.SpeakerVerifier: loads, and returns two numbers for two channels."""
    try:
        verifier = factory()
    except Exception as exc:
        return [f"could not be created: {type(exc).__name__}: {exc}"]

    if not isinstance(verifier, parts.SpeakerVerifier):
        return ["is not a subclass of nexus.parts.SpeakerVerifier"]

    if not verifier.enabled:
        return []

    try:
        verifier.load()
    except Exception as exc:
        return [f"load() failed: {type(exc).__name__}: {exc}"]

    audio = (np.random.default_rng(1).standard_normal(config.INPUT_RATE * 2) * 500).astype(np.int16)

    try:
        result = verifier.score(audio, audio.copy())
    except Exception as exc:
        return [f"score() failed: {type(exc).__name__}: {exc}"]

    if not (
        isinstance(result, tuple) and len(result) == 2 and all(isinstance(v, (int, float)) for v in result)
    ):
        return [f"score() must return two numbers (score_ch0, score_ch1), got {result!r}"]

    return []


def check_backend(factory):
    """Check a nexus.parts.VoiceBackend's shape. It does not connect."""
    try:
        backend = factory()
    except Exception as exc:
        return [f"could not be created: {type(exc).__name__}: {exc}"]

    if not isinstance(backend, parts.VoiceBackend):
        return ["is not a subclass of nexus.parts.VoiceBackend"]

    problems = []

    try:
        found = backend.check()

        if not isinstance(found, list):
            problems.append("check() must return a list of problems (empty when fine)")
    except Exception as exc:
        problems.append(f"check() failed: {type(exc).__name__}: {exc}")

    try:
        context = backend.connect()

        if not (hasattr(context, "__aenter__") and hasattr(context, "__aexit__")):
            problems.append(
                "connect() must return an async context manager (use contextlib.asynccontextmanager)"
            )

        close = getattr(context, "aclose", None) or getattr(getattr(context, "gen", None), "aclose", None)

        if close is not None:
            # Never entered; just discard it without a "never awaited" warning.
            try:
                close().close()
            except Exception:
                pass
    except Exception as exc:
        problems.append(f"connect() failed before connecting: {type(exc).__name__}: {exc}")

    return problems


def check_session_class(session_class):
    """Check that a VoiceSession subclass implements every method the engine calls."""
    if not (isinstance(session_class, type) and issubclass(session_class, parts.VoiceSession)):
        return ["is not a subclass of nexus.parts.VoiceSession"]

    missing = sorted(getattr(session_class, "__abstractmethods__", ()))

    return [f"does not implement: {', '.join(missing)}"] if missing else []


def check_plugin(register_function, name="plugin"):
    """Check a plugin's register() function without keeping what it registers."""
    saved = (dict(tools.TOOLS), list(tools.INSTRUCTIONS), list(tools.START_HOOKS), list(tools.STOP_HOOKS))
    saved_logs = (list(tools.LOADED), list(tools.FAILED))
    problems = []

    try:
        tools.clear()

        try:
            register_function(tools.PluginAPI(name))
        except Exception as exc:
            return [f"register() failed: {type(exc).__name__}: {exc}"]

        if not tools.TOOLS and not tools.START_HOOKS:
            problems.append("register() registered nothing: no tool and no start hook")

        for tool in tools.TOOLS.values():
            label = f"tool {tool.name!r}"

            if len(tool.description) < 15:
                problems.append(
                    f"{label}: the description is too short for the model to know when to call it"
                )

            try:
                json.dumps(tool.parameters)
            except (TypeError, ValueError) as exc:
                problems.append(f"{label}: parameters are not plain JSON ({exc})")
                continue

            properties = tool.parameters.get("properties", {})

            if not isinstance(properties, dict):
                problems.append(f'{label}: "properties" must be an object')
                continue

            for required in tool.parameters.get("required", []):
                if required not in properties:
                    problems.append(f"{label}: required argument {required!r} is not in properties")

            signature = inspect.signature(tool.handler)

            if len(signature.parameters) != 1:
                problems.append(f"{label}: the handler must take exactly one argument (the arguments dict)")

        for plugin_name, function in tools.START_HOOKS:
            if not inspect.iscoroutinefunction(function):
                problems.append("on_start() needs an async function")

        for plugin_name, function in tools.STOP_HOOKS:
            if inspect.iscoroutinefunction(function):
                problems.append("on_stop() needs a plain function, not an async one")

    finally:
        tools.clear()
        tools.TOOLS.update(saved[0])
        tools.INSTRUCTIONS.extend(saved[1])
        tools.START_HOOKS.extend(saved[2])
        tools.STOP_HOOKS.extend(saved[3])
        tools.LOADED.extend(saved_logs[0])
        tools.FAILED.extend(saved_logs[1])

    return problems


def check_plugin_file(path):
    """Load a plugin file the way Nexus does and check it."""
    import importlib.util
    from pathlib import Path

    path = Path(path)

    if not path.is_file():
        return [f"{path} is not a file"]

    try:
        spec = importlib.util.spec_from_file_location(f"nexus_plugin_check_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        return [f"the file could not be imported: {type(exc).__name__}: {exc}"]

    register_function = getattr(module, "register", None)

    if not callable(register_function):
        return ["the file has no register(nexus) function"]

    return check_plugin(register_function, path.stem)
