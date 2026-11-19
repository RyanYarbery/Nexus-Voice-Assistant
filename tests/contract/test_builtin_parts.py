"""The contract checks in nexus.testing, run against the parts and plugins that ship with Nexus.

Authors of new parts run the same checks against their own classes.
"""

import pathlib

import harness  # noqa: F401  (stubs sounddevice before nexus is imported)
from nexus import testing
from nexus.audio.fake import FakeAudio
from nexus.backends.fake import FakeBackend, FakeSession
from nexus.backends.openai_realtime import OpenAIRealtimeBackend, OpenAIRealtimeSession
from nexus.speaker_id.fake import FakeVerifier
from nexus.speaker_id.none import NoVerifier
from nexus.tools.builtin import alarms, clock, location, timers, weather
from nexus.wake.fake import FakeWakeEngine

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def test_fake_audio_follows_the_audio_contract():
    assert testing.check_audio(FakeAudio) == []


def test_fake_wake_engine_follows_the_wake_contract():
    assert testing.check_wake(FakeWakeEngine) == []


def test_verifiers_follow_the_verifier_contract():
    assert testing.check_verifier(FakeVerifier) == []
    assert testing.check_verifier(NoVerifier) == []


def test_backends_follow_the_backend_contract():
    assert testing.check_backend(FakeBackend) == []
    assert testing.check_backend(OpenAIRealtimeBackend) == []
    assert testing.check_session_class(FakeSession) == []
    assert testing.check_session_class(OpenAIRealtimeSession) == []


def test_built_in_plugins_follow_the_plugin_contract():
    for module in (timers, alarms, location, clock, weather):
        assert testing.check_plugin(module.register, module.__name__) == [], module.__name__


def test_example_plugins_follow_the_plugin_contract():
    assert testing.check_plugin_file(EXAMPLES / "dice_plugin.py") == []
    assert testing.check_plugin_file(EXAMPLES / "solar_plugin.py") == []


def test_the_checks_catch_a_part_that_breaks_the_contract():
    class WrongBlocks(FakeAudio):
        def start(self, on_block):
            import numpy as np

            on_block(np.zeros(100, np.float32), np.zeros(100, np.float32))
            on_block(np.zeros(100, np.float32), np.zeros(100, np.float32))
            on_block(np.zeros(100, np.float32), np.zeros(100, np.float32))

    problems = testing.check_audio(WrongBlocks, listen_seconds=0.5)
    assert any("int16" in problem for problem in problems)

    class OutOfRange(FakeWakeEngine):
        def score(self, channel, block):
            return 7.0

    assert any("between 0 and 1" in problem for problem in testing.check_wake(OutOfRange))


def test_the_templates_are_real_parts_that_say_they_are_unfinished():
    import importlib.util

    from nexus import parts

    for filename, class_name, base in (
        ("template_audio.py", "MyAudio", parts.AudioIO),
        ("template_wake.py", "MyWakeEngine", parts.WakeEngine),
    ):
        spec = importlib.util.spec_from_file_location(filename[:-3], EXAMPLES / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        part = getattr(module, class_name)()
        assert isinstance(part, base)
        assert part.check() and "template" in part.check()[0]
