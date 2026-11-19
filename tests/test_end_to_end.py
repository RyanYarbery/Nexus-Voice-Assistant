"""The whole assistant started the way `python -m nexus` starts it, on fake parts."""

import asyncio
import json
import subprocess
import sys

import harness as h
from harness import ROOT, ns
from nexus import engine, parts
from nexus.audio.fake import FakeAudio


def test_a_full_exchange_on_fake_parts(capsys, monkeypatch):
    monkeypatch.setattr(parts, "AUDIO_NAME", "fake")
    monkeypatch.setattr(parts, "WAKE_NAME", "fake")
    monkeypatch.setattr(parts, "SPEAKER_ID_NAME", "none")
    monkeypatch.setattr(parts, "BACKEND_NAME", "fake")
    monkeypatch.setattr(FakeAudio, "pace", 0.05)
    saved = (engine.AUDIO, engine.WAKE, engine.VERIFIER, engine.BACKEND, engine.SPEAKER_VERIFY_MODE)

    async def body():
        task = asyncio.create_task(engine.run())
        await h.wait_for(lambda: engine.active_controller is not None and engine.WAKE.loaded, what="start")
        controller = engine.active_controller
        await h.wait_for(lambda: controller.link.online, what="connection")
        await asyncio.sleep(0.4)
        engine.WAKE.next_score = 0.9  # someone says the wake phrase
        await h.wait_for(lambda: controller.request_count >= 1, what="wake")
        engine.WAKE.next_score = 0.0
        await h.wait_for(
            lambda: controller.active_origin == "followup", timeout=8, what="reply and follow-up"
        )
        await h.wait_for(lambda: ns["status"]["state"] == "ok", what="status ok")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    try:
        with h.setting(STATUS_OK_AFTER_SECONDS=0.1, FOLLOWUP_SECONDS=5.0, WAKE_RESUME_COOLDOWN_SECONDS=0.0):
            asyncio.run(asyncio.wait_for(body(), 30))
    finally:
        engine.AUDIO, engine.WAKE, engine.VERIFIER, engine.BACKEND, engine.SPEAKER_VERIFY_MODE = saved

    out = capsys.readouterr().out
    requests = [json.loads(line[8:]) for line in out.splitlines() if line.startswith("REQUEST ")]
    assert requests and requests[0]["origin"] == "wake" and requests[0]["outcome"] == "answered"
    assert requests[0]["backend"] == "fake"
    assert 'Waiting for "Hey Nexus"' in out


def run_cli(*arguments, **env):
    import os

    environment = {
        k: v for k, v in os.environ.items() if not k.startswith("NEXUS_") and k != "OPENAI_API_KEY"
    }
    environment.update(env)
    return subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60
    )


def test_doctor_reports_what_is_missing_instead_of_crashing(tmp_path):
    result = run_cli("-m", "nexus", "doctor", NEXUS_HOME=str(tmp_path), NEXUS_TYPO_SETTING="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert "PROBLEM  voice service (openai): OPENAI_API_KEY is not set" in result.stdout
    assert "NEXUS_TYPO_SETTING is set but nothing reads it" in result.stdout
    assert "OK       plugin timers" in result.stdout


def test_doctor_is_happy_with_fake_parts(tmp_path):
    result = run_cli(
        "-m", "nexus", "doctor",
        NEXUS_HOME=str(tmp_path), NEXUS_AUDIO="fake", NEXUS_WAKE="fake", NEXUS_BACKEND="fake",
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No problems found." in result.stdout


def test_an_unknown_part_name_is_explained_not_a_traceback(tmp_path):
    result = run_cli("-m", "nexus", "run", NEXUS_HOME=str(tmp_path), NEXUS_AUDIO="typo")
    assert result.returncode == 1
    assert "Nexus cannot start: NEXUS_AUDIO='typo' is not a known audio part" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_the_old_launcher_selects_the_original_build(tmp_path):
    code = (
        "import os, runpy, sys\n"
        "sys.argv = ['echo.py', 'help']\n"
        "try:\n    runpy.run_path('echo.py', run_name='__main__')\nexcept SystemExit:\n    pass\n"
        "print(os.environ['NEXUS_AUDIO'], os.environ['NEXUS_SPEAKER_ID'], os.environ['NEXUS_PERSONA'], "
        "os.environ['NEXUS_MIC_NAME'])\n"
    )
    result = run_cli("-c", code, NEXUS_MIC_NAME="my own mic")
    assert result.returncode == 0, result.stderr
    # Its defaults fill the gaps; what the env file already says is kept.
    assert result.stdout.strip().splitlines()[-1] == "xvf3800 ecapa sardonic my own mic"


def test_settings_doc_matches_the_code():
    documented = (ROOT / "docs" / "SETTINGS.md").read_text()
    generated = run_cli("-m", "nexus", "settings", "--markdown", NEXUS_HOME="/opt/nexus").stdout.strip()
    assert "NEXUS_WAKE_THRESHOLD" in generated and "NEXUS_TIMERS_FILE" in generated
    assert generated in documented, "docs/SETTINGS.md is out of date: run tools/write_settings_doc.sh"


def test_personas_and_prompt():
    with h.setting(PERSONA="plain", INSTRUCTIONS_FILE=""):
        text = ns["build_instructions"]()
    assert text.startswith("You are Nexus, a voice assistant")
    assert "'Hey Nexus' is the wake phrase" in text and "get_weather" in text
    with h.setting(PERSONA="sardonic"):
        assert "sardonic" in ns["build_instructions"]()
    with h.setting(PERSONA="no-such-persona"):
        assert ns["build_instructions"]().startswith("You are Nexus, a voice assistant")
