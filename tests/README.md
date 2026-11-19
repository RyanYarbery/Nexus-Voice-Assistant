# Tests

The tests run without a microphone, speakers, OpenAI, or the ML models.

```bash
python -m venv .venv-test && .venv-test/bin/pip install numpy scipy pytest
.venv-test/bin/python -m pytest tests
```

`harness.py` loads the assistant with `sounddevice`, `torch`, `websockets`,
`openwakeword`, and `speechbrain` replaced by stubs, then drives it with
scripted microphone blocks and scripted Realtime events.

- `test_units.py`: resampling, timers and small helpers.
- `test_flow.py`: wake, speaker check, streaming and buffered playback,
  barge-in, rejection and the follow-up window.
- `test_probes.py`: failure paths (errors, timeouts, a failing speaker, beeps
  during a request, a stray sound in the follow-up window).
