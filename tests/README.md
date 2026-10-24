# Tests

Both suites run without a microphone, speakers, OpenAI, or the ML models.

```bash
python -m venv .venv-test && .venv-test/bin/pip install numpy scipy
.venv-test/bin/python tests/test_units.py     # resampler, timers, helpers
.venv-test/bin/python tests/test_flow.py      # state machine end to end
```

`test_flow.py` loads `echo.py` with `sounddevice`, `torch`, `websockets`,
`openwakeword`, and `speechbrain` replaced by stubs, then drives the
controller with scripted microphone blocks and Realtime events. It
covers soft and strict speaker verification, streaming and buffered
playback, barge-in with server-side `response.cancel`, speaker
rejection, and the follow-up window.
