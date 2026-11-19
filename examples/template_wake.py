"""Template: your own wake-word detector.

Copy this file, fill in load() and score(), and select it with

    NEXUS_WAKE=my_wake:MyWakeEngine

Then check it:

    python -c "from nexus.testing import check_wake; from my_wake import MyWakeEngine; print(check_wake(MyWakeEngine) or 'OK')"

nexus/wake/openwakeword_engine.py is the working part to read first.
"""

from nexus import config
from nexus.parts import WakeEngine


class MyWakeEngine(WakeEngine):
    name = "my-wake"

    def load(self):
        # Load your model here (never at import time). Keep one detector per
        # microphone channel: each hears a different audio stream and must
        # keep its own memory of the last second or so.
        self.detectors = {0: None, 1: None}

    def score(self, channel, block):
        # `block` is 80 ms of int16 audio (config.INPUT_BLOCK_FRAMES samples
        # at config.INPUT_RATE) from microphone channel 0 or 1. You are
        # called for every block, in order, in every state, including while
        # Nexus is speaking. Return a float from 0.0 to 1.0: how sure you
        # are that the wake phrase has just been said. Nexus wakes when it
        # reaches NEXUS_WAKE_THRESHOLD. Take a few milliseconds at most.
        assert len(block) == config.INPUT_BLOCK_FRAMES
        return 0.0

    def check(self):
        # For `python -m nexus doctor`: is the model file there?
        return ["examples/template_wake.py is a template; fill in load() and score()."]
