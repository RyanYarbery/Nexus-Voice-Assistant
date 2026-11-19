"""A wake-word detector that never fires on its own (NEXUS_WAKE=fake).

Tests script it: give it a function, or set `next_score`.
"""

from ..parts import WakeEngine


class FakeWakeEngine(WakeEngine):
    name = "fake"

    def __init__(self, score_function=None):
        self.score_function = score_function
        self.next_score = 0.0
        self.loaded = False
        self.resets = 0

    def load(self):
        self.loaded = True

    def score(self, channel, block):
        if self.score_function is not None:
            return float(self.score_function(channel, block))

        return float(self.next_score)

    def reset(self):
        self.resets += 1
