"""A voice check that tests can script (NEXUS_SPEAKER_ID=fake)."""

from ..parts import SpeakerVerifier


class FakeVerifier(SpeakerVerifier):
    name = "fake"
    enabled = True

    def __init__(self, score_function=None):
        self.score_function = score_function
        self.next_score = 1.0
        self.calls = 0

    def score(self, audio_ch0, audio_ch1):
        self.calls += 1

        if self.score_function is not None:
            value = float(self.score_function())
        else:
            value = float(self.next_score)

        return (value, value)
