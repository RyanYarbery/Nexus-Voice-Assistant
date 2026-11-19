"""No voice check (NEXUS_SPEAKER_ID=none, the default): anyone who says the wake phrase is answered."""

from ..parts import SpeakerVerifier


class NoVerifier(SpeakerVerifier):
    name = "none"
    enabled = False

    def score(self, audio_ch0, audio_ch1):
        return (1.0, 1.0)
