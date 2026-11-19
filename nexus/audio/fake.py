"""A microphone that hears silence and a speaker that plays nothing.

For trying Nexus on a machine with no audio hardware (NEXUS_AUDIO=fake), and
for tests. Everything "played" is kept in `played` so a test can look at it.
"""

import threading
import time

import numpy as np

from .. import config
from ..parts import AudioIO, ReplyStream


class FakeReply(ReplyStream):
    def __init__(self, audio, rate):
        self.audio = audio
        self.rate = rate
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def write(self, pcm):
        self.audio.played.append(np.asarray(pcm).copy())
        time.sleep(len(pcm) / self.rate * self.audio.pace)

    def close(self, abort=False):
        self.closed = True
        self.audio.aborted += 1 if abort else 0


class FakeAudio(AudioIO):
    name = "fake"
    channels = 1

    # Fraction of real time that playback takes (1.0 = real time).
    pace = 1.0

    def __init__(self):
        self.played = []
        self.clips = []
        self.aborted = 0
        self._thread = None
        self._running = threading.Event()

    def start(self, on_block):
        self._running.set()

        def run():
            block_seconds = config.INPUT_BLOCK_FRAMES / config.INPUT_RATE

            while self._running.is_set():
                silence = np.zeros(config.INPUT_BLOCK_FRAMES, dtype=np.int16)
                on_block(silence, silence)
                time.sleep(block_seconds)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()

    def open_reply(self, rate):
        return FakeReply(self, rate)

    def play_clip(self, pcm, rate):
        self.clips.append(np.asarray(pcm).copy())
        time.sleep(min(0.05, len(pcm) / rate * self.pace))

    def describe(self):
        return ["Microphone: fake (silence)", "Speaker:    fake (plays nothing)"]
