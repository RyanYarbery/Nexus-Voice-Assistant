"""Template: your own microphone and speaker.

Copy this file, fill in the five methods, and select it with

    NEXUS_AUDIO=my_audio:MyAudio        (the file must be importable: put it
                                         next to echo.py or on PYTHONPATH)

Then check it on the machine with the hardware:

    python -c "from nexus.testing import check_audio; from my_audio import MyAudio; print(check_audio(MyAudio) or 'OK')"

Two working parts to read first: nexus/audio/sounddevice_io.py (any sound
card) and nexus/audio/xvf3800.py (a microphone array with echo cancelling,
built by subclassing the first). If your device shows up as a normal sound
card, subclass SoundDeviceAudio as xvf3800.py does; you may only need to
change a few class attributes. docs/ADDING-HARDWARE.md has the full guide.
"""

from nexus import config
from nexus.parts import AudioIO, PartError, ReplyStream


class MyReply(ReplyStream):
    """The speaker, held open for one spoken reply."""

    def __init__(self, rate):
        # `rate` is the sample rate of the audio you will be given (24000
        # for OpenAI). Open your output device here, but do not start it.
        self.rate = rate

    def start(self):
        # Start playback. Called once, just before the first write().
        raise NotImplementedError

    def write(self, pcm):
        # `pcm` is about 50 ms of mono int16 numpy audio at self.rate.
        # Hand it to the device and return when the device has taken it,
        # which should take about as long as the chunk lasts. Resample here
        # if your device needs another rate (nexus.dsp.StreamResampler).
        # If you have an echo canceller, give it this audio as well.
        raise NotImplementedError

    def close(self, abort=False):
        # Release the device. abort=True means the user interrupted: drop
        # whatever is queued so the sound stops at once.
        raise NotImplementedError


class MyAudio(AudioIO):
    name = "my-audio"

    # 1, or 2 if your device gives two useful channels (for example an
    # echo-cancelled beam and a raw mix). The engine picks the better one.
    channels = 1

    def start(self, on_block):
        # Open the microphone and call on_block(ch0, ch1) for every block:
        # int16 numpy arrays of config.INPUT_BLOCK_FRAMES (1280) samples at
        # config.INPUT_RATE (16000 Hz). With one channel pass the same array
        # twice. It may be called from your capture thread.
        #
        # If the device is missing, say so in plain words:
        raise PartError("MyAudio.start() is not written yet (examples/template_audio.py).")

    def stop(self):
        # Stop capturing and release everything. Must be safe to call twice.
        pass

    def open_reply(self, rate):
        return MyReply(rate)

    def play_clip(self, pcm, rate):
        # Play a short mono int16 clip (a timer beep, a notice) at `rate`
        # and return when it has finished.
        raise NotImplementedError

    def describe(self):
        return [f"Microphone and speaker: {self.name} at {config.INPUT_RATE} Hz in"]

    def check(self):
        # For `python -m nexus doctor`: return problems in plain words,
        # without leaving a device open. Empty list = fine.
        return ["examples/template_audio.py is a template; fill in its methods."]
