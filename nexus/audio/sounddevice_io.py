"""Any sound card PortAudio can see: a USB microphone, a headset, a laptop's own.

This is the default (NEXUS_AUDIO=generic). It has no echo cancelling, so
interrupting Nexus while it talks works only as well as your microphone
keeps its own speaker out. A speakerphone or conference microphone with
built-in echo cancelling is the easy way to get that.

It is also the base class to copy or subclass for other hardware; see
xvf3800.py for a subclass that adds an echo-cancelling reference stream.
"""

import numpy as np

from .. import config
from ..dsp import StreamResampler, resample_int16
from ..parts import AudioIO, PartError, ReplyStream

env_str, env_float, env_int, env_flag = config.group("audio: generic")

MIC_NAME = env_str(
    "NEXUS_MIC_NAME",
    "",
    "Part of the microphone's device name; empty for the system default.",
)
SPEAKER_NAME = env_str(
    "NEXUS_SPEAKER_NAME",
    "",
    "Part of the speaker's device name; empty for the system default.",
)
OUTPUT_RATE = env_int("NEXUS_OUTPUT_RATE", 48000, "Sample rate the speaker is opened at.")
MIC_CHANNELS = env_int(
    "NEXUS_MIC_CHANNELS",
    1,
    "Microphone channels to use: 1, or 2 if the device gives two useful channels.",
)


def sounddevice():
    """Import sounddevice only when audio is actually used."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise PartError(
            "The sounddevice package (and the PortAudio library it needs) is not installed: "
            f"{exc}. Install it with: pip install sounddevice  (Debian: apt install libportaudio2)."
        ) from exc

    return sd


def find_device(name_fragment, need_input=False, need_output=False):
    """Index of the first device whose name contains `name_fragment`; None for the default device."""
    sd = sounddevice()

    if not name_fragment.strip():
        return None

    names = []

    for index, device in enumerate(sd.query_devices()):
        if need_input and device["max_input_channels"] < 1:
            continue

        if need_output and device["max_output_channels"] < 1:
            continue

        names.append(device["name"])

        if name_fragment.lower() in device["name"].lower():
            return index

    kind = "microphone" if need_input else "speaker"
    listing = "; ".join(names) or "none found"

    raise PartError(f'No {kind} with "{name_fragment}" in its name. Devices available: {listing}.')


def device_label(index, need_input):
    sd = sounddevice()

    if index is None:
        try:
            default = sd.query_devices(kind="input" if need_input else "output")
            return f"system default ({default['name']})"
        except Exception:
            return "system default"

    return f"{index} - {sd.query_devices(index)['name']}"


class SoundDeviceReply(ReplyStream):
    """One reply on one output device, resampled from the service's rate to the device's."""

    def __init__(self, audio, rate):
        self.audio = audio
        self.sd = sounddevice()
        self.to_output = StreamResampler(rate, audio.output_rate)
        self.stream = self.sd.OutputStream(
            samplerate=audio.output_rate,
            device=audio.output_device,
            channels=1,
            dtype="int16",
            blocksize=int(audio.output_rate * 0.05),
        )

    def start(self):
        self.stream.start()

    def write(self, pcm):
        self.stream.write(self.to_output.process(pcm).reshape(-1, 1))

    def close(self, abort=False):
        try:
            if abort:
                # Discard queued audio immediately. stop() drains it and
                # can make a valid interruption appear to have been ignored.
                self.stream.abort()
            else:
                self.stream.stop()
        except Exception:
            pass

        try:
            self.stream.close()
        except Exception:
            pass


class SoundDeviceAudio(AudioIO):
    name = "generic"

    mic_name = MIC_NAME
    speaker_name = SPEAKER_NAME
    output_rate = OUTPUT_RATE
    channels = 2 if MIC_CHANNELS >= 2 else 1

    def __init__(self):
        self.input_device = None
        self.output_device = None
        self._stream = None
        self._on_block = None
        self._last_status_print = 0.0

    # -- microphone ---------------------------------------------------------

    def start(self, on_block):
        sd = sounddevice()

        self.input_device = find_device(self.mic_name, need_input=True)
        self.output_device = find_device(self.speaker_name, need_output=True)
        self._on_block = on_block

        try:
            stream = sd.InputStream(
                samplerate=config.INPUT_RATE,
                blocksize=config.INPUT_BLOCK_FRAMES,
                device=self.input_device,
                channels=self.channels,
                dtype="int16",
                callback=self._callback,
            )
            stream.start()
        except Exception as exc:
            raise PartError(
                f"Could not open the microphone ({device_label(self.input_device, True)}) at "
                f"{config.INPUT_RATE} Hz with {self.channels} channel(s): {exc}"
            ) from exc

        self._stream = stream

    def _callback(self, indata, frames, time_info, status):
        if status:
            import time

            now = time.monotonic()

            # At most one line per 10 s: under CPU pressure this fires on
            # every callback and would flood the log.
            if now - self._last_status_print >= 10.0:
                self._last_status_print = now
                print(f"Microphone status: {status}")

        ch0 = indata[:, 0].copy().astype(np.int16)
        ch1 = indata[:, 1].copy().astype(np.int16) if self.channels >= 2 else ch0

        self._on_block(ch0, ch1)

    def stop(self):
        stream = self._stream
        self._stream = None

        if stream is None:
            return

        for method in (stream.stop, stream.close):
            try:
                method()
            except Exception:
                pass

    # -- speaker ------------------------------------------------------------

    def open_reply(self, rate):
        return SoundDeviceReply(self, rate)

    def play_clip(self, pcm, rate):
        sd = sounddevice()

        if rate != self.output_rate:
            pcm = resample_int16(pcm, rate, self.output_rate)

        sd.play(pcm, self.output_rate, device=self.output_device, blocking=True)

    # -- reporting ----------------------------------------------------------

    def describe(self):
        return [
            f"Microphone: {device_label(self.input_device, True)} ({self.channels} channel(s))",
            f"Speaker:    {device_label(self.output_device, False)} at {self.output_rate} Hz",
        ]

    def check(self):
        problems = []

        try:
            sd = sounddevice()
        except PartError as exc:
            return [str(exc)]

        for fragment, is_input in ((self.mic_name, True), (self.speaker_name, False)):
            try:
                index = find_device(fragment, need_input=is_input, need_output=not is_input)

                if index is None:
                    sd.query_devices(kind="input" if is_input else "output")
            except PartError as exc:
                problems.append(str(exc))
            except Exception as exc:
                kind = "microphone" if is_input else "speaker"
                problems.append(
                    f"No default {kind} is set on this computer ({exc}). Name one with NEXUS_{'MIC' if is_input else 'SPEAKER'}_NAME."
                )

        if not problems:
            try:
                sd.check_input_settings(
                    device=find_device(self.mic_name, need_input=True),
                    channels=self.channels,
                    dtype="int16",
                    samplerate=config.INPUT_RATE,
                )
            except Exception as exc:
                problems.append(
                    f"The microphone does not accept {config.INPUT_RATE} Hz with {self.channels} channel(s): {exc}"
                )

        return problems
