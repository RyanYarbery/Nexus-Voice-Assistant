"""The ReSpeaker XVF3800 four-microphone USB array (NEXUS_AUDIO=xvf3800).

The array does its own echo cancelling: it subtracts what the speaker plays
from what the microphones hear, which is what lets the wake phrase interrupt
Nexus mid-sentence. To do that it needs a copy of the playback, the
"reference". This part plays every reply through your speaker and sends the
same audio, silently, to the array's own USB playback endpoint.

It captures two channels at 16 kHz: channel 0 is the array's processed,
echo-cancelled beam; channel 1 is whatever its right output is routed to.

Everything specific to this array lives in this file. Use it as the worked
example when writing a part for another array.
"""

import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .. import config
from ..dsp import StreamResampler, resample_int16
from .sounddevice_io import SoundDeviceAudio, SoundDeviceReply, device_label, find_device, sounddevice
from ..parts import PartError

env_str, env_float, env_int, env_flag = config.group("audio: xvf3800")

MIC_NAME = env_str("NEXUS_MIC_NAME", "respeaker xvf3800", "Part of the array's device name.")
SPEAKER_NAME = env_str(
    "NEXUS_SPEAKER_NAME",
    "",
    "Part of the speaker's device name; empty for the system default.",
)

# Verified USB playback format: 16 kHz, stereo, signed 16-bit PCM.
AEC_REFERENCE_RATE = 16000

AEC_REFERENCE_NAME = env_str(
    "NEXUS_AEC_REFERENCE_NAME",
    MIC_NAME,
    'Output that receives a copy of playback for echo cancelling; "off" for none.',
)

# The XVF3800 locks its master clock to the host playback stream and
# free-runs when none is open. Keeping one reference stream open for
# the life of the process (writing silence between replies) avoids a
# clock re-lock and AEC re-convergence at the start of every reply.
AEC_REFERENCE_CONTINUOUS = env_flag(
    "NEXUS_AEC_REFERENCE_CONTINUOUS",
    True,
    "Keep the echo-cancelling reference stream open all the time (0 = open one per reply).",
)

RESPEAKER_ROUTE_HELPER = env_str(
    "NEXUS_RESPEAKER_ROUTE_HELPER",
    "/usr/local/bin/nexus-respeaker-route",
    "Optional helper, run through sudo at start, that sets the array's channel routing.",
)


class ReferenceOutput:
    """Permanently open far-end reference stream for the XVF3800.

    PortAudio pulls from a queue of mono 16 kHz chunks in its callback
    and plays silence when the queue is empty, so the device always has
    a host playback stream to lock its clock to.
    """

    def __init__(self, device, rate):
        self.rate = rate
        self._lock = threading.Lock()
        self._chunks = deque()
        self._offset = 0
        self._stream = sounddevice().OutputStream(
            samplerate=rate,
            device=device,
            channels=2,
            dtype="int16",
            blocksize=int(rate * 0.05),
            callback=self._callback,
        )

    def start(self):
        self._stream.start()

    def close(self):
        for method in (self._stream.stop, self._stream.close):
            try:
                method()
            except Exception:
                pass

    def write(self, mono_int16):
        if len(mono_int16) == 0:
            return

        with self._lock:
            self._chunks.append(mono_int16)

    def flush(self):
        with self._lock:
            self._chunks.clear()
            self._offset = 0

    def _callback(self, outdata, frames, time_info, status):
        pos = 0

        with self._lock:
            while pos < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(len(chunk) - self._offset, frames - pos)
                segment = chunk[self._offset : self._offset + take]

                outdata[pos : pos + take, 0] = segment
                outdata[pos : pos + take, 1] = segment

                pos += take
                self._offset += take

                if self._offset >= len(chunk):
                    self._chunks.popleft()
                    self._offset = 0

        if pos < frames:
            outdata[pos:] = 0


def set_respeaker_route(
    mode,
):
    if mode not in (
        "barge-on",
        "barge-off",
        "status",
    ):
        raise ValueError(f"Invalid ReSpeaker route mode: {mode}")

    completed = subprocess.run(
        [
            "sudo",
            "-n",
            RESPEAKER_ROUTE_HELPER,
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3,
        check=False,
    )

    output = (completed.stdout or "").strip()

    if completed.returncode != 0:
        raise RuntimeError(f"ReSpeaker route command failed: {mode}; {output}")

    return output


class XVF3800Reply(SoundDeviceReply):
    """A reply on the speaker, mirrored to the array's reference input."""

    def __init__(self, audio, rate):
        super().__init__(audio, rate)

        self.shared_reference = audio.reference
        self.own_reference = None
        self.to_reference = None

        if audio.reference_device is None:
            return

        self.to_reference = StreamResampler(rate, AEC_REFERENCE_RATE)

        if self.shared_reference is None:
            # No continuous stream: open one for this reply only.
            self.own_reference = self.sd.OutputStream(
                samplerate=AEC_REFERENCE_RATE,
                device=audio.reference_device,
                channels=2,
                dtype="int16",
                blocksize=int(AEC_REFERENCE_RATE * 0.05),
            )

    def start(self):
        if self.own_reference is not None:
            self.own_reference.start()

        super().start()

    def write(self, pcm):
        if self.to_reference is not None:
            reference = self.to_reference.process(pcm)

            if self.shared_reference is not None:
                self.shared_reference.write(reference)
            else:
                self.own_reference.write(np.repeat(reference.reshape(-1, 1), 2, axis=1))

        super().write(pcm)

    def close(self, abort=False):
        if abort and self.shared_reference is not None:
            self.shared_reference.flush()

        super().close(abort=abort)

        if self.own_reference is not None:
            for method in (
                self.own_reference.abort if abort else self.own_reference.stop,
                self.own_reference.close,
            ):
                try:
                    method()
                except Exception:
                    pass


class XVF3800Audio(SoundDeviceAudio):
    name = "xvf3800"

    mic_name = MIC_NAME
    speaker_name = SPEAKER_NAME
    channels = 2

    def __init__(self):
        super().__init__()
        self.reference_device = None
        self.reference = None

    def _find_reference(self):
        if AEC_REFERENCE_NAME.strip().lower() in ("", "off", "none", "disabled"):
            return None

        try:
            return find_device(AEC_REFERENCE_NAME, need_output=True)
        except PartError as exc:
            print(f"WARNING: {exc} Continuing without an echo-cancelling reference.")
            return None

    def start(self, on_block):
        if Path(RESPEAKER_ROUTE_HELPER).exists():
            try:
                set_respeaker_route("barge-off")
                print("ReSpeaker channel routing set.")
            except Exception as exc:
                print(f"WARNING: could not set the ReSpeaker routing: {type(exc).__name__}: {exc}")
        else:
            print("ReSpeaker route helper not installed; skipping.")

        super().start(on_block)

        self.reference_device = self._find_reference()

        if self.reference_device is not None and AEC_REFERENCE_CONTINUOUS:
            try:
                reference = ReferenceOutput(self.reference_device, AEC_REFERENCE_RATE)
                reference.start()
                self.reference = reference
            except Exception as exc:
                print(
                    "WARNING: could not open a continuous echo-cancelling reference stream; "
                    f"falling back to one per reply: {type(exc).__name__}: {exc}"
                )
                self.reference = None

    def stop(self):
        reference = self.reference
        self.reference = None

        if reference is not None:
            reference.close()

        super().stop()

    def open_reply(self, rate):
        return XVF3800Reply(self, rate)

    def play_clip(self, pcm, rate):
        # Beeps and notices go to the reference too, so the wake phrase
        # can be heard over them.
        if self.reference is not None:
            self.reference.write(resample_int16(pcm, rate, AEC_REFERENCE_RATE))

        super().play_clip(pcm, rate)

    def describe(self):
        lines = super().describe()

        if self.reference_device is None:
            lines.append("Echo-cancelling reference: off")
        else:
            mode = "continuous" if self.reference is not None else "per reply"
            lines.append(f"Echo-cancelling reference: {device_label(self.reference_device, False)} ({mode})")

        return lines

    def check(self):
        problems = super().check()

        if not problems and AEC_REFERENCE_NAME.strip().lower() not in ("", "off", "none", "disabled"):
            try:
                find_device(AEC_REFERENCE_NAME, need_output=True)
            except PartError as exc:
                problems.append(f"{exc} Set NEXUS_AEC_REFERENCE_NAME=off to run without echo cancelling.")

        return problems
