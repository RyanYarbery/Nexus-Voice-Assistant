"""Small audio helpers: resampling, and lists of microphone blocks."""

import math

import numpy as np

from .config import INPUT_RATE

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover - scipy is normally present
    resample_poly = None


def resample_int16(
    audio,
    old_rate,
    new_rate,
):
    if old_rate == new_rate:
        return audio

    if len(audio) == 0:
        return audio

    if resample_poly is not None:
        # Polyphase FIR resampling. Linear interpolation has no
        # anti-aliasing filter, which matters most for the 48 kHz
        # to 16 kHz AEC reference and the 16 kHz to 24 kHz uplink.
        divisor = math.gcd(old_rate, new_rate)
        result = resample_poly(
            audio.astype(np.float32),
            new_rate // divisor,
            old_rate // divisor,
        )
        return np.clip(
            np.rint(result),
            -32768,
            32767,
        ).astype(np.int16)

    new_length = int(len(audio) * new_rate / old_rate)

    old_x = np.arange(
        len(audio),
        dtype=np.float64,
    )

    new_x = np.linspace(
        0,
        len(audio) - 1,
        new_length,
    )

    result = np.interp(
        new_x,
        old_x,
        audio.astype(np.float32),
    )

    return np.clip(
        result,
        -32768,
        32767,
    ).astype(np.int16)


class StreamResampler:
    """Stateful polyphase resampler for a stream of int16 chunks.

    Resampling each network delta independently would leave a filter
    transient at every chunk boundary and click audibly. This keeps the
    filter state and decimation phase between calls, so chunked output
    is identical to resampling the whole signal at once.
    """

    def __init__(self, old_rate, new_rate):
        divisor = math.gcd(old_rate, new_rate)
        self.up = new_rate // divisor
        self.down = old_rate // divisor
        self.identity = self.up == 1 and self.down == 1

        if self.identity:
            return

        half_len = 10 * max(self.up, self.down)
        taps = 2 * half_len + 1

        # Low-pass at the lower of the two Nyquist frequencies,
        # expressed as a fraction of the zero-stuffed signal's Nyquist.
        cutoff = 1.0 / max(self.up, self.down)
        n = np.arange(taps) - half_len
        kernel = cutoff * np.sinc(cutoff * n) * np.kaiser(taps, 5.0)
        kernel = kernel / kernel.sum() * self.up

        self.kernel = kernel.astype(np.float32)
        self.state = np.zeros(taps - 1, dtype=np.float32)
        self.phase = 0

    def process(self, audio_int16):
        if self.identity or len(audio_int16) == 0:
            return audio_int16

        x = audio_int16.astype(np.float32)

        if self.up > 1:
            upsampled = np.zeros(len(x) * self.up, dtype=np.float32)
            upsampled[:: self.up] = x
        else:
            upsampled = x

        buffer = np.concatenate([self.state, upsampled])
        filtered = np.convolve(buffer, self.kernel, mode="valid")
        self.state = buffer[len(upsampled) :]

        if self.down > 1:
            out = filtered[self.phase :: self.down]
            n = len(filtered)

            if n > self.phase:
                taken = (n - self.phase + self.down - 1) // self.down
                self.phase = self.phase + taken * self.down - n
            else:
                self.phase -= n
        else:
            out = filtered

        return np.clip(
            np.rint(out),
            -32768,
            32767,
        ).astype(np.int16)


def samples_in_blocks(
    blocks,
):
    return sum(len(block[0]) if isinstance(block, tuple) else len(block) for block in blocks)


def select_channel_blocks(
    blocks,
    channel,
):
    return [block[channel] if isinstance(block, tuple) else block for block in blocks]


def blocks_seconds(
    blocks,
):
    return samples_in_blocks(blocks) / INPUT_RATE


def blocks_have_seconds(
    blocks,
    seconds,
):
    needed = int(seconds * INPUT_RATE)

    return samples_in_blocks(blocks) >= needed
