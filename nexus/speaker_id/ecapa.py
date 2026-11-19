"""Speaker check with SpeechBrain's ECAPA-TDNN model (NEXUS_SPEAKER_ID=ecapa).

It compares the voice that spoke with an enrolled voice profile and returns
a similarity for each microphone channel. A profile is one speaker embedding
saved with torch.save (a tensor, or a dict with an "embedding" key). They
are personal data: keep them out of any repository.

The model runs on the CPU. On a small machine one check takes a second or
two, which is why the engine's default mode streams the request to the
voice service while the check runs beside it.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import config
from ..config import INPUT_RATE
from ..parts import PartError, SpeakerVerifier

env_str, env_float, env_int, env_flag = config.group("speaker_id: ecapa")

VOICE_PROFILE_CH0_PATH = env_str(
    "NEXUS_VOICE_PROFILE_CH0",
    str(config.BASE_DIR / "models" / "voice_ch0.pt"),
    "Enrolled voice profile for microphone channel 0.",
)
VOICE_PROFILE_CH1_PATH = env_str(
    "NEXUS_VOICE_PROFILE_CH1",
    str(config.BASE_DIR / "models" / "voice_ch1.pt"),
    "Enrolled voice profile for microphone channel 1 (channel 0's is used if this file is missing).",
)
SPEAKER_MODEL_CACHE = env_str(
    "NEXUS_SPEAKER_MODEL_CACHE",
    str(config.BASE_DIR / "models" / "spkrec-ecapa-voxceleb"),
    "Folder for the downloaded speaker-recognition model.",
)

# Trim the verification audio to the strongest speech region before
# scoring, so silence and room noise around the phrase do not dilute
# the embedding. 0 scores the raw window instead.
SPEAKER_SPEECH_AWARE = env_flag(
    "NEXUS_SPEAKER_SPEECH_AWARE", True, "Trim the audio to the speech before the voice check."
)

# Speech-aware speaker verification.  Instead of scoring arbitrary
# fixed windows, find the strongest contiguous speech region around
# the wake phrase, trim surrounding silence/noise, and verify that.
SPEAKER_VAD_FRAME_MS = 20
SPEAKER_VAD_HOP_MS = 10
SPEAKER_VAD_NOISE_MULTIPLIER = 2.2
SPEAKER_VAD_PEAK_FRACTION = 0.12
SPEAKER_VAD_ABSOLUTE_FLOOR = 20.0
SPEAKER_VAD_MAX_GAP_MS = 180
SPEAKER_VAD_MARGIN_MS = 120
SPEAKER_VAD_MIN_SECONDS = 0.45
SPEAKER_VAD_MAX_SECONDS = 1.35

# When the speech-aware trimmer finds less than this much speech, the
# untrimmed window is scored as well and the better score is used.
SPEAKER_VAD_FALLBACK_SECONDS = 0.8

# torch is imported in load(), never at import time.
torch = None


def speaker_similarity(
    recognizer,
    enrolled_embedding,
    audio_int16,
):
    if len(audio_int16) == 0:
        return 0.0

    audio_float = audio_int16.astype(np.float32) / 32768.0

    waveform = torch.from_numpy(audio_float).float().unsqueeze(0)

    with torch.no_grad():
        embedding = recognizer.encode_batch(waveform).squeeze().cpu().flatten()

    embedding = embedding / (torch.linalg.vector_norm(embedding) + 1e-8)

    return float(
        torch.dot(
            enrolled_embedding,
            embedding,
        )
    )


def extract_speech_region(
    audio_int16,
):
    """Return the strongest contiguous speech region and diagnostics.

    This is intentionally lightweight and local: it uses short-frame RMS,
    an adaptive threshold based on the recording's noise floor and peak,
    bridges brief intra-word gaps, then chooses the highest-energy region.
    """
    total_samples = len(audio_int16)

    if total_samples == 0:
        return (
            audio_int16,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    frame_samples = max(
        1,
        int(INPUT_RATE * SPEAKER_VAD_FRAME_MS / 1000),
    )

    hop_samples = max(
        1,
        int(INPUT_RATE * SPEAKER_VAD_HOP_MS / 1000),
    )

    if total_samples <= frame_samples:
        audio_float = audio_int16.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_float * audio_float)))
        return (
            audio_int16,
            0.0,
            total_samples / INPUT_RATE,
            rms,
            rms,
        )

    audio_float = audio_int16.astype(np.float32)

    starts = np.arange(
        0,
        total_samples - frame_samples + 1,
        hop_samples,
        dtype=np.int64,
    )

    frame_rms = np.empty(
        len(starts),
        dtype=np.float32,
    )

    for i, frame_start in enumerate(starts):
        frame = audio_float[frame_start : frame_start + frame_samples]

        frame_rms[i] = np.sqrt(np.mean(frame * frame))

    peak_rms = float(np.max(frame_rms))

    noise_floor = float(
        np.percentile(
            frame_rms,
            25,
        )
    )

    threshold = max(
        SPEAKER_VAD_ABSOLUTE_FLOOR,
        noise_floor * SPEAKER_VAD_NOISE_MULTIPLIER,
        peak_rms * SPEAKER_VAD_PEAK_FRACTION,
    )

    active = frame_rms >= threshold

    # Bridge short gaps between active frames so pauses between "Hey"
    # and "Nexus" remain one speech segment.
    max_gap_frames = max(
        1,
        int(SPEAKER_VAD_MAX_GAP_MS / SPEAKER_VAD_HOP_MS),
    )

    active_indices = np.flatnonzero(active)

    if len(active_indices) == 0:
        # Extremely quiet fallback: center a minimum-length region
        # around the loudest frame rather than scoring pure silence.
        peak_index = int(np.argmax(frame_rms))

        center_sample = int(starts[peak_index] + frame_samples // 2)

        target_samples = min(
            total_samples,
            max(
                frame_samples,
                int(SPEAKER_VAD_MIN_SECONDS * INPUT_RATE),
            ),
        )

        region_start = max(
            0,
            center_sample - target_samples // 2,
        )
        region_end = min(
            total_samples,
            region_start + target_samples,
        )
        region_start = max(
            0,
            region_end - target_samples,
        )

    else:
        groups = []
        group_start = int(active_indices[0])
        previous = group_start

        for index in active_indices[1:]:
            index = int(index)

            if index - previous > max_gap_frames:
                groups.append(
                    (
                        group_start,
                        previous,
                    )
                )
                group_start = index

            previous = index

        groups.append(
            (
                group_start,
                previous,
            )
        )

        # Prefer the group containing the most speech energy, not merely
        # the longest group. That keeps a low-level background sound from
        # beating the actual wake phrase.
        best_group = max(
            groups,
            key=lambda group: float(np.sum(frame_rms[group[0] : group[1] + 1] ** 2)),
        )

        region_start = int(starts[best_group[0]])

        region_end = int(starts[best_group[1]] + frame_samples)

        margin_samples = int(SPEAKER_VAD_MARGIN_MS * INPUT_RATE / 1000)

        region_start = max(
            0,
            region_start - margin_samples,
        )

        region_end = min(
            total_samples,
            region_end + margin_samples,
        )

        min_samples = int(SPEAKER_VAD_MIN_SECONDS * INPUT_RATE)

        if region_end - region_start < min_samples:
            center = (region_start + region_end) // 2

            region_start = max(
                0,
                center - min_samples // 2,
            )

            region_end = min(
                total_samples,
                region_start + min_samples,
            )

            region_start = max(
                0,
                region_end - min_samples,
            )

        max_samples = int(SPEAKER_VAD_MAX_SECONDS * INPUT_RATE)

        if region_end - region_start > max_samples:
            # Keep the strongest portion if speech/noise spans too much
            # of the verification buffer.
            group_slice = frame_rms[best_group[0] : best_group[1] + 1]

            peak_relative = int(np.argmax(group_slice))

            peak_frame = best_group[0] + peak_relative

            center = int(starts[peak_frame] + frame_samples // 2)

            region_start = max(
                0,
                center - max_samples // 2,
            )

            region_end = min(
                total_samples,
                region_start + max_samples,
            )

            region_start = max(
                0,
                region_end - max_samples,
            )

    speech_audio = audio_int16[region_start:region_end]

    return (
        speech_audio,
        region_start / INPUT_RATE,
        region_end / INPUT_RATE,
        noise_floor,
        threshold,
    )


def speaker_similarity_speech_aware(
    recognizer,
    enrolled_embedding,
    audio_int16,
):
    (
        speech_audio,
        speech_start,
        speech_end,
        noise_floor,
        threshold,
    ) = extract_speech_region(audio_int16)

    score = speaker_similarity(
        recognizer,
        enrolled_embedding,
        speech_audio,
    )

    # A very short region (one word, or the trimmer finding little
    # above the noise gate) gives a weak embedding. Score the whole
    # window too and keep the better result.
    if speech_end - speech_start < SPEAKER_VAD_FALLBACK_SECONDS and len(audio_int16) > len(speech_audio):
        score = max(
            score,
            speaker_similarity(
                recognizer,
                enrolled_embedding,
                audio_int16,
            ),
        )

    return (
        score,
        speech_start,
        speech_end,
        noise_floor,
        threshold,
    )


class EcapaVerifier(SpeakerVerifier):
    name = "ecapa"
    enabled = True

    def __init__(self):
        self.recognizer = None
        self.profiles = (None, None)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speaker-check")

    def check(self):
        problems = []

        if not Path(VOICE_PROFILE_CH0_PATH).is_file():
            problems.append(
                f"Voice profile not found at {VOICE_PROFILE_CH0_PATH}. Enroll your voice and save the "
                "embedding there, or set NEXUS_SPEAKER_ID=none to answer anyone."
            )

        for module in ("torch", "speechbrain"):
            try:
                __import__(module)
            except ImportError as exc:
                problems.append(f"The {module} package is not installed ({exc}).")

        return problems

    def load(self):
        global torch

        try:
            import torch as torch_module
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError as exc:
            raise PartError(
                f"NEXUS_SPEAKER_ID=ecapa needs torch and speechbrain ({exc}). "
                "Install them, or set NEXUS_SPEAKER_ID=none."
            ) from exc

        torch = torch_module

        print("Loading speaker-recognition model...")

        self.recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SPEAKER_MODEL_CACHE,
            run_opts={"device": "cpu"},
        )

        if not Path(VOICE_PROFILE_CH0_PATH).is_file():
            raise PartError(
                f"Voice profile not found at {VOICE_PROFILE_CH0_PATH}. "
                "Set NEXUS_VOICE_PROFILE_CH0, or NEXUS_SPEAKER_ID=none to answer anyone."
            )

        profile0 = self._load_profile(VOICE_PROFILE_CH0_PATH)
        profile1 = (
            self._load_profile(VOICE_PROFILE_CH1_PATH) if Path(VOICE_PROFILE_CH1_PATH).is_file() else profile0
        )

        self.profiles = (profile0, profile1)

    @staticmethod
    def _load_profile(path):
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Older profiles may contain pickled non-tensor objects.
            saved = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(saved, dict):
            saved = saved["embedding"]

        enrolled = saved.float().flatten()

        return enrolled / (torch.linalg.vector_norm(enrolled) + 1e-8)

    def _score_channel(self, channel, audio):
        if not SPEAKER_SPEECH_AWARE:
            return (speaker_similarity(self.recognizer, self.profiles[channel], audio), None)

        result = speaker_similarity_speech_aware(self.recognizer, self.profiles[channel], audio)

        return (result[0], result)

    def score(self, audio_ch0, audio_ch1):
        if audio_ch1 is audio_ch0 or (
            len(audio_ch0) == len(audio_ch1) and np.array_equal(audio_ch0, audio_ch1)
        ):
            # A one-channel microphone: one pass is enough.
            score0, detail0 = self._score_channel(0, audio_ch0)
            score1, detail1 = score0, detail0
        else:
            # Two passes side by side; the model releases the interpreter lock.
            future0 = self._pool.submit(self._score_channel, 0, audio_ch0)
            future1 = self._pool.submit(self._score_channel, 1, audio_ch1)
            score0, detail0 = future0.result()
            score1, detail1 = future1.result()

        if detail0 is not None and detail1 is not None:
            print(
                f"Speech region CH0: "
                f"{detail0[1]:.2f}-{detail0[2]:.2f}s "
                f"noise={detail0[3]:.1f} gate={detail0[4]:.1f}   "
                f"CH1: {detail1[1]:.2f}-{detail1[2]:.2f}s "
                f"noise={detail1[3]:.1f} gate={detail1[4]:.1f}"
            )

        return (score0, score1)
