#!/usr/bin/env python3
"""Record the wake-word test set at the box.

Saves both microphone channels of every round to a WAV file, scores each channel
live with the current wake model, and writes a JSON summary per round. The
recordings are the point: every later model, channel and threshold gets replayed
against the same files, without anyone at the box.

Plan and reasons: docs/WAKE-RECORDING.md (from the 2026-11-18 wake-word report).

Only one program can hold the microphone, so stop Nexus first:
    sudo systemctl stop nexus.service      # and start it again afterwards

Usage (on the box, from ~/echo):
    .venv/bin/python wake_record.py --list
    .venv/bin/python wake_record.py 1 2 2b 3 4 5 6 7
"""

import argparse
import datetime
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000            # the XVF3800 delivers 16 kHz; openWakeWord expects 16 kHz
BLOCK = 1280            # 80 ms, openWakeWord's frame size
OUT_RATE = 48000        # Nexus plays to the Dell jack at 48 kHz
REF_RATE = 16000        # echo-canceller reference to the array, stereo
ROUTE_HELPER = "/usr/local/bin/nexus-respeaker-route"
ENV_FILE = Path.home() / ".config/echo/env"
OUT_ROOT = Path.home() / "echo/wake-recordings"

# Attempt timing: beep, short pause, speaking window, gap.
BEEP_SECONDS = 0.12
WINDOW_SECONDS = 3.5
GAP_SECONDS = 1.0
SCORE_TAIL_SECONDS = 0.5   # the model reacts a little after the phrase ends

ROUNDS = {
    "1": dict(attempts=20, say="Stand about 1 m (3 ft) from the mic array. Quiet room."),
    "2": dict(attempts=20, say="Stand about 3 m (10 ft) away. Quiet room."),
    "2b": dict(attempts=20, asr_off=True,
               say="Same spot, 3 m. This round switches the chip's wake-word output OFF, for comparison."),
    "3": dict(attempts=20, say="Far side of the room (about 5 m / 16 ft). Quiet."),
    "4": dict(attempts=20, say="3 m away, TV on at normal volume."),
    "5": dict(attempts=20, playback=True,
              say="3 m away. The speakers play Nexus's voice the whole round; say the phrase over it."),
    "6": dict(seconds=300, say="Read any text aloud for 5 minutes. Do NOT say the wake phrase."),
    "7": dict(seconds=3600, say="Leave a TV show or podcast playing for an hour. Nobody needs to be here."),
}

FREE_RUN_THRESHOLDS = (0.1, 0.3, 0.5)
FREE_RUN_COOLDOWN = 1.5


def load_env_file():
    """Fill NEXUS_* settings from the service's env file when not already set."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def find_device(fragment, need_input=False, need_output=False):
    for i, dev in enumerate(sd.query_devices()):
        if fragment.lower() not in dev["name"].lower():
            continue
        if need_input and dev["max_input_channels"] < 2:
            continue
        if need_output and dev["max_output_channels"] < 1:
            continue
        return i
    return None


def resample(x, src, dst):
    if src == dst:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def tone(freq, seconds, rate, level=0.25):
    t = np.arange(int(seconds * rate)) / rate
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
    return (level * np.sin(2 * np.pi * freq * t) * fade).astype(np.float32)


def take_loop(x, pos, n):
    return x[(pos + np.arange(n)) % len(x)]


def route(mode):
    """Switch the chip's wake-word (ASR) output through the root-owned helper."""
    result = subprocess.run(["sudo", "-n", ROUTE_HELPER, mode], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{ROUTE_HELPER} {mode} failed: {result.stderr.strip() or result.stdout.strip()}")


def read_chip(setting):
    result = subprocess.run(["sudo", "-n", ROUTE_HELPER, "status"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(setting):
            return line.strip()
    return None


class Speaker:
    """Plays beeps (and, in round 5, looped speech) the way Nexus does: the Dell
    jack at 48 kHz plus a continuous 16 kHz stereo reference to the array, so the
    chip's echo canceller sees what is playing."""

    def __init__(self, out_dev, ref_dev, speech=None, speech_rate=None):
        self.lock = threading.Lock()
        self.beep48 = np.zeros(0, np.float32)
        self.beep16 = np.zeros(0, np.float32)
        self.speech48 = self.speech16 = None
        if speech is not None:
            self.speech48 = resample(speech, speech_rate, OUT_RATE)
            self.speech16 = resample(speech, speech_rate, REF_RATE)
        self.p48 = self.p16 = 0
        self.streams = [sd.OutputStream(samplerate=OUT_RATE, device=out_dev, channels=1, dtype="int16",
                                        blocksize=int(OUT_RATE * 0.05), callback=self._out)]
        if ref_dev is not None:
            self.streams.append(sd.OutputStream(samplerate=REF_RATE, device=ref_dev, channels=2, dtype="int16",
                                                blocksize=int(REF_RATE * 0.05), callback=self._ref))

    @staticmethod
    def _pcm(buf):
        return np.clip(buf * 32767, -32768, 32767).astype(np.int16)

    def _out(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, np.float32)
        if self.speech48 is not None:
            buf += take_loop(self.speech48, self.p48, frames)
            self.p48 = (self.p48 + frames) % len(self.speech48)
        with self.lock:
            n = min(frames, len(self.beep48))
            buf[:n] += self.beep48[:n]
            self.beep48 = self.beep48[n:]
        outdata[:, 0] = self._pcm(buf)

    def _ref(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, np.float32)
        if self.speech16 is not None:
            buf += take_loop(self.speech16, self.p16, frames)
            self.p16 = (self.p16 + frames) % len(self.speech16)
        with self.lock:
            n = min(frames, len(self.beep16))
            buf[:n] += self.beep16[:n]
            self.beep16 = self.beep16[n:]
        pcm = self._pcm(buf)
        outdata[:, 0] = pcm
        outdata[:, 1] = pcm

    def beep(self, freq=1000, seconds=BEEP_SECONDS):
        with self.lock:
            self.beep48 = np.concatenate([self.beep48, tone(freq, seconds, OUT_RATE)])
            self.beep16 = np.concatenate([self.beep16, tone(freq, seconds, REF_RATE)])

    def __enter__(self):
        for s in self.streams:
            s.start()
        return self

    def __exit__(self, *exc):
        for s in self.streams:
            s.stop()
            s.close()


class Recorder:
    """Writes both channels to WAV and scores each with its own wake model."""

    def __init__(self, mic_dev, model_path, wav_path):
        from openwakeword.model import Model  # slow import, keep it here

        self.models = [Model(wakeword_model_paths=[model_path]) for _ in range(2)]
        self.q = queue.Queue()
        self.frames = 0
        self.scores = []      # (frame_at_block_end, s0, s1, rms0, rms1)
        self.overflows = 0
        self.wav = sf.SoundFile(wav_path, "w", samplerate=RATE, channels=2, subtype="PCM_16")
        self.stream = sd.InputStream(device=mic_dev, samplerate=RATE, channels=2, dtype="int16",
                                     blocksize=BLOCK, callback=self._capture)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.running = False

    def _capture(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        self.q.put(indata.copy())

    @staticmethod
    def _score(model, audio):
        result = model.predict(audio)
        return max((float(v) for v in result.values()), default=0.0)

    def _work(self):
        while self.running or not self.q.empty():
            try:
                block = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.wav.write(block)
            self.frames += len(block)
            ch0 = block[:, 0].copy()
            ch1 = block[:, 1].copy()
            s0 = self._score(self.models[0], ch0)
            s1 = self._score(self.models[1], ch1)
            r0 = float(np.sqrt(np.mean(ch0.astype(np.float32) ** 2)))
            r1 = float(np.sqrt(np.mean(ch1.astype(np.float32) ** 2)))
            self.scores.append((self.frames, s0, s1, r0, r1))

    def best_between(self, start_frame, end_frame):
        rows = [r for r in self.scores if start_frame < r[0] <= end_frame]
        if not rows:
            return 0.0, 0.0
        return max(r[1] for r in rows), max(r[2] for r in rows)

    def wait_for_frame(self, frame):
        while self.frames < frame:
            time.sleep(0.02)

    def __enter__(self):
        self.running = True
        self.worker.start()
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
        self.running = False
        self.worker.join()
        self.wav.close()


def count_free_run(scores):
    """False-wake counts for rounds with no wake phrase: crossings per threshold."""
    counts = {}
    for thr in FREE_RUN_THRESHOLDS:
        for ch in (0, 1, "either"):
            last = -1e9
            n = 0
            for frame, s0, s1, _, _ in scores:
                s = max(s0, s1) if ch == "either" else (s0, s1)[ch]
                if s >= thr and (frame - last) / RATE >= FREE_RUN_COOLDOWN:
                    n += 1
                    last = frame
            counts[f"ch{ch}@{thr}" if ch != "either" else f"either@{thr}"] = n
    return counts


def run_round(name, spec, args, devices, out_dir):
    mic_dev, out_dev, ref_dev = devices
    print(f"\n=== Round {name}: {spec['say']}")
    if not args.yes:
        input("Press Enter to start (Ctrl+C to stop)... ")

    speech = speech_rate = None
    if spec.get("playback"):
        speech, speech_rate = sf.read(args.speech, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

    if spec.get("asr_off"):
        route("asr-off")
    meta = {
        "round": name,
        "instructions": spec["say"],
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "chip_asr_output": read_chip("AEC_ASROUTONOFF"),
        "speech_file": str(args.speech) if speech is not None else None,
    }
    wav_path = out_dir / f"round-{name}.wav"
    try:
        with Speaker(out_dev, ref_dev, speech, speech_rate) as spk, Recorder(mic_dev, args.model, wav_path) as rec:
            time.sleep(2.0)
            spk.beep(660, 0.1); time.sleep(0.2); spk.beep(660, 0.1); time.sleep(0.2); spk.beep(880, 0.15)
            time.sleep(1.5)
            attempts = []
            if "attempts" in spec:
                for i in range(1, spec["attempts"] + 1):
                    spk.beep()
                    time.sleep(BEEP_SECONDS + 0.15)
                    start = rec.frames
                    end = start + int(WINDOW_SECONDS * RATE)
                    rec.wait_for_frame(end + int(SCORE_TAIL_SECONDS * RATE))
                    s0, s1 = rec.best_between(start, end + int(SCORE_TAIL_SECONDS * RATE))
                    attempts.append({"attempt": i, "start_frame": start, "end_frame": end,
                                     "best_ch0": round(s0, 4), "best_ch1": round(s1, 4)})
                    print(f"  attempt {i:2d}/{spec['attempts']}:  CH0 {s0:.2f}   CH1 {s1:.2f}")
                    time.sleep(GAP_SECONDS)
            else:
                total = spec["seconds"]
                t0 = time.monotonic()
                while (left := total - (time.monotonic() - t0)) > 0:
                    print(f"\r  recording... {int(left // 60):02d}:{int(left % 60):02d} left", end="", flush=True)
                    time.sleep(min(1.0, left))
                print()
            spk.beep(440, 0.25)
            time.sleep(0.6)
    finally:
        if spec.get("asr_off"):
            route("asr-on")

    meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["seconds_recorded"] = round(rec.frames / RATE, 1)
    meta["input_overflows"] = rec.overflows
    if attempts:
        meta["attempts"] = attempts
        for ch in (0, 1):
            best = [a[f"best_ch{ch}"] for a in attempts]
            meta[f"ch{ch}_median"] = round(float(np.median(best)), 3)
            meta[f"ch{ch}_hits"] = {str(t): sum(b >= t for b in best) for t in (0.1, 0.3, 0.5)}
    else:
        meta["false_wake_counts"] = count_free_run(rec.scores)

    (out_dir / f"round-{name}.json").write_text(json.dumps(meta, indent=2))
    np.savetxt(out_dir / f"round-{name}-scores.csv", np.array(rec.scores), delimiter=",",
               header="frame,score_ch0,score_ch1,rms_ch0,rms_ch1", comments="", fmt="%.4f")
    if attempts:
        print(f"  median CH0 {meta['ch0_median']}  CH1 {meta['ch1_median']}  "
              f"| reaching 0.5: CH0 {meta['ch0_hits']['0.5']}/{len(attempts)}  CH1 {meta['ch1_hits']['0.5']}/{len(attempts)}")
    else:
        print(f"  wake-score crossings (either channel): {meta['false_wake_counts']['either@0.1']} at 0.1, "
              f"{meta['false_wake_counts']['either@0.5']} at 0.5")
    print(f"  saved {wav_path}")


def main():
    load_env_file()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rounds", nargs="*", help="round names in order, e.g. 1 2 2b 3")
    p.add_argument("--list", action="store_true", help="show the rounds and exit")
    p.add_argument("--yes", action="store_true", help="don't wait for Enter between rounds")
    p.add_argument("--out", type=Path, help="session folder (default: a new dated folder)")
    p.add_argument("--model", default=os.environ.get("NEXUS_WAKE_MODEL_PATH",
                                                     str(Path.home() / "echo/Hey_Nexus_20260328_154923.onnx")))
    p.add_argument("--speech", type=Path, default=Path.home() / "echo/marin_test.wav",
                   help="speech looped through the speakers in round 5")
    p.add_argument("--force", action="store_true", help="run even if nexus.service is active")
    p.add_argument("--attempts", type=int, help="override attempts per round (for a quick test)")
    p.add_argument("--seconds", type=int, help="override length of timed rounds (for a quick test)")
    args = p.parse_args()
    for spec in ROUNDS.values():
        if args.attempts and "attempts" in spec:
            spec["attempts"] = args.attempts
        if args.seconds and "seconds" in spec:
            spec["seconds"] = args.seconds

    if args.list or not args.rounds:
        for name, spec in ROUNDS.items():
            length = f"{spec['attempts']} attempts" if "attempts" in spec else f"{spec['seconds'] // 60} min"
            print(f"  {name:3s} {length:12s} {spec['say']}")
        return
    unknown = [r for r in args.rounds if r not in ROUNDS]
    if unknown:
        sys.exit(f"Unknown round(s): {', '.join(unknown)}. Use --list.")

    active = subprocess.run(["systemctl", "is-active", "--quiet", "nexus.service"]).returncode == 0
    if active and not args.force:
        sys.exit("Nexus is running and holds the microphone. Stop it first:\n"
                 "    sudo systemctl stop nexus.service\n"
                 "and start it again when you're done:\n"
                 "    sudo systemctl start nexus.service")

    mic_name = os.environ.get("NEXUS_MIC_NAME", "respeaker xvf3800")
    mic_dev = find_device(mic_name, need_input=True)
    out_dev = find_device(os.environ.get("NEXUS_SPEAKER_NAME", "alc3234 analog"), need_output=True)
    ref_name = os.environ.get("NEXUS_AEC_REFERENCE_NAME", mic_name)
    ref_dev = None if ref_name.strip().lower() in ("", "off", "none") else find_device(ref_name, need_output=True)
    if mic_dev is None or out_dev is None:
        sys.exit(f"Audio device missing: mic={mic_dev} speaker={out_dev}. Is Nexus still holding them?")
    if ref_dev is None:
        print("Note: no echo-canceller reference device; round 5 will not match how Nexus plays.")

    out_dir = args.out or OUT_ROOT / datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}")
    print(f"Mic: {sd.query_devices(mic_dev)['name']}   Speaker: {sd.query_devices(out_dev)['name']}")
    print("Each attempt: a beep, then say \"Hey Nexus\" once at normal volume. Three rising beeps start a round,"
          " one low beep ends it.")

    for name in args.rounds:
        run_round(name, ROUNDS[name], args, (mic_dev, out_dev, ref_dev), out_dir)
    print(f"\nDone. Recordings in {out_dir}. Start Nexus again: sudo systemctl start nexus.service")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Finished rounds are saved; start Nexus again: sudo systemctl start nexus.service")
