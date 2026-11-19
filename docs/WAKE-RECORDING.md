# Recording the wake-word test set

The wake model scores computer voices at 0.99 but real speech much lower, and it
fires on sound-alikes ("next", "connects us"). Every fix (picking the mic
channel, a personal second check, a retrain, barge-in tuning) needs the same set
of real recordings to train on and measure against. `tools/wake_record.py` makes
that set in one session at the box: about 15 minutes of saying the phrase, 5
minutes reading aloud, and an hour of TV with nobody there.

## Before you start

1. **Move the speakers to the mic array's 3.5 mm jack** if you can (powered
   speakers with a 3.5 mm plug). Then the chip's echo canceller works on one
   clock. In a test on 2026-11-19 with the speakers still on the Dell jack,
   Nexus's own voice came through both channels almost unreduced.
2. Stop Nexus. It holds the microphone:
   `sudo systemctl stop nexus.service`

## Run it

From `~/echo` on the box (SSH from a phone or laptop is fine):

```
.venv/bin/python wake_record.py --list           # what each round is
.venv/bin/python wake_record.py 1 2 2b 3 4 5 6 7
```

- Press Enter before each round. Three rising beeps start it, one low beep ends it.
- In rounds 1–5, say "Hey Nexus" once at normal volume after each beep. That's 20 times per round, a beep every ~5 s.
- Round 2b switches the chip's wake-word output off for comparison and turns it back on afterwards.
- Round 5 plays Nexus's voice through the speakers. Say the phrase over it.
- Round 6: read anything aloud for 5 minutes, without the phrase.
- Round 7: leave a TV show or podcast on for an hour and walk away.
- Ctrl+C stops at any time. Finished rounds stay saved, and rounds can be rerun on their own (e.g. `... wake_record.py 4`).

Afterwards: `sudo systemctl start nexus.service`

Recordings land in `~/echo/wake-recordings/<date_time>/`: one 2-channel 16 kHz WAV
per round, the live wake scores (`-scores.csv`) and a summary (`.json`). Expect
about 300 MB for a full session.

## Box setup this relies on (done 2026-11-19)

- `/usr/local/bin/nexus-respeaker-route` (root-owned, copy in `systemd/`) runs a
  root-owned copy of `xvf_host` in `/usr/local/lib/nexus/xvf_host/`, not the one
  in `~/echo`. A file `ryan` can overwrite must never run as root (review B12).
- The helper turns on the chip's wake-word (ASR) output (`AEC_ASROUTONOFF 1`,
  CH1 = AEC residual 3) at every boot (`nexus-respeaker.service` drop-in) and at
  every Nexus start. It was already on when read on 2026-11-19, but nothing set
  it, so a power loss could have reset it.
- `ryan` may run the helper's `asr-on`/`asr-off` and `systemctl stop|start
  nexus.service` without a password (`systemd/nexus-respeaker.sudoers`).
- Syncthing's journal output is capped at 200 lines per 30 s
  (`systemd/syncthing-log-limit.conf`), so a flood can't push Nexus's wake
  history out of the journal again.

## Decisions

- **2026-11-19, Ryan ("Let's move on to the wake word prep"):**
  - **Root-owned tool copy.** Use a root-owned copy of `xvf_host` rather than a udev rule. It's the smaller change and keeps the existing sudo flow.
  - **Syncthing rate cap.** Cap Syncthing's log rate instead of moving Nexus to its own journal namespace (`LogNamespace`). The flood had stopped, and a namespace would change every `journalctl -u nexus` command. Revisit if wake history is lost again.
