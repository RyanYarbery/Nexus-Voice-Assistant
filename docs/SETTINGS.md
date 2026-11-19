# Settings

Every setting is an environment variable, usually kept in an env file that
your service manager loads (see `.env.example`). `python -m nexus settings`
lists the ones in effect on your machine, and `python -m nexus doctor` points
out any `NEXUS_` variable that nothing reads.

Only one of each kind of part is loaded, so you only need the settings for the
parts you chose. Two parts may read the same variable with different defaults
(for example `NEXUS_MIC_NAME`).

`$NEXUS_HOME` below stands for your Nexus folder. The one secret, the API key,
is not a Nexus setting: put `OPENAI_API_KEY` in the env file. Nexus never
prints it.

This file is written by `tools/write_settings_doc.sh` from the code; a test
fails if it falls behind.

## core

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_ACCOUNT_RETRY_SECONDS` | `600` | Wait between attempts when the account, key or model is the problem. |
| `NEXUS_ACTIVE_SPEECH_TIMEOUT_SECONDS` | `10.0` | Give up on a request if no speech starts within this. |
| `NEXUS_AUDIO` | `generic` | Microphone and speaker: generic (any sound card), xvf3800, fake, or module:Class. |
| `NEXUS_BACKEND` | `openai` | Voice service: openai, fake, or module:Class. |
| `NEXUS_BARGE_CHANNEL` | `ch0` | Channel that can interrupt playback: ch0, ch1 or best. |
| `NEXUS_BARGE_DEBUG` | `0` | Print barge-in peak scores twice a second. |
| `NEXUS_BARGE_WAKE_THRESHOLD` | `0.3` | Wake-word score needed to interrupt Nexus while it is speaking. |
| `NEXUS_BEEP_HOLD_MAX_SECONDS` | `60` | Beeps wait up to this long for a request in progress to end (0 = ring at once). |
| `NEXUS_FOLLOWUP_GUARD_SECONDS` | `0.35` | Microphone ignored for this long after a reply ends, so the tail of Nexus's own voice is not heard. |
| `NEXUS_FOLLOWUP_KEEP_OPEN` | `1` | Keep the follow-up window open after a short or rejected sound (0 = one sound closes it). |
| `NEXUS_FOLLOWUP_MIN_SPEECH_SECONDS` | `0.4` | Sounds shorter than this in the follow-up window are ignored. |
| `NEXUS_FOLLOWUP_SECONDS` | `5.0` | How long the microphone stays open for a follow-up after a reply. |
| `NEXUS_HOME` | the folder Nexus is installed in | Folder holding models, plugins and saved state. |
| `NEXUS_HTTP_USER_AGENT` | `NexusHouseholdAssistant/1.0 (personal non-commercial assistant)` | User-Agent sent to web services (geocoding, weather, alerts). |
| `NEXUS_INSTRUCTIONS_FILE` | (empty) | A text file with your own personality prompt; replaces NEXUS_PERSONA. |
| `NEXUS_MIC_STALL_SECONDS` | `10` | Restart if the microphone delivers no audio for this long (0 = never). |
| `NEXUS_MISSED_ALARM_NOTES` | `1` | Have Nexus mention, with its next reply, an alarm or timer missed while it was off. |
| `NEXUS_NETWORK_PROBLEM_AFTER_FAILURES` | `3` | Failed connection attempts in a row before a network problem is reported. |
| `NEXUS_PERSONA` | `plain` | Personality: plain, sardonic, or any name with a file in nexus/personas/. |
| `NEXUS_PLAYBACK_PREBUFFER_SECONDS` | `0.3` | Audio buffered before the speaker starts. |
| `NEXUS_PLAYER_STALL_SECONDS` | `10` | Stop a reply's playback after this long with no audio arriving (0 = no limit). |
| `NEXUS_RECONNECT_MAX_SECONDS` | `15` | Longest wait between reconnect attempts after a network failure. |
| `NEXUS_REPLY_TIMEOUT_SECONDS` | `15` | Give up on a reply if no audio has arrived this long after asking (0 = wait for ever). |
| `NEXUS_REQUEST_LOG` | `1` | Print one REQUEST line with scores, timings and usage per request. |
| `NEXUS_REQUEST_MAX_SECONDS` | `30` | Longest one spoken request may last before it is dropped (0 = no limit). |
| `NEXUS_SOFT_VERIFY_SECONDS` | `2.5` | Soft mode: most audio used to judge the speaker. |
| `NEXUS_SOFT_VERIFY_SPEECH_SECONDS` | `1.0` | Soft mode: speech needed before a timed check. |
| `NEXUS_SPEAKER_FAIL_LIMIT` | `3` | Failed replies in a row before the speaker is reported broken and Nexus restarts (0 = never). |
| `NEXUS_SPEAKER_ID` | `none` | Voice check: none (anyone may speak), ecapa, fake, or module:Class. |
| `NEXUS_SPEAKER_THRESHOLD` | `0.25` | Voice-match score below which the speaker is rejected. |
| `NEXUS_SPEAKER_VERIFY_MODE` | `soft` | Voice check: soft, strict or off. |
| `NEXUS_SPEAKER_VERIFY_SECONDS` | `1.6` | Strict mode: audio gathered before the speaker is judged. |
| `NEXUS_STATUS_OK_AFTER_SECONDS` | `5` | A connection counts as healthy once it has lasted this long. |
| `NEXUS_STREAMING_PLAYBACK` | `1` | Start speaking as soon as audio arrives (0 = wait for the whole reply). |
| `NEXUS_WAKE` | `openwakeword` | Wake-word detector: openwakeword, fake, or module:Class. |
| `NEXUS_WAKE_ALWAYS_SCORE` | `1` | Feed the wake models every microphone block in every state (0 = only while waiting or speaking). |
| `NEXUS_WAKE_DEBUG` | `0` | Print wake scores above 0.02. |
| `NEXUS_WAKE_PHRASE` | `Hey Nexus` | The wake phrase as words, for log lines and the prompt (the wake model decides what is heard). |
| `NEXUS_WAKE_PREROLL_SECONDS` | `1.0` | Audio kept from before the wake word fired. |
| `NEXUS_WAKE_REJECT_COOLDOWN_SECONDS` | `1.0` | Wake word ignored for this long after a rejected speaker. |
| `NEXUS_WAKE_RESUME_COOLDOWN_SECONDS` | `0.3` | Wake word ignored for this long after each exchange. |
| `NEXUS_WAKE_THRESHOLD` | `0.1` | Wake-word score (0 to 1) at which Nexus wakes. |

## alerts

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_NOTICE_DIR` | `$NEXUS_HOME/notices` | Folder of your own notice recordings (quota.wav, network.wav, error.wav ...). |
| `NEXUS_NOTICE_MIN_GAP_SECONDS` | `20` | Shortest time between two notices. |
| `NEXUS_NOTICE_TTS` | `auto` | Speech for notices: auto (espeak-ng if installed), off (tones only), or a command that writes WAV to stdout. |
| `NEXUS_NTFY_URL` | (empty) | ntfy topic address for push alerts when the status changes; empty for none. |
| `NEXUS_OFFLINE_NOTICES` | `1` | Play a short notice when the wake phrase is heard but Nexus cannot answer. |
| `NEXUS_STATUS_FILE` | `$NEXUS_HOME/status.json` | Health file for a watchdog to read; "off" for none. |

## plugins

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_PLUGINS` | `timers,alarms,location,clock,weather` | Built-in features to load, comma separated; "none" for none. |
| `NEXUS_PLUGIN_DIR` | `$NEXUS_HOME/plugins` | Folder of your own plugin files; every *.py in it is loaded. |
| `NEXUS_TOOL_TIMEOUT_SECONDS` | `30` | A tool that runs longer than this is stopped and reported as failed. |

## plugin: timers and alarms

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_LATE_GRACE_SECONDS` | `600` | A timer or alarm that came due while Nexus was off still rings if it is no later than this. |

## plugin: timers

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_TIMERS_FILE` | `$NEXUS_HOME/timers.json` | Saved timers. |
| `NEXUS_TIMER_BEEP_COUNT` | `10` | Beeps when a timer ends. |
| `NEXUS_TIMER_RING_SECONDS` | `60` | A finished timer rings until the wake phrase or this long (0 = a fixed number of beeps). |
| `NEXUS_TIMER_VOLUME` | `0.25` | Timer beep volume, 0 to 1. |

## plugin: alarms

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_ALARMS_FILE` | `$NEXUS_HOME/alarms.json` | Saved alarms. |
| `NEXUS_ALARM_BEEP_COUNT` | `16` | Beeps when an alarm goes off. |
| `NEXUS_ALARM_RING_SECONDS` | `180` | An alarm rings until the wake phrase or this long (0 = a fixed number of beeps). |
| `NEXUS_ALARM_VOLUME` | `0.3` | Alarm beep volume, 0 to 1. |

## plugin: location

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_SETTINGS_FILE` | `$NEXUS_HOME/settings.json` | Saved location and preferences. |

## audio: generic

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_MIC_CHANNELS` | `1` | Microphone channels to use: 1, or 2 if the device gives two useful channels. |
| `NEXUS_MIC_NAME` | (empty) | Part of the microphone's device name; empty for the system default. |
| `NEXUS_OUTPUT_RATE` | `48000` | Sample rate the speaker is opened at. |
| `NEXUS_SPEAKER_NAME` | (empty) | Part of the speaker's device name; empty for the system default. |

## audio: xvf3800

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_AEC_REFERENCE_CONTINUOUS` | `1` | Keep the echo-cancelling reference stream open all the time (0 = open one per reply). |
| `NEXUS_AEC_REFERENCE_NAME` | `respeaker xvf3800` | Output that receives a copy of playback for echo cancelling; "off" for none. |
| `NEXUS_MIC_NAME` | `respeaker xvf3800` | Part of the array's device name. |
| `NEXUS_RESPEAKER_ROUTE_HELPER` | `/usr/local/bin/nexus-respeaker-route` | Optional helper, run through sudo at start, that sets the array's channel routing. |
| `NEXUS_SPEAKER_NAME` | (empty) | Part of the speaker's device name; empty for the system default. |

## wake: openwakeword

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_WAKE_MODEL_PATH` | `$NEXUS_HOME/models/hey_nexus.onnx` | openWakeWord .onnx model for your wake phrase. |

## speaker_id: ecapa

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_SPEAKER_MODEL_CACHE` | `$NEXUS_HOME/models/spkrec-ecapa-voxceleb` | Folder for the downloaded speaker-recognition model. |
| `NEXUS_SPEAKER_SPEECH_AWARE` | `1` | Trim the audio to the speech before the voice check. |
| `NEXUS_VOICE_PROFILE_CH0` | `$NEXUS_HOME/models/voice_ch0.pt` | Enrolled voice profile for microphone channel 0. |
| `NEXUS_VOICE_PROFILE_CH1` | `$NEXUS_HOME/models/voice_ch1.pt` | Enrolled voice profile for microphone channel 1 (channel 0's is used if this file is missing). |

## backend: openai

| Setting | Default | What it does |
|---|---|---|
| `NEXUS_NOISE_REDUCTION` | `off` | OpenAI input noise filter: off, near_field or far_field. |
| `NEXUS_PING_INTERVAL_SECONDS` | `5` | Seconds between keep-alive pings. |
| `NEXUS_PING_TIMEOUT_SECONDS` | `20` | Drop the connection after this long without a ping reply. |
| `NEXUS_TURN_DETECTION` | `server_vad` | How OpenAI decides you have finished: server_vad (fixed silence) or semantic_vad. |
| `NEXUS_UPLINK_STREAM_RESAMPLE` | `1` | Resample microphone audio as one stream (0 = block by block, which clicks at block edges). |
| `NEXUS_VAD_PREFIX_MS` | `300` | Audio kept from before speech was detected. |
| `NEXUS_VAD_SILENCE_MS` | `700` | Silence that ends your turn, in milliseconds. |
| `NEXUS_VAD_THRESHOLD` | `0.5` | OpenAI speech detector sensitivity, 0 to 1. |
| `NEXUS_VOICE` | `marin` | OpenAI voice used for replies. |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime-2.1` | OpenAI Realtime model name. |

