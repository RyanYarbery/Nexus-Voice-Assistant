# Phone shortcuts: stop + silence, restart, reboot

Nexus answers a few plain web addresses, so a phone shortcut can control it with one request.
The code is the "PHONE CONTROL" section of `echo.py`.

## Turn it on

1. Add a token to the env file the service reads (`~/.config/echo/env`), then restart Nexus:

   ```
   NEXUS_CONTROL_TOKEN=<output of: openssl rand -hex 12>
   ```

   No token = the endpoint is off.
2. For the reboot address only: install `systemd/nexus-control.sudoers` as
   `/etc/sudoers.d/nexus-control` (root:root, mode 440, check with `visudo -cf`).

Optional settings: `NEXUS_CONTROL_PORT` (8377), `NEXUS_CONTROL_HOST` (0.0.0.0),
`NEXUS_CONTROL_ALLOW` (networks allowed to call; default is Tailscale addresses and the box itself),
`NEXUS_SILENCE_FILE` (`silence.json` in the Nexus home folder).

## Addresses

Replace `nexus` with the box's Tailscale name or address and `TOKEN` with the token.

| Address | What it does |
| --- | --- |
| `http://nexus:8377/stop?token=TOKEN&minutes=20` | Stops talking at once and stays silent for 20 minutes |
| `http://nexus:8377/stop?token=TOKEN&for=1h30m` | Same, with `s`, `m`, `h`, `d` units |
| `http://nexus:8377/resume?token=TOKEN` | Ends the silence early |
| `http://nexus:8377/status?token=TOKEN` | Says whether it is silent and for how long |
| `http://nexus:8377/restart?token=TOKEN` | Restarts the assistant only (back in about 25 seconds) |
| `http://nexus:8377/reboot?token=TOKEN` | Reboots the whole box (back in about 95 seconds, measured 2026-11-23) |

While silent, Nexus ignores the wake word and drops any reply that was on its way. Timers and alarms
still ring; calling `/stop` again quiets one that is ringing. `/stop` with no time (or 0) silences for
5 seconds, enough to cut off a reply. The silence survives a restart or power cut. The most it will
accept is 7 days.

## iPhone (Shortcuts app)

The phone must be connected to Tailscale.

**Nexus quiet**

1. New shortcut. Add **Ask for Input**: type *Number*, prompt "Quiet for how many minutes?".
2. Add **Get Contents of URL**. Type `http://nexus:8377/stop?token=TOKEN&minutes=` and then insert
   the *Ask for Input* variable at the end of the address.
3. Optional: add **Show Notification** with the *Contents of URL* variable to see the answer.

**Nexus reboot**

1. New shortcut. Add **Get Contents of URL** with `http://nexus:8377/reboot?token=TOKEN`.

Name them, then *Add to Home Screen* (or say the name to Siri).
