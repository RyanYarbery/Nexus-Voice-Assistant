# Decisions

Each entry: date, who decided, what, why, what it replaces, what would justify revisiting.

## 2026-11-18: Fix first, then split into modules, on branch `modular`
- **Who:** Ryan: "I'll take any fixes you recommend. Also, i want you to make all of the pieces of
  nexus modular so another user can insert their own hardware and features with ease and not have
  issues."
- **What:** (1) The review's failing probes become tests and the bugs are fixed in `echo.py` one
  commit at a time. (2) Then `echo.py` is split into a `nexus/` package with swap-in parts:
  microphone/speaker, wake engine, speaker check, voice backend, tools (features), alerts.
  `echo.py` stays as a thin launcher so the existing service keeps working.
- **Why:** Fixing in the single file first keeps each fix small and checkable; the tests then guard
  the split. Parts are chosen by settings, and the default is plain hardware (any USB mic, default
  speaker, no speaker check) so a new user needs no XVF3800.
- **Replaces:** the single-file design.
- **Revisit if:** the split makes latency worse on the 2-core box, or an adapter needs something
  the interfaces can't express.

## 2026-11-18: Agensi listing = a skill that installs and extends Nexus
- **Who:** decided by Claude, Ryan to confirm. Ryan: "Agensi.io has come across the Nexus git and
  want me to put it on their site so help me shape it in a way to better fit agensi."
- **What:** Agensi lists SKILL.md packages (instructions an AI coding agent loads), not apps. The
  repo gets `skills/nexus-voice-assistant/` (SKILL.md + references + scripts) that walks a user's
  agent through installing Nexus, writing a hardware adapter and adding a feature. The app stays
  on GitHub; the skill points to it.
- **Why:** it is the only shape Agensi accepts (ZIP with SKILL.md; no undisclosed network calls,
  no credential handling, must match its listing), and a modular Nexus makes the skill useful.
- **Revisit if:** Agensi's message to Ryan asks for something else (I have not seen it).
