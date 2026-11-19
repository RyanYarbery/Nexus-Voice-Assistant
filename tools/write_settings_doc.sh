#!/bin/sh
# Rewrite docs/SETTINGS.md from the settings the code registers.
# Run from the repository root:  tools/write_settings_doc.sh [python]
set -e
PYTHON="${1:-python3}"
{
cat <<'HEAD'
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

HEAD
env -i PATH="$PATH" NEXUS_HOME=/opt/nexus "$PYTHON" -m nexus settings --markdown | sed 's#/opt/nexus#$NEXUS_HOME#g'
} > docs/SETTINGS.md
echo "wrote docs/SETTINGS.md"
