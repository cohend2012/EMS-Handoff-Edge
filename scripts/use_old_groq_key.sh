#!/usr/bin/env bash
# use_old_groq_key.sh — reuse the Groq key from the old local EMS-Handoff checkout, if it still works.
#
#   bash scripts/use_old_groq_key.sh [path/to/old/option_a_ems_handoff.py]
#
# Reads the key from the old prototype file on this machine (default
# ~/EMS-Handoff/option_a_ems_handoff.py), tests it against Groq, and if Groq
# accepts it (HTTP 200) saves it in the one shared key file,
# ~/.config/handoff-ai/secrets.env, that every copy of the app reads.
# Nothing is committed; the key only moves between local files.
set -euo pipefail
cd "$(dirname "$0")/.."
OLD="${1:-$HOME/EMS-Handoff/option_a_ems_handoff.py}"

KEY=$(grep -o 'gsk_[A-Za-z0-9]*' "$OLD" 2>/dev/null | head -1 || true)
if [ -z "$KEY" ]; then
  echo "No Groq key found in $OLD."
  exit 1
fi
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
  https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY" || true)
CODE="${CODE:-000}"
if [ "$CODE" != "200" ]; then
  echo "The Groq key in $OLD does not work (HTTP $CODE)."
  echo "Create a new key at https://console.groq.com/keys and set GROQ_API_KEY in .env."
  exit 1
fi
SECRETS="${HANDOFF_SECRETS:-$HOME/.config/handoff-ai/secrets.env}"
mkdir -p "$(dirname "$SECRETS")"
touch "$SECRETS" && chmod 600 "$SECRETS"
sed -i '/^GROQ_API_KEY=/d' "$SECRETS"
echo "GROQ_API_KEY=$KEY" >> "$SECRETS"
echo "Groq accepted the key from $OLD. Saved once, in $SECRETS."
