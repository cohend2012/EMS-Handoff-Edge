#!/usr/bin/env bash
# use_old_groq_key.sh — reuse the Groq key from the old local EMS-Handoff checkout, if it still works.
#
#   bash scripts/use_old_groq_key.sh [path/to/old/option_a_ems_handoff.py]
#
# Reads the key from the old prototype file on this machine (default
# ~/EMS-Handoff/option_a_ems_handoff.py), tests it against Groq, and if Groq
# accepts it (HTTP 200) writes GROQ_API_KEY into this app's .env.
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
[ -f .env ] || cp .env.example .env
sed -i '/^GROQ_API_KEY=/d' .env
echo "GROQ_API_KEY=$KEY" >> .env
echo "Groq accepted the key from $OLD. Saved to .env."
