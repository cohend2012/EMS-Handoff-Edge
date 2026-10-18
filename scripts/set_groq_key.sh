#!/usr/bin/env bash
# set_groq_key.sh — set the Groq API key ONCE for every copy of HANDOFF.AI on this machine.
#
#   bash scripts/set_groq_key.sh
#
# The key is stored in one file:  ~/.config/handoff-ai/secrets.env   (readable only by you)
#   * the new app (~/EMS-Handoff-app) reads that file automatically
#   * the old app (~/EMS-Handoff) has its .env turned into a link to that file, so it
#     reads the same key. Any other settings in the old .env are kept (moved into the
#     shared file) and the original is backed up as .env.bak.
# To change the key later, run this script again. That is the only place to change it.
#
# You see the key as you paste it, then a masked summary, then Groq's verdict.
# Nothing is saved unless Groq accepts the key.

SECRETS="${HANDOFF_SECRETS:-$HOME/.config/handoff-ai/secrets.env}"
OLD_APP="$HOME/EMS-Handoff"
NEW_APP="$HOME/EMS-Handoff-app"

echo
echo "Step 1 of 3: paste your Groq API key (it starts with gsk_), then press Enter."
read -rp "Key: " KEY
KEY=$(printf '%s' "$KEY" | tr -d '[:space:]')

echo
echo "Step 2 of 3: checking what was pasted"
if [ -z "$KEY" ]; then
  echo "  Nothing was pasted. Run the script again."
  exit 1
fi
echo "  Received: ${KEY:0:8}...${KEY: -4}  (${#KEY} characters)"
case "$KEY" in
  gsk_*) echo "  Format looks right (starts with gsk_)." ;;
  *) echo "  This does not start with gsk_, so it is not a Groq key. Nothing was saved."; exit 1 ;;
esac

echo
echo "Step 3 of 3: asking Groq if the key works"
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
  https://api.groq.com/openai/v1/models -H "Authorization: Bearer $KEY" || true)
CODE="${CODE:-000}"
case "$CODE" in
  200) echo "  Groq accepted the key." ;;
  401|403) echo "  Groq rejected the key (HTTP $CODE). Check you copied all of it. Nothing was saved."; exit 1 ;;
  000) echo "  Could not reach Groq. Check the internet connection. Nothing was saved."; exit 1 ;;
  *) echo "  Unexpected answer from Groq (HTTP $CODE). Nothing was saved."; exit 1 ;;
esac

# --- the one place the key lives -------------------------------------------
mkdir -p "$(dirname "$SECRETS")"
touch "$SECRETS"
chmod 600 "$SECRETS"

# old app: keep its other settings, then point its .env at the shared file
if [ -d "$OLD_APP" ]; then
  if [ -f "$OLD_APP/.env" ] && [ ! -L "$OLD_APP/.env" ]; then
    grep -v '^GROQ_API_KEY=' "$OLD_APP/.env" | grep -v '^[[:space:]]*$' | while IFS= read -r line; do
      grep -qxF "$line" "$SECRETS" || echo "$line" >> "$SECRETS"
    done
    mv "$OLD_APP/.env" "$OLD_APP/.env.bak"
  fi
  ln -sfn "$SECRETS" "$OLD_APP/.env"
fi

sed -i '/^GROQ_API_KEY=/d' "$SECRETS"
echo "GROQ_API_KEY=$KEY" >> "$SECRETS"

# new app: make sure no stale key in its own .env competes with the shared one
if [ -f "$NEW_APP/.env" ] && [ ! -L "$NEW_APP/.env" ]; then
  sed -i '/^GROQ_API_KEY=/d' "$NEW_APP/.env"
fi

echo
echo "Saved once, in $SECRETS"
[ -d "$OLD_APP" ] && echo "  old app ($OLD_APP/.env) -> links to it"
[ -d "$NEW_APP" ] && echo "  new app ($NEW_APP) -> reads it automatically"
exit 0
