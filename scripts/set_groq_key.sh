#!/usr/bin/env bash
# set_groq_key.sh — paste a Groq API key, check it with Groq, and save it for both apps.
#
#   bash scripts/set_groq_key.sh
#
# Shows the key while you paste, confirms what it received, tests it with Groq,
# and only then writes GROQ_API_KEY into:
#     ~/EMS-Handoff/.env       (the old app)
#     ~/EMS-Handoff-app/.env   (the new app)
# Any older GROQ_API_KEY line is replaced. The files are made readable only by you.

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

echo
SAVED=0
for DIR in "$HOME/EMS-Handoff" "$HOME/EMS-Handoff-app"; do
  [ -d "$DIR" ] || continue
  touch "$DIR/.env"
  sed -i '/^GROQ_API_KEY=/d' "$DIR/.env"
  echo "GROQ_API_KEY=$KEY" >> "$DIR/.env"
  chmod 600 "$DIR/.env"
  echo "Saved to $DIR/.env"
  SAVED=1
done
[ "$SAVED" = 1 ] || echo "Neither ~/EMS-Handoff nor ~/EMS-Handoff-app exists, so nothing was saved."
