#!/usr/bin/env bash
# install_jetson.sh — one-time setup of HANDOFF.AI on the Jetson (works on any Ubuntu).
#
#   bash scripts/install_jetson.sh
#
# What it does
#   1. installs system packages: PortAudio (mic), venv, unzip
#   2. creates .venv and installs requirements.txt
#   3. creates .env from .env.example if there is none (then edit it: GROQ_API_KEY)
#   4. downloads the speech models (Vosk captions + Whisper) for offline use
#   5. adds a "HANDOFF.AI" launcher to the app menu and the desktop
#   6. runs the setup check
#
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"

echo "== 1/6 system packages"
sudo apt-get update -qq
sudo apt-get install -y libportaudio2 python3-venv python3-pip unzip

echo "== 2/6 python environment"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

echo "== 3/6 settings"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env."
fi
if ! grep -q '^GROQ_API_KEY=gsk_' .env; then
  bash scripts/use_old_groq_key.sh || echo "Set GROQ_API_KEY in $APP_DIR/.env (nano $APP_DIR/.env)."
fi

echo "== 4/6 speech models"
.venv/bin/python scripts/download_models.py || echo "Some downloads failed; the app downloads them on first use."

echo "== 5/6 launcher"
LAUNCHER="$HOME/.local/share/applications/handoff-ai.desktop"
mkdir -p "$(dirname "$LAUNCHER")"
cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=HANDOFF.AI
Comment=EMS radio report to hospital handoff
Exec=bash -c 'cd "$APP_DIR" && .venv/bin/python -m ems_handoff >> output/launcher.log 2>&1'
Icon=audio-input-microphone
Terminal=false
Categories=Medical;AudioVideo;
EOF
chmod +x "$LAUNCHER"
if [ -d "$HOME/Desktop" ]; then
  cp "$LAUNCHER" "$HOME/Desktop/handoff-ai.desktop"
  chmod +x "$HOME/Desktop/handoff-ai.desktop"
  gio set "$HOME/Desktop/handoff-ai.desktop" metadata::trusted true 2>/dev/null || true
fi
mkdir -p output

echo "== 6/6 setup check"
.venv/bin/python -m ems_handoff --check || true
echo
echo "Done. Start HANDOFF.AI from the app menu or desktop icon, or run:"
echo "  cd $APP_DIR && .venv/bin/python -m ems_handoff"
