# HANDOFF.AI in a container: the server only (no app window). Open http://<host>:5050/ in a browser.
#
#   docker build -t handoff-ai .
#   docker run --rm --network host --device /dev/snd \
#       -v ~/.config/handoff-ai:/root/.config/handoff-ai:ro \
#       -v handoff-output:/app/output -v handoff-hf:/root/.cache/huggingface handoff-ai
#
# --network host   the app is on port 5050 of the machine, and Ollama on localhost:11434 is reachable
# --device /dev/snd  the microphone (Linux only; on Windows/macOS use the venv install, see README)
# ~/.config/handoff-ai  the shared Groq key written by scripts/set_groq_key.sh (or pass --env-file .env)
# handoff-hf       keeps the Whisper models (about 1.6 GB) between runs instead of fetching them each start

FROM python:3.11-slim

WORKDIR /app

# System packages: ffmpeg for faster-whisper, libportaudio2 for the microphone (sounddevice).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

# Dependencies first so a source edit does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app, its scripts, and the files it reads from the project root.
COPY ems_handoff/ ems_handoff/
COPY scripts/ scripts/
COPY pyproject.toml README.md .env.example ./

# The Vosk caption model (about 40 MB) is baked in so live captions work offline.
# Whisper models download on first use into /root/.cache/huggingface (mount a volume to keep them).
RUN python scripts/download_models.py --vosk && mkdir -p output

ENV HANDOFF_HOST=0.0.0.0 \
    HANDOFF_PORT=5050 \
    PYTHONUNBUFFERED=1

EXPOSE 5050

CMD ["python", "-m", "ems_handoff", "--no-window"]
