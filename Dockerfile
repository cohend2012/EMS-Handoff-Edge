# Start from an official Python image. "slim" is a stripped-down Debian
# Linux with Python already installed — smaller than the full image, but
# still has apt so we can add system packages.
FROM python:3.11-slim

# Everything after this runs inside /app in the container.
# Think of it as `cd /app` for the rest of the build.
WORKDIR /app

# ffmpeg is a SYSTEM package (not pip). faster-whisper needs it to read
# audio files. This is exactly the kind of dependency a venv can't handle
# and a container can.
#   --no-install-recommends  = skip optional extras, keeps image smaller
#   rm -rf /var/lib/apt/lists = delete apt's cache, also keeps image smaller
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements.txt first, then install.
# This is a caching trick: Docker caches each step, and only re-runs a step
# if its inputs changed. Since requirements.txt changes rarely but your .py
# files change constantly, doing it this way means editing handoff.py
# doesn't force a slow re-install of every pip package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual source code into /app/src inside the container.
COPY src/ ./src/

# Your app writes logs and JSON to output/. Create it so the app doesn't
# fail on first run.
RUN mkdir -p output

# The command that runs when the container starts. Same command you'd
# type locally — because inside the container, you ARE on Linux with
# everything installed.
CMD ["python", "-m", "src.main"]
