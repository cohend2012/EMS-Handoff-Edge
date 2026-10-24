# HANDOFF.AI — EMS radio report → hospital handoff

A paramedic gives their radio report out loud. HANDOFF.AI shows the words live, and when the paramedic
presses **Done** (or says **"confirm report ready"**) it turns the report into a structured
hospital handoff: alert level, vitals, interventions, history, and what the ER should prepare before
the patient arrives.

It runs as a desktop app on an NVIDIA Jetson (or any Ubuntu machine). Speech recognition can run fully
on the device; the report-writing AI can be Groq (cloud, fastest), Ollama (on the device, offline), or OpenRouter.

> Decision support only. Clinical judgment is always required.

---

## Quick start (Jetson)

```bash
git clone -b handoff-app https://github.com/cohend2012/EMS-Handoff-Edge ~/EMS-Handoff-app
cd ~/EMS-Handoff-app
bash scripts/install_jetson.sh        # packages, venv, speech models, desktop icon, setup check
# the installer reuses the Groq key from ~/EMS-Handoff if Groq still accepts it;
# otherwise create a key at console.groq.com/keys and run: bash scripts/set_groq_key.sh
# (the key is stored once, in ~/.config/handoff-ai/secrets.env, for every copy of the app)
.venv/bin/python -m ems_handoff --check
```

Then start **HANDOFF.AI** from the desktop icon or app menu, or:

```bash
.venv/bin/python -m ems_handoff
```

An app window opens. No terminal menus. The terminal prints the address, normally
**http://127.0.0.1:5050/**. The header reads *"new app v2.0.0"*.

**http://127.0.0.1:5000 is the old EMS-Handoff app** (the intern version or its Docker container), not this
one. The new app never uses port 5000. If 5050 is taken, it uses the next free port and prints it.

## Using the app

| Screen | What you do |
|---|---|
| **Home** | *Start recording* for a live report (optionally pick a practice script), or pick a sample / paste text and press *Make report*. |
| **Live** | Talk. Your words appear in the right-hand panel within about a second. Press **Done — make report** when finished, or say **"confirm report ready"**. *Cancel* discards the recording. |
| **Working** | Shows each step (transcribing, checking medical terms, writing the handoff) with timings. |
| **Report** | The hospital dashboard. *Record another* or *New report*. Every report is also saved to `output/`. |
| **Settings** | Pick the AI provider (with a *Check connection* button) and the microphone. Saved for next time. |

If anything fails, the app says what went wrong in plain words and returns to the home screen.

---

## How live listening works (and why)

```
mic ─▶ AudioCapture: one stream, 16 kHz, full recording kept
          ├─▶ live captions thread     Vosk streaming recognizer → words on screen ~5×/s
          └─▶ stop-phrase thread       Vosk grammar recognizer: "confirm report ready" or nothing
DONE (button, phrase, or 10-min limit)
          ─▶ final transcript, once    Groq whisper-large-v3-turbo, or local faster-whisper
          ─▶ report pipeline           corrector → EMS lexicon → LLM (JSON) → validation → optional textbook guidance
          ─▶ dashboard + output/handoff_report.json
```

* **Live captions use a streaming recognizer, not Whisper.** Whisper is a batch model: it transcribes a
  finished clip. The first version re-ran Whisper on overlapping 6-second clips every 3 s; on the Jetson
  CPU one clip could take 35 s, and while it ran nothing else was heard. It also kept Whisper's default
  *temperature fallback* — re-decoding a "bad-looking" clip up to 6 times at temperatures 0.0, 0.2 … 1.0
  (Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv:2212.04356) —
  which is right for offline accuracy and wrong for live use. Vosk (Kaldi) is built to stream on small
  ARM CPUs. If Vosk captions are ever not good enough, the published path to real-time Whisper is
  Whisper-Streaming (Macháček, Dabre, Bojar, IJCNLP-AACL 2023, arXiv:2307.14743).
* **The stop phrase has its own listener.** A grammar-limited recognizer can only answer "the phrase" or
  "something else", in its own thread, so it never waits on captions. The captions are checked for the
  phrase too; either one finishes the recording.
* **Done always works.** The phrase is a hands-free shortcut, not the only way. A recording is never
  thrown away except by *Cancel*; the 10-minute limit processes it.
* **Whisper runs once, at the end,** on the whole recording, with the EMS vocabulary hint and a single
  decoding pass. With a Groq key this is Groq's hosted Whisper (about 1–2 s per minute of audio).
  If Groq is unreachable the app retries with local Whisper, and as a last resort uses the live captions.
* **Without the Vosk model** captions fall back to Whisper `tiny.en` on a gap-free window (each pass covers
  all audio since the last commit, single pass, bounded time). Install Vosk for the best experience.

---

## Settings (`.env`)

Copy `.env.example` to `.env`. Everything has a sensible default except the API key.

| Variable | Default | Meaning |
|---|---|---|
| `HANDOFF_PROVIDER` | `groq` | AI that writes the report: `groq`, `ollama`, `openrouter` (also changeable in the app) |
| `GROQ_API_KEY` | — | Groq key. Set it once with `bash scripts/set_groq_key.sh`; it lives in `~/.config/handoff-ai/secrets.env`, shared by every copy of the app. Never commit it. |
| `OLLAMA_MODEL` | `qwen3:8b` | Local model. If it does not fit in free memory, the app uses the largest pulled smaller model that does (`qwen3:4b`, `qwen3:1.7b`, ...) and says so |
| `HANDOFF_TRANSCRIBER` | `auto` | Final transcript: `groq`, `local`, or `auto` (Groq if a key is set) |
| `LOCAL_WHISPER_MODEL` | `large-v3-turbo` | Offline transcript model (`small.en` is ~4× faster on CPU) |
| `HANDOFF_MIC_DEVICE` | auto (USB) | Device number or part of its name (`PnP`). Names survive reboots; numbers may not. |
| `HANDOFF_CAPTIONS` | `auto` | `vosk`, `whisper`, or `off` |
| `HANDOFF_STOP_PHRASE` | `confirm report ready` | Hands-free finish phrase |
| `HANDOFF_MAX_RECORDING_S` | `600` | Auto-finish (and process) after this long |
| `HANDOFF_LLM_TIMEOUT_S` | `180` | Per-request timeout for the AI |

---

## Testing

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest                 # 77 tests: unit, audio engine, server, UI (headless Chromium); no hardware needed
.venv/bin/python -m pytest -m jetson -s    # on the device: real mic, caption speed, AI provider, end-to-end report
.venv/bin/python scripts/mic_test.py       # talk and watch live captions + stop phrase in the terminal
.venv/bin/python -m ems_handoff --check    # setup report: packages, mic, models, AI provider
.venv/bin/python -m ems_handoff.cli --case stemi   # headless report from a sample (also --text-file, --wav)
```

The default test run uses fakes for the mic, recognizers, transcriber and AI, so it exercises the real
threads, state machine, API and screens anywhere (UI screenshots land in `tests/artifacts/`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No microphone found" | Plug in the USB mic; `python -m sounddevice` lists devices; set `HANDOFF_MIC_DEVICE` to part of its name. |
| Level bar does not move | Mic muted or gain low: `alsamixer -c 2` (card number from `arecord -l`). |
| Captions say "Whisper tiny.en (fallback)" | Run `python scripts/download_models.py --vosk`. |
| "groq rejected the API key" | Create a new key (console.groq.com/keys, org *Personal*) and run `bash scripts/set_groq_key.sh`. |
| "cannot reach Ollama" / "model not pulled" | `ollama serve` / `ollama pull qwen3:8b`. A crash-looping Docker `ollama` container fighting for port 11434 can be stopped with `docker update --restart no ollama && docker stop ollama`. |
| "Ollama ran out of memory loading ..." | The Jetson shares its 8 GB between CPU and GPU. `qwen3:8b` needs about 7.3 GB free, `qwen3:4b` about 3.9 GB, `qwen3:1.7b` about 2.6 GB. Pull a smaller model (`ollama pull qwen3:4b`): the app switches to the largest pulled model that fits. Close browsers and the Docker `ollama` container to free memory; `free -h` shows what is available. On an 8 GB Orin Nano with the desktop running, `qwen3:4b` is the practical ceiling and needs swap to be reliable. |
| Local model is killed or times out although it "fits" | Add swap so the desktop can be paged out while the model runs: `sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`, then `echo '/swapfile none swap sw 0 0' \| sudo tee -a /etc/fstab`. Also set Ollama to use less memory per model: `sudo systemctl edit ollama` and add `[Service]` / `Environment=OLLAMA_FLASH_ATTENTION=1` / `Environment=OLLAMA_KV_CACHE_TYPE=q8_0` / `Environment=OLLAMA_CONTEXT_LENGTH=4096`, then `sudo systemctl restart ollama`. |
| Slow final transcript with no Groq key | Use Groq, or set `LOCAL_WHISPER_MODEL=small.en`. |
| Logs | `output/handoff.log` |

---

## Project layout

```
ems_handoff/
  __main__.py      app launcher (server + app window)        doctor.py   setup check
  config.py        all settings                              cli.py      headless command line
  server.py        local API + state machine + report worker
  web/             app screens: index.html, app.css, app.js (no internet needed)
  audio/           capture.py (mic), listeners.py (captions, stop phrase), session.py (one recording)
  transcribe.py    final transcript (Groq / local Whisper)
  pipeline.py      transcript → validated report             llm.py      Groq / Ollama / OpenRouter client
  phrase.py        stop-phrase matching                      prompt.py   LLM prompts
  transcript_corrector.py, lexicon.py, lexicon_rag.py        EMS language helpers
  textbook_rag.py, vector_store.py, embed_documents.py       optional textbook guidance
scripts/           install_jetson.sh, download_models.py, mic_test.py
tests/             pytest suite (+ test_jetson.py for the real device)
legacy/            the original single-file prototype, kept for reference
```

Every source file starts with a header: purpose, inputs, outputs, and threads.

## Textbook guidance (optional)

Put EMS reference PDFs in `dataset/`, then `ollama pull nomic-embed-text` and
`python -m ems_handoff.embed_documents`. Reports then include cited, textbook-grounded suggestions.

## Privacy

`output/` holds recordings, transcripts and reports. That is fine for demos with scripted reports;
real patient data needs encryption at rest and a retention policy before any clinical use.

## What changed from v1 (the intern prototype)

Terminal menus → app window · live captions that keep up (streaming) · DONE button that always works ·
dedicated stop-phrase listener · Groq Whisper for the final transcript with local fallback · bounded
timeouts and clear errors · settings in one place (no hard-coded mic 24 / model names) · large transcript
panel · 77 automated tests · leaked Groq key removed from the code (it was revoked by Groq on 2026-07-23).
