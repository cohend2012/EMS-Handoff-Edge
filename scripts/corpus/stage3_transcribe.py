#!/usr/bin/env python3
"""Stage 3a: transcribe every kept WAV once with Groq Whisper (verbose_json, word +
segment timestamps). Resumable: a WAV whose ~/transcripts/<stem>.json exists is skipped.
Rate limits: honor retry-after, else exponential backoff from 10 s; never drop a file.
If Groq keeps blocking for more than 10 min, the remaining files go to local faster-whisper.
Each transcript JSON records which transcriber produced it."""
import csv, io, json, re, sys, time, wave
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home()/"EMS-Handoff-app"))
from ems_handoff.config import load_env_files, load_settings
from ems_handoff.lexicon_rag import whisper_hint
import openai
from openai import OpenAI

HOME = Path.home(); WAVS = HOME/"calls_wav"; OUT = HOME/"transcripts"; LABELS = HOME/"labels"
OUT.mkdir(exist_ok=True)
LOCAL_MODEL = "small"          # cached; large-v3-turbo on the Jetson CPU is too slow for a fallback
GIVE_UP_AFTER_S = 600          # continuous blocking before switching to local
load_env_files(); settings = load_settings()
if not settings.groq_api_key: sys.exit("no GROQ_API_KEY in ~/.config/handoff-ai/secrets.env")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key, timeout=120, max_retries=0)
HINT = whisper_hint()[-800:]

rows = [r for r in csv.DictReader(open(LABELS/"inventory.csv")) if r["status"] == "kept"]
rows.sort(key=lambda r: int(r["download_index"]))
todo = [r for r in rows if not (OUT/(Path(r["filename"]).stem+".json")).exists()]
mins = sum(float(r["duration_s"]) for r in todo)/60
est = max(len(todo)*2.5 + mins*2.0, len(todo)/20*60)   # per-request overhead + ~2 s per audio minute; free tier 20 req/min
print(f"files to transcribe: {len(todo)} of {len(rows)} kept ({mins:.1f} min of audio, hourly cap 120 min). "
      f"Estimated batch time with Groq: ~{est/60:.1f} min if not rate-limited.", flush=True)

def parse_retry(e):
    h = getattr(getattr(e, "response", None), "headers", None) or {}
    ra = h.get("retry-after")
    if ra:
        try: return float(ra)
        except ValueError: pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
    if m: return int(m.group(1) or 0)*60 + float(m.group(2))
    return None

def groq_transcribe(wav_path):
    raw = client.audio.transcriptions.with_raw_response.create(
        file=(wav_path.name, wav_path.read_bytes()), model=settings.groq_whisper_model,
        prompt=HINT, language="en", temperature=0.0, response_format="verbose_json",
        timestamp_granularities=["word", "segment"])
    j = json.loads(raw.text)
    return {"transcriber": f"groq {settings.groq_whisper_model}", "text": (j.get("text") or "").strip(),
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                          "no_speech_prob": s.get("no_speech_prob"), "avg_logprob": s.get("avg_logprob")} for s in j.get("segments") or []],
            "words": [{"start": w["start"], "end": w["end"], "word": w["word"]} for w in j.get("words") or []]}

_local = None
def local_transcribe(wav_path):
    global _local
    from faster_whisper import WhisperModel
    if _local is None: _local = WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")
    with wave.open(str(wav_path)) as w: audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)/32768
    segs, _ = _local.transcribe(audio, language="en", beam_size=1, temperature=0.0, initial_prompt=HINT, word_timestamps=True, vad_filter=False)
    segments, words = [], []
    for s in segs:
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip(), "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob})
        words += [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
    return {"transcriber": f"local faster-whisper {LOCAL_MODEL}", "text": " ".join(s["text"] for s in segments), "segments": segments, "words": words}

use_local = False; blocked = 0.0; backoff = 10.0; t_batch = time.time()
for r in todo:
    wav = WAVS/(Path(r["filename"]).stem+".wav"); out = OUT/(wav.stem+".json")
    while True:
        t0 = time.time()
        try:
            res = local_transcribe(wav) if use_local else groq_transcribe(wav)
            blocked = 0.0; backoff = 10.0
            break
        except openai.RateLimitError as e:
            wait = parse_retry(e) or backoff; backoff = min(backoff*2, 300)
            print(f"  429 rate limit on {wav.name}: waiting {wait:.0f}s ({e})", flush=True)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.APIStatusError) as e:
            wait = parse_retry(e) or backoff; backoff = min(backoff*2, 300)
            print(f"  Groq error on {wav.name}: waiting {wait:.0f}s ({type(e).__name__}: {e})", flush=True)
        blocked += wait
        if blocked > GIVE_UP_AFTER_S:
            print(f"  Groq blocked for {blocked:.0f}s > {GIVE_UP_AFTER_S}s: switching to local faster-whisper {LOCAL_MODEL} for the rest", flush=True)
            use_local = True; continue
        time.sleep(wait)
    res.update({"source_file": r["filename"], "wav": wav.name, "download_index": int(r["download_index"]),
                "file_timestamp": r["download_time"], "duration_s": float(r["duration_s"]), "took_s": round(time.time()-t0, 2),
                "whisper_prompt": HINT})
    out.write_text(json.dumps(res, indent=1))
    print(f"[{r['download_index']:>2}] {wav.name:<36} {res['duration_s']:>6.1f}s audio  {res['took_s']:>5.1f}s  {res['transcriber']:<30} {len(res['words'])} words", flush=True)
print(f"done: {len(todo)} files in {(time.time()-t_batch)/60:.1f} min; transcripts in {OUT}")
