#!/usr/bin/env python3
"""Stage 2: order the ~/calls recordings by download time, find duplicates
(SHA256 + decoded-audio fingerprint), copy duplicates to ~/duplicates/,
convert kept files to 16 kHz mono 16-bit PCM WAV in ~/calls_wav/.
Never writes inside ~/calls."""
import csv, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home()
CALLS, DUPS, WAVS, LABELS = HOME/"calls", HOME/"duplicates", HOME/"calls_wav", HOME/"labels"
for d in (DUPS, WAVS, LABELS): d.mkdir(exist_ok=True)
SR = 16000

def ffprobe(p):
    out = subprocess.run(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(p)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out); f = j["format"]; s = j["streams"][0]
    return {"duration": float(f.get("duration") or s.get("duration") or 0),
            "codec": s.get("codec_name"), "rate": s.get("sample_rate"), "ch": s.get("channels"),
            "tag_time": (f.get("tags") or {}).get("creation_time")}

def decode(p):
    raw = subprocess.run(["ffmpeg","-v","error","-i",str(p),"-f","f32le","-ac","1","-ar",str(SR),"-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def sha(b): return hashlib.sha256(b).hexdigest()

def envelope(x, frame=320):  # 20 ms RMS frames
    n = len(x)//frame
    e = np.sqrt((x[:n*frame].reshape(n, frame)**2).mean(1) + 1e-12)
    return np.log(e)

def corr(a, b, max_lag=50):  # lag search ±1 s over 20 ms frames
    best = -1.0
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0: x, y = a[lag:], b[:len(b)-lag] if lag else b
        else:        x, y = a[:len(a)+lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 25: continue
        x, y = x[:n]-x[:n].mean(), y[:n]-y[:n].mean()
        d = np.sqrt((x*x).sum()*(y*y).sum())
        if d > 0: best = max(best, float((x*y).sum()/d))
    return best

def birth(p):
    try:
        v = int(subprocess.run(["stat","-c","%W",str(p)],capture_output=True,text=True).stdout.strip())
        return v if v > 0 else None
    except Exception: return None

files = sorted(CALLS.iterdir())
rows = []
for p in files:
    if not p.is_file(): continue
    st = p.stat(); info = ffprobe(p); data = p.read_bytes(); audio = decode(p)
    m = re.search(r"-(\d{10})(?: \(\d+\))?\.m4a$", p.name)
    rows.append({"filename": p.name, "size": st.st_size, "mtime": st.st_mtime, "birth": birth(p),
                 "name_epoch": int(m.group(1)) if m else None, "tag_time": info["tag_time"],
                 "duration": round(len(audio)/SR, 2), "probe_duration": round(info["duration"],2),
                 "codec": f'{info["codec"]} {info["rate"]}Hz {info["ch"]}ch',
                 "sha256": sha(data), "pcm_sha256": sha(audio.tobytes()), "_env": envelope(audio)})

rows.sort(key=lambda r: (r["mtime"], r["filename"]))
for i, r in enumerate(rows, 1): r["download_index"] = i

# --- duplicates: exact hash first, then audio match
kept_of = {}   # dup filename -> (kept filename, match_type, reason)
by_hash = {}
for r in rows:
    h = r["sha256"]
    if h in by_hash: kept_of[r["filename"]] = (by_hash[h], "exact_hash", "identical bytes (SHA256)")
    else: by_hash[h] = r["filename"]
kept_rows = [r for r in rows if r["filename"] not in kept_of]
for i, a in enumerate(kept_rows):
    if a["filename"] in kept_of: continue
    for b in kept_rows[i+1:]:
        if b["filename"] in kept_of: continue
        dd = abs(a["duration"] - b["duration"])
        if dd > 1.0: continue
        if a["pcm_sha256"] == b["pcm_sha256"]:
            kept_of[b["filename"]] = (a["filename"], "audio_match", "identical decoded PCM, different container bytes"); continue
        c = corr(a["_env"], b["_env"])
        if c >= 0.97:
            kept_of[b["filename"]] = (a["filename"], "audio_match", f"duration diff {dd:.2f}s, envelope correlation {c:.3f}")
kept = [r for r in rows if r["filename"] not in kept_of]

# --- outputs
with open(LABELS/"inventory.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["download_index","filename","download_time","birth_time","filename_epoch_utc","duration_s","size_bytes","codec","sha256","status","kept_file"])
    for r in rows:
        k = kept_of.get(r["filename"])
        w.writerow([r["download_index"], r["filename"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["mtime"])),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["birth"])) if r["birth"] else "",
                    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["name_epoch"])) if r["name_epoch"] else "",
                    r["duration"], r["size"], r["codec"], r["sha256"], "duplicate" if k else "kept", k[0] if k else ""])
with open(LABELS/"duplicates.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["kept_file","duplicate_file","match_type","reason"])
    for r in rows:
        k = kept_of.get(r["filename"])
        if k:
            w.writerow([k[0], r["filename"], k[1], k[2]])
            shutil.copy2(CALLS/r["filename"], DUPS/r["filename"])
for r in kept:
    out = WAVS/(Path(r["filename"]).stem + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg","-v","error","-y","-i",str(CALLS/r["filename"]),"-ac","1","-ar",str(SR),"-c:a","pcm_s16le",str(out)], check=True)

print(f"{'idx':>3} {'filename':<40} {'download(mtime)':<19} {'rec time (name,UTC)':<19} {'dur s':>7} {'bytes':>7} status")
for r in rows:
    k = kept_of.get(r["filename"])
    print(f"{r['download_index']:>3} {r['filename']:<40} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['mtime'])):<19} "
          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(r['name_epoch'])):<19} {r['duration']:>7.2f} {r['size']:>7} {'DUP of '+k[0] if k else 'kept'}")
tot = sum(r["duration"] for r in rows); tk = sum(r["duration"] for r in kept)
print(f"\nfiles {len(rows)}, duplicates {len(kept_of)}, kept {len(kept)}, wavs {len(list(WAVS.glob('*.wav')))}")
print(f"audio: all files {tot/60:.1f} min, kept files {tk/60:.1f} min")
print("birth times present:", sum(1 for r in rows if r['birth']), "; mtime==birth for", sum(1 for r in rows if r['birth'] and abs(r['birth']-r['mtime'])<2))
print("container tag creation_time present:", sum(1 for r in rows if r['tag_time']))
print("codecs:", sorted({r['codec'] for r in rows}))
print("\nduplicates.csv:"); print(open(LABELS/"duplicates.csv").read())
