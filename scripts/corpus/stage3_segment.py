#!/usr/bin/env python3
"""Stage 3b: find consultation boundaries inside each recording and cut segments.
Signals (scored at every Whisper segment start):
  silence gap before it (>=2 s weak, >=4 s strong), a steady tone in that gap (squelch /
  dispatch tone), an opening phrase right after a closing phrase, and a change of unit id,
  patient age or sex versus what this file already said. Score >= 0.6 -> new segment.
Outputs ~/labels/segments.csv, ~/labels/segment_signals.txt, ~/segments/<id>.wav + .txt."""
import csv, json, re, wave
from pathlib import Path
import numpy as np

HOME = Path.home(); WAVS = HOME/"calls_wav"; TR = HOME/"transcripts"; SEG = HOME/"segments"; LABELS = HOME/"labels"
SEG.mkdir(exist_ok=True); SR = 16000; SPLIT_AT = 0.6
OVERRIDES = json.load(open(LABELS/"segment_overrides.json")) if (LABELS/"segment_overrides.json").exists() else {}

OPENING = re.compile(r"^\W*(?:(?:hi|hello|hey|good (?:morning|evening|afternoon))\b|this is (?:medic|ambulance|unit|paramedic|dr|doctor|nurse)|(?:medic|ambulance)s? ?\d{3}|.{0,40}\b(?:online|on the line|go ahead|do you need a physician)\b)", re.I)
CLOSING = re.compile(r"\b(?:copy|copies|see you (?:on|in|at) arrival|see you (?:in|at) \w+|clear(?:ing)? (?:of )?the (?:line|channel)|is clear|any (?:other |further )?questions|thank you|thanks|bye)\b", re.I)
UNIT = re.compile(r"\b(?:medic|medics|medix|ambulance|unit|paramedic)s?\s*(\d{3})\b", re.I)
AGE = re.compile(r"\b(\d{1,3})[- ]?(?:year|month|week)s?[- ]old\b", re.I)
SEX = re.compile(r"\b(male|female)\b", re.I)

def rms_frames(x, n=320):
    k = len(x)//n; return 20*np.log10(np.sqrt((x[:k*n].reshape(k, n)**2).mean(1)) + 1e-9)

def silence_runs(db, floor_db, min_s=1.5, n=320):
    thr = max(floor_db + 6, -50); runs = []; start = None
    for i, v in enumerate(np.append(db, 0)):
        if v < thr and start is None: start = i
        elif v >= thr and start is not None:
            if (i-start)*n/SR >= min_s: runs.append((start*n/SR, i*n/SR))
            start = None
    return runs

def tones(x, frame=800, hop=400, min_frames=6):
    """steady narrow-band tone 300-2500 Hz: one spectral peak >= 25x the mean, same bin for >= 300 ms.
    (A looser version fired on voiced speech at every segment edge; see segment_signals.txt history.)"""
    events, run, last_bin = [], 0, None
    for i in range(0, len(x)-frame, hop):
        seg = x[i:i+frame]*np.hanning(frame); sp = np.abs(np.fft.rfft(seg))[5:]
        if sp.mean() <= 0: continue
        b = int(sp.argmax()); f = (b+5)*SR/frame; pure = 300 <= f <= 2500 and sp[b]/sp.mean() > 25 and 20*np.log10(sp[b]+1e-9) > -20
        if pure and (last_bin is None or abs(b-last_bin) <= 1): run += 1; last_bin = b
        else:
            if run >= min_frames: events.append(((i-run*hop)/SR, i/SR, round((last_bin+5)*SR/frame)))
            run, last_bin = (1, b) if pure else (0, None)
    return events

def in_gap(events, a, b): return [e for e in events if e[0] >= a-0.3 and e[1] <= b+0.3]

transcripts = sorted((json.load(open(p)) for p in TR.glob("*.json")), key=lambda j: j["download_index"])
rows, log = [], []
for j in transcripts:
    wav = WAVS/j["wav"]
    with wave.open(str(wav)) as w: pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    x = pcm.astype(np.float32)/32768; dur = len(x)/SR
    db = rms_frames(x); floor = float(np.percentile(db, 10))
    sil = silence_runs(db, floor); tn = tones(x)
    segs = j["segments"]; words = j["words"]
    log.append(f"\n[{j['download_index']}] {j['source_file']} dur={dur:.1f}s noise_floor={floor:.0f}dB silences>=1.5s={[(round(a,1),round(b,1)) for a,b in sil]} tones={[(round(a,1),round(b,1),f) for a,b,f in tn]}")
    seen_units, seen_ages, seen_sex = set(), set(), set(); cuts = []; conf_reason = []
    for i, s in enumerate(segs):
        text = s["text"]; score = 0.0; why = []
        gap = s["start"] - (segs[i-1]["end"] if i else 0.0)
        if i:
            if gap >= 4: score += 0.3; why.append(f"gap {gap:.1f}s")
            elif gap >= 2: score += 0.15; why.append(f"gap {gap:.1f}s")
            t = in_gap(tn, segs[i-1]["end"], s["start"])
            if t: score += 0.15; why.append(f"tone {t[0][2]}Hz")
            if OPENING.search(text) and CLOSING.search(segs[i-1]["text"]): score += 0.35; why.append("closing then opening")
        u = {m.group(1) for m in UNIT.finditer(text)}; a = {m.group(1) for m in AGE.finditer(text)}; sx = {m.group(1).lower() for m in SEX.finditer(text)}
        if i and seen_units and u and not (u & seen_units): score += 0.5; why.append(f"unit {sorted(u)} vs {sorted(seen_units)}")
        if i and seen_ages and a and not (a & seen_ages): score += 0.5; why.append(f"age {sorted(a)} vs {sorted(seen_ages)}")
        if i and seen_sex and sx and not (sx & seen_sex): score += 0.25; why.append(f"sex {sorted(sx)} vs {sorted(seen_sex)}")
        seen_units |= u; seen_ages |= a; seen_sex |= sx
        if i and score > 0: log.append(f"   cand @{s['start']:6.1f}s score={score:.2f} [{', '.join(why)}] -> {'SPLIT' if score >= SPLIT_AT else 'keep'} | {text[:70]}")
        if i and score >= SPLIT_AT: cuts.append((s["start"], ", ".join(why), score))
    ov = OVERRIDES.get(str(j["download_index"])); auto_cuts = cuts
    if ov is not None:
        cuts = [(c, "reviewed override", 0.8) for c in ov["cuts"]]
        log.append(f"   OVERRIDE: auto cuts {[round(c[0],1) for c in auto_cuts]} -> {ov['cuts']}: {ov['note']}")
    bounds = [0.0] + [c[0] for c in cuts] + [dur]
    for k in range(len(bounds)-1):
        a, b = bounds[k], bounds[k+1]
        # cut in the middle of the preceding silence when splitting
        if k: a = (segs_prev_end + a)/2 if (segs_prev_end := max(s["end"] for s in segs if s["end"] <= a)) else a
        sid = f"S{j['download_index']:03d}_{k+1:02d}"
        wtxt = " ".join(w["word"].strip() for w in words if w["start"] >= a-0.05 and w["start"] < b) or " ".join(s["text"] for s in segs if s["start"] >= a-0.05 and s["start"] < b)
        if not cuts:
            conf, reason = 0.9, "single consultation: no split signal reached threshold"
            if ov is not None: conf, reason = 0.85, f"single consultation; {ov['note']}"
        else: conf, reason = (0.6 if k else 0.7), (f"split at {bounds[k]:.1f}s: {cuts[k-1][1]}" if k else f"ends at split {bounds[1]:.1f}s: {cuts[0][1]}")
        with wave.open(str(SEG/f"{sid}.wav"), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR); out.writeframes(pcm[int(a*SR):int(b*SR)].tobytes())
        (SEG/f"{sid}.txt").write_text(wtxt.strip()+"\n")
        rows.append({"download_index": j["download_index"], "source_file": j["source_file"], "file_timestamp": j["file_timestamp"],
                     "segment_id": sid, "start_sec": round(a, 2), "end_sec": round(b, 2), "confidence": conf, "reason": reason,
                     "transcriber": j["transcriber"], "words": len(wtxt.split()), "text": wtxt.strip()})
with open(LABELS/"segments.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["download_index","source_file","file_timestamp","segment_id","start_sec","end_sec","confidence","reason","transcriber"], extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
for old in SEG.glob('S*'):
    if old.stem not in {r['segment_id'] for r in rows}: old.unlink()
(LABELS/"segment_signals.txt").write_text("\n".join(log)+"\n")
print(f"{'segment':<9} {'idx':>3} {'source_file':<36} {'start':>6} {'end':>6} conf  words  transcriber")
for r in rows: print(f"{r['segment_id']:<9} {r['download_index']:>3} {r['source_file']:<36} {r['start_sec']:>6.1f} {r['end_sec']:>6.1f} {r['confidence']:.2f} {r['words']:>5}  {r['transcriber']}")
print(f"\n{len(rows)} segments from {len(transcripts)} recordings; files split: {len({r['download_index'] for r in rows if r['segment_id'].endswith('_02')})}")
print("\ncandidates that scored (from segment_signals.txt):"); print("\n".join(l for l in log if "cand @" in l))
