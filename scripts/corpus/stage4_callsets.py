#!/usr/bin/env python3
"""Stage 4: group segments into call-sets. Per feed (4390/4395/4430), walk bursts in
ascending epoch (recording time from the filename). Each burst is compared ONLY with the
open call-set on its own feed: same_patient or new_patient, biased toward new.
Evidence: unit id, patient age/sex, chief-complaint keywords, destination hospital, ETA,
continuation cues (copy/go ahead/repeat/closings), opening cues, and the time gap
(>5 min = strong new-patient signal). Cap: CAP bursts per set (flagged when hit).
Cross-feed: bursts never join across feeds automatically. A reviewed entry in
callset_relations.json with "join": true places that burst into the set holding the
related segment (a patched channel carrying the other side of the same consult), in
time order. Set IDs are stable across re-runs: a set keeps the ID its earliest burst
had in the previous callsets.csv; an ID whose set was absorbed is retired, never reused,
and its folder moves to callsets/_retired/. Report copies already in a set folder are kept."""
import csv, json, re, shutil, time
from pathlib import Path

HOME = Path.home(); LABELS = HOME/"labels"; SEG = HOME/"segments"; OUT = HOME/"callsets"
OVER_CAP = 6; STRONG_GAP = 300; FRAGMENT_WORDS = 15   # cap is a flag (>6 bursts), not a limit
RELATIONS = json.load(open(LABELS/"callset_relations.json")) if (LABELS/"callset_relations.json").exists() else {}
OUT.mkdir(exist_ok=True)

UNIT = re.compile(r"\b(?:medic|medics|medix|ambulance|unit|paramedic|m)s?\s*(7\d\d)\b", re.I)
BARE_UNIT = re.compile(r"(?<![\d.])(7[0-5]\d)[a-z]?\b(?!\s*(?:over|percent|%|micro|mg|milli))", re.I)
WORDNUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
AGE = re.compile(r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[- ]?(year|month|week)s?[- ]old\b|\bage (?:about |of |approximately )?(\d{1,3})\b", re.I)
SEX = re.compile(r"\b(male|female)\b", re.I)
ETA = re.compile(r"\b(\d{1,2})[- ]?(?:minute|min)s?\b", re.I)
HOSPITALS = {"holy cross germantown": ["holy cross germantown", "holycroft", "germantown"], "holy cross": ["holy cross"],
             "shady grove": ["shady grove", "shagos", "shady grubble"], "suburban": ["suburban"]}
COMPLAINT = {"stroke": ["stroke", "lam score", "lams", "cincinnati", "last known well", "last-known well", "facial droop", "slurred", "arm drift", "vision"],
             "chest pain": ["chest pain", "aspirin", "nitro", "12 lead", "12-lead", "stemi"],
             "arrhythmia": ["afib", "a-fib", "rvr", "svt", "adenosine", "palpitation"],
             "seizure": ["seizure", "versed", "febrile"],
             "respiratory": ["respiratory distress", "trouble breathing", "shortness of breath", "wheez", "albuterol", "duoneb", "saliva", "drool"],
             "trauma": ["trauma", "fell", "fall", "laceration", "motorcycle", "scooter", "struck", "tackled", "head injury", "backboard", "c-collar", "collar"],
             "overdose": ["overdose", "narcan", "opioid", "pinpoint"],
             "altered/glucose": ["dementia", "hyperglycemia", "hypoglycemia", "diabetes", "diabetic", "altered"],
             "refusal": ["refus", "risks", "transport the patient"]}
CONT_CUES = re.compile(r"^\W*(?:(?:i don'?t |we |yeah,? i )?cop(?:y|ied|ies)|go ahead|(?:please )?continue|can you (?:say again|repeat)|say again|repeat|yes|no\b|correct|okay|all right|alright|thank you|sorry|so,? (?:your|just)|that is a possibility|right\.)", re.I)
CLOSING = re.compile(r"\b(?:see you (?:on|in|at) arrival|clear(?:ing)? (?:of )?the (?:line|channel)|is clear|\d+ clear|call (?:us )?(?:with|if)|thank you for your time)\b", re.I)
CALL_OPEN = re.compile(r"\b(?:online|on the line|go ahead|do you need a physician|this is (?:medic|ambulance|medix)|good (?:morning|evening|afternoon)|(?:medic|ambulance|medics)s? ?\d{3} (?:awaiting|standing|calling|is on))\b", re.I)
CALL_CLOSE = re.compile(r"\b(?:see you (?:on|in|at) arrival|see you (?:in|at) (?:about )?\w+ minutes?|clear|thank you|thanks|bye|call (?:us )?(?:with|if))\b", re.I)
OPENING = re.compile(r"^\W*(?:.{0,40}\b(?:online|on the line|do you need a physician)\b|(?:hi|hello|good (?:morning|evening|afternoon))\b.{0,30}\b(?:this is|suburban|shady|holy)|this is (?:medic|ambulance|dr|doctor|nurse))", re.I)
EXPLICIT_REF = re.compile(r"\b(?:update on|the patient we called|same (?:patient|event)|called (?:you )?(?:about|earlier))\b", re.I)

def features(text):
    t = text.lower()
    f = {"units": {m.group(1) for m in UNIT.finditer(text)}, "bare_units": {m.group(1) for m in BARE_UNIT.finditer(text)},
         "ages": {WORDNUM.get((m.group(1) or m.group(3)).lower(), (m.group(1) or m.group(3))) + ("" if not m.group(2) else m.group(2)[0].lower()) for m in AGE.finditer(text)},
         "sex": {m.group(1).lower() for m in SEX.finditer(text)},
         "hosp": {h for h, al in HOSPITALS.items() if any(a in t for a in al)},
         "complaints": {c for c, kws in COMPLAINT.items() if any(k in t for k in kws)},
         "eta": {m.group(1) for m in ETA.finditer(text)},
         "cont": bool(CONT_CUES.search(text)), "closing_end": bool(CLOSING.search(text[-120:])),
         "opening": bool(OPENING.search(text)), "explicit": bool(EXPLICIT_REF.search(text))}
    if "holy cross germantown" in f["hosp"]: f["hosp"].discard("holy cross")
    return f

def merge(acc, f):
    for k in ("units", "bare_units", "ages", "sex", "hosp", "complaints", "eta"): acc.setdefault(k, set()).update(f[k])
    acc["closing_end"] = f["closing_end"]; return acc

def hosp_compatible(a, b):
    return any(x == y or x.startswith(y) or y.startswith(x) for x in a for y in b)

def decide(acc, f, gap, n_in_set):
    return _decide(acc, f, gap)

def _decide(acc, f, gap):
    ev, conflicts, positives = [], [], []
    if f["units"] & acc["units"]: positives.append(f"unit {sorted(f['units'] & acc['units'])} match")
    elif f["units"] and acc["units"]: conflicts.append(f"unit {sorted(f['units'])} vs {sorted(acc['units'])}")
    elif f["bare_units"] & (acc["units"] | acc["bare_units"]): positives.append(f"unit {sorted(f['bare_units'] & (acc['units'] | acc['bare_units']))} match (bare number)")
    if f["ages"] & acc["ages"]: positives.append(f"age {sorted(f['ages'] & acc['ages'])} match")
    elif f["ages"] and acc["ages"]: conflicts.append(f"age {sorted(f['ages'])} vs {sorted(acc['ages'])}")
    if f["sex"] and acc["sex"]:
        if f["sex"] & acc["sex"]: ev.append(f"sex {sorted(f['sex'] & acc['sex'])} consistent")
        else: conflicts.append(f"sex {sorted(f['sex'])} vs {sorted(acc['sex'])}")
    if f["hosp"] and acc["hosp"]:
        if hosp_compatible(f["hosp"], acc["hosp"]): positives.append(f"hospital {sorted(f['hosp'])} match")
        else: conflicts.append(f"hospital {sorted(f['hosp'])} vs {sorted(acc['hosp'])}")
    if f["complaints"] & acc["complaints"]: positives.append(f"complaint {sorted(f['complaints'] & acc['complaints'])} match")
    if f["eta"] & acc["eta"]: positives.append(f"ETA {sorted(f['eta'] & acc['eta'])} min repeated")
    if f["explicit"]: positives.append("explicit reference to prior call")
    if f["cont"]: ev.append("continuation cue at start")
    if f["opening"]: ev.append("opening phrase at start")
    if acc.get("closing_end"): ev.append("previous burst closed the call")
    gap_s = f"gap {gap:.0f}s"
    if gap > STRONG_GAP:
        if f["explicit"] and not conflicts: return "same_patient", 0.6, f"{gap_s} > 5 min BUT explicit reference; " + "; ".join(positives+ev), True
        return "new_patient", 0.95, f"{gap_s} > 5 min" + ("; " + "; ".join(conflicts) if conflicts else ""), False
    if conflicts: return "new_patient", 0.9, f"{gap_s}; " + "; ".join(conflicts + positives + ev), False
    strong = any(p.startswith(("unit", "age", "hospital", "explicit")) for p in positives)
    if strong or len(positives) >= 2: return "same_patient", 0.9, f"{gap_s}; " + "; ".join(positives + ev), False
    if positives: return "same_patient", 0.8, f"{gap_s}; " + "; ".join(positives + ev), False
    if acc.get("closing_end") and not f["cont"]: return "new_patient", 0.7, f"{gap_s}; previous burst closed the call; no identifiers link this burst", False
    if f["opening"]: return "new_patient", 0.7, f"{gap_s}; opening phrase at start; no identifiers link this burst", False
    if f["cont"]: return "same_patient", 0.75, f"{gap_s}; continuation cue at start; no conflicting identifiers", False
    if gap <= 30: return "same_patient", 0.7, f"{gap_s} (<=30 s); no identifiers either way, no conflict", False
    if gap <= 120: return "same_patient", 0.6, f"{gap_s} (30-120 s); no identifiers either way; weak", True
    return "new_patient", 0.6, f"{gap_s} (2-5 min); no identifiers link this burst; weak", True

segs = list(csv.DictReader(open(LABELS/"segments.csv")))
JOINS = {sid: rel for sid, rel in RELATIONS.items() if rel.get("join")}
for s in segs:
    m = re.search(r"mocomdps-(\d+)-(\d{10})", s["source_file"]); s["feed"], s["epoch"] = m.group(1), int(m.group(2))
    s["dur"] = float(s["end_sec"]) - float(s["start_sec"]); s["text"] = (SEG/f"{s['segment_id']}.txt").read_text().strip()
    s["f"] = features(s["text"])
sets = []  # each: {"feed", "members": [(seg, decision, conf, evidence, gap, flag)]}
for feed in sorted({s["feed"] for s in segs}):
    stream = sorted((s for s in segs if s["feed"] == feed and s["segment_id"] not in JOINS), key=lambda s: (s["epoch"], s["segment_id"]))
    cur = None; prev_end = None
    for s in stream:
        gap = (s["epoch"] - prev_end) if prev_end is not None else None
        if cur is None:
            cur = {"feed": feed, "acc": merge({}, s["f"]), "members": [(s, "new_patient", 1.0, "first burst on feed " + feed, gap, False)]}; sets.append(cur)
        else:
            d, c, ev, flag = decide(cur["acc"], s["f"], gap, len(cur["members"]))
            if d == "same_patient":
                cur["acc"] = merge(cur["acc"], s["f"]); cur["members"].append((s, d, c, ev, gap, flag or c < 0.7))
            else:
                cur = {"feed": feed, "acc": merge({}, s["f"]), "members": [(s, d, c, ev, gap, flag or c < 0.7)]}; sets.append(cur)
        prev_end = s["epoch"] + s["dur"]
# reviewed cross-feed joins: the burst goes into the set that holds its related segment, in time order
seg2set = {m[0]["segment_id"]: cs for cs in sets for m in cs["members"]}
for sid, rel in JOINS.items():
    s = next(x for x in segs if x["segment_id"] == sid); cs = seg2set[rel["related_to_segment"]]
    cs["members"].append((s, "same_patient", rel.get("confidence", 0.85), "joined across feeds on review: " + rel["evidence"], None, False))
    cs["members"].sort(key=lambda m: (m[0]["epoch"], m[0]["segment_id"])); cs["acc"] = merge(cs["acc"], s["f"])
    prev_end = None; fixed = []
    for (m, dec, conf, ev, gap, flag) in cs["members"]:   # gaps follow the new order
        fixed.append((m, dec, conf, ev, None if prev_end is None else m["epoch"] - prev_end, flag)); prev_end = m["epoch"] + m["dur"]
    cs["members"] = fixed; seg2set[sid] = cs
sets.sort(key=lambda cs: cs["members"][0][0]["epoch"])
# stable IDs: keep the ID the set's earliest burst had last time; new sets get fresh numbers; absorbed IDs retire
prev_id = {r["segment_id"]: r["callset_id"] for r in csv.DictReader(open(LABELS/"callsets.csv"))} if (LABELS/"callsets.csv").exists() else {}
used, retired = set(), []
for cs in sets:
    cs["id"] = next((prev_id[m[0]["segment_id"]] for m in cs["members"] if m[0]["segment_id"] in prev_id and prev_id[m[0]["segment_id"]] not in used), None)
    if cs["id"]: used.add(cs["id"])
n = 0
for cs in sets:
    if cs["id"] is None:
        while True:
            n += 1
            if f"CS{n:03d}" not in used and f"CS{n:03d}" not in prev_id.values(): break
        cs["id"] = f"CS{n:03d}"; used.add(cs["id"])
for old in sorted(set(prev_id.values()) - used):
    retired.append(old); (OUT/"_retired").mkdir(exist_ok=True)
    if (OUT/old).exists(): shutil.move(str(OUT/old), str(OUT/"_retired"/old))
sets.sort(key=lambda cs: cs["id"])
rows = []
for cs in sets:
    cid = cs["id"]; d = OUT/cid
    d.mkdir(exist_ok=True)
    for stale in list(d.glob("S*.wav")) + list(d.glob("S*.txt")) + list(d.glob("combined_transcript*.txt")): stale.unlink()
    combined, annotated = [], []; over_cap = len(cs["members"]) > OVER_CAP
    cs["id"] = cid
    for k, (s, dec, conf, ev, gap, flag) in enumerate(cs["members"], 1):
        flags = []
        if flag: flags.append("low_confidence")
        if over_cap and k == OVER_CAP + 1: flags.append("over_cap")
        s["fragment"] = len(s["text"].split()) < FRAGMENT_WORDS
        rows.append({"callset_id": cid, "segment_id": s["segment_id"], "order_in_set": k, "download_index": s["download_index"], "source_file": s["source_file"],
                     "feed": s["feed"], "epoch": s["epoch"], "gap_from_previous_sec": "" if gap is None else round(gap, 1),
                     "decision": dec, "confidence": conf, "evidence": ev, "flags": ",".join(flags), "fragment": str(s["fragment"]).lower(),
                     "words": len(s["text"].split()), "related_callset": ""})
        shutil.copy2(SEG/f"{s['segment_id']}.wav", d/f"{s['segment_id']}.wav"); shutil.copy2(SEG/f"{s['segment_id']}.txt", d/f"{s['segment_id']}.txt")
        combined.append(s["text"])
        annotated.append(f"--- {s['segment_id']} | {s['source_file']} | {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(s['epoch']))} | {s['dur']:.1f}s | {len(s['text'].split())} words{' | FRAGMENT' if s['fragment'] else ''}\n{s['text']}\n")
    (d/"combined_transcript.txt").write_text("\n\n".join(combined) + "\n")          # plain text: this is what the pipeline reads
    (d/"combined_transcript_annotated.txt").write_text("\n".join(annotated))        # same bursts with headers, for humans
    acc = cs["acc"]; first, last = cs["members"][0][0], cs["members"][-1][0]
    has_open = bool(CALL_OPEN.search(first["text"][:200])); has_close = bool(CALL_CLOSE.search(last["text"][-200:]))
    cs["info"] = {"callset_id": cid, "feed": cs["feed"], "hospital": sorted(acc["hosp"]) or "not stated", "units": sorted(acc["units"] | acc["bare_units"]) or "not stated",
            "patient": {"ages": sorted(acc["ages"]), "sex": sorted(acc["sex"])}, "complaint_keywords": sorted(acc["complaints"]),
            "first_burst_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(first["epoch"])), "last_burst_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(last["epoch"] + last["dur"])),
            "num_bursts": len(cs["members"]), "total_words": sum(len(m[0]["text"].split()) for m in cs["members"]),
            "total_audio_sec": round(sum(m[0]["dur"] for m in cs["members"]), 1), "over_cap": over_cap,
            "has_clear_opening": has_open, "has_clear_closing": has_close,
            "appears_truncated": ("start" if not has_open else "") + ("+" if not has_open and not has_close else "") + ("end" if not has_close else "") or "no",
            "fragments": [m[0]["segment_id"] for m in cs["members"] if m[0]["fragment"]],
            "segments": [m[0]["segment_id"] for m in cs["members"]], "related_callset": "", "related_evidence": ""}
# related call-sets (cross-feed hints), resolved after numbering
seg2set = {m[0]["segment_id"]: cs for cs in sets for m in cs["members"]}
for sid, rel in RELATIONS.items():
    if rel.get("join"): continue
    me, other = seg2set[sid], seg2set[rel["related_to_segment"]]
    me["info"]["related_callset"] = other["id"]; me["info"]["related_evidence"] = rel["evidence"]
    for r in rows:
        if r["callset_id"] == me["id"]: r["related_callset"] = other["id"]
for cs in sets: (OUT/cs["id"]/"callset_info.json").write_text(json.dumps(cs["info"], indent=1))
# fragment column into segments.csv too
seg_rows = list(csv.DictReader(open(LABELS/"segments.csv"))); frag = {r["segment_id"]: r["fragment"] for r in rows}
for r in seg_rows: r["fragment"] = frag[r["segment_id"]]
with open(LABELS/"segments.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys())); w.writeheader(); w.writerows(seg_rows)
with open(LABELS/"callsets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# report
print(f"{'set':<6} {'feed':<5} {'seg':<8} {'#':>2} {'rec UTC':<8} {'gap s':>6} {'decision':<12} conf  flags            evidence")
for r in rows:
    print(f"{r['callset_id']:<6} {r['feed']:<5} {r['segment_id']:<8} {r['order_in_set']:>2} {time.strftime('%H:%M:%S', time.gmtime(r['epoch']))} {str(r['gap_from_previous_sec']):>6} {r['decision']:<12} {r['confidence']:.2f}  {r['flags']:<16} {r['evidence']}")
print(f"\n{len(sets)} call-sets from {len(rows)} segments; sizes:", {n: sum(1 for cs in sets if len(cs['members']) == n) for n in sorted({len(cs['members']) for cs in sets})})
print("fragments (<15 words):", [r['segment_id'] for r in rows if r['fragment'] == 'true'])
print("over-cap sets:", [cs['id'] for cs in sets if cs['info']['over_cap']], "| related:", {cs['id']: cs['info']['related_callset'] for cs in sets if cs['info']['related_callset']})
print("joined across feeds on review:", {sid: seg2set[sid]['id'] for sid in JOINS}, "| retired IDs:", retired)
print("truncated:", {cs['id']: cs['info']['appears_truncated'] for cs in sets if cs['info']['appears_truncated'] != 'no'})
