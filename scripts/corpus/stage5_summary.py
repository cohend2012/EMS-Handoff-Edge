#!/usr/bin/env python3
"""Summarize the Stage 5 baseline: successes/failures with causes, wall time, LLM time stats,
data_quality.confidence distribution for the combined (per-set) reports. Appends to BASELINE.md."""
import csv, json, statistics, time
from collections import Counter
from pathlib import Path
HOME = Path.home(); LABELS = HOME/"labels"; SETS = HOME/"callsets"; REPORTS = HOME/"reports"
log = list(csv.DictReader(open(LABELS/"stage5_log.csv")))
ok = [r for r in log if r["status"] == "ok"]; bad = [r for r in log if r["status"] not in ("ok", "skipped")]
def secs(hms): h, m, s = map(int, hms.split(":")); return h*3600+m*60+s
lines = []
lines.append(f"runs logged: {len(log)}; ok {len(ok)} (primary {sum(1 for r in ok if r['kind']=='primary')}, secondary {sum(1 for r in ok if r['kind']=='secondary')}); not ok {len(bad)}")
for r in bad: lines.append(f"  {r['status']:<22} {r['kind']:<9} {r['target']:<16} {r['model']:<18} {r['error']}")
models = Counter(r["model"] for r in ok); lines.append(f"models in accepted reports: {dict(models)}")
wall = sum(int(r["wall_s"]) for r in log if r["status"] != "skipped"); lines.append(f"total wall time, all attempts incl. rejected: {wall/60:.1f} min ({wall} s); accepted runs only: {sum(int(r['wall_s']) for r in ok)/60:.1f} min")
batch_span = secs(max(r["when"] for r in log if r["target"] != "CS001_combined")) - secs(min(r["when"] for r in log if r["target"] != "CS001_combined"))
lines.append(f"batch clock span (first to last log entry after CS001): {batch_span/60:.1f} min, including the stop/restart")
for kind in ("primary", "secondary"):
    ls = [float(r["llm_s"]) for r in ok if r["kind"] == kind and r["llm_s"]]
    if ls:
        worst = max((r for r in ok if r["kind"] == kind and r["llm_s"]), key=lambda r: float(r["llm_s"]))
        lines.append(f"{kind:<9} LLM time: mean {statistics.mean(ls):.1f} s, median {statistics.median(ls):.1f} s, worst {max(ls):.1f} s ({worst['target']}), best {min(ls):.1f} s; n={len(ls)}")
ws = [int(r["wall_s"]) for r in ok if r["kind"] == "secondary"]
if ws: lines.append(f"secondary wall (incl. Groq transcription): mean {statistics.mean(ws):.0f} s, worst {max(ws)} s")
conf = Counter(); alerts = Counter(); levels = Counter(); per_set = []
for d in sorted(SETS.glob("CS*")):
    p = d/"combined_report.json"
    if not p.exists(): per_set.append((d.name, "MISSING", "", "")); continue
    rep = json.load(open(p)); c = (rep.get("data_quality") or {}).get("confidence") or "null"
    conf[c] += 1; alerts[str(rep.get("alert_type"))] += 1; levels[str(rep.get("alert_level"))] += 1
    per_set.append((d.name, c, rep.get("alert_type"), rep.get("alert_level")))
lines.append(f"data_quality.confidence over {sum(conf.values())} combined reports: {dict(conf)}")
lines.append(f"alert_type: {dict(alerts)}"); lines.append(f"alert_level: {dict(levels)}")
lines.append("per set: " + ", ".join(f"{n}={c}/{a}/{l}" for n, c, a, l in per_set))
text = "\n".join(lines); print(text)
with open(REPORTS/"BASELINE.md", "a") as f: f.write(f"\n## Results ({time.strftime('%Y-%m-%d %H:%M')})\n\n```\n{text}\n```\n")
