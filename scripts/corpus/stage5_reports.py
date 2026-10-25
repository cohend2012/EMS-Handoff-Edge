#!/usr/bin/env python3
"""Stage 5: run the HANDOFF.AI pipeline (Ollama) over the call-sets.
  primary   : each call-set's combined_transcript.txt -> cli.py --text-file
              -> ~/callsets/<id>/combined_report.json + ~/reports/<id>_combined.json
  secondary : every burst, however short -> cli.py --wav (Groq transcribes, Ollama writes)
              -> ~/reports/<segment_id>.json + copy in the set folder
Sequential, one at a time, resumable (existing outputs are skipped), failures logged
to the log csv and the batch continues. Same order every run: primary over the set IDs
sorted, then every burst in callsets.csv order.
  usage: stage5_reports.py --sets CS001            (primary only, named sets)
         stage5_reports.py --all                   (every primary, then every secondary)
         stage5_reports.py --all --out ~/reports_postfix --log ~/labels/stage5_postfix_log.csv
             (a second run beside the baseline: reports only in --out, nothing written into
              ~/callsets, which keeps the baseline copies there)"""
import argparse, csv, json, re, shutil, subprocess, sys, time
from pathlib import Path

HOME = Path.home(); APP = HOME/"EMS-Handoff-app"; PY = APP/".venv/bin/python"
SETS, LABELS = HOME/"callsets", HOME/"labels"; TIMEOUT = 900
BASELINE_MODEL = "qwen3:4b-instruct"; MODEL_RETRIES = 3   # OLLAMA_MODEL pins the model in the app (an error if it cannot be used); the check here stays as a second guard
import os; ENV = dict(os.environ, OLLAMA_MODEL=BASELINE_MODEL)
ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="*"); ap.add_argument("--all", action="store_true")
ap.add_argument("--secondary", action="store_true", help="with --sets: also run that set's bursts")
ap.add_argument("--out", default=str(HOME/"reports"), help="report folder (default ~/reports, the baseline)")
ap.add_argument("--log", default=None, help="log csv (default ~/labels/stage5_log.csv; with --out, <out>/stage5_log.csv)")
ap.add_argument("--app", default=str(APP), help="checkout to run (default ~/EMS-Handoff-app)")
args = ap.parse_args()
REPORTS = Path(args.out).expanduser(); REPORTS.mkdir(parents=True, exist_ok=True); APP = Path(args.app).expanduser()
BASELINE_RUN = REPORTS == HOME/"reports"
LOG = Path(args.log).expanduser() if args.log else (LABELS/"stage5_log.csv" if BASELINE_RUN else REPORTS/"stage5_log.csv")
def git(*a):
    try: return subprocess.run(["git", *a], cwd=APP, capture_output=True, text=True).stdout.strip()
    except OSError: return "?"
RUN_INFO = f"{git('rev-parse', '--abbrev-ref', 'HEAD')} {git('rev-parse', '--short', 'HEAD')}{' dirty' if git('status', '--porcelain', '--', 'ems_handoff') else ''}"

def log(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["when", "kind", "callset", "target", "status", "wall_s", "model", "llm_s", "error"])
        if new: w.writeheader()
        w.writerow(row)

def run_cli(cli_args, dest_paths, kind, cid, target):
    if all(p.exists() for p in dest_paths):
        print(f"  skip {target}: already done", flush=True); return "skipped", 0
    t0 = time.time(); attempt = 0
    cmd = [str(PY), "-m", "ems_handoff.cli", *cli_args, "--provider", "ollama", "--json"]
    while True:
        attempt += 1
        with open(LABELS/"run_log.txt", "a") as rl: rl.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] (cwd {APP} {RUN_INFO}, OLLAMA_MODEL={BASELINE_MODEL}, out {REPORTS}) {' '.join(cmd)}\n")
        try:
            r = subprocess.run(cmd, cwd=APP, capture_output=True, text=True, timeout=TIMEOUT, env=ENV)
        except subprocess.TimeoutExpired:
            wall = time.time()-t0; log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "timeout", "wall_s": round(wall), "model": "", "llm_s": "", "error": f"no result in {TIMEOUT}s"})
            print(f"  TIMEOUT {target} after {wall:.0f}s", flush=True); return "timeout", wall
        wall = time.time()-t0
        m = re.search(r"^Saved: (.+)$", r.stdout, re.M)
        if r.returncode != 0 or not m:
            err = (r.stderr.strip().splitlines() or ["?"])[-1][:300]
            log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "failed", "wall_s": round(wall), "model": "", "llm_s": "", "error": err})
            print(f"  FAILED {target} ({wall:.0f}s): {err}", flush=True); return "failed", wall
        report = json.load(open(m.group(1).strip())); meta = report.get("_meta", {})
        if meta.get("model") != BASELINE_MODEL:
            rej = REPORTS/"_rejected"/f"{target}.{str(meta.get('model')).replace(':', '-')}.attempt{attempt}.json"; rej.parent.mkdir(exist_ok=True); rej.write_text(json.dumps(report, indent=2))
            log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "rejected_wrong_model", "wall_s": round(wall), "model": meta.get("model", ""), "llm_s": "", "error": f"app picked {meta.get('model')}; attempt {attempt}"})
            print(f"  WRONG MODEL {target}: {meta.get('model')} (attempt {attempt}); output set aside", flush=True)
            if attempt < MODEL_RETRIES: time.sleep(20); continue
            return "failed", wall
        report["_run"] = {"app": str(APP), "git": RUN_INFO, "pinned_model": BASELINE_MODEL, "out": str(REPORTS)}
        for p in dest_paths: p.write_text(json.dumps(report, indent=2))
        tm = meta.get("timings", {}); llm_s = tm.get("llm_s") or tm.get("extract_s") or ""
        log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "ok", "wall_s": round(wall), "model": meta.get("model", ""), "llm_s": llm_s, "error": ""})
        print(f"  ok {target}: {wall:.0f}s wall, model {meta.get('model')}, timings {tm}", flush=True); return "ok", wall

rows = list(csv.DictReader(open(LABELS/"callsets.csv")))
set_ids = sorted({r["callset_id"] for r in rows})
targets = args.sets if args.sets else set_ids
t_batch = time.time(); results = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0}
print(f"app {APP} ({RUN_INFO}); model {BASELINE_MODEL}; reports -> {REPORTS}; log {LOG}" + ("" if BASELINE_RUN else "; set folders untouched"))
def dests(in_set, in_reports): return [SETS/in_set, REPORTS/in_reports] if BASELINE_RUN else [REPORTS/in_reports]
print("PRIMARY (combined transcript -> text path)")
for cid in targets:
    print(f"[{cid}]", flush=True)
    st, _ = run_cli(["--text-file", str(SETS/cid/"combined_transcript.txt")], dests(f"{cid}/combined_report.json", f"{cid}_combined.json"), "primary", cid, f"{cid}_combined")
    results[st] += 1
if args.all or args.secondary:
    print("\nSECONDARY (every burst, WAV -> transcribe -> report)")
    bursts = [r for r in rows if r["callset_id"] in targets]   # no length floor: a short burst is a real input
    print(f"{len(bursts)} bursts to run ({sum(int(r['words']) < 15 for r in bursts)} fragments under 15 words)")
    for r in bursts:
        sid, cid = r["segment_id"], r["callset_id"]; print(f"[{cid} / {sid}] {r['words']} words", flush=True)
        st, _ = run_cli(["--wav", str(HOME/"segments"/f"{sid}.wav")], dests(f"{cid}/{sid}_report.json", f"{sid}.json"), "secondary", cid, sid)
        results[st] += 1
print(f"\nbatch done in {(time.time()-t_batch)/60:.1f} min: {results}")
