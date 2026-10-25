#!/usr/bin/env python3
"""Stage 5: run the HANDOFF.AI pipeline over the call-sets.
  primary   : each call-set's combined_transcript.txt -> cli.py --text-file
              -> ~/callsets/<id>/combined_report.json + ~/reports/<id>_combined.json
  secondary : every burst, however short -> cli.py --wav (Groq transcribes, the model writes)
              -> ~/reports/<segment_id>.json + copy in the set folder
              with --text the burst's Stage 3 transcript (~/segments/<id>.txt) is used instead
              of the wav, so the run is extraction only and every run sees the same words
Sequential, one at a time, resumable (existing outputs are skipped), failures logged
to the log csv and the batch continues. Same order every run: primary over the set IDs
sorted, then every burst in callsets.csv order.
Provider: --provider ollama (default, model pinned through OLLAMA_MODEL) or --provider groq
(model pinned through GROQ_LLM_MODEL). A report from any other model is set aside and the
target retried. HTTP 429 from Groq: honor 'try again in', else back off from 10 s doubling
to 300 s, and retry the same target. A per-day token cap stops the batch cleanly and says
how many targets remain; the same command later continues from there.
  usage: stage5_reports.py --sets CS001            (primary only, named sets)
         stage5_reports.py --all                   (every primary, then every secondary)
         stage5_reports.py --all --out ~/reports_postfix --log ~/labels/stage5_postfix_log.csv
             (a second run beside the baseline: reports only in --out, nothing written into
              ~/callsets, which keeps the baseline copies there)
         stage5_reports.py --all --provider groq --text --out ~/reports_groq --log ~/labels/stage5_groq_log.csv"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

HOME = Path.home(); APP = HOME/"EMS-Handoff-app"; PY = APP/".venv/bin/python"
SETS, LABELS, SEGMENTS = HOME/"callsets", HOME/"labels", HOME/"segments"; TIMEOUT = 900
DEFAULT_MODEL = {"ollama": "qwen3:4b-instruct", "groq": "openai/gpt-oss-120b"}; MODEL_RETRIES = 3
ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="*"); ap.add_argument("--all", action="store_true")
ap.add_argument("--secondary", action="store_true", help="with --sets: also run that set's bursts")
ap.add_argument("--text", action="store_true", help="bursts from their Stage 3 transcript on disk instead of the wav")
ap.add_argument("--provider", choices=list(DEFAULT_MODEL), default="ollama")
ap.add_argument("--model", default=None, help=f"pinned model (default per provider: {DEFAULT_MODEL})")
ap.add_argument("--out", default=str(HOME/"reports"), help="report folder (default ~/reports, the baseline)")
ap.add_argument("--log", default=None, help="log csv (default ~/labels/stage5_log.csv; with --out, <out>/stage5_log.csv)")
ap.add_argument("--app", default=str(APP), help="checkout to run (default ~/EMS-Handoff-app)")
args = ap.parse_args()
PROVIDER = args.provider; MODEL = args.model or DEFAULT_MODEL[PROVIDER]
ENV = dict(os.environ, **({"OLLAMA_MODEL": MODEL} if PROVIDER == "ollama" else {"GROQ_LLM_MODEL": MODEL}))
REPORTS = Path(args.out).expanduser(); REPORTS.mkdir(parents=True, exist_ok=True); APP = Path(args.app).expanduser()
BASELINE_RUN = REPORTS == HOME/"reports"
LOG = Path(args.log).expanduser() if args.log else (LABELS/"stage5_log.csv" if BASELINE_RUN else REPORTS/"stage5_log.csv")
def git(*a):
    try: return subprocess.run(["git", *a], cwd=APP, capture_output=True, text=True).stdout.strip()
    except OSError: return "?"
RUN_INFO = f"{git('rev-parse', '--abbrev-ref', 'HEAD')} {git('rev-parse', '--short', 'HEAD')}{' dirty' if git('status', '--porcelain', '--', 'ems_handoff') else ''}"
PIN = f"{'OLLAMA_MODEL' if PROVIDER == 'ollama' else 'GROQ_LLM_MODEL'}={MODEL}"

class DailyCap(Exception): pass

def log(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["when", "kind", "callset", "target", "status", "wall_s", "model", "llm_s", "error"])
        if new: w.writeheader()
        w.writerow(row)

def retry_after_s(err):
    """seconds from a Groq 'Please try again in 3h21m5.1s' / '12.3s' / '450ms' hint, else None"""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+m?s)?)", err)
    if not m or not m.group(1): return None
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(h|ms|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total or None

def run_cli(cli_args, dest_paths, kind, cid, target):
    if all(p.exists() for p in dest_paths):
        print(f"  skip {target}: already done", flush=True); return "skipped", 0
    t0 = time.time(); attempt = 0; backoff = 10.0
    cmd = [str(PY), "-m", "ems_handoff.cli", *cli_args, "--provider", PROVIDER, "--json"]
    while True:
        attempt += 1
        with open(LABELS/"run_log.txt", "a") as rl: rl.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] (cwd {APP} {RUN_INFO}, {PIN}, out {REPORTS}) {' '.join(cmd)}\n")
        try:
            r = subprocess.run(cmd, cwd=APP, capture_output=True, text=True, timeout=TIMEOUT, env=ENV)
        except subprocess.TimeoutExpired:
            wall = time.time()-t0; log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "timeout", "wall_s": round(wall), "model": "", "llm_s": "", "error": f"no result in {TIMEOUT}s"})
            print(f"  TIMEOUT {target} after {wall:.0f}s", flush=True); return "timeout", wall
        wall = time.time()-t0
        m = re.search(r"^Saved: (.+)$", r.stdout, re.M)
        if r.returncode != 0 or not m:
            err = (r.stderr.strip().splitlines() or ["?"])[-1][:400]
            if re.search(r"HTTP 429|rate limit", err, re.I):
                if re.search(r"per day|TPD|RPD|daily", err, re.I):
                    log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "daily_cap", "wall_s": round(wall), "model": "", "llm_s": "", "error": err})
                    raise DailyCap(err)
                wait = retry_after_s(err) or backoff; backoff = min(backoff*2, 300); wait = min(max(wait + 1, 2), 900)
                log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "rate_limited", "wall_s": round(wall), "model": "", "llm_s": "", "error": err[:200]})
                print(f"  429 on {target}: waiting {wait:.0f}s", flush=True); time.sleep(wait); continue
            log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "failed", "wall_s": round(wall), "model": "", "llm_s": "", "error": err[:300]})
            print(f"  FAILED {target} ({wall:.0f}s): {err[:300]}", flush=True); return "failed", wall
        report = json.load(open(m.group(1).strip())); meta = report.get("_meta", {})
        if meta.get("model") != MODEL:
            rej = REPORTS/"_rejected"/f"{target}.{str(meta.get('model')).replace(':', '-').replace('/', '-')}.attempt{attempt}.json"; rej.parent.mkdir(exist_ok=True); rej.write_text(json.dumps(report, indent=2))
            log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "rejected_wrong_model", "wall_s": round(wall), "model": meta.get("model", ""), "llm_s": "", "error": f"app picked {meta.get('model')}; attempt {attempt}"})
            print(f"  WRONG MODEL {target}: {meta.get('model')} (attempt {attempt}); output set aside", flush=True)
            if attempt < MODEL_RETRIES: time.sleep(20); continue
            return "failed", wall
        report["_run"] = {"app": str(APP), "git": RUN_INFO, "provider": PROVIDER, "pinned_model": MODEL, "out": str(REPORTS), "burst_input": "text" if args.text else "wav"}
        for p in dest_paths: p.write_text(json.dumps(report, indent=2))
        tm = meta.get("timings", {}); llm_s = tm.get("llm_s") or tm.get("extract_s") or ""
        log({"when": time.strftime("%H:%M:%S"), "kind": kind, "callset": cid, "target": target, "status": "ok", "wall_s": round(wall), "model": meta.get("model", ""), "llm_s": llm_s, "error": ""})
        print(f"  ok {target}: {wall:.0f}s wall, model {meta.get('model')}, timings {tm}", flush=True); return "ok", wall

rows = list(csv.DictReader(open(LABELS/"callsets.csv")))
set_ids = sorted({r["callset_id"] for r in rows})
targets = args.sets if args.sets else set_ids
t_batch = time.time(); results = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0}
print(f"app {APP} ({RUN_INFO}); provider {PROVIDER}, model {MODEL}; reports -> {REPORTS}; log {LOG}" + ("" if BASELINE_RUN else "; set folders untouched") + ("; bursts from Stage 3 text" if args.text else ""))
def dests(in_set, in_reports): return [SETS/in_set, REPORTS/in_reports] if BASELINE_RUN else [REPORTS/in_reports]
plan = [("primary", cid, f"{cid}_combined", ["--text-file", str(SETS/cid/"combined_transcript.txt")], dests(f"{cid}/combined_report.json", f"{cid}_combined.json"), None) for cid in targets]
if args.all or args.secondary:
    bursts = [r for r in rows if r["callset_id"] in targets]   # no length floor: a short burst is a real input
    for r in bursts:
        sid = r["segment_id"]; src = ["--text-file", str(SEGMENTS/f"{sid}.txt")] if args.text else ["--wav", str(SEGMENTS/f"{sid}.wav")]
        plan.append(("secondary", r["callset_id"], sid, src, dests(f"{r['callset_id']}/{sid}_report.json", f"{sid}.json"), r["words"]))
    print(f"{len(bursts)} bursts to run ({sum(int(r['words']) < 15 for r in bursts)} fragments under 15 words)")
print("PRIMARY (combined transcript -> text path)")
try:
    for i, (kind, cid, target, src, dest, words) in enumerate(plan):
        if kind == "secondary" and (i == 0 or plan[i-1][0] == "primary"): print(f"\nSECONDARY (every burst, {'Stage 3 text' if args.text else 'WAV -> transcribe'} -> report)")
        print(f"[{cid}]" if kind == "primary" else f"[{cid} / {target}] {words} words", flush=True)
        st, _ = run_cli(src, dest, kind, cid, target); results[st] += 1
except DailyCap as e:
    remaining = [t for _, _, t, _, dest, _ in plan if not all(p.exists() for p in dest)]
    print(f"\nDAILY CAP HIT at {time.strftime('%H:%M:%S')}: {e}\n{len(remaining)} of {len(plan)} targets remain: {', '.join(remaining)}\n"
          f"re-run the same command when the cap resets; finished targets are skipped. batch so far: {results}", flush=True)
    sys.exit(2)
print(f"\nbatch done in {(time.time()-t_batch)/60:.1f} min: {results}")
