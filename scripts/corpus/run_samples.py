#!/usr/bin/env python3
"""Run the four built-in sample cases through the pipeline and save one report per case.

    run_samples.py --out ~/reports/samples_baseline [--cases stemi trauma] [--app DIR] [--model qwen3:4b-instruct]

Every claimed run ends in exactly one of two files in --out:
    <case>.json         the report, with a "_run" block (app commit, branch, pinned model, times)
    <case>.error.txt    the exit code, stderr and stdout of the failed run
and one line in ~/labels/run_log.txt written AFTER the run, with its status and the file.
(The stemi run logged on 2026-10-24 16:36 had its log line written before the run and the
run never finished, so the log claimed a file that did not exist. This runner cannot do that.)
The Ollama model is pinned through OLLAMA_MODEL; a report from any other model is an error.
An existing <case>.json is kept unless --force."""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HOME = Path.home(); LABELS = HOME / "labels"
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", required=True); ap.add_argument("--cases", nargs="*", default=["quick", "stemi", "trauma", "sepsis"])
ap.add_argument("--app", default=str(Path(__file__).resolve().parents[2]), help="checkout to run (default: this repo)")
ap.add_argument("--model", default="qwen3:4b-instruct"); ap.add_argument("--provider", default="ollama")
ap.add_argument("--timeout", type=int, default=900); ap.add_argument("--force", action="store_true")
args = ap.parse_args()
APP = Path(args.app).resolve(); PY = APP / ".venv/bin/python"
if not PY.exists(): PY = Path(__file__).resolve().parents[2] / ".venv/bin/python"   # a worktree shares the main venv
OUT = Path(args.out).expanduser(); OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, OLLAMA_MODEL=args.model)

def git(*a):
    try: return subprocess.run(["git", *a], cwd=APP, capture_output=True, text=True).stdout.strip()
    except OSError: return "?"
commit, branch = git("rev-parse", "--short", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
dirty = bool(git("status", "--porcelain", "--", "ems_handoff"))

def log(line):
    LABELS.mkdir(exist_ok=True)
    with open(LABELS / "run_log.txt", "a") as f: f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {line}\n")

failures = 0
for case in args.cases:
    dest, err = OUT / f"{case}.json", OUT / f"{case}.error.txt"
    if dest.exists() and not args.force:
        print(f"skip {case}: {dest} exists"); continue
    cmd = [str(PY), "-m", "ems_handoff.cli", "--case", case, "--provider", args.provider, "--json"]
    started = time.strftime("%Y-%m-%dT%H:%M:%S"); t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=APP, capture_output=True, text=True, timeout=args.timeout, env=ENV)
        rc, out, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, se = "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), f"no result in {args.timeout}s"
    wall = round(time.time() - t0)
    m = re.search(r"^Saved: (.+)$", out, re.M)
    report = None; problem = None
    if rc != 0 or not m: problem = f"exit {rc}: {(se.strip().splitlines() or ['?'])[-1][:300]}"
    else:
        try: report = json.load(open(m.group(1).strip()))
        except (OSError, ValueError) as e: problem = f"saved file unreadable: {e}"
    if report is not None and args.provider == "ollama" and report.get("_meta", {}).get("model") != args.model:
        problem = f"wrong model: app used {report.get('_meta', {}).get('model')!r}, pinned {args.model!r}"
    if problem:
        failures += 1
        err.write_text(f"case {case}\napp {APP} ({branch} {commit}{' dirty' if dirty else ''})\ncmd {' '.join(cmd)}\nstarted {started}, wall {wall}s\n"
                       f"problem {problem}\n\n--- stderr ---\n{se}\n--- stdout ---\n{out}\n")
        if dest.exists(): dest.unlink()
        log(f"(app {APP} {branch} {commit}{' dirty' if dirty else ''}, OLLAMA_MODEL={args.model}) {' '.join(cmd)}  # FAILED after {wall}s: {problem} -> {err}")
        print(f"FAILED {case} ({wall}s): {problem} -> {err}", flush=True); continue
    report["_run"] = {"case": case, "app": str(APP), "branch": branch, "commit": commit, "dirty": dirty,
                      "pinned_model": args.model, "started": started, "wall_s": wall}
    dest.write_text(json.dumps(report, indent=2))
    if err.exists(): err.unlink()
    tm = report["_meta"].get("timings", {})
    log(f"(app {APP} {branch} {commit}{' dirty' if dirty else ''}, OLLAMA_MODEL={args.model}) {' '.join(cmd)}  # ok in {wall}s, model {report['_meta'].get('model')} -> {dest}")
    print(f"ok {case}: {wall}s wall, model {report['_meta'].get('model')}, timings {tm} -> {dest}", flush=True)
sys.exit(1 if failures else 0)
