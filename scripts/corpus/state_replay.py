"""
state_replay.py — replay the real call sets through the incident manager the way the live app runs them
(session-cumulative reports, quiet-period and closing-cue triggers, the ask rule), with the real writer, and
record the patient state after every report.

    python scripts/corpus/state_replay.py                       # every call set under ~/callsets, local writer
    python scripts/corpus/state_replay.py --callsets CS001 CS008 --out ~/reports_state

For each call set the bursts are the saved transcripts (<CS>/<segment>.txt) at their real epochs from
callset_info.json; a fake clock advances to each burst's start, the manager's quiet period (25 s offline) and
closing cues fire exactly as live, and every queued extraction runs on this thread. Output per call set:
<out>/<CS>.json with the bursts, every report (cumulative), the session events (triggers, asks) and the final
state; <out>/summary.md with one row per call set. Resumable: a call set with a finished JSON is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ems_handoff.config import load_env_files, load_settings  # noqa: E402
from ems_handoff.incident import IncidentManager, QUIET_OFFLINE_S  # noqa: E402
from ems_handoff.llm import LLMClient  # noqa: E402
from ems_handoff.pipeline import ReportPipeline  # noqa: E402
from ems_handoff.server import default_engines  # noqa: E402


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def replay_callset(cs_dir: Path, extract, quiet_s: float) -> dict:
    info = json.loads((cs_dir / "callset_info.json").read_text())
    bursts = info.get("bursts") or [{"segment_id": s, "epoch": None, "duration_sec": 0.0} for s in info["segments"]]
    clock = Clock()
    events: list = []
    man = IncidentManager(extract, quiet_s=quiet_s, clock=clock, emit=events.append, start_worker=False)
    t0 = bursts[0].get("epoch") or 0.0
    fed = []
    for i, b in enumerate(bursts, 1):
        seg = b["segment_id"]
        txt = cs_dir / f"{seg}.txt"
        if not txt.exists():
            continue
        text = txt.read_text().strip()
        at = float(b.get("epoch") or (t0 + 60.0 * (i - 1)))
        clock.now = at
        man.poll()                       # a quiet period that passed before this burst fires first
        man.run_pending(timeout=1800)
        man.add_burst(text, at=at, duration_s=float(b.get("duration_sec") or 0.0), feed=str(b.get("feed") or ""), source_id=seg)
        man.run_pending(timeout=1800)
        fed.append({"n": i, "segment": seg, "at": at, "duration_s": b.get("duration_sec"), "words": len(text.split())})
    clock.now = (clock.now or 0.0) + max(b.get("duration_sec") or 0.0 for b in bursts) + quiet_s + 1.0
    man.poll()
    man.run_pending(timeout=1800)
    out = {"callset": cs_dir.name, "bursts": fed, "sessions": []}
    for s in man.sessions:
        snap = s.snapshot()
        out["sessions"].append({"id": s.id, "bursts": len(s.bursts), "reports": [{k: v for k, v in r.items() if k != "_run"} for r in s.reports],
                                "events": list(s.events), "closing": s.closing, "ask": s.ask, "state": snap["state"],
                                "change_lines": snap["change_lines"], "error": s.last_error})
    out["events"] = [{k: v for k, v in e.items() if k != "session"} for e in events]
    return out


def summarize(results: list) -> str:
    rows = ["| call set | bursts | sessions | reports | alert | readings | flags | crossings | missing at close | ask | conflicts |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"calls": 0, "reports": 0, "flags": 0, "asks": 0, "missing": 0, "conflicts": 0}
    for r in results:
        tot["calls"] += 1
        for s in r["sessions"]:
            st = s["state"] or {}
            flags = ", ".join(f["name"] for f in st.get("flags", [])) or "-"
            cross = sum(len(x.get("crossings", [])) for x in st.get("series", []))
            tot["reports"] += len(s["reports"]); tot["flags"] += len(st.get("flags", [])); tot["asks"] += 1 if s["ask"] else 0
            tot["missing"] += len(st.get("missing", [])); tot["conflicts"] += len(st.get("conflicts", []))
            rows.append(f"| {r['callset']} | {s['bursts']} | {len(r['sessions'])} | {len(s['reports'])} | {st.get('alert_type') or '-'} | "
                        f"{len(st.get('readings', []))} | {flags} | {cross} | {', '.join(st.get('missing', [])) or '-'} | "
                        f"{s['ask'] or ('-' if not s['closing'] else 'nothing missing')} | {'; '.join(st.get('conflicts', [])) or '-'} |")
    head = (f"calls {tot['calls']}, reports {tot['reports']}, flags {tot['flags']}, asks {tot['asks']}, "
            f"missing items at close {tot['missing']}, conflicts {tot['conflicts']}\n\n")
    return head + "\n".join(rows) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--callsets", nargs="*", help="call set ids; default every CS* folder")
    ap.add_argument("--out", type=Path, default=Path.home() / "reports_state")
    ap.add_argument("--quiet", type=float, default=QUIET_OFFLINE_S)
    args = ap.parse_args(argv)
    load_env_files()
    settings = load_settings()
    settings = settings.__class__(**{**settings.__dict__, "provider": "ollama"}) if hasattr(settings, "__dict__") else settings
    engines = default_engines(settings)
    llm = LLMClient(settings)
    pipeline = ReportPipeline(llm, stop_phrase="", lexicon_store=engines.lexicon_store, textbook_rag=engines.textbook_rag,
                              on_stage=lambda m: None, attribution=True)

    def extract(text: str, meta: dict) -> dict:
        return pipeline.run(text, source=f"state_replay:{meta.get('session')}", timings={})

    args.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in args.callsets_dir.glob("CS*") if d.is_dir() and (d / "callset_info.json").exists())
    if args.callsets:
        dirs = [d for d in dirs if d.name in set(args.callsets)]
    results = []
    for d in dirs:
        dst = args.out / f"{d.name}.json"
        if dst.exists():
            results.append(json.loads(dst.read_text()))
            print(f"[{d.name}] done already", flush=True)
            continue
        t0 = time.perf_counter()
        res = replay_callset(d, extract, args.quiet)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        dst.write_text(json.dumps(res, indent=1, default=str))
        results.append(res)
        s = res["sessions"][-1] if res["sessions"] else {}
        print(f"[{d.name}] {len(res['bursts'])} bursts, {sum(len(x['reports']) for x in res['sessions'])} reports, "
              f"{res['seconds']} s; ask: {s.get('ask') or '-'}", flush=True)
    (args.out / "summary.md").write_text(f"# State replay (session-cumulative, {settings.ollama_model})\n\n" + summarize(results))
    print(summarize(results).splitlines()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
