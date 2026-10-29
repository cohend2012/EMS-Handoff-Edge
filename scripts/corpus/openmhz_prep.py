#!/usr/bin/env python3
"""openmhz_prep.py — find more calls like the ones we have on OpenMHz, without leaning on OpenMHz.

Three offline commands and one guarded online one:

    profile                     what our corpus is (system, talkgroups, hours, lengths), from ~/calls and ~/labels
    plan  [--days 3] [--hours]  the OpenMHz pages to open BY HAND for the same talkgroups and the hours that
                                produced our calls, as a list of links (nothing is fetched)
    ingest <folder>             move hand-downloaded .m4a files into ~/calls: strip " (1)" copies, drop sha256
                                duplicates, keep the OpenMHz name <system>-<talkgroup>-<epoch>.m4a, write a manifest
    fetch                       the polite API downloader. OFF unless --i-accept-the-rules is given. One request
                                at a time, never faster than --pace seconds (default 15), at most --max-requests
                                per run (default 20), a User-Agent that names us, a cache so nothing is asked twice,
                                and a full stop on the first 429 or 5xx. Not run by default and not to be looped.

Why the care: OpenMHz is a volunteer archive. The pipeline already starts from a folder of hand-downloaded
files (stage2_inventory.py reads ~/calls), so by-hand is the normal route; fetch exists so the plan can be
executed the same way a person would, slower.

What the files tell us (profile on 2026-10-29): system short name `mocomdps` (Montgomery County, MD public
safety); 44 of 51 kept calls on talkgroup 4390 and 6 on 4395 (one stray on 4430); all between 2026-10-23
21:41 and 2026-10-24 16:45 UTC; hospitals named in the audio are Shady Grove, Holy Cross Germantown, Suburban,
so 4390 and 4395 are the county's EMS-to-hospital consult channels. Those two talkgroups are the target.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

HOME = Path.home()
CALLS = HOME / "calls"
LABELS = HOME / "labels"
MANIFEST = LABELS / "openmhz_manifest.csv"
NAME = re.compile(r"^(?P<system>[a-z0-9]+)-(?P<tg>\d+)-(?P<epoch>\d{9,10})(?: \((?P<copy>\d+)\))?\.m4a$")
# Link shapes written from memory of the OpenMHz site, not checked against it (nothing here contacts the site
# unless fetch is armed). If a link does not open the archive at the hour, open the system page and use its own filter.
WEB = "https://openmhz.com/system/{system}?filter-type=talkgroup&filter-code={tg}&time={ms}"
API = "https://api.openmhz.com/{system}/calls?filter-type=talkgroup&filter-code={tg}&time={ms}"
UA = "EMS-Handoff-Edge corpus tool (research, one request at a time; contact cohend2012@gmail.com)"


def parse(name: str):
    m = NAME.match(name)
    if not m:
        return None
    d = m.groupdict()
    return d["system"], d["tg"], int(d["epoch"]), d["copy"]


def our_calls():
    out = []
    for p in sorted(CALLS.glob("*.m4a")):
        got = parse(p.name)
        if got:
            out.append((p,) + got)
    return out


def cmd_profile(args):
    calls = our_calls()
    inv = {}
    if (LABELS / "inventory.csv").exists():
        inv = {r["filename"]: r for r in csv.DictReader(open(LABELS / "inventory.csv"))}
    by_tg = collections.Counter(); hours = collections.Counter(); lengths = []
    for p, system, tg, epoch, copy in calls:
        r = inv.get(p.name, {})
        if r.get("status") == "duplicate" or copy:
            continue
        by_tg[(system, tg)] += 1
        hours[dt.datetime.fromtimestamp(epoch, dt.timezone.utc).hour] += 1
        if r.get("duration_s"):
            lengths.append(float(r["duration_s"]))
    print("kept calls:", sum(by_tg.values()))
    for (system, tg), n in by_tg.most_common():
        eps = [e for _, s, t, e, c in calls if s == system and t == tg]
        lo, hi = (dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in (min(eps), max(eps)))
        print(f"  {system} talkgroup {tg}: {n} calls, {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC")
    if lengths:
        lengths.sort()
        print(f"lengths s: min {lengths[0]:.0f}, median {lengths[len(lengths)//2]:.0f}, max {lengths[-1]:.0f}; "
              f"{sum(1 for x in lengths if x >= 20)} of {len(lengths)} are 20 s or longer (the handoff-shaped ones)")
    print("calls per UTC hour:", " ".join(f"{h:02d}:{hours[h]}" for h in sorted(hours)))
    if (LABELS / "labels.csv").exists():
        sc = collections.Counter(r["scenario"] for r in csv.DictReader(open(LABELS / "labels.csv")))
        print("scenarios in the labelled call-sets:", dict(sc))


def cmd_plan(args):
    calls = our_calls()
    by_tg = collections.Counter((s, t) for _, s, t, e, c in calls if not c)
    targets = [(s, t) for (s, t), n in by_tg.most_common() if n >= args.min_calls]
    hours = sorted({dt.datetime.fromtimestamp(e, dt.timezone.utc).hour for _, s, t, e, c in calls}) if not args.hours else args.hours
    start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    print(f"# OpenMHz pages to open by hand: {len(targets)} talkgroup(s) x {args.days} day(s) x {len(hours)} hour(s)")
    print("# Each link opens the archive at that hour; scroll back from there. Download only calls 20 s or longer")
    print("# with a unit talking to a hospital. Save into one folder, then: openmhz_prep.py ingest <folder>")
    print("# Our hours (UTC):", " ".join(f"{h:02d}" for h in hours))
    for (system, tg) in targets:
        print(f"\n## {system} talkgroup {tg}")
        for d in range(1, args.days + 1):
            day = start - dt.timedelta(days=d)
            for h in hours[:: max(1, len(hours) // args.links_per_day)]:
                t = day.replace(hour=h)
                print(f"{t:%Y-%m-%d %H:00} UTC  " + WEB.format(system=system, tg=tg, ms=int(t.timestamp() * 1000)))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_ingest(args):
    src = Path(args.folder).expanduser()
    have = {}
    for p in CALLS.glob("*.m4a"):
        have[sha256(p)] = p.name
    rows = []
    MANIFEST.parent.mkdir(exist_ok=True)
    new_manifest = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as mf:
        w = csv.writer(mf)
        if new_manifest:
            w.writerow(["ingested_at", "source_name", "kept_as", "system", "talkgroup", "epoch_utc", "sha256", "status"])
        for p in sorted(src.glob("*.m4a")):
            got = parse(p.name)
            if not got:
                print(f"skip (not an OpenMHz name): {p.name}"); continue
            system, tg, epoch, copy = got
            clean = f"{system}-{tg}-{epoch}.m4a"
            digest = sha256(p)
            if digest in have:
                status = f"duplicate of {have[digest]}"
            elif (CALLS / clean).exists():
                status = "name taken, content differs; kept with sha suffix"
                clean = f"{system}-{tg}-{epoch}-{digest[:8]}.m4a"
            else:
                status = "kept"
            if status.startswith("kept") or status.startswith("name taken"):
                if not args.dry_run:
                    shutil.copy2(p, CALLS / clean)
                have[digest] = clean
            if not args.dry_run:
                w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), p.name, clean if "kept" in status else "",
                            system, tg, dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(), digest, status])
            rows.append(status); print(f"{p.name}: {status}")
    print(f"\n{sum(1 for s in rows if 'kept' in s)} kept, {sum(1 for s in rows if s.startswith('duplicate'))} duplicates; "
          f"manifest {MANIFEST}. Next: python ~/labels/scripts/stage2_inventory.py")


def cmd_fetch(args):
    if not args.i_accept_the_rules:
        print("fetch is off. It contacts api.openmhz.com. Re-run with --i-accept-the-rules only when Daniel has decided to,\n"
              "and prefer the by-hand plan (openmhz_prep.py plan). Rules it enforces: one request at a time, --pace seconds\n"
              "between requests (15), --max-requests per run (20), stop on the first 429 or 5xx, nothing asked twice.")
        return 2
    import urllib.request, urllib.error
    cache = LABELS / "openmhz_cache"; cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    t_ms = int((args.before or time.time()) * 1000)
    made = 0; got_files = 0
    for tg in args.talkgroups:
        url = API.format(system=args.system, tg=tg, ms=t_ms)
        key = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".json")
        if key.exists():
            data = json.loads(key.read_text()); print(f"cache: {url}")
        else:
            if made >= args.max_requests:
                print("request budget spent; stop"); break
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code} on {url}: stopping this run entirely"); return 1
            except Exception as e:  # noqa: BLE001
                print(f"{e}: stopping"); return 1
            made += 1; key.write_text(json.dumps(data)); time.sleep(args.pace)
        calls = data.get("calls", [])
        print(f"{args.system} tg {tg}: {len(calls)} calls listed before {dt.datetime.fromtimestamp(t_ms/1000, dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
        for c in calls:
            length = float(c.get("len", 0) or 0)
            if length < args.min_len:
                continue
            start = c.get("time") or c.get("start_time")
            try:
                epoch = int(dt.datetime.fromisoformat(str(start).replace("Z", "+00:00")).timestamp()) if isinstance(start, str) else int(start)
            except Exception:  # noqa: BLE001
                epoch = int(time.time())
            name = f"{args.system}-{tg}-{epoch}.m4a"
            dest = out / name
            if dest.exists() or (CALLS / name).exists():
                continue
            if made >= args.max_requests:
                print("request budget spent; stop"); return 0
            audio_url = c.get("url")
            if not audio_url:
                continue
            req = urllib.request.Request(audio_url, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    dest.write_bytes(r.read())
            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code} on audio: stopping"); return 1
            made += 1; got_files += 1; print(f"  saved {name} ({length:.0f} s)"); time.sleep(args.pace)
    print(f"{made} requests, {got_files} files into {out}. Next: openmhz_prep.py ingest {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profile")
    p = sub.add_parser("plan"); p.add_argument("--days", type=int, default=3); p.add_argument("--hours", type=int, nargs="*")
    p.add_argument("--min-calls", type=int, default=2, help="talkgroups with fewer of our calls are not targets")
    p.add_argument("--links-per-day", type=int, default=6)
    p = sub.add_parser("ingest"); p.add_argument("folder"); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("fetch"); p.add_argument("--system", default="mocomdps"); p.add_argument("--talkgroups", nargs="+", default=["4390", "4395"])
    p.add_argument("--before", type=float, help="unix seconds; list calls before this time (default now)")
    p.add_argument("--min-len", type=float, default=20.0); p.add_argument("--pace", type=float, default=15.0)
    p.add_argument("--max-requests", type=int, default=20); p.add_argument("--out", default=str(HOME / "calls_incoming"))
    p.add_argument("--i-accept-the-rules", action="store_true")
    args = ap.parse_args(argv)
    return {"profile": cmd_profile, "plan": cmd_plan, "ingest": cmd_ingest, "fetch": cmd_fetch}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
