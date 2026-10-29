# Sourcing more calls from OpenMHz (prepared 2026-10-29, nothing fetched)

OpenMHz is a volunteer archive and we want to stay welcome there. Rule: no scraping, no crawling, no parallel
requests. Calls are downloaded by hand from the site, or, only if Daniel decides so, by `openmhz_prep.py fetch`
one request at a time with a 15 s pace and a 20-request budget per run. This document and the tool were
prepared without contacting the site.

## Where our corpus came from

The 51 kept recordings in `~/calls` carry OpenMHz's own file names, `<system>-<talkgroup>-<unix time>.m4a`:

| system | talkgroup | kept calls | window (UTC) | what it is |
|---|---|---|---|---|
| `mocomdps` | 4390 | 44 | 2026-10-23 21:41 to 2026-10-24 16:43 | EMS to hospital consult (Shady Grove, Holy Cross Germantown, Suburban are named in the audio) |
| `mocomdps` | 4395 | 6 | 2026-10-24 00:52 to 16:45 | same kind of traffic, second channel |
| `mocomdps` | 4430 | 1 | 2026-10-24 07:41 | one stray; not a target |

`mocomdps` is Montgomery County, Maryland public safety. The files were saved by hand in the browser on
2026-10-24 between 09:53 and 10:30 local (seven of them twice, the " (1)" copies that stage 2 caught as
duplicates). Talkgroup names on the site were not recorded; the numbers above are the search key.

What a useful call looks like: 20 s or longer, one unit talking to a hospital (age, sex, complaint, vitals,
ETA). 23 of our 51 bursts are 20 s or longer, median 18 s; the labelled call-sets split stroke 8, trauma 5,
other 6, sepsis 1, respiratory 1, unclear 1. Our calls cluster at 00 to 04 and 12 to 16 UTC (evening and
morning local).

## The by-hand route (default)

```
.venv/bin/python scripts/corpus/openmhz_prep.py profile         # what we have, from disk
.venv/bin/python scripts/corpus/openmhz_prep.py plan --days 3   # links to open by hand, per talkgroup and hour
#   ... open the links, listen, save the calls that look like ours into one folder (keep the site's file names)
.venv/bin/python scripts/corpus/openmhz_prep.py ingest ~/Downloads/openmhz_2026-11   # into ~/calls, de-duplicated, manifest written
python ~/labels/scripts/stage2_inventory.py                      # then stages 3 to 6 as before
```

The link shape the plan prints was written from memory of the site and is not verified; if a link does not land
on the archive at that hour, open the `mocomdps` system page and use its own talkgroup filter and time picker.

## The guarded route

`openmhz_prep.py fetch` is off unless `--i-accept-the-rules` is passed. It identifies us in the User-Agent, asks
once per talkgroup for the list of calls before a time, skips anything under 20 s or already on disk, downloads
one file at a time with the pace between every request, caches every answer so a re-run asks nothing twice,
and stops the whole run on the first 429 or 5xx. It is for executing the plan the way a person would, slower;
it is not to be looped or scheduled.
