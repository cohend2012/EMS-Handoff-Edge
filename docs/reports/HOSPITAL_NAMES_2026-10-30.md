# Base-hospital names: corrector rules scored against human reads (2026-10-30 night)

Lois Sayrs named base-hospital names and unit numbers as the most misheard words. On her 22 read bursts the names fail
in radio frames: "CineGroves online", "Holy cow, Sturm and Tau copies" (Holy Cross, Germantown copies), "We crashed German
town copies", "Polycross Germantown", "Suburban L9" (Suburban online), "Medic729", "Medics 715".

Change (ems_handoff/transcript_corrector.py): five narrow context rules (Germantown spacing; Holy Cross before
Germantown/copies/online/this is; a burst-opening "Holy cow, ... copies"; "online" heard as L9 or "on 9" before a comma;
unit name spacing) and a frame pass `_apply_hospital_frames`: a span at a sentence start before "online", "copies" or "is
clear" is rewritten to the base hospital whose stem it ends with (grove, cross, suburb, germ...town, white oak, sibley),
only when exactly one stem matches and the span is not already a hospital name. Tests in tests/test_transcript_corrector.py.

Scored by applying only the new rules to every saved transcript of the 22 bursts. Name recall counts the 17 hospital
names in her reads; two of her reads carry a transcript's mishear verbatim ("Polycross", "CineGroves"), so for name recall
those two words are read as Holy Cross and Shady Grove. WER is against her reads unchanged.

| transcript | WER before | after | names before | after | rule firings |
|---|---|---|---|---|---|
| gpu_large_v3_ours (the live system) | 16.1 % | 15.7 % | 11/17 | 13/17 | 4 |
| cloud_large_ours | 14.7 % | 14.6 % | 10/17 | 13/17 | 3 |
| local_ours | 20.7 % | 20.1 % | 9/17 | 12/17 | 4 |
| local_base | 21.0 % | 20.6 % | 8/17 | 11/17 | 3 |
| cloud_base | 18.1 % | 18.5 % | 9/17 | 12/17 | 4 |
| cloud_ours | 20.8 % | 20.8 % | 12/17 | 12/17 | 2 |

cloud_base WER rises because the fix turns "Polycross" into Holy Cross where her read kept "Polycross".

False-fire check: on the 130 saved transcripts of the 12 unread calls the rules fired 9 times, all on true hospital names
(CS014 "ShadyGroves"/"Cady Grove's" online, CS023 "German Town"); one firing turns "Shady Grove is online" into "Shady
Grove online" (a dropped "is", now skipped). On 1,830 synthetic transcripts: 0 firings. Full suite 282 passed.

Limits: the rules were written from these same 22 bursts, so the gain is in-sample; it needs confirming on bursts Lois has
not read yet. Unit numbers are mostly heard right already (Medic 714/731, Ambulance 733/742); her own reads use forms like
"AM1-721" and "E1793", so unit-number fixes beyond spacing wait for a list of local unit designations. Openers garbled
past the stem ("Why don't you call the line", "Wide open line" for "Shady Grove online") are not touched.
