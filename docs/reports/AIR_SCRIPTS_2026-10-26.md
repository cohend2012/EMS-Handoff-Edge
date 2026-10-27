# The three practice scripts over the air (2026-10-26)

`scripts/air_check.py all --play [--variant _radio]`: each script is read by Piper (en_US-lessac-medium, spoken
form of the abbreviations) and played on the JMTek USB dongle's speaker (volume 0.8) into a live app session on
the same dongle's mic: Vosk captions with Groq Whisper repair, Groq whisper-large-v3-turbo transcript,
qwen3:4b-instruct report. Server dabd424 (150 Hz mic high-pass). "radioeq" runs play the same audio after
`scripts/corpus/radioize.py`: equalised to the corpus' mean speech spectrum and brick-walled at 4.6 kHz
(SPECTRAL_2026-10-26.md). Runs are under `~/reports_air/`, one line each in `INDEX.tsv`.

| script | audio | played | report after | WER transcript | WER captions | report | script says |
|---|---|---|---|---|---|---|---|
| sepsis | usb-dongle-radioeq | 81 s | 80 s | 10% (161/161 words) | 20% | ENDOCRINE RED ETA 12 | ETA 12, no alert word |
| sepsis | usb-dongle | 81 s | 80 s | 10% (161/161 words) | n/a | ENDOCRINE RED ETA 12 | ETA 12, no alert word |
| stemi | usb-dongle-radioeq | 78 s | 75 s | 8% (171/169 words) | 14% | STEMI RED ETA 8 | ETA 8, STEMI, cath lab |
| stemi | usb-dongle | 78 s | 65 s | 35% (119/169 words) | n/a | STEMI RED ETA 12 | ETA 8, STEMI, cath lab |
| trauma | usb-dongle-radioeq | 77 s | 74 s | 5% (157/155 words) | 24% | TRAUMA RED ETA 6 | ETA 6, trauma activation |
| trauma | usb-dongle | 77 s | 76 s | 6% (154/155 words) | n/a | TRAUMA RED ETA 6 | ETA 6, trauma activation |

(captions WER n/a: the first run's caption text was joined wrongly by the script; fixed before the radioeq runs.)

## What each run dropped or misheard

**sepsis / usb-dongle-radioeq**: dropped: over | disease. Replaced: squad -> quad | 24 spo2 -> sbo2 | rhonchi -> ronti | metoprolol furosemide donepezil -> metaprolol furosmide dumpyzil | eliquis -> deliquis | iv -> roman 4 | 500ml -> 500 ml | polst -> pulse.

**sepsis / usb-dongle**: dropped: over. Replaced: squad -> fwad | 24 spo2 -> sbo2 | skin -> gin | lungs -> luns | rhonchi -> ronti | metoprolol furosemide donepezil -> metaprolol furosmide dumpizil | eliquis -> deliquis | iv -> roman 4 | 500ml -> 500 ml | polst -> pulsed.

**stemi / usb-dongle-radioeq**: dropped: over. Replaced: on scene -> unseen | spo2 94 -> sbo 294 | lisinopril -> listenaprol | administered 324mg -> administer 324 mg | 4mg -> 4 mg.

**stemi / usb-dongle**: dropped: over. Replaced: on scene -> unseen | spo2 -> sbo2 | lisinopril -> lisinaprol | iv -> roman 4 | administered 324mg aspirin chewed and 0 4mg sublingual nitro times 2 with some relief currently running a saline lock patient is alert and oriented pain currently 6 out of 10 down from 9 out of 10 our eta to your facility is approximately 8 minutes requesting cath lab activation unit 41 clear -> i don't know.

**trauma / usb-dongle-radioeq**: dropped: over. Replaced: medic -> medics | spo2 -> sbo 2 | bore ivs -> borrome and 4.

**trauma / usb-dongle**: dropped: nothing. Replaced: medic -> medics | bp 88 over -> bp88 | spo2 91 -> sbo 291 | bore ivs -> borrome and 4.

## What it says

- **The plain Piper voice does not suit the speaker-and-mic path.** Its 95 % energy roll-off is 719 Hz
  (the corpus' is 1.5 kHz); through the small dongle speaker and the 150 Hz high-pass the consonants went,
  and on the STEMI script Whisper gave up on the last third ("I don't know") while the Vosk captions still
  carried aspirin, nitro, cath lab and "Unit 41 clear". Equalised to the corpus' speech spectrum the same
  audio scored 8 %, 5 % and 10 % word error with the right ETA every time.
- **That is a statement about our lab audio, not about radio speech.** The runs show the test chain works
  end to end and that test audio must be shaped like the channel before it is played; judging the
  transcriber and the report needs a voice cloned from the medics with the corpus' own noise under it,
  then a larger generated gold set with a train/test split (SPECTRAL_2026-10-26.md, "Towards a cloned
  radio voice"). `scripts/corpus/radioize.py --noise-file ~/noise_bank/feed4390_floor.wav --snr 20`
  now overlays 76 s of the real between-words floor of feed 4390 instead of synthetic noise.
- **ETA from "12-lead".** With the ETA sentence lost, the local model reported ETA 12 from "12-lead".
  The pipeline now clears an ETA whose number is not spoken as a time (commit c6308c1); on the equalised
  run the sentence came through and ETA 8 stood.
- **Sepsis script -> ENDOCRINE.** Both sepsis runs (10 % WER, every word of the vitals heard) came back
  ENDOCRINE RED from qwen3:4b-instruct, not SEPSIS: fever 102.4, hypotension, tachycardia, altered mental
  status from a nursing facility, cloudy urine. A model judgment, not a hearing problem; the alert-type
  rules in the prompt are the place to look, and the gold set should score it.
- **Piper's reading of abbreviations** cost words that no mic would: "IV" was read as the Roman numeral
  ("Roman 4"), "SpO2" as a word ("sbo2"). The scripts are now synthesised from a spoken form (I V, S P O 2,
  158 over 92, 324 milligrams); the runs above were made before that change.
- **The Vosk captions are a safety net.** On every run the captions carried words the Whisper transcript
  lost; a transcript much shorter than its captions should fall back to them for the missing stretch.
