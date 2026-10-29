# AI on the radio: an off-grid, radio-driven handoff system

Asked 2026-10-29: evaluate running the AI directly on GMRS or on the radios EMTs use, and whether an
off-grid system can keep the AI "alive" through the radio alone.

## 1. The shape that works

The AI cannot live *over* the radio: the local model is 2.5 GB (qwen3:4b-instruct) and whisper medium is
1.5 GB, while any legal radio data link we can use carries a few hundred bytes a second. So the AI lives
*on* the Jetson, powered off-grid, and the radio is its only input and output:

| direction | carrier | what crosses the air | status |
|---|---|---|---|
| in | the EMS trunked system (receive only, Part 90 public safety) | the crew's own radio traffic, as today | done through OpenMHz archives; live reception needs an SDR |
| in | GMRS / FRS handheld near the patient | the crew's or the AI operator's voice | testable now with two handhelds |
| out, voice | GMRS / FRS | the AI reading the handoff report back (Piper TTS, already built) | testable now |
| out, data | LoRa / Meshtastic, 915 MHz ISM | the report as compressed JSON, a few hundred bytes | needs two LoRa nodes |

Power: the Orin Nano draws 7 to 15 W, so a 100 Wh pack runs the whole AI for 7 to 12 hours and a
50 W panel sustains it. That is the "off-grid" part and it is cheap.

## 2. What the law allows (to verify in 47 CFR Part 95 before anything transmits)

- **Public-safety frequencies (what EMTs use, e.g. mocomdps):** receive only. We never transmit there;
  that needs the agency's licence and its radios.
- **GMRS:** licence required ($35, ten years, no exam, covers the family). Voice is normal use; the AI's
  spoken read-back under a human operator's control is voice. Station identification every 15 minutes
  and at the end. **Digital data on GMRS is heavily restricted** (short text and location only), so the
  report cannot go over GMRS as data. One-way broadcasting is prohibited; an automatic responder that
  answers on its own needs a reading of the rules first. The safe form is a human-triggered read-back.
- **FRS:** no licence, 2 W, fixed antenna. Fine for every indoor and short-range test below.
- **MURS (VHF, 2 W, no licence):** data allowed; a fallback for the data link if LoRa range is short.
- **LoRa / Meshtastic (915 MHz ISM):** no licence; small packets, kilometres of range line of sight;
  the natural carrier for the report as data and for a mesh of several Jetsons.
- Amateur radio would allow more but forbids encryption and needs an operator licence per person.

## 3. Experiments, cheapest first, each with a number

| # | experiment | hardware | measure | against |
|---|---|---|---|---|
| R1 | gold samples through a real FM link: play the Piper script on one FRS handheld, the Jetson's mic hears the other | two FRS/GMRS handhelds (about $60) | WER and the gold-field score through real narrowband FM | our radio EQ simulation (STEMI 8 % WER with EQ) |
| R2 | wired tap: handheld speaker jack to a USB sound card into the Jetson | $15 adapter | WER against R1 | the mic path |
| R3 | round trip: the AI's spoken report over the link, transcribed again on the far side | same radios | field-level correctness after the round trip | the report as written |
| R4 | live scanner feed: RTL-SDR with trunk-recorder on the Jetson for the mocomdps talkgroups | RTL-SDR dongle (about $35) | seconds from end of call to report, fully local | the OpenMHz archive route |
| R5 | report as data over LoRa | two Meshtastic nodes (about $80) | bytes, delivery time, range | nothing to beat; a feasibility row |
| R6 | off-grid run: Jetson on a battery pack for a shift | a USB-PD pack | hours per 100 Wh with the pipeline busy | the 7 to 15 W spec |

R1 to R3 reuse the air-test tools (`scripts/corpus/run_samples.py`, the radio EQ, the gold scorer) and
need no code. R4 is the one that changes the product: it removes the website from the loop.

## 4. Decision rules

- The radio path is accepted only if the gold-field score through the real link is within 2 points of
  the simulated path; if not, the EQ simulation is wrong and gets re-fitted from the R1 recordings.
- Nothing transmits automatically until the Part 95 reading in section 2 is settled; tests use FRS.
- The data link ships only if a full report arrives in under 30 s at the ranges the crews need.

## 5. Not in scope

Transmitting on public-safety systems; carrying model weights or updates over radio (a 4B model at
1200 bit/s would take years); encryption on amateur bands.
