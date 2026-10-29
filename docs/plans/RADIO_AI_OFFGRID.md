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

## 6. Follow-up questions (2026-10-29): Airspy, encryption, compute in the radios, splitting the model, jamming

**Airspy.** An Airspy (R2 or Mini) is a receive-only SDR with 10 MHz of bandwidth, enough to watch a whole
700/800 MHz P25 site at once. With `trunk-recorder` on the Jetson it records every call on the talkgroups we
already use (mocomdps 4390, 4395) live, which replaces the OpenMHz download step (experiment R4). It is the
right first purchase. It never transmits.

**Encryption and compression.** Where encryption is legal decides the carrier:

| carrier | encryption | data | note |
|---|---|---|---|
| GMRS / FRS / MURS (Part 95) | not allowed (no obscured or coded messages) | GMRS: short text only; MURS: data allowed | voice to humans only |
| amateur (Part 97) | not allowed | allowed | needs a licence per operator |
| public safety P25 (Part 90) | AES-256 is standard | yes | only with the agency's licence and keys |
| 915 MHz ISM, LoRa / Meshtastic (Part 15) | allowed; Meshtastic encrypts by default (AES) | yes, 0.3 to 27 kbit/s | our encrypted data link |
| Wi-Fi / Ethernet between our own boxes | allowed | fast | how the two Jetsons already talk |

So the encrypted, compressed channel is LoRa. The handoff report as JSON is 1 to 2 KB and gzips to 300
to 600 bytes: one or two LoRa packets, a few seconds. Voice can be compressed with Codec2 (700 to 3200
bit/s, the open codec behind FreeDV) and sent encrypted over LoRa at the slowest spreading factors, but
only on ISM; encrypted digital voice on GMRS is illegal.

**Is there compute in the radios?** Not enough to matter:

| device | processor and memory | neural net it can hold |
|---|---|---|
| consumer handheld (Baofeng class) | small MCU, kilobytes | none |
| Meshtastic node (ESP32 or nRF52, up to 8 MB PSRAM) | about 1 MHz-class tasks | about 0.1 to 1 M parameters int8: a wake word or a VAD, no language model |
| public-safety radio (APX class) | ARM, tens of MB, closed firmware | nothing we can load |
| Raspberry Pi 5, 8 GB | 4 cores | about 3 B parameters int4, 2 to 5 tokens/s |
| Jetson Orin Nano 8 GB (ours) | GPU, 8 GB shared | 4 B comfortably, 8 B at the edge (measured 10-29) |
| Jetson Orin NX 16 GB | GPU, 16 GB | 8 to 14 B |
| AGX Orin 64 GB | GPU, 64 GB | 70 B int4 |
| phone, 8 to 12 GB | NPU | 3 to 4 B |

The radio is a modem. The compute sits next to it, and the "radio node" is a Jetson plus an Airspy, a
LoRa node, a handheld and a battery.

**Can the net be split across the radio network and weights fetched when needed?** No, and the arithmetic
says why. A transformer passes its hidden state between layers for every token: for the 4B model that is
2560 values, about 5 KB per token per device boundary. Over Ethernet (100 Mbit/s and up) that is nothing,
which is why llama.cpp's RPC mode can pipeline a model across our two Jetsons on the wire. Over LoRa at
20 kbit/s it is 2 s per token per hop: a 300-token report across two nodes takes 10 minutes and across
four nodes 30. Fetching weights on demand is worse: the 4B model is 2.5 GB, 20 Gbit, which at 20 kbit/s is
11 days for one forward pass. So each node must hold the whole model it uses. What the mesh carries is
the small products: transcripts, reports, patient context, model-free state. The two-Jetson split we have
(whisper on one box, the writer on the other) is this pattern on a fast link.

**How many parameters can a node have without cloud or cellular?** As many as its memory holds: 4 B on
the Orin Nano, 8 to 14 B on an NX, 70 B on an AGX. The choice is per node, not per network. For this
product the evidence so far says 4 B with the grounding rules is within 1 to 3 weighted points of the
120 B cloud writer, and the 8 B buys half a point at three times the time on an 8 GB board.

**Keeping the AI alive via radio, properly.** A mesh of whole models with shared state: every node has the
same models; the LoRa mesh replicates transcripts and reports (store-and-forward); any node can finish a
report another started; a node that dies loses nothing. That is resilience, which is what "alive" can
mean here, and it is measurable: pull a node's battery mid-call and time how long until another node
delivers the same report.

**Jamming and frequency hopping.** LoRa's chirp spread spectrum already resists narrowband interference,
and frequency hopping is legal in the 915 MHz ISM band (Part 15.247 was written for it). Meshtastic does
not hop today, so hopping means our own firmware or a different ISM radio. GMRS and FRS are fixed
channels on type-accepted radios, so no hopping there. Public-safety P25 does not hop either; agencies
rely on trunking, encryption and many sites. Our best anti-jam measures are diversity (voice on one band,
data on another, store-and-forward on both) and the mesh's many paths; hopping is a later layer.

**Next purchases, in order:** Airspy Mini or R2 (receive the EMS system live), two Meshtastic LoRa nodes
(the encrypted data mesh), two FRS handhelds (the real-FM tests in section 3). About 300 dollars.

**Meshtastic as distributed compute, specifically.** The numbers: a Meshtastic node has 0.25 to 8 MB of
memory and its LoRa link moves 0.3 to 27 kbit/s. Holding the 4B writer (2.5 GB at int4) across nodes would
take 300 to 10,000 of them, and every generated token would need the 5 KB hidden state to cross a LoRa hop
(2 s per hop at the fastest setting, minutes at the long-range settings). Whisper medium is the same
story. So Meshtastic cannot be the compute. What it is good at is the control and data plane of a
distributed system: carrying transcripts, reports, patient context and heartbeats between Jetson nodes,
each holding its own whole models, with store-and-forward so the work survives any node. The one model
that fits on a node itself is tiny (under a million parameters): a wake word, a voice-activity gate, or a
keyword alert ("STEMI", "trauma alert") that wakes the nearest Jetson. That is a useful edge trigger, not
the AI.
