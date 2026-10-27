#!/usr/bin/env python3
"""
spectral_survey.py — what the radio corpus sounds like, frame by frame: speech, tones, static, hum, silence.

    python scripts/corpus/spectral_survey.py ~/calls --out ~/reports_spectral
    python scripts/corpus/spectral_survey.py ~/calls/mocomdps-4390-1792791667.m4a --out ~/reports_spectral

Method
    Each file is decoded with ffmpeg to 16 kHz mono. 32 ms Hann frames every 16 ms give a power spectrum
    (31.25 Hz per bin, 0-8 kHz). From each frame: level in dBFS, spectral centroid, spectral flatness
    (0 = one pure tone, 1 = white noise), the share of energy in six bands (0-150 Hz mains hum, 150-300,
    300-1000, 1000-3000 where speech lives, 3000-4000 the top of a radio channel, 4000-8000 above it),
    the frequency below which 95 % of the energy lies (roll-off), the strongest bin's share of the energy
    (a siren or a beep is one narrow line), and a voicing cue (normalised autocorrelation peak for a pitch
    between 60 and 400 Hz, high for a voice, low for noise).

    Flatness and the peak are judged inside the radio channel (100-4500 Hz): the feed carries nothing
    above about 4.6 kHz, and over the full 8 kHz every frame would look tonal.
    A first-pass rule labels every frame, then a 5-frame majority vote smooths the labels:
        silence  more than 25 dB below the median voiced level (squelch closed, gaps)
        hum      more than 60 % of the energy under 150 Hz
        tone     one bin above 300 Hz holds more than 45 % of the in-channel energy and holds still
                 for three frames (siren, alert tone, squelch tone)
        speech   voicing cue above 0.30 with the centroid between 150 and 3000 Hz
        noise    in-channel spectral flatness above 0.30 (static, engine, wind)
        other    the rest (unvoiced consonants, mixtures)
    The rules are a starting point to listen against, not a classifier; the per-frame CSV lets the
    thresholds be re-cut without re-decoding.

Outputs (in --out)
    files.csv      one row per file: duration, % of frames per label, speech level, noise floor, SNR,
                   median roll-off and centroid during speech, hum level
    frames/<file>.csv   every frame's features and label
    spectra.csv    the mean spectrum (dB) of each label across the corpus, one column per label
    summary.md     the table above plus corpus-wide averages
    png/<file>.png spectrogram with the label strip under it (needs matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SR = 16000
FRAME = 512          # 32 ms
HOP = 256            # 16 ms
NBINS = FRAME // 2 + 1
FREQS = np.fft.rfftfreq(FRAME, 1 / SR)
BANDS = [(0, 150), (150, 300), (300, 1000), (1000, 3000), (3000, 4000), (4000, 8000)]
LABELS = ("silence", "speech", "tone", "noise", "hum", "other")


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def frames_of(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)[None, :]


def features(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fr = frames_of(x)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12          # power per bin
    total = spec.sum(axis=1)
    level_db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    centroid = (spec * FREQS[None, :]).sum(axis=1) / total
    cum = np.cumsum(spec, axis=1) / total[:, None]
    rolloff = FREQS[np.argmax(cum >= 0.95, axis=1)]
    # the radio channel carries nothing above about 4.5 kHz, so flatness and the peak are judged
    # inside the channel (100-4500 Hz); over the full 8 kHz every frame looks "tonal"
    ib = (FREQS >= 100) & (FREQS <= 4500)
    sib = spec[:, ib]
    flatness = np.exp(np.mean(np.log(sib), axis=1)) / np.mean(sib, axis=1)
    peak_share = sib.max(axis=1) / sib.sum(axis=1)
    peak_hz = FREQS[ib][np.argmax(sib, axis=1)]
    bands = {}
    for lo, hi in BANDS:
        m = (FREQS >= lo) & (FREQS < hi)
        bands[f"band_{lo}_{hi}"] = spec[:, m].sum(axis=1) / total
    # voicing: normalised autocorrelation peak for lags of 2.5-16.7 ms (60-400 Hz)
    ac = np.fft.irfft(spec, n=FRAME, axis=1)
    lo, hi = SR // 400, SR // 60
    voicing = ac[:, lo:hi].max(axis=1) / (ac[:, 0] + 1e-12)
    feats = {"level_db": level_db, "centroid": centroid, "flatness": flatness, "rolloff": rolloff,
             "peak_share": peak_share, "peak_hz": peak_hz, "voicing": voicing, **bands}
    return spec, feats


def label_frames(f: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(f["level_db"])
    voiced = f["voicing"] > 0.30
    speech_ref = float(np.percentile(f["level_db"][voiced], 50)) if voiced.any() else float(np.percentile(f["level_db"], 90))
    # a tone holds still: the peak bin is within 2 bins (62 Hz) of its neighbours on both sides
    ph = f["peak_hz"]
    still = np.zeros(n, dtype=bool)
    if n > 2:
        still[1:-1] = (np.abs(ph[1:-1] - ph[:-2]) <= 62.5) & (np.abs(ph[1:-1] - ph[2:]) <= 62.5)
    lab = np.full(n, "other", dtype=object)
    lab[f["flatness"] > 0.30] = "noise"
    lab[voiced & (f["centroid"] > 150) & (f["centroid"] < 3000)] = "speech"
    lab[(f["peak_share"] > 0.45) & (ph >= 300) & still] = "tone"
    lab[f["band_0_150"] > 0.60] = "hum"
    lab[f["level_db"] < speech_ref - 25] = "silence"
    # 5-frame majority vote
    out = lab.copy()
    for i in range(n):
        window = lab[max(0, i - 2): i + 3]
        out[i] = Counter(window).most_common(1)[0][0]
    return out


def survey_file(path: Path, out: Path, plot: bool) -> Tuple[dict, Dict[str, np.ndarray]]:
    x = decode(path)
    spec, f = features(x)
    lab = label_frames(f)
    n = len(lab)
    counts = Counter(lab)
    speech = lab == "speech"
    floor = float(np.percentile(f["level_db"][lab != "silence"], 10)) if (lab != "silence").any() else float("nan")
    speech_level = float(np.median(f["level_db"][speech])) if speech.any() else float("nan")
    hum = lab == "hum"
    row = {"file": path.name, "seconds": round(len(x) / SR, 1), "frames": n,
           **{f"pct_{k}": round(100 * counts.get(k, 0) / n, 1) for k in LABELS},
           "speech_level_db": round(speech_level, 1), "noise_floor_db": round(floor, 1),
           "snr_db": round(speech_level - floor, 1),
           "speech_rolloff_hz": round(float(np.median(f["rolloff"][speech])), 0) if speech.any() else "",
           "speech_centroid_hz": round(float(np.median(f["centroid"][speech])), 0) if speech.any() else "",
           "speech_flatness": round(float(np.median(f["flatness"][speech])), 3) if speech.any() else "",
           "hum_level_db": round(float(np.median(f["level_db"][hum])), 1) if hum.any() else ""}
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    with (fdir / (path.stem + ".csv")).open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(f.keys())
        w.writerow(["t_s"] + keys + ["label"])
        for i in range(n):
            w.writerow([round(i * HOP / SR, 3)] + [round(float(f[k][i]), 4) for k in keys] + [lab[i]])
    spectra = {k: 10 * np.log10(spec[lab == k].mean(axis=0)) for k in LABELS if (lab == k).any()}
    if plot:
        plot_file(path, spec, lab, out / "png")
    return row, spectra


def plot_file(path: Path, spec: np.ndarray, lab: np.ndarray, pdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(lab)) * HOP / SR
    colors = {"silence": "#bbbbbb", "speech": "#2a9d8f", "tone": "#e76f51", "noise": "#8d6e63",
              "hum": "#7b61ff", "other": "#f4a261"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [6, 1]})
    a1.imshow(10 * np.log10(spec).T, origin="lower", aspect="auto", cmap="magma",
              extent=[0, t[-1] if len(t) else 1, 0, SR / 2], vmin=-110, vmax=-30)
    a1.set_ylabel("Hz")
    a1.set_title(path.name)
    for k, c in colors.items():
        m = lab == k
        if m.any():
            a2.scatter(t[m], np.zeros(m.sum()), s=12, c=c, marker="|", label=k)
    a2.set_yticks([])
    a2.set_xlabel("s")
    a2.legend(loc="upper right", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(pdir / (path.stem + ".png"), dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("paths", nargs="+", type=Path, help="audio files or folders (wav, m4a, mp3)")
    ap.add_argument("--out", type=Path, default=Path("~/reports_spectral").expanduser())
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    files: List[Path] = []
    for p in args.paths:
        files += sorted(q for q in (p.glob("*") if p.is_dir() else [p]) if q.suffix.lower() in (".wav", ".m4a", ".mp3"))
    if not files:
        print("no audio files", file=sys.stderr)
        return 1
    plot = not args.no_plot
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        plot = False
    args.out.mkdir(parents=True, exist_ok=True)

    rows, spectra_sum, spectra_n = [], {k: np.zeros(NBINS) for k in LABELS}, Counter()
    for i, path in enumerate(files, 1):
        row, spectra = survey_file(path, args.out, plot)
        rows.append(row)
        for k, s in spectra.items():
            spectra_sum[k] += s
            spectra_n[k] += 1
        print(f"[{i}/{len(files)}] {row['file']}: {row['seconds']} s, speech {row['pct_speech']}%, "
              f"tone {row['pct_tone']}%, noise {row['pct_noise']}%, hum {row['pct_hum']}%, "
              f"SNR {row['snr_db']} dB, roll-off {row['speech_rolloff_hz']} Hz", flush=True)

    with (args.out / "files.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "spectra.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        labels = [k for k in LABELS if spectra_n[k]]
        w.writerow(["hz"] + labels)
        for b in range(NBINS):
            w.writerow([round(float(FREQS[b]), 2)] + [round(float(spectra_sum[k][b] / spectra_n[k]), 2) for k in labels])

    def mean(key):
        vals = [r[key] for r in rows if r[key] != ""]
        return sum(vals) / len(vals) if vals else float("nan")

    secs = sum(r["seconds"] for r in rows)
    lines = [f"# Spectral survey: {len(rows)} files, {secs / 60:.1f} min\n",
             "| label | share of frames |", "|---|---|"]
    lines += [f"| {k} | {mean(f'pct_{k}'):.1f}% |" for k in LABELS]
    lines += ["", f"During speech: median roll-off (95 % energy) {mean('speech_rolloff_hz'):.0f} Hz, centroid "
                  f"{mean('speech_centroid_hz'):.0f} Hz, flatness {mean('speech_flatness'):.2f}. Speech level "
                  f"{mean('speech_level_db'):.1f} dBFS over a noise floor of {mean('noise_floor_db'):.1f} dBFS "
                  f"(SNR {mean('snr_db'):.1f} dB).", "",
              "| file | s | speech | tone | noise | hum | silence | SNR dB | roll-off Hz | centroid Hz |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['file']} | {r['seconds']} | {r['pct_speech']}% | {r['pct_tone']}% | {r['pct_noise']}% | "
              f"{r['pct_hum']}% | {r['pct_silence']}% | {r['snr_db']} | {r['speech_rolloff_hz']} | {r['speech_centroid_hz']} |"
              for r in rows]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}/summary.md, files.csv, spectra.csv, frames/" + (", png/" if plot else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
