#!/usr/bin/env python3
"""
radioize.py — make a clean (synthetic or studio) voice sound like the radio corpus.

    python scripts/corpus/radioize.py $HANDOFF_DATASETS/scripts_v0/stemi.wav $HANDOFF_DATASETS/scripts_v0/stemi_radio.wav
    python scripts/corpus/radioize.py in.wav out.wav --spectra $HANDOFF_DATASETS/spectral/spectra.csv --noise -45

How
    1. The target is the corpus' mean speech spectrum from spectral_survey.py (spectra.csv, "speech"
       column). The source's own mean speech spectrum is measured the same way (voiced frames).
    2. The ratio target/source, smoothed over a third of an octave and limited to +-24 dB, is the
       equaliser; it is applied in the STFT domain (32 ms Hann frames, 16 ms hop, overlap-add).
       That gives the band limit (nothing above ~4.6 kHz) and the tilt (95 % of energy under 1.5 kHz).
    3. A soft limiter mimics the transmitter's compression, and a noise bed is added: real corpus noise
       with --noise-file (a $HANDOFF_DATASETS/noise/*.wav bed of the between-words floor, looped, at --snr dB under
       the speech), otherwise synthetic noise shaped like the carrier-open "silence" spectrum
       (--noise, dBFS; 0 = none).
    Output: 16 kHz mono WAV.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

from spectral_survey import FRAME, FREQS, HOP, NBINS, SR, decode, features  # noqa: E402


def mean_speech_spectrum_db(x: np.ndarray) -> np.ndarray:
    spec, f = features(x)
    voiced = (f["voicing"] > 0.30) & (f["centroid"] > 150) & (f["centroid"] < 3000)
    if voiced.sum() < 20:
        voiced = f["level_db"] > np.percentile(f["level_db"], 50)
    return 10 * np.log10(spec[voiced].mean(axis=0))


def read_spectra(path: Path):
    rows = list(csv.DictReader(path.open()))
    speech = np.array([float(r["speech"]) for r in rows])
    silence = np.array([float(r["silence"]) for r in rows]) if "silence" in rows[0] else None
    return speech, silence


def smooth_third_octave(db: np.ndarray) -> np.ndarray:
    out = db.copy()
    for i in range(1, NBINS):
        lo, hi = FREQS[i] / 2 ** (1 / 6), FREQS[i] * 2 ** (1 / 6)
        m = (FREQS >= lo) & (FREQS <= hi)
        out[i] = db[m].mean()
    return out


def stft_filter(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    win = np.hanning(FRAME)
    n = 1 + max(0, (len(x) - FRAME)) // HOP
    pad = n * HOP + FRAME
    xp = np.pad(x, (0, max(0, pad - len(x))))
    out = np.zeros_like(xp)
    norm = np.zeros_like(xp)
    g = 10 ** (gain_db / 20)
    for i in range(n + 1):
        a = i * HOP
        seg = xp[a: a + FRAME] * win
        y = np.fft.irfft(np.fft.rfft(seg) * g, n=FRAME) * win
        out[a: a + FRAME] += y
        norm[a: a + FRAME] += win ** 2
    norm[norm < 1e-6] = 1.0
    return (out / norm)[: len(x)]


def soft_limit(x: np.ndarray, drive: float = 2.0) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def shaped_noise(n: int, shape_db: np.ndarray, level_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    shaped = stft_filter(white, shape_db - shape_db.max())
    shaped /= np.sqrt(np.mean(shaped ** 2)) + 1e-9
    return shaped * 10 ** (level_dbfs / 20)


def channel_eq(x: np.ndarray, target_db: np.ndarray, max_gain: float = 24.0) -> np.ndarray:
    """The equaliser that takes this clip's speech spectrum to the corpus' (dB per STFT bin)."""
    source_db = mean_speech_spectrum_db(x)
    eq = smooth_third_octave(target_db - source_db)
    eq -= eq[(FREQS > 300) & (FREQS < 1000)].mean()          # unity gain where the voice lives
    eq = np.clip(eq, -max_gain, max_gain)
    eq[FREQS > 4600] = -60.0                                  # the channel's brick wall
    return eq


def radioize(x: np.ndarray, target_db: np.ndarray, bed: Optional[np.ndarray] = None, snr: Optional[float] = 20.0,
             level: float = -26.0, max_gain: float = 24.0, silence_db: Optional[np.ndarray] = None,
             synth_noise_dbfs: float = -48.0, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Clean 16 kHz speech -> the radio channel: EQ to the corpus spectrum and brick wall, level, soft limiter,
    then a noise bed. `bed` is real corpus floor looped under the speech at `snr` dB (snr None = no noise);
    without a bed, synthetic noise shaped like the carrier-open floor at synth_noise_dbfs (0 = none).
    Returns (audio, eq_db)."""
    eq = channel_eq(x, target_db, max_gain)
    y = stft_filter(x, eq)
    y = y / (np.sqrt(np.mean(y[np.abs(y) > 1e-4] ** 2)) + 1e-9) * 10 ** (level / 20)
    y = soft_limit(y)
    if bed is not None and snr is not None:
        reps = int(np.ceil(len(y) / len(bed)))
        b = np.tile(bed, reps)[: len(y)]
        b = b / (np.sqrt(np.mean(b ** 2)) + 1e-9) * 10 ** ((level - snr) / 20)
        y = y + b
    elif bed is None and synth_noise_dbfs < 0 and silence_db is not None:
        y = y + shaped_noise(len(y), silence_db, synth_noise_dbfs, np.random.default_rng(seed))
    return np.clip(y, -0.99, 0.99), eq


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


def eq_summary(eq: np.ndarray) -> str:
    return " ".join(f"{eq[np.argmin(np.abs(FREQS - hz))]:+.0f}" for hz in (250, 500, 1000, 2000, 3000, 4000)) + " dB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--spectra", type=Path, default=paths.spectral() / "spectra.csv")
    ap.add_argument("--noise", type=float, default=-48.0, help="synthetic noise bed level in dBFS (0 = none)")
    ap.add_argument("--noise-file", type=Path, help="real noise bed WAV to loop under the speech instead")
    ap.add_argument("--snr", type=float, default=20.0, help="speech-to-noise ratio in dB for --noise-file (corpus ~20)")
    ap.add_argument("--level", type=float, default=-26.0, help="target speech level in dBFS (corpus median -26)")
    ap.add_argument("--max-gain", type=float, default=24.0)
    args = ap.parse_args()

    x = decode(args.src)
    target_db, silence_db = read_spectra(args.spectra)
    bed = decode(args.noise_file) if args.noise_file else None
    y, eq = radioize(x, target_db, bed, args.snr, args.level, args.max_gain, silence_db, args.noise)
    write_wav(args.dst, y)
    rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
    print(f"[OK] {args.dst} ({len(y) / SR:.1f} s, {rms:.1f} dBFS); EQ at 250/500/1k/2k/3k/4k Hz: {eq_summary(eq)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
