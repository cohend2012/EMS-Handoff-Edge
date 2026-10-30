#!/usr/bin/env python3
"""
trajectories.py — Phase 1 of docs/plans/LYAPUNOV_STT_PLAN.md: run the PyTorch Whisper (Hugging Face) over
generated-set clips with hooks, and save, per emitted token, the decoder's last hidden state, log-probability,
entropy and cross-attention statistics, together with a label from the alignment against the gold script:

    correct     the token belongs to a word that matches the reference
    substituted the token belongs to a word that replaced a reference word
    inserted    the token belongs to a word with no reference counterpart (a hallucination)
    loop        the token lies inside a run of three or more repeated n-grams

Deleted reference words (dropped stretches) have no token to label; they are saved as spans of the reference.

    # inside the PyTorch container on the GPU Jetson (torch + transformers), repo at /app, set at /data
    python3 scripts/corpus/trajectories.py --set /data --split train --snr inf 5 --model openai/whisper-small \
        --out /data/trajectories/small_train --limit 60

Output: <out>/<id>_<snr>.npz with arrays tokens (int32), pieces (str), logprob, entropy, attn_peak, attn_entropy,
attn_mass_window (float32), hidden (float16, tokens x d_model), label (str), plus the decoded text and the
reference; <out>/summary.tsv with one row per clip (WER, token counts per label, deletion spans).
Nothing here is a verdict about the model; it is the dataset Phase 2 learns V from and tests it on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "corpus"))

from score_callsets import normalize, wer  # noqa: E402

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0 if width == 2 else np.frombuffer(raw, dtype=np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        from ems_handoff.audio.capture import resample
        x = resample(x, rate, SAMPLE_RATE)
    return x


def align_words(ref: list, hyp: list):
    """Levenshtein backtrace: per hypothesis word a label (correct|substituted|inserted), and the deleted reference spans."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    labels = [""] * m
    deleted = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            labels[j - 1] = "correct" if ref[i - 1] == hyp[j - 1] else "substituted"; i -= 1; j -= 1
        elif j > 0 and D[i, j] == D[i, j - 1] + 1:
            labels[j - 1] = "inserted"; j -= 1
        else:
            deleted.append(i - 1); i -= 1
    deleted.sort()
    spans = []
    for k in deleted:
        if spans and spans[-1][1] == k - 1:
            spans[-1][1] = k
        else:
            spans.append([k, k])
    return labels, [(a, b, " ".join(ref[a:b + 1])) for a, b in spans]


def loop_words(hyp: list, n: int = 3, repeats: int = 3) -> set:
    """Indices of hypothesis words inside a run of `repeats` or more identical n-grams."""
    out = set()
    for size in range(1, n + 1):
        i = 0
        while i + size * repeats <= len(hyp):
            gram = hyp[i:i + size]
            k = 1
            while hyp[i + k * size:i + (k + 1) * size] == gram:
                k += 1
            if k >= repeats:
                out.update(range(i, i + k * size)); i += k * size
            else:
                i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", type=Path, default=Path.home() / "datasets" / "synth_v1")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--snr", nargs="+", default=["inf", "5"])
    ap.add_argument("--voice", default="piper")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N scripts of the split")
    ap.add_argument("--hint", action="store_true", help="prepend the app's whisper hint as the prompt (default: no prompt, so the trajectories are the model's own)")
    args = ap.parse_args(argv)

    import torch  # noqa: WPS433 - the container has it, the dev box does not
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    tok = processor.tokenizer
    prompt_ids = None
    if args.hint:
        from ems_handoff.lexicon_rag import whisper_hint
        prompt_ids = torch.tensor(processor.get_prompt_ids(whisper_hint()), device=device)

    splits = json.loads((args.set / "splits.json").read_text())
    ids = sorted(splits[args.split])
    if args.limit:
        ids = ids[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.perf_counter()
    for sid in ids:
        ref_text = (args.set / "scripts" / f"{sid}.txt").read_text().strip()
        for snr in args.snr:
            tag = f"snr{snr}"
            wav = args.set / "audio" / args.voice / tag / f"{sid}.wav"
            if not wav.exists():
                print(f"[skip] {wav} missing", flush=True); continue
            x = load_wav(wav)
            feats = processor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device, dtype)
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(feats, return_dict_in_generate=True, output_scores=True, output_hidden_states=True,
                                     output_attentions=True, max_new_tokens=440, language="en", task="transcribe",
                                     prompt_ids=prompt_ids)
            seq = gen.sequences[0].tolist()
            steps = len(gen.scores)
            gen_tokens = seq[-steps:]                       # the tokens the steps produced, in order
            logp, ent, peak, aent, mass = [], [], [], [], []
            hidden = []
            for s in range(steps):
                sc = gen.scores[s][0].float()
                lp = torch.log_softmax(sc, dim=-1)
                logp.append(float(lp[gen_tokens[s]]))
                p = lp.exp()
                ent.append(float(-(p * lp).sum()))
                h = gen.decoder_hidden_states[s][-1][0, -1, :].float().cpu().numpy()   # last layer, last position
                hidden.append(h.astype(np.float16))
                ca = gen.cross_attentions[s][-1][0].float()      # last layer: heads x q x T_enc
                a = ca[:, -1, :].mean(dim=0)                      # mean over heads, last query position
                a = a / (a.sum() + 1e-9)
                peak.append(int(a.argmax()))
                aent.append(float(-(a * (a + 1e-12).log()).sum()))
                w = 25                                            # 25 frames = 0.5 s of encoder time around the peak
                lo, hi = max(0, peak[-1] - w), min(a.numel(), peak[-1] + w)
                mass.append(float(a[lo:hi].sum()))
            text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            pieces = [tok.decode([t]) for t in gen_tokens]
            # token -> word index (a piece starting with a space, or the first content piece, starts a word)
            word_of_token, widx = [], -1
            for pc, t in zip(pieces, gen_tokens):
                if t >= tok.eos_token_id and pc.strip() == "":
                    word_of_token.append(-1); continue
                if pc.startswith(" ") or widx < 0:
                    widx += 1
                word_of_token.append(widx)
            hyp_words_raw = [""] * (widx + 1)
            for pc, wi in zip(pieces, word_of_token):
                if wi >= 0:
                    hyp_words_raw[wi] += pc
            # align on normalised words; a raw word that normalises to several words keeps its first label
            ref_norm = normalize(ref_text)
            hyp_norm_per_word = [normalize(w) for w in hyp_words_raw]
            flat = [w for ws in hyp_norm_per_word for w in ws]
            labels_flat, deleted = align_words(ref_norm, flat)
            loops = loop_words(flat)
            lab_word, k = [], 0
            for ws in hyp_norm_per_word:
                if not ws:
                    lab_word.append("punct"); continue
                lab = labels_flat[k]
                if any((k + i) in loops for i in range(len(ws))):
                    lab = "loop"
                lab_word.append(lab); k += len(ws)
            label = [lab_word[wi] if wi >= 0 else "special" for wi in word_of_token]
            rate, edits, nref = wer(ref_text, text)
            np.savez_compressed(args.out / f"{sid}_{tag}.npz", tokens=np.array(gen_tokens, dtype=np.int32),
                                pieces=np.array(pieces, dtype=object), logprob=np.array(logp, dtype=np.float32),
                                entropy=np.array(ent, dtype=np.float32), attn_peak=np.array(peak, dtype=np.int32),
                                attn_entropy=np.array(aent, dtype=np.float32), attn_mass_window=np.array(mass, dtype=np.float32),
                                hidden=np.stack(hidden) if hidden else np.zeros((0, 1), np.float16),
                                label=np.array(label, dtype=object), text=text, reference=ref_text,
                                deleted=json.dumps(deleted))
            counts = {l: label.count(l) for l in ("correct", "substituted", "inserted", "loop")}
            rows.append({"id": sid, "snr": tag, "wer": rate, "edits": edits, "ref_words": nref, "tokens": steps, **counts,
                         "deleted_spans": len(deleted), "deleted_words": sum(b - a + 1 for a, b, _ in deleted),
                         "seconds": round(time.perf_counter() - t0, 2)})
            print(f"{sid} {tag:<7} wer {rate:.3f} tokens {steps} correct {counts['correct']} sub {counts['substituted']} "
                  f"ins {counts['inserted']} loop {counts['loop']} del {rows[-1]['deleted_words']} ({rows[-1]['seconds']} s)", flush=True)
    if rows:
        keys = list(rows[0].keys())
        with open(args.out / "summary.tsv", "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"done: {len(rows)} clips in {time.perf_counter() - t_all:.0f} s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
