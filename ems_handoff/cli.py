"""
cli.py — headless HANDOFF.AI, for scripts, tests and quick checks over SSH.

    python -m ems_handoff.cli --case stemi              sample case -> report
    python -m ems_handoff.cli --text-file report.txt    your own text -> report
    python -m ems_handoff.cli --wav recording.wav       audio file -> transcript -> report
    add --provider ollama|groq|openrouter to override .env, --json to print raw JSON

The report is printed as a short summary and saved to output/handoff_report.json.
Exit code 0 on success, 1 on failure (the reason is printed).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

from .config import PROJECT_ROOT, PROVIDERS, load_settings


def summarize(report: dict) -> str:
    s = report.get("handoff_summary", {})
    v = report.get("vital_signs", {})
    p = report.get("patient", {})
    hr = report.get("hospital_recommendations", {})
    lines = [
        f"{report.get('alert_type')} ALERT · {report.get('alert_level')} · ETA {report.get('eta_minutes')} min",
        f"  {s.get('one_liner', '')}",
        f"  Patient: {p.get('age')} {p.get('age_unit', 'years')} {p.get('sex', '')}; {report.get('chief_complaint', '')}",
        "  Vitals: " + ", ".join(f"{k}={v[k]}" for k in v if v.get(k) not in (None, "", "null")),
        f"  Activation: {hr.get('activation_requested')} -> {hr.get('suggested_destination')}",
    ]
    for action in hr.get("time_critical_actions") or []:
        lines.append(f"  ! {action}")
    meta = report.get("_meta", {})
    lines.append(f"  ({meta.get('provider')} · {meta.get('model')} · {meta.get('timings')})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ems_handoff.cli", description=__doc__.split("\n")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", choices=["quick", "stemi", "trauma", "sepsis"])
    src.add_argument("--text-file")
    src.add_argument("--wav")
    ap.add_argument("--provider", choices=PROVIDERS)
    ap.add_argument("--show-context", action="store_true",
                    help="print the lexicon terms and textbook excerpts the model was given")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    args = ap.parse_args(argv)

    from .config import load_env_files
    load_env_files()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from .server import SAMPLES, default_engines, save_report
    from .llm import LLMClient, LLMError
    from .pipeline import PipelineError, ReportPipeline

    settings = load_settings()
    if args.provider:
        settings = dataclasses.replace(settings, provider=args.provider)
    engines = default_engines(settings)
    timings = {}

    if args.case:
        text = next(s["text"] for s in SAMPLES if s["id"] == args.case)
    elif args.text_file:
        text = open(args.text_file, encoding="utf-8").read()
    else:
        import numpy as np
        import scipy.io.wavfile as wavfile
        from .audio.capture import resample
        from .transcribe import TranscriptionError, make_transcriber
        rate, data = wavfile.read(args.wav)
        audio = data.astype(np.float32)
        if data.dtype == np.int16:
            audio /= 32768.0
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = resample(audio, rate)
        t0 = time.perf_counter()
        try:
            text = make_transcriber(settings).transcribe(audio)
        except TranscriptionError as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            return 1
        timings["transcribe_s"] = round(time.perf_counter() - t0, 2)
        print(f"Transcript ({timings['transcribe_s']} s):\n{text}\n")

    try:
        llm = LLMClient(settings)
        pipeline = ReportPipeline(llm, stop_phrase=settings.stop_phrase,
                                  lexicon_store=engines.lexicon_store,
                                  textbook_rag=engines.textbook_rag,
                                  on_stage=lambda m: print(f"... {m}", file=sys.stderr))
        report = pipeline.run(text, source="cli", timings=timings)
    except (LLMError, PipelineError) as e:
        print(f"Report failed: {e}", file=sys.stderr)
        return 1
    if args.show_context:
        print("=" * 60 + "\nCONTEXT GIVEN TO THE MODEL\n" + "=" * 60)
        print("Lexicon terms + flagged suspects (in the extraction prompt):")
        print(pipeline.last_context or "  (none: HANDOFF_USE_LEXICON=0)")
        rag = engines.textbook_rag
        if rag is None:
            print("\nTextbook guidance: off (HANDOFF_USE_TEXTBOOK_RAG=0)")
        elif not rag.available:
            print(f"\nTextbook guidance: no vector store at {settings.textbook_dir} "
                  "(put PDFs in dataset/ and run: python -m ems_handoff.embed_documents)")
        else:
            print("\nTextbook excerpts (in the guidance prompt):")
            print(pipeline.last_reference or "  (store present, nothing relevant found)")
        print("=" * 60 + "\n")
    path = save_report(settings.output_dir, report)
    print(json.dumps(report, indent=2) if args.json else summarize(report))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
