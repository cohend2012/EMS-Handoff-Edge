"""
ems_handoff — HANDOFF.AI: paramedic radio report -> structured hospital handoff.

Start the app:      python -m ems_handoff
Check the setup:    python -m ems_handoff --check
Headless text run:  python -m ems_handoff.cli --case stemi

Module map
    config.py               every setting (from .env / environment)
    server.py               local web server + state machine behind the app window
    web/                    the app screens (HTML/CSS/JS, no internet needed)
    audio/                  mic capture, live captions, stop-phrase spotter, live session
    transcribe.py           the final, accurate transcript (Groq or local Whisper)
    pipeline.py             transcript -> validated handoff report
    llm.py                  Groq / Ollama / OpenRouter client
    phrase.py               stop-phrase matching
    transcript_corrector.py fixes known speech-recognition mistakes in drug names
    lexicon.py, lexicon_rag.py   EMS terminology and prompt context
    textbook_rag.py, vector_store.py, embed_documents.py   optional textbook guidance
    prompt.py, samples.py   prompts and sample radio reports
    doctor.py               setup checks (python -m ems_handoff --check)
    cli.py                  headless command line
"""

__version__ = "2.0.0"
