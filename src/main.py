#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HANDOFF.AI - Option A: EMS to Hospital Handoff Demo
====================================================
Transforms paramedic radio reports into structured hospital handoff data.

This is the CORRECT flow:
  AMBULANCE (Paramedic) --> HANDOFF.AI --> HOSPITAL ER
  
NOT 911 dispatch, but the actual paramedic transporting the patient.
"""

import json
import os
import sys
import logging
import codecs
from dotenv import load_dotenv
from typing import Optional, Tuple

from .handoff import HandoffAI
from .samples import SAMPLE_STEMI, SAMPLE_TRAUMA, SAMPLE_SEPSIS
from .web_ui import start_web_ui
from .transcript_corrector import correct_transcript, assumption_notes

# Load environment variables from a local .env file if present
load_dotenv()

# All artifacts (logs, JSON reports, recordings)
os.makedirs("output", exist_ok=True)

# Root logging configuration for the whole app
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("output/handoff.log", mode='w'),   # overwrite on each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Two output channels are used throughout this module:
#   print(...)  -> user-facing terminal UI. Prefix tags signal the kind of message:
#                  [*] status/info, [!] warning/attention, [X] error,
#                  [OK] success, [SAVE] a file was written.
#   logger.*    -> the audit trail in output/handoff.log (and mirrored to stdout)
# Prints drive the interactive experience; log calls record what happened

MAX_REPORT_LENGTH = 5000
MIN_REPORT_LENGTH = 50

CASES = {
    "1": ("STEMI demo", SAMPLE_STEMI),
    "2": ("TRAUMA demo", SAMPLE_TRAUMA),
    "3": ("SEPSIS demo", SAMPLE_SEPSIS),
}

PROVIDERS = {
    "1": "groq",
    "2": "ollama",
    "3": "openrouter",
}

USE_LEXICON = os.getenv("HANDOFF_USE_LEXICON", "1") != "0"

def _fix_windows_encoding() -> None:
    """Fix UTF-8 encoding on Windows."""
    if sys.platform == "win32":
        try:
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
        except Exception as e:
            logger.warning(f"Could not fix Windows encoding: {e}")

def _prompt_with_validation(prompt: str, validator: callable, retry_msg: str) -> Optional[str]:
    """Generic prompt with validation."""
    while True:
        try:
            value = input(prompt).strip()
            if validator(value):
                return value
            print(retry_msg)
        except (KeyboardInterrupt, EOFError):
            return None

def _select_provider() -> Tuple[str, Optional[str]]:
    """Interactive provider selection."""
    print("\nSelect AI Provider:\n")
    print("  1. Groq (Cloud - FAST & FREE) [Recommended]")
    print("  2. Ollama (Local - requires install)")
    print("  3. OpenRouter (Cloud)\n")

    choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    
    if choice not in PROVIDERS:
        print("[!] Invalid choice. Defaulting to Groq.")
        choice = "1"
    
    provider = PROVIDERS[choice]
    api_key = None

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            print("\n[!] GROQ_API_KEY not set in environment.")
            print("    Options: export GROQ_API_KEY='gsk_...' or create .env file\n")
            api_key = input("Paste your Groq API key (or press Enter to skip): ").strip()
            if not api_key:
                print("[!] No API key provided.")
                return _select_provider()
            if not api_key.startswith("gsk_"):
                print(f"[!] Warning: Groq keys usually start with 'gsk_', yours starts with '{api_key[:10]}...'")
                if input("Continue anyway? (y/n): ").strip().lower() != 'y':
                    return _select_provider()
        logger.info(f"Using Groq with key: {api_key[:20]}...")
        
    elif provider == "ollama":
        print("\n[*] Using Ollama (Local)")
        print("    Make sure Ollama is running: 'ollama serve'")
        print("    Model: qwen3:8b")
        input("    Press Enter to continue...")
        logger.info("Using Ollama (local)")
        
    else:  # openrouter
        print("\n[*] Using OpenRouter")
        api_key = input("Paste your OpenRouter API key: ").strip()
        if not api_key:
            print("[!] No key provided. Try again.")
            return _select_provider()
        logger.info(f"Using OpenRouter with key: {api_key[:20]}...")

    return provider, api_key

def _prompt_custom_report() -> Optional[str]:
    """Prompt user to paste their own EMS report with validation."""
    print("\n[*] Paste your EMS radio report (max 5000 characters)")
    print("    Press Enter twice when done:\n")
    
    lines = []
    blank_count = 0
    
    while blank_count < 2:
        try:
            line = input()
            if line == "":
                blank_count += 1
            else:
                blank_count = 0
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            return None
    
    report = "\n".join(lines).strip()

    # Validation
    if len(report) < MIN_REPORT_LENGTH:
        print(f"[!] Report too short ({len(report)} chars, min {MIN_REPORT_LENGTH}).")
        if input("    Try again? (y/n): ").strip().lower() == 'y':
            return _prompt_custom_report()
        return None
    
    if len(report) > MAX_REPORT_LENGTH:
        print(f"[!] Report too long ({len(report)} chars, max {MAX_REPORT_LENGTH}).")
        print("    First 50 chars: " + report[:50])
        if input("    Try again with shorter text? (y/n): ").strip().lower() == 'y':
            return _prompt_custom_report()
        return None
    
    logger.info(f"Custom report entered: {len(report)} characters")
    return report

def _record_live_audio(handoff: HandoffAI) -> Optional[str]:
    """Record live audio via web UI. Transcription runs locally via
    faster-whisper, so this works regardless of which LLM provider is selected."""
    logger.info("Starting live audio recording")
    # Hand the chunk-transcriber to the web UI until a recording is captured
    filename = start_web_ui(transcribe_chunk_fn=handoff.transcribe_audio_chunk)

    # No file means the user cancelled or recording failed
    if not filename: 
        print("\n[!] No recording captured.")
        logger.warning("Live audio recording failed or cancelled")
        return None
    
    # Re-transcribe the complete recording
    print("\n[*] Transcribing full audio...")
    ems_report = handoff.transcribe_audio_final(filename)
    
    if not ems_report:
        print("\n[!] Transcription failed.")
        logger.error("Audio transcription returned None")
        return None
    
    print("\n" + "="*60)
    print("TRANSCRIBED REPORT:")
    print("="*60)
    print(ems_report)
    print("\n")
    return ems_report

def _select_case(handoff: HandoffAI) -> Optional[str]:
    """Interactive case selection."""
    print("Select a sample case to process:\n")
    print("  1. STEMI (Heart Attack) - 67yo male, chest pain, cath lab activation")
    print("  2. TRAUMA (MVC) - 34yo female, rollover crash, trauma activation")
    print("  3. SEPSIS (Medical) - 82yo female, nursing home, altered mental status")
    print("  4. Enter your own EMS report")
    print("  5. Record Live Audio (Microphone)")
    print("  6. Exit\n")

    choice = input("Enter choice (1-6, default=1): ").strip() or "1"

    if choice in CASES:
        logger.info(f"Selected: {CASES[choice][0]}")
        return CASES[choice][1]
    elif choice == "4":
        return _prompt_custom_report()
    elif choice == "5":
        return _record_live_audio(handoff)
    elif choice == "6":
        print("\n[OK] Goodbye!")
        return None
    else:
        print("\n[!] Invalid choice. Defaulting to STEMI.")
        logger.info("Selected: STEMI demo (default)")
        return SAMPLE_STEMI

def process_case(handoff: HandoffAI, ems_report: str) -> bool:
    """Process a single EMS report. Returns True on success, else False."""
    fix = correct_transcript(ems_report)
    corrected = fix.text

    for c in fix.corrections:
        print(f"[*] Corrected: {c.original} -> {c.replacement}")
    for fl in fix.flagged:
        print(f"[?] Possible mis-hearing: {fl.token} -> {fl.best_guess} "
              f"({fl.canonical})? — left in place for the LLM to judge")

    n = len(fix.corrections)
    header = (f"CORRECTED REPORT ({n} fix{'es' if n != 1 else ''} applied)"
              if n else "EMS RADIO REPORT (INPUT)")

    print("\n" + "=" * 60)
    print(header + ":")
    print("=" * 60)
    print(corrected)

    try:
        handoff_data = handoff.process_ems_report(corrected, suspects=fix.flagged)

        # Surface each correction as an assumption the ED can veto.
        notes = assumption_notes(fix)
        if notes:
            handoff_data.setdefault("data_quality", {}) \
                        .setdefault("assumptions", []).extend(notes)

        guidance = handoff.suggest_courses_of_action(handoff_data)
        if guidance:
            handoff_data["clinical_guidance"] = guidance

        handoff.display_hospital_view(handoff_data)
        if guidance:
            handoff.display_courses_of_action(guidance)

        output_file = "output/handoff_report.json"
        with open(output_file, "w") as f:
            json.dump(handoff_data, f, indent=2)
        print(f"\n[SAVE] Full JSON: {output_file}")
        logger.info(f"Report saved to {output_file}")
        return True

    except json.JSONDecodeError as e:
        print(f"\n[!] LLM returned invalid JSON: {e}")
        logger.error(f"JSON decode error: {e}")
        return False

    except KeyboardInterrupt:
        print("\n\n[!] Cancelled by user.")
        logger.info("User cancelled processing")
        return False
    except Exception as e:
        print(f"\n[!] Unexpected error: {type(e).__name__}: {e}")
        logger.exception("Unexpected error during processing")
        return False

def main() -> None:
    """Main loop with retry and loop-back."""
    _fix_windows_encoding()
 
    print("""
        ╔═══════════════════════════════════════════════════════════════╗
        ║                       HANDOFF.AI v2                           ║       
        ║           Option A: EMS -> Hospital Handoff Demo              ║               
        ║═══════════════════════════════════════════════════════════════║
        ║   This transforms what paramedics SAY into structured data    ║ 
        ║   that hospitals can ACT ON before the patient arrives.       ║ 
        ║                                                               ║
        ║   Now using FREE AI providers!                                ║
        ╚═══════════════════════════════════════════════════════════════╝
    """)
 
    # Initialize the AI provider
    handoff = None
    for attempt in range(3):
        try: 
            provider, api_key = _select_provider()
            handoff = HandoffAI(api_key=api_key, provider=provider, use_lexicon=USE_LEXICON)
            logger.info(f"Provider initialized: {provider}")
            break
        except Exception as e: 
            attempt_num = attempt + 1
            print(f"\n[!] Provider setup failed (attempt {attempt_num}/3)")
            logger.error(f"Provider setup failed: {e}")
            
            if attempt_num < 3 and input("Try a different provider? (y/n): ").strip().lower() == 'y':
                continue
            print("\n[!] Max attempts reached. Exiting.")
            return
    
    if not handoff:
        return
            
    # Main loop
    while True: 
        ems_report = _select_case(handoff)
        if ems_report is None: 
            break

        success = process_case(handoff, ems_report)
        prompt = "Process another case?" if success else "Try again?"
        
        print("\n" + "="*60)
        if input(f"{prompt} (y/n): ").strip().lower() != 'y':
            print("[OK] Thank you for using HANDOFF.AI")
            logger.info("User exited main loop")
            break 

    print("\n Handoff.AI over.\n")

if __name__ == "__main__":
    main()