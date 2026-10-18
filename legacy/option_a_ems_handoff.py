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

from openai import OpenAI
import json
import os
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import webbrowser
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from typing import Optional

# =============================================================================
# SAMPLE EMS RADIO REPORTS (What paramedics actually say to hospitals)
# =============================================================================

SAMPLE_STEMI = """
Unit 41 to Memorial Hospital, how do you copy?

We're inbound with a 67-year-old male, chief complaint chest pain radiating to left arm. 

Patient was at home watching TV when symptoms started approximately 45 minutes ago. 
Wife called 911 when he became diaphoretic and short of breath.

Vitals on scene: BP 158/92, heart rate 98 and irregular, respiratory rate 22, 
SpO2 94% on room air, now 98% on 4 liters nasal cannula.

12-lead shows ST elevation in leads V1 through V4, consistent with anterior STEMI.

Patient has history of hypertension and type 2 diabetes. Takes metformin and 
lisinopril daily. No known allergies.

We've established an 18-gauge IV left AC, administered 324mg aspirin chewed, 
and 0.4mg sublingual nitro times two with some relief. Currently running a saline lock.

Patient is alert and oriented, pain currently 6 out of 10, down from 9 out of 10.

Our ETA to your facility is approximately 8 minutes. Requesting cath lab activation.

Unit 41 clear.
"""

SAMPLE_TRAUMA = """
Medic 7 to County General trauma, priority traffic.

Inbound with a 34-year-old female, restrained driver, high-speed MVC with rollover. 
Positive loss of consciousness at scene, approximately 2 minutes per bystanders.

Patient was extricated by fire, c-spine immobilized, on backboard.

Vitals: BP 88/60, heart rate 124, respiratory rate 28 and shallow, 
SpO2 91% on 15 liters non-rebreather.

GCS is 13, eyes 3, verbal 4, motor 6. Pupils equal and reactive. 
Patient confused but following commands.

Visible injuries include a large scalp laceration with controlled bleeding, 
deformity to right forearm likely fracture, and she's guarding her abdomen 
with bruising across lower quadrants consistent with seatbelt sign.

Two large bore IVs established, running wide open. No medications administered. 
Bilateral breath sounds present but diminished on right.

Patient is a visitor from out of state, no medical history available, 
no medications known, allergy status unknown.

ETA 6 minutes, requesting trauma activation.
"""

SAMPLE_SEPSIS = """
Squad 12 to Riverside Emergency.

We have an 82-year-old female from Sunrise Nursing Facility, 
chief complaint altered mental status and fever.

Per nursing staff, patient was in her usual state of dementia baseline 
until this morning when she became increasingly confused, not recognizing staff, 
and had a temperature of 102.4.

Vitals: BP 92/58, heart rate 108, respiratory rate 24, SpO2 93% on 2 liters, 
blood glucose 186.

Patient is oriented to name only, not following commands consistently. 
Skin is warm and flushed. Lungs have coarse rhonchi bilateral bases.

History per facility transfer sheet: Alzheimer's dementia, CHF, 
atrial fibrillation, chronic kidney disease stage 3. 

Medications include metoprolol, furosemide, donepezil, and eliquis. 
Allergies to penicillin and sulfa drugs.

She has a foley catheter in place, urine appears cloudy.

IV established 20-gauge right hand, 500ml bolus initiated. No other interventions.

Patient is DNR but full treatment per POLST form, which we have with us.

ETA 12 minutes.
"""

# =============================================================================
# AI PROMPT FOR STRUCTURED EXTRACTION
# =============================================================================

EXTRACTION_PROMPT = """You are HANDOFF.AI, an AI system that transforms EMS paramedic radio reports into structured hospital handoff data.

The following is a paramedic's verbal radio report to a hospital while transporting a patient. Extract ALL relevant information and structure it for the receiving Emergency Department.

EMS RADIO REPORT:
{ems_report}

Generate a JSON response with this EXACT structure (return ONLY valid JSON, no markdown):
{{
  "alert_type": "STEMI" | "STROKE" | "TRAUMA" | "SEPSIS" | "CARDIAC_ARREST" | "PEDIATRIC" | "OBSTETRIC" | "MEDICAL" | "PSYCHIATRIC",
  "alert_level": "RED" | "YELLOW" | "GREEN",
  "eta_minutes": number,
  
  "patient": {{
    "age": number,
    "age_unit": "years" | "months",
    "sex": "Male" | "Female" | "Unknown"
  }},
  
  "chief_complaint": "Brief one-line summary",
  "mechanism_or_onset": "How/when this started",
  
  "vital_signs": {{
    "blood_pressure": "systolic/diastolic or null",
    "heart_rate": "number or null",
    "respiratory_rate": "number or null", 
    "spo2": "percentage and oxygen delivery method",
    "temperature": "if mentioned",
    "blood_glucose": "if mentioned",
    "gcs": "total and breakdown if mentioned"
  }},
  
  "assessment_findings": {{
    "level_of_consciousness": "Alert/Verbal/Pain/Unresponsive and orientation",
    "airway": "Patent/Compromised/Secured",
    "breathing": "Description of respiratory effort and lung sounds",
    "circulation": "Skin signs, pulses, bleeding",
    "neuro": "Pupils, movement, deficits",
    "other_findings": ["List of other significant findings"]
  }},
  
  "ems_interventions": [
    {{
      "intervention": "What was done",
      "details": "Specifics (dose, route, size, etc.)",
      "response": "Patient response if mentioned"
    }}
  ],
  
  "history": {{
    "past_medical": ["List of conditions"],
    "medications": ["List of medications"],
    "allergies": ["List or NKDA"],
    "code_status": "Full code / DNR / DNI / Other if mentioned"
  }},
  
  "hospital_recommendations": {{
    "activation_requested": "What the paramedic requested (cath lab, trauma team, etc.)",
    "suggested_destination": "Cath Lab" | "Trauma Bay" | "Resuscitation Bay" | "Critical Care" | "General ED",
    "resources_to_prepare": ["List specific resources, teams, equipment"],
    "time_critical_actions": ["Actions to initiate before patient arrival"]
  }},
  
  "handoff_summary": {{
    "one_liner": "Single sentence for quick verbal handoff",
    "key_concerns": ["Top 2-3 clinical concerns"],
    "pending_needs": ["What patient will need immediately on arrival"]
  }},
  
  "data_quality": {{
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "missing_info": ["Important information not provided"],
    "assumptions": ["Any assumptions made from context"]
  }}
}}"""


# =============================================================================
# MAIN HANDOFF CLASS
# =============================================================================

class HandoffAI:
    """
    Transforms EMS radio reports into structured hospital handoff data.
    
    The Problem We Solve:
    - Paramedics give verbal reports that are hard to capture
    - Hospital staff miss critical info, have to ask repeat questions
    - Time-critical patients (STEMI, Stroke, Trauma) need instant prep
    
    Our Solution:
    - AI listens to paramedic's natural speech
    - Extracts structured data (vitals, history, interventions)
    - Sends actionable report to hospital BEFORE patient arrives
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "ollama"):
        """
        Initialize with your choice of FREE AI provider.
        
        Args:
            api_key: API key (not needed for Ollama)
            provider: "ollama" (local, free) | "groq" (cloud, free tier) | "openrouter" (cloud, some free models)
        """
        self.provider = provider
        
        if provider == "ollama":
            # Ollama runs locally - completely FREE!
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't need a real key
            )
            # Use smaller 1B model as it uses less RAM and runs on all laptops
            self.model = "llama3.2:1b"
            
        elif provider == "groq":
            # Groq - FREE tier available at console.groq.com
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.getenv("GROQ_API_KEY")
            )
            self.model = "llama-3.3-70b-versatile"  # Free on Groq
            
        elif provider == "openrouter":
            # OpenRouter - some free models available
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key or os.getenv("OPENROUTER_API_KEY")
            )
            self.model = "meta-llama/llama-3.2-3b-instruct:free"  # Free model
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'groq', or 'openrouter'")
    
    def process_ems_report(self, ems_report: str) -> dict:
        """
        Process a paramedic's verbal radio report and extract structured data.
        
        Args:
            ems_report: The raw text of what the paramedic said on the radio
            
        Returns:
            Structured handoff data for the hospital
        """
        print("\n[*] Processing EMS report...")
        print(f"   Using: {self.provider.upper()} - {self.model}")
        print("-" * 50)
        
        try:
            # Call AI to extract structured data (OpenAI-compatible API)
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(ems_report=ems_report)
                }]
            )
            
            # Parse the JSON response
            json_text = response.choices[0].message.content
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            handoff_data = json.loads(json_text)
            
            return handoff_data
            
        except Exception as e:
            error_type = type(e).__name__
            
            # Provide helpful error messages based on provider
            if "Connection" in error_type or "ConnectError" in str(e):
                print("\n" + "="*60)
                print("[X] CONNECTION ERROR")
                print("="*60)
                
                if self.provider == "ollama":
                    print("\n[!]  Ollama is not running or not installed.\n")
                    print("Quick Fix Options:\n")
                    print("Option 1: Install Ollama (Local - Free)")
                    print("  1. Download from: https://ollama.ai/download")
                    print("  2. Install it")
                    print("  3. Run: ollama pull llama3.2")
                    print("  4. Ollama should auto-start (or run: ollama serve)")
                    print("  5. Re-run this script\n")
                    
                    print("Option 2: Use Groq Instead (Cloud - Free)")
                    print("  1. Get API key from: https://console.groq.com")
                    print("  2. Re-run this script")
                    print("  3. Choose option 2 (Groq)")
                    print("  4. Paste your API key\n")
                    
                elif self.provider == "groq":
                    print("\n[!]  Cannot connect to Groq API.\n")
                    print("Possible issues:")
                    print("  - Check your internet connection")
                    print("  - Verify your API key is correct")
                    print("  - Groq service might be down (check status.groq.com)\n")
                    
                elif self.provider == "openrouter":
                    print("\n[!]  Cannot connect to OpenRouter API.\n")
                    print("Possible issues:")
                    print("  - Check your internet connection")
                    print("  - Verify your API key is correct")
                    print("  - OpenRouter service might be down\n")
                    
                print("="*60)
                
            elif "API" in error_type or "Auth" in error_type:
                print("\n[X] API Authentication Error")
                print("Your API key might be invalid or expired.")
                print(f"Provider: {self.provider}")
                print("\nGet a new API key:")
                if self.provider == "groq":
                    print("  https://console.groq.com")
                elif self.provider == "openrouter":
                    print("  https://openrouter.ai/keys")
                    
            else:
                print(f"\n[X] Unexpected error: {error_type}")
                print(f"Details: {str(e)}")
            
            raise




    def start_web_ui(self):
        """Start the Flask server for the Web GUI."""
        app = Flask(__name__)
        CORS(app)
        
        # Shared state
        self.web_state = {
            "is_recording": False,
            "transcript": "",
            "status": "ready", # ready, recording, confirmed
            "final_file": None
        }
        
        @app.route("/")
        def index():
            return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMS Handoff - Live Mode</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden; transition: background-color 0.5s; }
        .container { background-color: #1e1e1e; padding: 2rem; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); width: 80%; max-width: 800px; text-align: center; border: 1px solid #333; position: relative; z-index: 10; }
        h1 { color: #fff; margin-bottom: 0.5rem; font-weight: 300; letter-spacing: 1px; }
        .status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1px; }
        .status-ready { background-color: #333; color: #aaa; }
        .status-recording { background-color: #cf222e; color: white; animation: pulse 1.5s infinite; }
        .status-confirmed { background-color: #238636; color: white; }
        
        .script-box { background-color: #252525; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #58a6ff; text-align: left; margin-bottom: 1.5rem; font-size: 1.1rem; line-height: 1.6; color: #fff; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); }
        .transcript-box { background-color: #0d1117; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; height: 150px; overflow-y: auto; text-align: left; font-family: 'Consolas', monospace; color: #8b949e; margin-bottom: 2rem; transition: all 0.3s; }
        .transcript-box.active { border-color: #58a6ff; color: #c9d1d9; }
        .transcript-box .new-text { color: #fff; font-weight: bold; }
        
        .controls { display: flex; gap: 1rem; justify-content: center; }
        button { padding: 0.8rem 2rem; border: none; border-radius: 6px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-record { background-color: #238636; color: white; }
        .btn-record:hover { background-color: #2ea043; transform: translateY(-2px); }
        .btn-stop { background-color: #da3633; color: white; }
        .btn-stop:hover { background-color: #f85149; }
        
        .instruction { color: #8b949e; font-size: 0.9rem; margin-top: 1rem; }
        .magic-word { color: #58a6ff; font-weight: bold; }

        /* Flash Animation for Confirmation */
        @keyframes flashRed {
            0% { background-color: #121212; }
            50% { background-color: #8a1c1c; }
            100% { background-color: #121212; }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(207, 34, 46, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(207, 34, 46, 0); }
            100% { box-shadow: 0 0 0 0 rgba(207, 34, 46, 0); }
        }
        
        .flashing { animation: flashRed 2s ease-in-out infinite; }
        .confirmed-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(35, 134, 54, 0.95); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 100; opacity: 0; pointer-events: none; transition: opacity 0.5s; }
        .confirmed-overlay.show { opacity: 1; pointer-events: auto; }
        .confirmed-text { color: white; font-size: 3rem; font-weight: bold; margin-bottom: 1rem; text-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    </style>
</head>
<body>
    <div id="flash-layer"></div>
    <div class="confirmed-overlay" id="confirmOverlay">
        <div class="confirmed-text">REPORT CONFIRMED</div>
        <div style="color: #fff; font-size: 1.2rem;">Generating Handoff...</div>
    </div>

    <div class="container">
        <h1>EMS HANDOFF AI</h1>
        <div id="status-badge" class="status-badge status-ready">Ready</div>
        
        <div class="script-box">
            <strong>📄 READ THIS SCRIPT:</strong><br><br>
            "Medic 21 to General Hospital.<br>
            We are inbound with a 45-year-old male found down.<br>
            GCS is 14. History of Seizures.<br>
            Vitals: BP 140 over 90, Heart Rate 110, SpO2 98%.<br>
            We have established an IV and given 5mg Versed.<br>
            ETA is 5 minutes."
        </div>
        
        <div class="transcript-box active" id="transcript">
            (Transcript will appear here...)
        </div>
        
        <div class="controls">
            <button id="btnAction" class="btn-record" onclick="toggleRecording()">START RECORDING</button>
        </div>
        
        <p class="instruction">
            Say <span class="magic-word">"CONFIRM REPORT READY"</span> to send automatically.
        </p>
    </div>

    <script>
        let isRecording = false;
        let pollInterval;

        async function toggleRecording() {
            const btn = document.getElementById('btnAction');
            const status = document.getElementById('status-badge');
            
            if (!isRecording) {
                // Start
                await fetch('/start_recording');
                isRecording = true;
                btn.textContent = "STOP RECORDING";
                btn.className = "btn-stop";
                status.textContent = "🔴 RECORDING";
                status.className = "status-badge status-recording";
                startPolling();
            } else {
                // Stop manually
                await fetch('/stop_recording');
                isRecording = false;
                btn.textContent = "START RECORDING";
                btn.className = "btn-record";
                status.textContent = "Ready";
                status.className = "status-badge status-ready";
                stopPolling();
            }
        }

        function startPolling() {
            pollInterval = setInterval(async () => {
                const response = await fetch('/get_state');
                const data = await response.json();
                
                // Update transcript
                const textBox = document.getElementById('transcript');
                if(data.transcript) {
                    textBox.innerText = data.transcript;
                    textBox.scrollTop = textBox.scrollHeight;
                }
                
                // Check status
                if (data.status === 'confirmed') {
                    stopPolling();
                    document.body.className = "flashing"; // Red flash on body if needed, or overlay
                    document.getElementById('confirmOverlay').className = "confirmed-overlay show";
                    setTimeout(() => {
                        window.close(); // Optional: close window after delay
                    }, 3000);
                }
            }, 1000);
        }

        function stopPolling() {
            clearInterval(pollInterval);
        }
    </script>
</body>
</html>
"""

        @app.route("/start_recording")
        def start_recording():
            if not self.web_state["is_recording"]:
                self.web_state["is_recording"] = True
                self.web_state["status"] = "recording"
                self.web_state["transcript"] = ""
                # Start recording thread
                threading.Thread(target=self._recording_worker).start()
            return jsonify({"status": "started"})

        @app.route("/stop_recording")
        def stop_recording():
            self.web_state["is_recording"] = False
            self.web_state["status"] = "ready"
            return jsonify({"status": "stopped"})

        @app.route("/get_state")
        def get_state():
            return jsonify(self.web_state)
            
        # Run Flask in a separate thread
        print("[WEB] Starting local content server...")
        threading.Thread(target=lambda: app.run(port=5000, use_reloader=False), daemon=True).start()
        
        # Open Browser
        time.sleep(1)
        webbrowser.open("http://localhost:5000")
        
        # Block main thread until we have a file or user exits
        print("[WEB] Waiting for recording from Web UI...")
        try:
            while True:
                if self.web_state["status"] == "confirmed" and self.web_state["final_file"]:
                    return self.web_state["final_file"]
                time.sleep(0.5)
        except KeyboardInterrupt:
            return None

    def _recording_worker(self):
        """Background thread to handle audio recording loop."""
        threshold_text = "confirm report"
        sample_rate = 16000
        chunk_duration = 3.0
        
        full_audio_data = np.array([], dtype=np.float32)
        
        try:
            while self.web_state["is_recording"]:
                # Record chunk
                chunk = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1)
                sd.wait()
                
                # Append
                chunk_flat = chunk.flatten()
                full_audio_data = np.concatenate((full_audio_data, chunk_flat))
                
                # Check keyword
                check_duration = 6.0
                samples_to_check = int(check_duration * sample_rate)
                if len(full_audio_data) > samples_to_check:
                    check_data = full_audio_data[-samples_to_check:]
                else:
                    check_data = full_audio_data
                
                wav.write("temp_check.wav", sample_rate, check_data)
                
                # Transcribe
                new_text = self.transcribe_audio_chunk("temp_check.wav")
                
                if new_text:
                    # Append to transcript for UI (simplistic)
                    # Ideally we wouldn't just append, but for demo this shows activity
                    if len(self.web_state["transcript"]) == 0:
                         self.web_state["transcript"] = new_text
                    elif len(new_text) > 3: # Avoid noise
                         self.web_state["transcript"] += f" {new_text}"
                    
                    # Check keyword
                    if threshold_text.lower() in new_text.lower() or "ready to send" in new_text.lower():
                        self.web_state["status"] = "confirmed"
                        self.web_state["is_recording"] = False
                        break
                        
            # Save final
            final_filename = "final_recording.wav"
            wav.write(final_filename, sample_rate, full_audio_data)
            self.web_state["final_file"] = final_filename
            print("[REC] Recording finished.")
            
        except Exception as e:
            print(f"[REC] Error: {e}")
            self.web_state["is_recording"] = False

    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def transcribe_audio_chunk(self, filename):
        """Helper to transcribe a small chunk for keyword detection"""
        try:
            with open(filename, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3-turbo",  # UPDATED MODEL
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
            return transcription.text.strip()
        except:
            return ""

    def transcribe_audio_final(self, filename):
        """Transcribe the full audio file"""
        print("\n[CLIPBOARD] Transcribing full report with Whisper...")
        try:
            with open(filename, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3-turbo", # UPDATED MODEL
                    prompt="The following is a paramedic radio report. Terminology: BP, HR, SpO2, STEMI, GCS, IV.",
                    response_format="json",
                    language="en"
                )
            return transcription.text
        except Exception as e:
            print(f"[X] Transcription failed: {e}")
            return None
    
    def display_hospital_view(self, data: dict):
        """
        Display what the hospital ER would see on their dashboard.
        This is the OUTPUT that makes us valuable.
        """
        alert_colors = {
            "RED": "\033[91m",      # Red
            "YELLOW": "\033[93m",   # Yellow  
            "GREEN": "\033[92m",    # Green
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        color = alert_colors.get(data.get("alert_level", "GREEN"), "")
        
        print("\n" + "=" * 60)
        print(f"{BOLD}[HOSPITAL] HOSPITAL ER DASHBOARD - INCOMING PATIENT{RESET}")
        print("=" * 60)
        
        # Alert Banner
        print(f"\n{color}{BOLD}[!]  {data.get('alert_type', 'UNKNOWN')} ALERT - {data.get('alert_level', 'UNKNOWN')} PRIORITY{RESET}")
        print(f"{BOLD}ETA: {data.get('eta_minutes', '?')} MINUTES{RESET}")
        
        # One-liner summary
        summary = data.get("handoff_summary", {})
        print(f"\n \"{summary.get('one_liner', 'No summary available')}\"")
        
        # Patient Info
        patient = data.get("patient", {})
        print(f"\n{BOLD}PATIENT:{RESET}")
        print(f"  Age/Sex: {patient.get('age', '?')} {patient.get('age_unit', 'years')} {patient.get('sex', 'Unknown')}")
        print(f"  Chief Complaint: {data.get('chief_complaint', 'Unknown')}")
        print(f"  Mechanism/Onset: {data.get('mechanism_or_onset', 'Unknown')}")
        
        # Vital Signs
        vitals = data.get("vital_signs", {})
        print(f"\n{BOLD}VITAL SIGNS:{RESET}")
        if vitals.get("blood_pressure"):
            print(f"  BP: {vitals['blood_pressure']}")
        if vitals.get("heart_rate"):
            print(f"  HR: {vitals['heart_rate']}")
        if vitals.get("respiratory_rate"):
            print(f"  RR: {vitals['respiratory_rate']}")
        if vitals.get("spo2"):
            print(f"  SpO2: {vitals['spo2']}")
        if vitals.get("gcs"):
            print(f"  GCS: {vitals['gcs']}")
        if vitals.get("temperature"):
            print(f"  Temp: {vitals['temperature']}")
        if vitals.get("blood_glucose"):
            print(f"  BGL: {vitals['blood_glucose']}")
        
        # Assessment
        assessment = data.get("assessment_findings", {})
        print(f"\n{BOLD}ASSESSMENT:{RESET}")
        print(f"  LOC: {assessment.get('level_of_consciousness', 'Unknown')}")
        print(f"  Airway: {assessment.get('airway', 'Unknown')}")
        print(f"  Breathing: {assessment.get('breathing', 'Unknown')}")
        print(f"  Circulation: {assessment.get('circulation', 'Unknown')}")
        if assessment.get("other_findings"):
            print(f"  Other: {', '.join(assessment['other_findings'])}")
        
        # EMS Interventions
        interventions = data.get("ems_interventions", [])
        if interventions:
            print(f"\n{BOLD}EMS INTERVENTIONS:{RESET}")
            for i, item in enumerate(interventions, 1):
                print(f"  {i}. {item.get('intervention', '')} - {item.get('details', '')}")
                if item.get("response"):
                    print(f"     -> Response: {item['response']}")
        
        # History
        history = data.get("history", {})
        print(f"\n{BOLD}HISTORY:{RESET}")
        print(f"  PMH: {', '.join(history.get('past_medical', ['Unknown']))}")
        print(f"  Medications: {', '.join(history.get('medications', ['Unknown']))}")
        print(f"  Allergies: {', '.join(history.get('allergies', ['NKDA']))}")
        print(f"  Code Status: {history.get('code_status', 'Full Code')}")
        
        # Hospital Prep - THE KEY VALUE PROPOSITION
        hospital = data.get("hospital_recommendations", {})
        print(f"\n{color}{BOLD}[!] HOSPITAL PREPARATION:{RESET}")
        print(f"  Requested: {hospital.get('activation_requested', 'None specified')}")
        print(f"  Destination: {hospital.get('suggested_destination', 'General ED')}")
        if hospital.get("resources_to_prepare"):
            print(f"  Resources Needed:")
            for resource in hospital["resources_to_prepare"]:
                print(f"    * {resource}")
        if hospital.get("time_critical_actions"):
            print(f"  {BOLD}TIME-CRITICAL ACTIONS:{RESET}")
            for action in hospital["time_critical_actions"]:
                print(f"    [TIME]  {action}")
        
        # Key Concerns
        if summary.get("key_concerns"):
            print(f"\n{BOLD}[!] KEY CONCERNS:{RESET}")
            for concern in summary["key_concerns"]:
                print(f"  * {concern}")
        
        # Pending Needs
        if summary.get("pending_needs"):
            print(f"\n{BOLD}PENDING NEEDS ON ARRIVAL:{RESET}")
            for need in summary["pending_needs"]:
                print(f"  -> {need}")
        
        # Data Quality
        quality = data.get("data_quality", {})
        print(f"\n{BOLD}DATA QUALITY:{RESET} {quality.get('confidence', 'Unknown')}")
        if quality.get("missing_info"):
            print(f"  Missing: {', '.join(quality['missing_info'])}")
        
        print("\n" + "=" * 60)


# =============================================================================
# DEMO / TESTING
# =============================================================================

def run_demo():
    """Run the demo with sample EMS reports."""
    
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == "win32":
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass  # If it fails, continue anyway
    
    print("""
===============================================================
                      HANDOFF.AI                              
          Option A: EMS -> Hospital Handoff Demo               
===============================================================
 This transforms what paramedics SAY into structured data     
 that hospitals can ACT ON before the patient arrives.        
                                                               
 Now using FREE AI providers!                              
===============================================================
    """)
    
    # Choose your FREE provider
    print("\nSelect AI Provider:\n")
    print("  1. Groq (Cloud - FAST & FREE) [Recommended]")
    print("  2. Ollama (Local - requires install)")
    print("  3. OpenRouter (Cloud)\n")
    
    provider_choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    
    if provider_choice == "1":
        provider = "groq"
                api_key = os.getenv("GROQ_API_KEY")  # revoked key removed; set GROQ_API_KEY in .env
        print("\n[OK] Using Groq cloud API (key from GROQ_API_KEY)")
    elif provider_choice == "2":
        provider = "ollama"
        api_key = None
        print("\n[OK] Using Ollama (local). Make sure Ollama is running: 'ollama serve'")
    elif provider_choice == "3":
        provider = "openrouter"
        api_key = input("Enter your OpenRouter API key (or press Enter to use OPENROUTER_API_KEY env var): ").strip() or None
        print("\n[OK] Using OpenRouter API")
    else:
        provider = "ollama"
        api_key = None
        print("\n[OK] Defaulting to Ollama (local)")
    
    # Initialize
    try:
        handoff = HandoffAI(api_key=api_key, provider=provider)
    except Exception as e:
        print(f"\n[X] Error initializing AI provider: {e}")
        print("\nTroubleshooting:")
        print("  - For Ollama: Install from https://ollama.ai and run 'ollama pull llama3.2:1b'")
        print("  - For Groq: Get free API key from https://console.groq.com")
        print("  - For OpenRouter: Get API key from https://openrouter.ai")
        return
    
    # Menu
    print("Select a sample case to process:\n")
    print("  1. STEMI (Heart Attack) - 67yo male, chest pain, cath lab activation")
    print("  2. TRAUMA (MVC) - 34yo female, rollover crash, trauma activation")
    print("  3. SEPSIS (Medical) - 82yo female, nursing home, altered mental status")
    print("  4. Enter your own EMS report")
    print("  5. [NEW] Record Live Audio (Microphone)")
    print("  6. Exit\n")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        ems_report = SAMPLE_STEMI
        print("\n[RADIO] Loading STEMI case...")
    elif choice == "2":
        ems_report = SAMPLE_TRAUMA
        print("\n[RADIO] Loading TRAUMA case...")
    elif choice == "3":
        ems_report = SAMPLE_SEPSIS
        print("\n[RADIO] Loading SEPSIS case...")
    elif choice == "4":
        print("\nPaste the EMS radio report (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        ems_report = "\n".join(lines)
    elif choice == "5":
        if provider != "groq":
            print("\n[!] Error: Live audio requires Groq provider. Please restart and choose Option 1.")
            return
            
        # Launch Web GUI
        filename = handoff.start_web_ui()
        
        if not filename:
             print("\n[!] No recording captured.")
             return
             
        ems_report = handoff.transcribe_audio_final(filename)
        if not ems_report:
             return
             
        print("\n" + "=" * 60)
        print("[RADIO] FULL TRANSCRIBED REPORT:")
        print("=" * 60)
        print(ems_report)
             
        print("\n" + "=" * 60)
        print("[RADIO] FULL TRANSCRIBED REPORT:")
        print("=" * 60)
        print(ems_report)
        
    elif choice == "6":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Running STEMI demo...")
        ems_report = SAMPLE_STEMI
    
    # Show input
    print("\n" + "=" * 60)
    print("[RADIO] EMS RADIO REPORT (INPUT):")
    print("=" * 60)
    print(ems_report)
    
    # Process
    try:
        handoff_data = handoff.process_ems_report(ems_report)
        
        # Display hospital view
        handoff.display_hospital_view(handoff_data)
        
        # Save JSON to current directory
        output_file = "handoff_report.json"
        with open(output_file, "w") as f:
            json.dump(handoff_data, f, indent=2)
        print(f"\n[SAVE] Full JSON saved to: {output_file}")
        print("\n[OK] Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n[!]  Demo cancelled by user.")
        return
    except Exception as e:
        # Error details already printed by process_ems_report
        print("\n[!]  Demo failed. See error details above.")
        return


if __name__ == "__main__":
    run_demo()
