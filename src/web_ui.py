import time
import threading
import webbrowser
import tempfile
import os
import shutil
import atexit
import logging 

from typing import Optional, Callable
from difflib import SequenceMatcher
 
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from flask import Flask, jsonify
from flask_cors import CORS
from typing import Optional

_INDEX_HTML = """
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

TIMEOUT_SECONDS = 10 * 60
WARNING_THRESHOLD = 3 * 60
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0
MIN_AUDIO_LEVEL = 0.05
KEYWORD_WINDOW_SECONDS = 6.0

logger = logging.getLogger(__name__)

class RecordingManager:
    """Manages audio recording with timeout and cleanup."""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="handoff_")
        self.files_to_cleanup = []
        logger.info(f"Created temp directory: {self.temp_dir}")
        atexit.register(self.cleanup)
    
    def create_temp_file(self, suffix: str = ".wav") -> str:
        """Create a temp file path."""
        filepath = os.path.join(self.temp_dir, f"recording{suffix}")
        self.files_to_cleanup.append(filepath)
        return filepath
    
    def save_final(self, final_file: Optional[str]) -> None:
        """Save final recording to output folder."""
        if not final_file or not os.path.exists(final_file):
            return
        
        try:
            # Copy the temp recording into the persistent output folder
            os.makedirs("output", exist_ok=True)
            output_filename = "output/final_recording.wav"
            shutil.copy(final_file, output_filename)
            logger.info(f"Saved final recording to: {output_filename}")
            print(f"[SAVE] Recording: {output_filename}")
        except Exception as e:
            logger.warning(f"Could not save to output/: {e}")
    
    def cleanup(self, final_file: Optional[str] = None) -> None: 
        """Clean up all temp files except the final recording."""
        for filepath in self.files_to_cleanup: 
            if final_file and filepath == final_file:
                continue
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Cleaned up: {filepath}")
            except Exception as e: 
                logger.warning(f"Could not clean up {filepath}: {e}")

        try: 
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e: 
            logger.warning(f"Could not clean up temp directory: {e}")

class KeywordDetector: 
    """Detects confirmation keyword with fuzzy matching."""

    def __init__(self, keyword: str = "confirm report ready", threshold: float = 0.85):
        """
        Initialize keyword detector. 

        Args: 
            keyword: phrase to detect
            threshold: similarity threshold 
        """
        self.keywords = [
            keyword.lower(),
            "confirm report",
            "report ready"
        ]
        self.threshold = threshold

    def _matches(self, phrase: str, text: str) -> bool:
        """Check if phrase matches text (exact or fuzzy)."""
        if phrase in text:
            return True
        similarity = SequenceMatcher(None, phrase, text).ratio()
        return similarity > self.threshold

    def detect(self, text: str) -> bool: 
        """
        Detect if text contains confirmation keyword (with fuzzy matching).

        Args: 
            text: transcribed text to check 

        Returns: 
            True or False 
        """
        text = text.lower()
        
        for keyword in self.keywords:
            if self._matches(keyword, text):
                logger.info(f"Keyword detected: {keyword}")
                return True
        
        return False
    
class AudioValidator: 
    """Validates audio quality during recording."""
    @staticmethod
    def get_rms_level(audio_chunk: np.ndarray) -> float:
        """Calculate RMS (root mean square) level of audio chunk."""
        return float(np.sqrt(np.mean(audio_chunk**2)))
    
    @staticmethod
    def is_silent(audio_chunk: np.ndarray, threshold: float = MIN_AUDIO_LEVEL) -> bool: 
        """Check if audio chunk is essentially silent."""
        rms = AudioValidator.get_rms_level(audio_chunk)
        return rms < threshold

recording_manager = RecordingManager()

class RecordingState:
    """Manages state of audio recording session."""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear all fields back to a fresh pre-recording state.

        Called at the start of every start_web_ui() run so a second case
        does not inherit 'confirmed' status or the previous final_file.
        """
        self.is_recording = False
        self.transcript = ""
        self.status = "ready"  # ready, recording, confirmed
        self.final_file = None
        self.audio_level = 0.0
        self.start_time = None

def _run_recording_worker(state, transcribe_chunk_fn) -> None:
    """
    Background thread: record continuously, transcribe periodically.
 
    Audio capture and transcription are decoupled. The InputStream callback
    appends to `frames` on PortAudio's thread; the loop below snapshots that
    buffer every CHUNK_DURATION seconds to check for the stop keyword. A slow
    transcription delays the next *check*, never the *recording*.
    """
    frames = []                     
    lock = threading.Lock()
    silent_checks = 0
    mic_issue_warned = False
 
    def on_audio(indata, n_frames, time_info, status):
        if status:
            logger.debug(f"Audio stream status: {status}")
        with lock:
            frames.append(indata.copy())
 
    def snapshot() -> np.ndarray:
        with lock:
            if not frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(frames).flatten()
 
    try:
        logger.info("Recording started (continuous stream)")
        keyword_detector = KeywordDetector()         
 
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=on_audio,
        ):
            next_check = time.time() + CHUNK_DURATION
 
            while state.is_recording:
                # Poll cheaply so a stop request is honored promptly
                time.sleep(0.1)
 
                if state.start_time and time.time() - state.start_time > TIMEOUT_SECONDS:
                    logger.warning("Recording timeout - auto-stopping")
                    state.is_recording = False
                    break
 
                if time.time() < next_check:
                    continue
                next_check = time.time() + CHUNK_DURATION
 
                audio = snapshot()
                if audio.size == 0:
                    continue
 
                recent = audio[-int(CHUNK_DURATION * SAMPLE_RATE):]
                level = float(np.sqrt(np.mean(recent ** 2))) if recent.size else 0.0
                state.audio_level = level
                if level < MIN_AUDIO_LEVEL:
                    silent_checks += 1
                    if silent_checks > 2 and not mic_issue_warned:
                        logger.warning("Audio level very low - check microphone")
                        mic_issue_warned = True
                else:
                    silent_checks = 0
 
                # keyword detection on a trailing window 
                window = int(KEYWORD_WINDOW_SECONDS * SAMPLE_RATE)
                check_data = audio[-window:] if audio.size > window else audio
 
                temp_check = recording_manager.create_temp_file("_check.wav")  
                wav.write(temp_check, SAMPLE_RATE, check_data)
 
                new_text = transcribe_chunk_fn(temp_check)   # stream keeps running
                if not new_text:
                    continue
 
                if len(state.transcript) == 0:
                    state.transcript = new_text
                elif len(new_text) > 3:
                    state.transcript += f" {new_text}"
 
                if keyword_detector.detect(new_text):
                    logger.info("Confirmation keyword detected - ending recording")
                    state.status = "confirmed"
                    state.is_recording = False
                    break
 
        # write the complete, gap-free audio stream
        final_audio = snapshot()
        duration = final_audio.size / SAMPLE_RATE
        final_filename = recording_manager.create_temp_file("_final.wav")  
        wav.write(final_filename, SAMPLE_RATE, final_audio)
        state.final_file = final_filename
        logger.info(f"Recording saved: {final_filename} ({duration:.1f}s captured)")
        print(f"[REC] Recording finished ({duration:.1f}s).")
 
    except Exception as e:
        logger.error(f"Recording error: {e}")
        print(f"[REC] Error: {e}")
        state.is_recording = False

_SERVER_STARTED = False
_SHARED_STATE = RecordingState()
_TRANSCRIBE_FN = None


def start_web_ui(transcribe_chunk_fn: Callable) -> Optional[str]:
    """
    Start the Flask server for the Web GUI (first call only) and wait for a
    recording.

    The server is a module-level singleton: Flask binds port 5000 once and
    that thread lives for the rest of the process. Re-creating the app on a
    second call would silently fail to bind the port, leaving the browser
    talking to the *old* server while this function polled a *new* state
    object that nothing ever updated — an infinite hang. Instead we reuse
    one app and one state object, resetting the state each run.

    Args: 
        transcribe_chunk_fn: function to transcribe audio chunks

    Returns: path to final recording file or None
    """
    global _SERVER_STARTED, _TRANSCRIBE_FN

    state = _SHARED_STATE
    state.reset()

    # Rebind each call so the current HandoffAI instance's transcriber is used
    _TRANSCRIBE_FN = transcribe_chunk_fn

    def _recording_worker() -> None:
        """Background thread wrapper."""
        _run_recording_worker(state, _TRANSCRIBE_FN)

    if not _SERVER_STARTED:
        app = Flask(__name__)
        CORS(app)

        @app.route("/")
        def index():
            return _INDEX_HTML
        
        @app.route("/health")
        def health() -> dict:
            return jsonify({"status": "ok"})

        @app.route("/start_recording")
        def start_recording() -> dict:
            if not state.is_recording:
                state.is_recording = True
                state.status = "recording"
                state.transcript = ""
                state.start_time = time.time()
                state.audio_level = 0.0
                threading.Thread(target=_recording_worker, daemon=True).start()
                logger.info("Recording started via web UI")
            return jsonify({"status": "started"})

        @app.route("/stop_recording")
        def stop_recording() -> dict:
            state.is_recording = False
            state.status = "ready"
            logger.info("Recording stopped via web UI")
            return jsonify({"status": "stopped"})

        @app.route("/get_state")
        def get_state() -> dict:
            return jsonify({
                "is_recording": state.is_recording,
                "transcript": state.transcript,
                "status": state.status,
                "audio_level": state.audio_level
            })

        logger.info("Starting Flask web UI server...")
        print("[WEB] Starting local content server...")
        threading.Thread(target=lambda: app.run(port=5000, use_reloader=False), daemon=True).start()
        _SERVER_STARTED = True
        time.sleep(1)
    else:
        logger.info("Reusing existing Flask web UI server")
        print("[WEB] Reusing local content server...")

    webbrowser.open("http://127.0.0.1:5000")
    logger.info("Opened browser to http://127.0.0.1:5000")
    
    # Block the main thread, polling shared state until a recording is confirmed
    print("[WEB] Waiting for recording from Web UI...")
    timeout_count = 0
    try:
        while True:
            timeout_count += 1
            if timeout_count >= 10:
                logger.debug(f"Waiting... status={state.status}, has_file={bool(state.final_file)}")
                timeout_count = 0
            
            # Recording is done once the worker marks it confirmed and a file exists
            if state.status == "confirmed" and state.final_file:
                logger.info(f"Recording complete: {state.final_file}")
                recording_manager.save_final(state.final_file)
                return state.final_file
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Web UI canceled by user")
        return None