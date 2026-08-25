import copy
import json
import os
import re
import logging 
import time 
 
from openai import OpenAI, APIConnectionError, APIError
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, List
 
from .prompt import EXTRACTION_PROMPT, GUIDANCE_PROMPT
from .lexicon_rag import LexiconStore, whisper_hint          
from .database_rag import TextbookRAG                
from .transcript_corrector import format_suspects_for_llm, Flag

logger = logging.getLogger(__name__)

# Resolve the vector store relative to the project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    # Required fields that must be in LLM response
    REQUIRED_FIELDS = {
        "alert_type", "alert_level", "patient", "chief_complaint",
        "vital_signs", "assessment_findings", "history", "handoff_summary"
    }
    
    # Optional fields with sensible defaults
    OPTIONAL_FIELDS = {
        "eta_minutes": 0,
        "mechanism_or_onset": "Unknown",
        "ems_interventions": [],
        "hospital_recommendations": {},
        "data_quality": {"confidence": "MEDIUM"}
    }
    
    def __init__(self, api_key=None, provider="ollama", use_lexicon=True, use_textbook_rag=True, textbook_store_dir=None):
        """
        Initialize with your choice of FREE AI provider.
        
        Args:
            api_key: API key (not needed for Ollama)
            provider: "ollama" (local, free) | "groq" (cloud, free tier) | "openrouter" (cloud, some free models)
        """
        self.provider = provider
        self.use_lexicon = use_lexicon
        self.lexicon_store = LexiconStore() if use_lexicon else None
        self.use_textbook_rag = use_textbook_rag
        store_dir = textbook_store_dir or os.path.join(_PROJECT_ROOT, "database")
        self._textbook_rag = TextbookRAG(store_dir=store_dir) if use_textbook_rag else None
        self._whisper_prompt = whisper_hint()
        self.max_retries = 3
        self.retry_delay = 1.0
        self._colors = ColorScheme() 

        # Two Whisper models, each loaded lazily and cached by size:
        #   - a small/fast model for live keyword detection (short chunks)
        #   - a large, accurate model for the final full-report transcription
        self._whisper_models: Dict[str, WhisperModel] = {}
        self.chunk_model_size = "small"           # fast; catches the stop phrase live
        self.final_model_size = "large-v3-turbo"  # accurate; the real transcript
        self.whisper_device = "cpu"               
        self.whisper_compute_type = "int8"
        
        if provider == "ollama":
            # Ollama runs locally - completely FREE!
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't need a real key
            )
            # qwen3:8b — good extraction quality, but an ~8B model (~5 GB RAM quantized)
            self.model = "qwen3:8b"
            logger.info(f"Initialized Ollama client (model: {self.model})")
            
        elif provider == "groq":
            # Groq - FREE tier available at console.groq.com
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.getenv("GROQ_API_KEY")
            )
            self.model = "llama-3.3-70b-versatile"  # Free on Groq
            logger.info(f"Initialized Groq client (model: {self.model})")
            
        elif provider == "openrouter":
            # OpenRouter - some free models available
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key or os.getenv("OPENROUTER_API_KEY")
            )
            self.model = "meta-llama/llama-3.2-3b-instruct:free"  # Free model
            logger.info(f"Initialized OpenRouter client (model: {self.model})")
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'groq', or 'openrouter'")
    
    def process_ems_report(self, ems_report: str, suspects: Optional[List[Flag]] = None) -> dict:
        """
        Process a paramedic's verbal radio report and extract structured data.

        Args:
            ems_report: The raw text of what the paramedic said on the radio.
                        Expected to be already run through the transcript
                        corrector by the caller.
            suspects:   Flagged mis-hearings from the corrector, left in place
                        for the LLM to resolve from context. Injected into the
                        extraction prompt; pass None if there are none.

        Returns:
            Structured handoff data for the hospital
        """
        if not ems_report or len(ems_report.strip()) == 0:
            raise ValueError("EMS report is empty")

        print("\n[*] Processing EMS report...")
        print(f"    Using: {self.provider.upper()} — {self.model}")
        print("-" * 50)

        # Ask the LLM for structured JSON, then parse and validate it
        json_response = self._call_llm(ems_report, suspects)

        try: 
            handoff_data = json.loads(json_response)
        except json.JSONDecodeError as e:
            # Log the parse error plus a truncated preview of the raw output for debugging
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {json_response[:500]}")
            raise ValueError(f"LLM returned invalid JSON. {json_response[:500]}")

        # Ensure required fields exist and fill in optional defaults
        self._validate_response(handoff_data)

        logger.info("Report processed successfully.")
        return handoff_data
    
    def _call_llm(self, ems_report: str, suspects: Optional[List[Flag]] = None) -> str:
        """
        Call LLM to extract structured data with exponential backoff retry.

        Args: 
            ems_report: Raw EMS report text
            suspects:   Flagged mis-hearings to surface to the model, or None

        Returns: 
            Raw JSON string from LLM 
        """
        if self.lexicon_store:
            lexicon_context = self.lexicon_store.build_context(ems_report, top_k=25)
            logger.info(f"Lexicon context: {len(lexicon_context)} chars")
        else:
            lexicon_context = ""

        suspect_block = format_suspects_for_llm(suspects or [])
        if suspect_block:
            lexicon_context = (f"{lexicon_context}\n{suspect_block}"
                               if lexicon_context else suspect_block)
        if lexicon_context:
            logger.info(f"Lexicon/suspect context: {len(lexicon_context)} chars")

        prompt = EXTRACTION_PROMPT.format(
            ems_report=ems_report,
            lexicon_context=lexicon_context,
        )

        for attempt in range(self.max_retries):
            try: 
                response = self.client.chat.completions.create(
                    model=self.model, 
                    max_tokens=4096, 
                    temperature=0.03, 
                    messages=[{"role": "user", "content": prompt}]
                )

                json_text = self._strip_llm_json(response.choices[0].message.content)

                # A "successful" call can still yield nothing to parse: a
                # reasoning model (qwen3) may spend the whole max_tokens budget
                # inside <think>...</think> and stop right after the close tag,
                # leaving _strip_llm_json with an empty string. That is not a
                # transport error, so it never hits the except blocks below —
                # guard it here, and retry rather than returning "" straight
                # into json.loads (which surfaces as a confusing "invalid JSON").
                if not json_text.strip():
                    attempt_num = attempt + 1
                    logger.warning(
                        "Empty completion (reasoning-only, no JSON) on attempt "
                        f"{attempt_num}/{self.max_retries}."
                    )
                    if attempt_num < self.max_retries:
                        wait = self.retry_delay * (2 ** attempt)  # 1s, 2s, 4s, ...
                        print(f"    [!] Empty response. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    raise ValueError(
                        "LLM returned an empty response after "
                        f"{self.max_retries} attempts (reasoning trace likely "
                        "consumed the token budget)."
                    )

                logger.info(f"LLM call successful on attempt {attempt + 1}")
                return json_text
            
            except APIConnectionError as e:
                attempt_num = attempt + 1
                if attempt_num < self.max_retries:
                    wait = self.retry_delay * (2 ** attempt)  # 1s, 2s, 4s, ...
                    logger.warning(
                        f"Connection error (attempt {attempt_num}/{self.max_retries}). "
                        f"Retrying in {wait}s..."
                    )
                    print(f"    [!] Connection issue. Retrying in {wait}s...")
                    time.sleep(wait)
                else: 
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    self._print_error(e)
                    raise
            except APIError as e: 
                logger.error(f"API error: {e}")
                self._print_error(e)
                raise

    @staticmethod
    def _strip_llm_json(text: str) -> str:
        """
        Normalize raw LLM output down to (hopefully) parseable JSON.

        Handles the two things every provider/model here can add around the
        JSON: a reasoning-model ``<think>...</think>`` trace (qwen3) and
        Markdown code fences. Shared by both the extraction and guidance paths
        so the cleanup can't drift between them. Array isolation and the
        object-comma repair stay in the guidance path, since they're specific
        to its array-of-objects output.
        """
        if not text:
            return ""
        if "</think>" in text:                    
            text = text.split("</think>", 1)[-1]
        return text.replace("```json", "").replace("```", "").strip()

    def _validate_response(self, data: Dict[str, Any]) -> None:
        """
        Validate that LLM response has required fields and fill in optional ones.
        
        Args:
            data: Parsed JSON response from LLM (will be modified in place)
        """
        if not isinstance(data, dict):
            raise ValueError(f"Response is not a dict: {type(data)}")
        
        # Check required fields
        missing_fields = self.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(sorted(missing_fields))}")
        
        # Fill in missing optional fields with sensible defaults. deepcopy is
        # essential: OPTIONAL_FIELDS holds mutable objects ([], {}), so assigning
        # the raw default would share one object across every handoff. Callers
        # (e.g. main.process_case appending to data_quality.assumptions) mutate
        # these in place, which would otherwise leak one patient's data into the
        # next report that happens to omit the same field.
        for field, default_value in self.OPTIONAL_FIELDS.items():
            if field not in data:
                data[field] = copy.deepcopy(default_value)
                logger.info(f"Filled missing optional field '{field}' with default value")
        
        if data.get("alert_type") not in [
            "STEMI", "STROKE", "TRAUMA", "SEPSIS", "CARDIAC_ARREST",
            "PEDIATRIC", "OBSTETRIC", "MEDICAL", "PSYCHIATRIC", "ENDOCRINE"
        ]:
            logger.warning(f"Unexpected alert_type: {data.get('alert_type')}")
        
        if data.get("alert_level") not in ["RED", "YELLOW", "GREEN"]:
            raise ValueError(f"Invalid alert_level: {data.get('alert_level')}")
        
        if not isinstance(data.get("patient"), dict):
            raise ValueError("patient field is not a dict")
        
        logger.info("Response validation passed (all required + optional fields present)")

    def _get_whisper_model(self, size: str) -> WhisperModel:
        """
        Lazily load (and cache) a local faster-whisper model by size.

        Models are cached per size, so the small keyword-detection model and the
        large final-transcription model each load at most once, and only if
        actually used. (If both sizes were set to the same value they'd share a
        single instance.)

        device="cpu" works everywhere. On a Jetson or any NVIDIA+CUDA machine,
        set self.whisper_device = "cuda" for a large speedup — same code otherwise.
        """
        model = self._whisper_models.get(size)
        if model is None:
            logger.info(f"Loading local faster-whisper model ({size})...")
            model = WhisperModel(
                size,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
            self._whisper_models[size] = model
            logger.info(f"faster-whisper model loaded ({size}).")
        return model

    def transcribe_audio_chunk(self, filename: str) -> str: 
        """
        Transcribe a small audio chunk for keyword detection.
        Runs fully locally now — no longer requires Groq.
        
        Args: 
            filename: path to WAV file
        
        Returns: 
            Transcribed text or empty string
        """
        try:
            model = self._get_whisper_model(self.chunk_model_size)
            segments, _ = model.transcribe(filename, language="en", beam_size=1)
            return " ".join(segment.text for segment in segments).strip()
        except Exception as e:
            logger.warning(f"Chunk transcription failed: {e}")
            return ""

    def transcribe_audio_final(self, filename: str) -> Optional[str]:
        """
        Transcribe the full audio file to text using local faster-whisper.
        Runs fully locally now — no longer requires Groq.

        Args: 
            filename: path to WAV file

        Returns: 
            Transcribed text or None 
        """
        print("\n[*] Transcribing full report with local faster-whisper...")
        try:
            model = self._get_whisper_model(self.final_model_size)
            # self._whisper_prompt comes from rag.py's whisper_hint(), which
            # builds the EMS terminology list from lexicon.py dynamically —
            # unchanged from before, just passed as initial_prompt instead of
            # Whisper API's `prompt` param (faster-whisper's naming).
            segments, info = model.transcribe(
                filename,
                language="en",
                beam_size=5,
                initial_prompt=self._whisper_prompt
            )
            text = " ".join(segment.text for segment in segments).strip()
            logger.info("Audio transcribed successfully (local faster-whisper)")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            print(f"[X] Transcription failed: {e}")
            return None

    def display_hospital_view(self, data: Dict[str, Any]) -> None:
        """
        Display what the hospital ER would see on their dashboard.
        This is the OUTPUT that makes us valuable.

        Args: 
            data: structured handoff data from LLM
        """
        colors = self._colors

        print("\n" + "=" * 60)
        print(f"{colors.bold}[HOSPITAL] HOSPITAL ER DASHBOARD - INCOMING PATIENT{colors.reset}")
        print("=" * 60)

        alert_type = data.get("alert_type", "UNKNOWN")
        alert_level = data.get("alert_level", "UNKNOWN")
        color = colors.get_alert_color(alert_level)

        # Alert Banner
        print(f"\n{color}{colors.bold}[!]  {data.get('alert_type', 'UNKNOWN')} ALERT - {data.get('alert_level', 'UNKNOWN')} PRIORITY{colors.reset}")
        print(f"{colors.bold}ETA: {data.get('eta_minutes', '?')} MINUTES{colors.reset}")

        # One-liner summary
        summary = data.get("handoff_summary", {})
        print(f"\n \"{summary.get('one_liner', 'No summary available')}\"")
        
        # Patient demographics
        self._display_patient_section(data, colors)
        self._display_vitals_section(data, colors)
        self._display_assessment_section(data, colors)
        self._display_interventions_section(data, colors)
        self._display_history_section(data, colors)
        self._display_hospital_prep_section(data, colors)
        self._display_summary_section(data, colors)
        self._display_data_quality_section(data, colors)
        
        print("\n" + "="*60)
    
    def _print_section(self, title: str, colors: 'ColorScheme') -> None:
        """Print a section header."""
        print(f"\n{colors.bold}{title}:{colors.reset}")
    
    def _print_field(self, label: str, value: str) -> None:
        """Print a labeled field."""
        if value:
            print(f"  {label}: {value}")

    def _display_patient_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display patient demographics."""
        patient = data.get("patient", {})
        self._print_section("PATIENT", colors)
        print(f"  Age: {patient.get('age', '?')} {patient.get('age_unit', 'years')}")
        print(f"  Sex: {patient.get('sex', 'Unknown')}")
        print(f"  Chief Complaint: {data.get('chief_complaint', 'Unknown')}")
        print(f"  Onset/Mechanism: {data.get('mechanism_or_onset', 'Unknown')}")
    
    def _display_vitals_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display vital signs."""
        vitals = data.get("vital_signs", {})
        self._print_section("VITAL SIGNS", colors)
        
        vital_labels = [
            ("blood_pressure", "BP"),
            ("heart_rate", "HR"),
            ("respiratory_rate", "RR"),
            ("spo2", "SpO2"),
            ("temperature", "Temp"),
            ("blood_glucose", "BGL"),
            ("gcs", "GCS")
        ]
        
        for key, label in vital_labels:
            self._print_field(label, vitals.get(key))
    
    def _display_assessment_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display clinical assessment findings."""
        assessment = data.get("assessment_findings", {})
        self._print_section("ASSESSMENT", colors)
        self._print_field("LOC", assessment.get('level_of_consciousness'))
        self._print_field("Airway", assessment.get('airway'))
        self._print_field("Breathing", assessment.get('breathing'))
        self._print_field("Circulation", assessment.get('circulation'))
        if assessment.get("other_findings"):
            print(f"  Other: {', '.join(assessment['other_findings'])}")
    
    def _display_interventions_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display EMS interventions."""
        interventions = data.get("ems_interventions", [])
        if interventions:
            self._print_section("EMS INTERVENTIONS", colors)
            for i, item in enumerate(interventions, 1):
                print(f"  {i}. {item.get('intervention', '')} - {item.get('details', '')}")
                if item.get("response"):
                    print(f"     → {item['response']}")
    
    def _display_history_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display patient history."""
        history = data.get("history", {})
        self._print_section("HISTORY", colors)
        print(f"  PMH: {', '.join(history.get('past_medical', ['Unknown']))}")
        print(f"  Meds: {', '.join(history.get('medications', ['Unknown']))}")
        print(f"  Allergies: {', '.join(history.get('allergies', ['NKDA']))}")
        print(f"  Code: {history.get('code_status', 'Full Code')}")
    
    def _display_hospital_prep_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display hospital preparation actions (the KEY value prop)."""
        hosp = data.get("hospital_recommendations", {})
        alert_level = data.get("alert_level", "GREEN")
        color = colors.get_alert_color(alert_level)
        
        print(f"\n{color}{colors.bold}[!] HOSPITAL PREPARATION:{colors.reset}")
        print(f"  Activation: {hosp.get('activation_requested', 'None')}")
        print(f"  Destination: {hosp.get('suggested_destination', 'General ED')}")
        
        if hosp.get("resources_to_prepare"):
            print(f"  Resources:")
            for res in hosp["resources_to_prepare"]:
                print(f"    • {res}")
        
        if hosp.get("time_critical_actions"):
            print(f"  {colors.bold}TIME-CRITICAL:{colors.reset}")
            for action in hosp["time_critical_actions"]:
                print(f"    ! {action}")
    
    def _display_summary_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display summary concerns and pending needs."""
        summary = data.get("handoff_summary", {})
        
        if summary.get("key_concerns"):
            print(f"\n{colors.bold}[!] KEY CONCERNS:{colors.reset}")
            for concern in summary["key_concerns"]:
                print(f"  • {concern}")
        
        if summary.get("pending_needs"):
            print(f"\n{colors.bold}PENDING ON ARRIVAL:{colors.reset}")
            for need in summary["pending_needs"]:
                print(f"  → {need}")

    def _display_data_quality_section(self, data: Dict[str, Any], colors: 'ColorScheme') -> None:
        """Display data quality metrics and missing information."""
        quality = data.get("data_quality", {})
        self._print_section("DATA QUALITY", colors)
        print(f"  Confidence: {quality.get('confidence', 'Unknown')}")
        if quality.get("missing_info"):
            print(f"  Missing: {', '.join(quality['missing_info'])}")
        if quality.get("assumptions"):
            print(f"  Assumptions: {', '.join(quality['assumptions'])}")
            
    def suggest_courses_of_action(self, handoff_data, top_k: int = 6):
        """Retrieve textbook references for this handoff and generate grounded,
        cited courses of action. Returns a list of dicts, or None if no store."""
        if not self.use_textbook_rag or self._textbook_rag is None:
            return None
        if not self._textbook_rag.available:
            logger.info("Textbook vector store not found; skipping guidance.")
            return None
        try:
            reference = self._textbook_rag.guidance_context(handoff_data, top_k=top_k)
        except Exception as e:
            logger.warning(f"Textbook retrieval failed: {e}")
            return None
        if not reference:
            return None

        prompt = GUIDANCE_PROMPT.format(
            handoff_json=json.dumps(handoff_data, indent=2),
            reference=reference,
        )

        # Same failure mode as extraction: qwen3 doesn't always honor the
        # /no_think in GUIDANCE_PROMPT, and when it thinks anyway the reasoning
        # trace can eat the whole token budget and stop right at </think>,
        # leaving _strip_llm_json with an empty string. A single empty response
        # then surfaced as a confusing "Expecting value" JSON error and dropped
        # guidance entirely. Retry the call on an empty completion (as _call_llm
        # does) before giving up. Unlike extraction, guidance fails soft: if
        # every attempt is empty or the parse fails, return None so the ER
        # dashboard still renders without the (optional) courses of action.
        text = ""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, max_tokens=4096, temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = self._strip_llm_json(resp.choices[0].message.content or "")

                if not text.strip():
                    attempt_num = attempt + 1
                    logger.warning(
                        "Guidance empty completion (reasoning-only, no JSON) on "
                        f"attempt {attempt_num}/{self.max_retries}."
                    )
                    if attempt_num < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))  # 1s, 2s, 4s
                        continue
                    logger.warning("Guidance skipped: empty after "
                                   f"{self.max_retries} attempts.")
                    return None

                # isolate the JSON array in case the model added prose around it
                start, end = text.find("["), text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    text = text[start:end + 1]

                # qwen3 sometimes omits the comma between array objects: "} {" -> "}, {"
                text = re.sub(r"\}\s*\{", "}, {", text)

                guidance = json.loads(text)
                if isinstance(guidance, dict):
                    guidance = guidance.get("courses_of_action") or guidance.get("actions") or [guidance]
                logger.info(f"Generated {len(guidance)} textbook-grounded recommendations")
                return guidance
            except Exception as e:
                logger.warning(f"Guidance generation failed: {e}")
                logger.warning(f"Raw guidance output (first 600 chars): {text[:600]!r}")
                return None

    def display_courses_of_action(self, guidance) -> None:
        if not guidance:
            return
        colors = self._colors
        print(f"\n{colors.bold}[+] SUGGESTED COURSES OF ACTION (textbook-grounded):{colors.reset}")
        for i, g in enumerate(guidance, 1):
            if isinstance(g, dict):
                print(f"  {i}. {g.get('action', '')}")
                if g.get("rationale"):
                    print(f"     Rationale: {g['rationale']}")
                if g.get("source"):
                    print(f"     Source: {g['source']}")
            else:
                print(f"  {i}. {g}")
        print("  (Decision support only — clinical judgment required.)")

    def _print_error(self, e: Exception) -> None:
        """Print provider-specific error message."""
        error_type = type(e).__name__
        error_str = str(e)
        
        print("\n" + "=" * 60)
        
        if "Connection" in error_type or "ConnectError" in error_str:
            print("[X] CONNECTION ERROR")
            print("=" * 60)
            
            provider_msgs = {
                "ollama": (
                    "[!] Ollama is not running or not installed.\n\n"
                    "Quick Fix:\n"
                    "  1. Download from: https://ollama.ai/download\n"
                    "  2. Install and run: ollama pull llama3.2\n"
                    "  3. Start: ollama serve\n"
                    "  4. Re-run this script\n"
                ),
                "groq": (
                    "[!] Cannot connect to Groq API.\n\n"
                    "Check:\n"
                    "  - Internet connection\n"
                    "  - API key is correct\n"
                    "  - Groq status: status.groq.com\n"
                ),
                "openrouter": (
                    "[!] Cannot connect to OpenRouter API.\n\n"
                    "Check:\n"
                    "  - Internet connection\n"
                    "  - API key is correct\n"
                    "  - OpenRouter service status\n"
                ),
            }
            print("\n" + provider_msgs.get(self.provider, "[!] Connection failed\n"))
            
        elif "API" in error_type or "Auth" in error_type:
            print("[X] API AUTHENTICATION ERROR")
            print("=" * 60)
            print("\nYour API key might be invalid or expired.")
            print(f"Provider: {self.provider}")
            urls = {
                "groq": "https://console.groq.com",
                "openrouter": "https://openrouter.ai/keys",
            }
            if self.provider in urls:
                print(f"Get a new key: {urls[self.provider]}\n")
        else:
            print(f"[X] {error_type}")
            print("=" * 60)
            print(f"\nDetails: {error_str}\n")
        
        print("=" * 60)

class ColorScheme:
    """Terminal color codes for nice output."""
    
    def __init__(self):
        self.reset = "\033[0m"
        self.bold = "\033[1m"
        self.red = "\033[91m"
        self.yellow = "\033[93m"
        self.green = "\033[92m"
    
    def get_alert_color(self, level: str) -> str:
        """Get color code for alert level."""
        colors = {
            "RED": self.red,
            "YELLOW": self.yellow,
            "GREEN": self.green,
        }
        return colors.get(level, self.reset)