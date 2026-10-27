"""
prompt.py — the two prompts sent to the language model.

EXTRACTION_PROMPT
    Turns a paramedic radio report into the structured handoff JSON shown on the
    dashboard. Placeholders: {ems_report}, {lexicon_context}.
GUIDANCE_PROMPT
    Optional second call. It suggests courses of action grounded only in
    retrieved textbook excerpts. Placeholders: {handoff_json}, {reference}.
    Returns {"courses_of_action": [...]}. That is a JSON object, so providers
    with JSON mode can enforce it.

Both prompts end with "/no_think" for qwen3 models. The LLM client also turns
thinking off in the request itself.

extraction_prompt(attribution)
    EXTRACTION_PROMPT as is, or with the burst-attribution text added (see the
    end of this file). Incident sessions ask for it; the live path does not.
"""

# =============================================================================
# AI PROMPT FOR STRUCTURED EXTRACTION
# =============================================================================

EXTRACTION_PROMPT = """
You are HANDOFF.AI, an AI system that transforms EMS paramedic radio reports into structured hospital handoff data.

The following is a paramedic's verbal radio report to a hospital while transporting a patient. Extract ALL relevant information and structure it for the receiving Emergency Department.

{lexicon_context}
EMS RADIO REPORT:
{ems_report}

Generate a JSON response with this EXACT structure (return ONLY valid JSON, no markdown):
{{
  "alert_type": "STEMI" | "STROKE" | "TRAUMA" | "SEPSIS" | "CARDIAC_ARREST" | "PEDIATRIC" | "OBSTETRIC" | "ENDOCRINE" | "MEDICAL" | "PSYCHIATRIC" | "UNCLEAR",
  "alert_level": "RED" | "YELLOW" | "GREEN",
  "eta_minutes": number,
  
  "patient": {{
    "age": number,
    "age_unit": "years" | "months",
    "sex": "Male" | "Female" | null
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
    "allergies": ["each allergy exactly as the paramedic said it"] | null,
    "code_status": "the code status exactly as the paramedic said it" | null
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
}}

Guidance:
- Use "ENDOCRINE" as the alert_type for metabolic/endocrine emergencies such as diabetic ketoacidosis (DKA), hyperosmolar hyperglycemic state, severe hypo/hyperglycemia, thyroid storm, or adrenal crisis.
- For any field with no information, use null or omit it — never output the literal word "Null".
- Patient sex: when it is not said outright, infer it from the pronouns used for the patient ("he", "his", "him" = Male; "she", "her" = Female) and record that in data_quality.assumptions. Use null only when there are no such pronouns.
- history.allergies and history.code_status are null unless the paramedic actually spoke them. A report that never mentions allergies has allergies null; one that never mentions a code status or directive has code_status null. When the paramedic did speak them, write what was said (for example "no known allergies" or "DNR"). Never fill either field with a default, and list "allergies" or "code status" in missing_info when null.
- The report may be a fragment: a few words, a single question, or a reply with no clinical content. Still return the full structure. Use "UNCLEAR" only when no chief complaint, no mechanism of injury, and no alert word (stroke, CVA, brain bleed, trauma, STEMI, cardiac arrest, sepsis, seizure, overdose, chest pain and the like) was stated at all. If any one of those was stated, even in a fragment such as "trauma, priority 2", choose the matching alert_type and leave every unstated field null. When UNCLEAR does apply, set alert_level to null, data_quality.confidence to "LOW", and list what was not provided in missing_info. Never invent a patient, vitals, history, or interventions that were not said.
/no_think
"""

GUIDANCE_PROMPT = """
You are HANDOFF.AI's clinical decision-support module. A patient handoff has already
been extracted from a paramedic's radio report. Using ONLY the textbook reference
excerpts provided below, suggest concise, evidence-based courses of action for the
RECEIVING hospital team to prepare or consider before the patient arrives.

Rules:
- Ground every recommendation in the provided excerpts. Do NOT introduce doses,
  facts, or protocols not supported by the excerpts.
- If the excerpts don't support a recommendation for some concern, say so rather
  than inventing one.
- This is decision support, not a directive. Keep each action specific and brief.
- cite the specific bracketed source that supports each action; do not reuse one citation for all.

PATIENT HANDOFF (structured):
{handoff_json}

{reference}

Return ONLY a valid JSON object (no markdown) with this shape, holding 3-6 actions:
{{"courses_of_action": [
  {{"action": "...", "rationale": "grounded in the excerpts", "source": "citation"}}
]}}
Do not use double quotes inside any string value — paraphrase instead. No trailing commas, no text before or after the object.
/no_think
"""

# =============================================================================
# BURST ATTRIBUTION (incident sessions only; the live path never sends it)
# =============================================================================
# A "sources" entry in the schema (burst number per filled field) and one guidance
# line. The four sample cases scored the same or better with it on 2026-10-25, but the
# extra output cost about 25 s per report on the Jetson and one score moved, so it is
# added only when the pipeline is built with attribution=True (incident sessions).

ATTRIBUTION_SCHEMA = (
    "  }},\n"
    "\n"
    '  "sources": {{"field path such as patient.age or vital_signs.spo2": burst number that stated it}}\n'
    "}}\n"
)

ATTRIBUTION_GUIDANCE = (
    "- The report may arrive as numbered bursts, each under a header such as "
    '"[Burst 3 · 07:43:10 UTC · feed 4390]". In "sources", give for every field you filled '
    "the number of the burst that stated it, keyed by the field path. With no burst headers, "
    "every source is 1.\n"
    "/no_think\n"
)

_SCHEMA_END = "  }}\n}}\n"


def extraction_prompt(attribution: bool = False) -> str:
    """The extraction prompt template; with ``attribution`` the "sources" entry and its guidance line."""
    if not attribution:
        return EXTRACTION_PROMPT
    assert EXTRACTION_PROMPT.count(_SCHEMA_END) == 1
    text = EXTRACTION_PROMPT.replace(_SCHEMA_END, ATTRIBUTION_SCHEMA, 1)
    return text.replace("/no_think\n", ATTRIBUTION_GUIDANCE, 1)
