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
  "alert_type": "STEMI" | "STROKE" | "TRAUMA" | "SEPSIS" | "CARDIAC_ARREST" | "PEDIATRIC" | "OBSTETRIC" | "ENDOCRINE" | "MEDICAL" | "PSYCHIATRIC",
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
}}

Guidance:
- Use "ENDOCRINE" as the alert_type for metabolic/endocrine emergencies such as diabetic ketoacidosis (DKA), hyperosmolar hyperglycemic state, severe hypo/hyperglycemia, thyroid storm, or adrenal crisis.
- For any field with no information, use null or omit it — never output the literal word "Null".
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

Return ONLY a valid JSON array (no markdown) of 3-6 objects with this shape:
[
  {{"action": "...", "rationale": "grounded in the excerpts", "source": "citation"}}
]
Do not use double quotes inside any string value — paraphrase instead. No trailing commas, no text before or after the array. Separate every object in the array with a comma.
/no_think
"""