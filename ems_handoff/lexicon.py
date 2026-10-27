"""
lexicon.py — EMS terminology: canonical terms, the aliases paramedics say, and short clinical notes.

Structure
    LEXICON         {category: {canonical_term: [alias, ...]}}  (14 categories, ~850 surface forms)
    CLINICAL_NOTES  {canonical_term: one-line note shown to the LLM}

Used by
    lexicon_rag.py            finds the terms present in a report and builds prompt context,
                              and builds the vocabulary hint given to Whisper
    transcript_corrector.py   the correction targets for mis-heard drug names

Data only. Edit freely, then run the tests (tests/test_corrector_lexicon.py).
"""

from typing import Dict, List, Tuple

# =============================================================================
# THE LEXICON
# =============================================================================
# Structure:  { category: { canonical_term: [alias, alias, ...] } }

LEXICON: Dict[str, Dict[str, List[str]]] = {
 
    # -------------------------------------------------------------------------
    # VITAL SIGNS
    # -------------------------------------------------------------------------
    "vitals": {
        "blood pressure": ["BP", "B/P", "pressure"],
        "systolic blood pressure": ["SBP", "systolic"],
        "diastolic blood pressure": ["DBP", "diastolic"],
        "mean arterial pressure": ["MAP"],
        "heart rate": ["HR", "pulse", "pulse rate"],
        "respiratory rate": ["RR", "resp rate", "respirations", "resps", "respiratory"],
        "oxygen saturation": ["SpO2", "O2 sat", "SP02", "O2 sats", "sat", "sats",   # SP02 stays an alias for the corrector but out of the hint (aliases[:2])
                              "pulse ox", "pulse oximetry", "oxygen sat"],
        "temperature": ["temp", "T", "tympanic temp", "core temp"],
        "blood glucose": ["BGL", "BG", "blood sugar", "blood glucose level",
                         "glucose", "dextrostick", "fingerstick", "accucheck",
                         "accu-chek", "CBG"],
        "end-tidal carbon dioxide": ["EtCO2", "ETCO2", "end tidal", "end-tidal CO2",
                                    "capnography", "capno"],
        "Glasgow Coma Scale": ["GCS", "Glasgow", "Glasgow coma score"],
        "level of consciousness": ["LOC", "mental status", "level of responsiveness"],
        "pain score": ["pain scale", "pain level", "out of ten", "0 to 10"],
        "capillary refill": ["cap refill", "CRT", "cap refill time"],
    },
 
    # -------------------------------------------------------------------------
    # ASSESSMENT TOOLS & MNEMONICS
    # -------------------------------------------------------------------------
    "assessment_tools": {
        "AVPU scale": ["AVPU", "alert verbal pain unresponsive"],
        "alert and oriented": ["A&O", "AOx4", "AAOx4", "AOx3", "oriented times four",
                             "awake alert oriented"],
        "SAMPLE history": ["SAMPLE"],
        "OPQRST": ["OPQRST", "onset provocation quality radiation severity timing"],
        "DCAP-BTLS": ["DCAP BTLS", "DCAP-BTLS"],
        "pediatric assessment triangle": ["PAT"],
        "mechanism of injury": ["MOI"],
        "nature of illness": ["NOI"],
        "field impression": ["working impression", "clinical impression"],
        "chief complaint": ["CC", "chief concern", "presenting complaint"],
        "Cincinnati stroke scale": ["Cincinnati", "CPSS", "FAST exam", "FAST"],
        "Revised Trauma Score": ["RTS", "trauma score"],
    },
 
    # -------------------------------------------------------------------------
    # ALERT TYPES (time-critical syndromes the ED pre-activates for)
    # -------------------------------------------------------------------------
    "alert_types": {
        "ST-elevation myocardial infarction": ["STEMI", "ST elevation MI",
                                              "anterior STEMI", "inferior STEMI"],
        "non-ST-elevation myocardial infarction": ["NSTEMI", "non-STEMI"],
        "acute myocardial infarction": ["AMI", "MI", "heart attack",
                                       "myocardial infarction"],
        "acute coronary syndrome": ["ACS"],
        "cerebrovascular accident": ["CVA", "stroke", "brain attack"],
        "ischemic stroke": ["ischemic CVA"],
        "hemorrhagic stroke": ["hemorrhagic CVA", "brain bleed"],
        "transient ischemic attack": ["TIA", "mini stroke"],
        "large vessel occlusion": ["LVO"],
        "sepsis": ["septic", "septic shock", "severe sepsis"],
        "trauma alert": ["trauma activation", "trauma team", "multisystem trauma"],
        "cardiac arrest": ["arrest", "code", "full arrest", "coded",
                         "cardiopulmonary arrest"],
        "return of spontaneous circulation": ["ROSC"],
        "motor vehicle collision": ["MVC", "MVA", "motor vehicle accident",
                                  "car crash", "car accident"],
        "STEMI alert": ["cath lab activation", "cath lab alert"],
        "stroke alert": ["code stroke"],
    },
 
    # -------------------------------------------------------------------------
    # CARDIAC (rhythms & conditions)
    # -------------------------------------------------------------------------
    "cardiac": {
        "12-lead electrocardiogram": ["12-lead", "twelve lead", "ECG", "EKG"],
        "ST elevation": ["ST elevations", "STE"],
        "ST depression": ["ST depressions"],
        "atrial fibrillation": ["AFib", "A-fib", "A fib", "afib"],
        "atrial flutter": ["A-flutter", "flutter"],
        "ventricular fibrillation": ["VF", "V-fib", "V fib", "vfib"],
        "ventricular tachycardia": ["VT", "V-tach", "V tach", "vtach"],
        "supraventricular tachycardia": ["SVT"],
        "pulseless electrical activity": ["PEA"],
        "asystole": ["flatline", "flat line"],
        "premature ventricular contraction": ["PVC", "PVCs"],
        "bradycardia": ["brady", "slow heart rate"],
        "tachycardia": ["tachy", "fast heart rate"],
        "congestive heart failure": ["CHF", "heart failure", "HF"],
        "cardiogenic shock": [],
        "aortic aneurysm": ["AAA", "triple A", "abdominal aortic aneurysm"],
        "cardiac tamponade": ["pericardial tamponade", "tamponade"],
        "dysrhythmia": ["arrhythmia", "irregular rhythm"],
    },
 
    # -------------------------------------------------------------------------
    # RESPIRATORY
    # -------------------------------------------------------------------------
    "respiratory": {
        "shortness of breath": ["SOB", "dyspnea", "difficulty breathing",
                             "trouble breathing", "respiratory distress"],
        "chronic obstructive pulmonary disease": ["COPD"],
        "asthma": ["reactive airway", "asthma exacerbation"],
        "respiratory failure": ["resp failure"],
        "respiratory arrest": ["resp arrest", "apnea", "apneic", "not breathing"],
        "pulmonary edema": ["flash pulmonary edema"],
        "pulmonary embolism": ["PE"],
        "pneumothorax": ["collapsed lung", "PTX"],
        "tension pneumothorax": ["tension pneumo"],
        "pneumonia": ["PNA", "chest infection"],
        "wheezing": ["wheezes", "wheeze"],
        "rales": ["crackles"],
        "rhonchi": [],
        "stridor": [],
        "accessory muscle use": ["accessory muscles", "retractions"],
        "bilateral breath sounds": ["breath sounds equal", "BBS", "clear bilaterally"],
    },
 
    # -------------------------------------------------------------------------
    # NEUROLOGIC
    # -------------------------------------------------------------------------
    "neurologic": {
        "altered mental status": ["AMS", "altered", "altered LOC",
                               "change in mental status"],
        "loss of consciousness": ["LOC", "passed out", "syncope", "syncopal",
                               "fainted", "unconscious"],
        "traumatic brain injury": ["TBI", "head injury"],
        "increased intracranial pressure": ["increased ICP", "elevated ICP", "ICP"],
        "seizure": ["convulsion", "seizing", "post-ictal", "postictal"],
        "status epilepticus": [],
        "pupils equal and reactive": ["PERRL", "PERRLA", "pupils equal reactive to light",
                                   "pupils reactive"],
        "cerebrospinal fluid": ["CSF"],
        "hemiparesis": ["one-sided weakness", "facial droop"],
        "aphasia": ["slurred speech", "dysarthria", "trouble speaking"],
        "Cushing triad": ["Cushing's triad"],
    },
 
    # -------------------------------------------------------------------------
    # EXAM FINDINGS (skin signs, trauma patterns, narrative descriptors)
    # -------------------------------------------------------------------------
    "exam_findings": {
        "diaphoretic": ["diaphoresis", "sweaty", "clammy"],
        "guarding": ["abdominal guarding", "voluntary guarding"],
        "seatbelt sign": ["seat belt sign", "seatbelt mark"],
        "laceration": ["lac", "cut"],
        "fracture": ["fx", "deformity", "broken"],
        "hemorrhage": ["bleeding", "active bleeding", "hemorrhaging"],
        "flushed skin": ["flushed", "warm and flushed"],
        "pale": ["pallor", "ashen"],
        "loss of consciousness at scene": ["positive LOC", "LOC at scene",
                                          "witnessed LOC"],
    },
 
    # -------------------------------------------------------------------------
    # CONDITIONS (other medical history / presentations)
    # -------------------------------------------------------------------------
    "conditions": {
        "hypertension": ["HTN", "high blood pressure"],
        "hypotension": ["low blood pressure"],
        "diabetes mellitus": ["DM", "diabetes", "diabetic"],
        "type 2 diabetes": ["T2DM", "type two diabetes"],
        "type 1 diabetes": ["T1DM", "type one diabetes"],
        "diabetic ketoacidosis": ["DKA"],
        "hyperglycemia": ["high blood sugar"],
        "hypoglycemia": ["low blood sugar"],
        "chronic kidney disease": ["CKD", "renal failure", "kidney disease"],
        "end-stage renal disease": ["ESRD", "dialysis patient"],
        "coronary artery disease": ["CAD", "coronary disease"],
        "gastrointestinal bleed": ["GI bleed", "GIB"],
        "deep vein thrombosis": ["DVT", "blood clot"],
        "anaphylaxis": ["anaphylactic shock", "severe allergic reaction"],
        "hypovolemic shock": ["hypovolemia"],
        "overdose": ["OD", "poisoning", "intoxication", "tox"],
        "nausea and vomiting": ["N/V", "nausea vomiting", "emesis"],
        "no known drug allergies": ["NKDA", "NKA", "no known allergies",
                                 "no allergies"],
        "penicillin allergy": ["penicillin", "PCN"],
        "sulfa allergy": ["sulfa", "sulfa drugs", "sulfonamide"],
        "urinary tract infection": ["UTI", "bladder infection", "urosepsis"],
        "fever": ["febrile", "pyrexia", "temperature of", "running a temp"],
        "dementia": ["Alzheimer's", "Alzheimers", "Alzheimer's dementia",
                    "cognitive decline"],
        "skilled nursing facility": ["SNF", "nursing home", "nursing facility",
                                    "long-term care", "LTC", "assisted living"],
        "delirium": ["acute confusion"],
    },
 
    # -------------------------------------------------------------------------
    # ANATOMY & POSITIONAL TERMS
    # -------------------------------------------------------------------------
    "anatomy": {
        "antecubital": ["AC", "AC fossa", "antecubital fossa"],
        "right upper quadrant": ["RUQ"],
        "left upper quadrant": ["LUQ"],
        "right lower quadrant": ["RLQ"],
        "left lower quadrant": ["LLQ"],
        "cervical spine": ["C-spine", "c spine", "cervical"],
        "thoracic spine": ["T-spine"],
        "lumbar spine": ["L-spine", "lumbar"],
        "external jugular": ["EJ", "external jugular vein"],
        "intercostal": ["ICS", "intercostal space"],
        "midclavicular line": ["MCL"],
        "anterior": ["front", "ventral"],
        "posterior": ["back", "dorsal"],
        "bilateral": ["bilat", "both sides"],
        "unilateral": ["one-sided"],
        "distal": [],
        "proximal": [],
        "midline": [],
    },
 
    # -------------------------------------------------------------------------
    # MEDICATION ADMINISTRATION ROUTES
    # -------------------------------------------------------------------------
    "routes": {
        "intravenous": ["IV", "I.V.", "intravenously"],
        "intraosseous": ["IO", "I.O."],
        "intramuscular": ["IM", "I.M."],
        "subcutaneous": ["SubQ", "SQ", "SC", "sub-q"],
        "sublingual": ["SL", "under the tongue"],
        "per os": ["PO", "oral", "orally", "by mouth"],
        "per rectum": ["PR", "rectal", "rectally"],
        "intranasal": ["IN", "intranasally", "nasal atomizer", "MAD"],
        "endotracheal": ["ET", "down the tube"],
        "nebulized": ["neb", "nebulizer", "nebulized treatment"],
        "transdermal": ["patch", "transcutaneous"],
        "inhaled": ["inhalation", "MDI", "metered dose inhaler", "puffer"],
    },
 
    # -------------------------------------------------------------------------
    # INTERVENTIONS, PROCEDURES & EQUIPMENT
    # -------------------------------------------------------------------------
    "interventions": {
        "intravenous access": ["IV access", "IV line", "line", "saline lock",
                            "IV established", "large bore IV", "18-gauge", "gauge"],
        "nasal cannula": ["NC", "cannula"],
        "non-rebreather mask": ["NRB", "non-rebreather", "non-rebreathing mask"],
        "bag-valve mask": ["BVM", "bag-mask", "bagging", "bag valve mask"],
        "continuous positive airway pressure": ["CPAP"],
        "endotracheal intubation": ["ETI", "intubation", "intubated", "ET tube",
                                 "tube", "definitive airway"],
        "supraglottic airway": ["SGA", "King airway", "iGel", "i-gel", "LMA"],
        "oropharyngeal airway": ["OPA", "oral airway"],
        "nasopharyngeal airway": ["NPA", "nasal airway"],
        "cardiopulmonary resuscitation": ["CPR", "compressions", "chest compressions"],
        "automated external defibrillator": ["AED"],
        "defibrillation": ["defib", "shock", "shocked"],
        "cervical collar": ["C-collar", "collar", "cervical immobilization"],
        "backboard": ["long board", "spine board", "LSB"],
        "spinal motion restriction": ["SMR", "c-spine precautions",
                                   "spinal precautions"],
        "tourniquet": ["TQ", "TK"],
        "indwelling urinary catheter": ["Foley", "Foley catheter", "urinary catheter"],
        "supplemental oxygen": ["O2", "oxygen", "on oxygen", "liters",
                             "liters per minute", "LPM", "blow-by", "blow-by oxygen",
                             "blow by oxygen"],
        "normal saline bolus": ["fluid bolus", "saline bolus", "fluids wide open",
                             "running wide open", "500 mL bolus"],
    },
 
    # -------------------------------------------------------------------------
    # DRUGS  (canonical generic -> brand names, slang, common abbreviations)
    # From the RI EMS Pharmacology Guide, plus common field names.
    # -------------------------------------------------------------------------
    "drugs": {
        "acetaminophen": ["Tylenol", "APAP", "paracetamol"],
        "activated charcoal": ["charcoal"],
        "adenosine": ["Adenocard"],
        "albuterol": ["salbutamol", "Ventolin", "Proventil"],
        "amiodarone": ["Cordarone", "Pacerone"],
        "aspirin": ["ASA", "acetylsalicylic acid", "baby aspirin"],
        "atropine sulfate": ["atropine"],
        "calcium chloride": ["CaCl", "cal chloride"],
        "calcium gluconate": ["cal gluconate"],
        "cefazolin": ["Ancef", "Kefzol"],
        "dexamethasone": ["Decadron", "dex"],
        "dextrose": ["D50", "D-50", "D10", "D25", "D5", "amp of dextrose", "D50W"],
        "diazepam": ["Valium"],
        "diltiazem": ["Cardizem"],
        "diphenhydramine": ["Benadryl"],
        "dopamine": ["Intropin", "dopamine drip"],
        "droperidol": ["Inapsine"],
        "enalaprilat": ["Vasotec"],
        "epinephrine": ["epi", "adrenaline", "epi 1:1000", "epi 1:10000"],
        "etomidate": ["Amidate"],
        "famotidine": ["Pepcid"],
        "fentanyl": ["Sublimaze"],
        "furosemide": ["Lasix"],
        "glucagon": ["GlucaGen"],
        "oral glucose": ["glucose", "glucose gel", "oral glucose gel"],
        "haloperidol": ["Haldol"],
        "hydrocortisone": ["Solu-Cortef"],
        "hydroxocobalamin": ["Cyanokit"],
        "ibuprofen": ["Motrin", "Advil"],
        "intravenous fat emulsion": ["Intralipid", "IFE", "lipid emulsion",
                                  "lipid rescue"],
        "ipratropium bromide": ["Atrovent"],
        "ketamine": ["Ketalar"],
        "ketorolac": ["Toradol"],
        "labetalol": ["Trandate", "Normodyne"],
        "lactated Ringers": ["LR", "Ringer's lactate", "RL", "lactated Ringer's"],
        "levalbuterol": ["Xopenex"],
        "lidocaine": ["Xylocaine", "lignocaine", "lido"],
        "lorazepam": ["Ativan"],
        "magnesium sulfate": ["mag", "MgSO4", "magnesium"],
        "methylprednisolone": ["Solu-Medrol"],
        "metoprolol": ["Lopressor", "Toprol"],
        "midazolam": ["Versed"],
        "naloxone": ["Narcan"],
        "nicardipine": ["Cardene"],
        "nitroglycerin": ["nitro", "NTG", "GTN", "glyceryl trinitrate", "SL nitro"],
        "nitrous oxide": ["nitrous", "N2O", "laughing gas", "Nitronox"],
        "norepinephrine": ["Levophed", "noradrenaline", "levo"],
        "ondansetron": ["Zofran"],
        "oxygen": ["O2"],
        "oxymetazoline": ["Afrin"],
        "phenobarbital": ["phenobarb"],
        "phenylephrine": ["Neo-Synephrine", "neo"],
        "oxytocin": ["Pitocin"],
        "pralidoxime": ["2-PAM", "Protopam", "pralidoxime chloride"],
        "racemic epinephrine": ["racemic epi", "rac epi", "racemic", "nebulized epinephrine"],
        "prednisone": [],
        "procainamide": ["Pronestyl"],
        "promethazine": ["Phenergan"],
        "proparacaine": [],
        "pseudoephedrine": ["Sudafed"],
        "rocuronium": ["Zemuron", "roc"],
        "sodium bicarbonate": ["bicarb", "NaHCO3", "sodium bicarb", "amp of bicarb"],
        "sodium chloride 0.9%": ["normal saline", "NS", "saline", "0.9 saline"],
        "sodium chloride 3%": ["hypertonic saline", "3% saline"],
        "sodium thiosulfate": [],
        "succinylcholine": ["sux", "Anectine", "sch", "succ"],
        "terbutaline": ["Brethine"],
        "tetracaine": [],
        "thiamine": ["vitamin B1", "B1"],
        "tissue plasminogen activator": ["tPA", "TPA", "alteplase", "Activase"],
        "tranexamic acid": ["TXA", "Cyclokapron"],
        "vecuronium": ["Norcuron", "vec"],
    },
 
    # -------------------------------------------------------------------------
    # HOME / MAINTENANCE MEDICATIONS
    # Drugs patients report taking (vs. EMS-administered drugs above). Common
    # in the PMH portion of a report; useful for the model to recognize as meds.
    # -------------------------------------------------------------------------
    "home_medications": {
        "metformin": ["Glucophage"],
        "lisinopril": ["Prinivil", "Zestril"],
        "apixaban": ["Eliquis"],
        "rivaroxaban": ["Xarelto"],
        "warfarin": ["Coumadin", "blood thinner"],
        "clopidogrel": ["Plavix"],
        "atorvastatin": ["Lipitor"],
        "donepezil": ["Aricept"],
        "levothyroxine": ["Synthroid"],
        "amlodipine": ["Norvasc"],
        "insulin": ["Lantus", "Humalog", "Novolog"],
        "losartan": ["Cozaar"],
        "gabapentin": ["Neurontin"],
        "levetiracetam": ["Keppra"],
        "albuterol inhaler": ["rescue inhaler"],
    },
 
    # -------------------------------------------------------------------------
    # CODE STATUS & ADVANCE DIRECTIVES
    # -------------------------------------------------------------------------
    "code_status": {
        "do not resuscitate": ["DNR", "DNAR", "do not attempt resuscitation",
                              "no code"],
        "do not intubate": ["DNI"],
        "full code": ["full resuscitation", "all measures"],
        # "full treatment" is a POLST section, not a code status: a patient can be DNR with full treatment
        "physician orders for life-sustaining treatment": ["POLST", "MOLST"],
        "advance directive": ["living will", "health care directive",
                             "health care proxy", "durable power of attorney"],
        "comfort measures only": ["CMO", "comfort care", "palliative"],
        "surrogate decision maker": ["next of kin", "power of attorney", "POA"],
    },
}
 
 
# =============================================================================
# CLINICAL ANNOTATIONS
# =============================================================================
 
CLINICAL_NOTES: Dict[str, str] = {
    # vitals
    "blood pressure": "Systolic/diastolic in mmHg. Normal ~120/80. Hypotension suggests shock.",
    "heart rate": "Beats/min. Normal 60-100. >100 suggests shock, pain, or cardiac cause.",
    "respiratory rate": "Breaths/min. Normal 12-20. >20 suggests distress, acidosis, or pain.",
    "oxygen saturation": "Normal >=95%. <94% may need intervention; <90% is critical. Always record the delivery method.",
    "Glasgow Coma Scale": "Score 3-15 = Eye(1-4) + Verbal(1-5) + Motor(1-6). <8 implies airway compromise.",
    "blood glucose": "mg/dL. <70 hypoglycemic, >200 hyperglycemic. Reversible cause of altered mental status.",
 
    # alert types
    "ST-elevation myocardial infarction": "TIME-CRITICAL. Target door-to-balloon <90 min. Activate cath lab.",
    "non-ST-elevation myocardial infarction": "Serious but not an immediate cath lab activation.",
    "acute coronary syndrome": "Umbrella term covering STEMI, NSTEMI, and unstable angina.",
    "cerebrovascular accident": "TIME-CRITICAL. Establish last-known-well time. Activate stroke team.",
    "transient ischemic attack": "Symptoms resolve <24h. Still needs workup; predicts future stroke.",
    "urinary tract infection": "Common sepsis source in older adults. Cloudy or foul urine plus altered mental status is suggestive.",
    "indwelling urinary catheter": "Infection risk; note cloudy or foul urine. A common sepsis source.",
    "skilled nursing facility": "Request the transfer sheet and baseline mental status; residents carry resistant organisms.",
    "sepsis": "TIME-CRITICAL. Infection plus organ dysfunction. Cultures then antibiotics; fluids.",
    "cardiac arrest": "Record downtime, witnessed status, bystander CPR, and any ROSC.",
    "motor vehicle collision": "Assess for multi-system injury. Rollover, ejection, or high speed implies trauma activation.",
 
    # cardiac / respiratory
    "atrial fibrillation": "Irregular rhythm; raises stroke risk. Note rate control and anticoagulation.",
    "ventricular fibrillation": "Shockable arrest rhythm.",
    "ventricular tachycardia": "Shockable if pulseless; may deteriorate to arrest.",
    "congestive heart failure": "Prone to fluid overload; be cautious with aggressive fluid boluses.",
    "chronic obstructive pulmonary disease": "High exacerbation risk. Titrate oxygen; avoid over-oxygenation.",
    "tension pneumothorax": "Immediately life-threatening. Needs decompression.",
 
    # neuro
    "altered mental status": "Compare against the patient's baseline, especially in dementia.",
    "traumatic brain injury": "Watch for Cushing triad (rising BP, falling HR, irregular respirations).",
 
    # interventions
    "intravenous access": "Record gauge and site. 14-18G = large bore, 20G standard, 22-24G small.",
    "endotracheal intubation": "Definitive airway. Record tube size, depth, and confirmation method.",
    "supplemental oxygen": "Always pair the saturation with the delivery device and flow rate. Blow-by = mask or tubing held near a child's face, low effective FiO2.",
    "racemic epinephrine": "Nebulized for croup or stridor in children. Symptoms can rebound in 2-4 hours; the child needs observation.",
    "dexamethasone": "Steroid given for croup (0.6 mg/kg, max 10-16 mg), asthma, and allergic reactions; onset in hours.",
    "tourniquet": "Record anatomical site and time applied -- the ED needs the clock.",
 
    # code status
    "do not resuscitate": "Legally binding. Requires documentation on scene. Does NOT mean withhold all treatment.",
    "physician orders for life-sustaining treatment": "Portable medical order. May permit full treatment while still declining CPR -- read it, do not assume.",
    "full code": "Only when stated. A report that says nothing about a directive has no code status.",
}
 