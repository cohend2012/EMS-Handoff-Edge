"""
samples.py — three realistic paramedic radio reports (STEMI, trauma, sepsis).

Used as practice scripts on the live screen, as one-click text cases in the
app, by the command line (--case), and as test fixtures.
"""

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
Patient has history of hypertension and type 2 diabetes. Takes metformin and lisinopril daily. No known allergies.
We've established an 18-gauge IV left AC, administered 324mg aspirin chewed, and 0.4mg sublingual nitro times two with some relief. Currently running a saline lock.
Patient is alert and oriented, pain currently 6 out of 10, down from 9 out of 10.
Our ETA to your facility is approximately 8 minutes. Requesting cath lab activation.
Unit 41 clear.
"""

SAMPLE_TRAUMA = """
Medic 7 to County General trauma, priority traffic.
Inbound with a 34-year-old female, restrained driver, high-speed MVC with rollover. 
Positive loss of consciousness at scene, approximately 2 minutes per bystanders.
Patient was extricated by fire, c-spine immobilized, on backboard.
Vitals: BP 88/60, heart rate 124, respiratory rate 28 and shallow, SpO2 91% on 15 liters non-rebreather.
GCS is 13, eyes 3, verbal 4, motor 6. Pupils equal and reactive. 
Patient confused but following commands.
Visible injuries include a large scalp laceration with controlled bleeding, deformity to right forearm likely fracture, and she's guarding her abdomen 
with bruising across lower quadrants consistent with seatbelt sign.
Two large bore IVs established, running wide open. No medications administered. 
Bilateral breath sounds present but diminished on right.
Patient is a visitor from out of state, no medical history available, no medications known, allergy status unknown.
ETA 6 minutes, requesting trauma activation.
"""

SAMPLE_SEPSIS = """
Squad 12 to Riverside Emergency.
We have an 82-year-old female from Sunrise Nursing Facility, 
chief complaint altered mental status and fever.
Per nursing staff, patient was in her usual state of dementia baseline until this morning when she became increasingly confused, not recognizing staff, and had a temperature of 102.4.
Vitals: BP 92/58, heart rate 108, respiratory rate 24, SpO2 93% on 2 liters, blood glucose 186.
Patient is oriented to name only, not following commands consistently. 
Skin is warm and flushed. Lungs have coarse rhonchi bilateral bases.
History per facility transfer sheet: Alzheimer's dementia, CHF, atrial fibrillation, chronic kidney disease stage 3. 
Medications include metoprolol, furosemide, donepezil, and eliquis. 
Allergies to penicillin and sulfa drugs.
She has a foley catheter in place, urine appears cloudy.
IV established 20-gauge right hand, 500ml bolus initiated. No other interventions.
Patient is DNR but full treatment per POLST form, which we have with us.
ETA 12 minutes.
"""