"""The intern-era medical language helpers, now under test."""

from ems_handoff.lexicon_rag import LexiconStore, whisper_hint
from ems_handoff.samples import SAMPLE_SEPSIS, SAMPLE_STEMI, SAMPLE_TRAUMA
from ems_handoff.transcript_corrector import assumption_notes, correct_transcript


def test_known_misspellings_are_fixed_and_case_kept():
    r = correct_transcript("Meds include Ferosemide and aliquis.")
    assert "Furosemide" in r.text and "apixaban" in r.text.lower()
    assert assumption_notes(r) == ["Furosemide (assumed from 'Ferosemide')", "Apixaban (assumed from 'aliquis')"]
    assert correct_transcript("She takes Eliquis for AFib.").text == "She takes Eliquis for AFib."   # the brand name itself is fine


def test_radio_corpus_mishears_are_fixed():
    # CS014 (Medics 729, 3-month-old with croup) as Whisper heard it
    heard = ("sats low 90s, we're currently administering glow-by-oxygen, and we administer 2 mg of "
             "texamethasone, looking for orders for 2.5 mg for CMA-kepi.")
    r = correct_transcript(heard)
    low = r.text.lower()
    assert "blow-by-oxygen" in low and "dexamethasone" in low and "racemic epinephrine" in low
    assert "glow" not in r.text and "texa" not in r.text and "kepi" not in r.text.lower()
    assert len(r.corrections) == 3
    store = LexiconStore()
    found = {t.canonical for t in store.retrieve(r.text, top_k=0)}
    assert {"supplemental oxygen", "dexamethasone", "racemic epinephrine"} <= found


def test_context_rules():
    r = correct_transcript("SP02 is 91. Coarse bronchi bilaterally. DNR with a post form.")
    assert "SpO2" in r.text and "Coarse rhonchi" in r.text and "POLST form" in r.text


def test_samples_are_left_alone():
    for sample in (SAMPLE_STEMI, SAMPLE_TRAUMA, SAMPLE_SEPSIS):
        assert correct_transcript(sample).text == sample


def test_stop_phrase_words_are_never_flagged():
    r = correct_transcript("ETA 5 minutes confirm report ready")
    assert not any(f.token.lower() in {"confirm", "report", "ready"} for f in r.flagged)


def test_lexicon_finds_the_important_terms():
    store = LexiconStore()
    found = {t.canonical for t in store.retrieve(SAMPLE_STEMI, top_k=0)}
    assert {"ST-elevation myocardial infarction", "aspirin", "nitroglycerin", "antecubital"} <= found
    assert "MEDICAL LEXICON" in store.build_context(SAMPLE_STEMI)
    assert store.build_context("") == ""


def test_whisper_hint_is_bounded():
    hint = whisper_hint()
    assert hint.startswith("The following is a paramedic radio report.") and len(hint) <= 800
