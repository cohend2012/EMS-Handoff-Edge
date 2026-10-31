

def test_spelled_nkda_variants_become_nkda():
    from ems_handoff.transcript_corrector import _apply_context_rules
    for heard in ("Medications include metformin. NKD day. Family states full code.",
                  "lisinopril, and aliquiz. NKD-8. Family states",
                  "hypertension. NKDSA. Requesting a STEMI alert",
                  "alert and oriented. NKD, hey, we're about four minutes out",
                  "No past history. NKD is. ETA five",
                  "history includes hypertension. NKD is A. Requesting a STEMI alert"):
        out, applied = _apply_context_rules(heard)
        assert "NKDA" in out and "NKD " not in out and "NKD-" not in out and "NKDSA" not in out, (heard, out)
        assert applied and applied[-1].category == "NKDA spelled"
    out, applied = _apply_context_rules("the NKD is a hospital network")   # "is" mid-sentence is left alone
    assert out == "the NKD is a hospital network" and not applied
    out, _ = _apply_context_rules("NKDA, how do you copy?")               # already right: untouched
    assert out == "NKDA, how do you copy?"


def test_base_hospital_names_from_human_reads():
    from ems_handoff.transcript_corrector import _apply_context_rules, _apply_hospital_frames

    def fix(t):
        t, _ = _apply_context_rules(t)
        return _apply_hospital_frames(t)[0]
    assert fix("CineGroves online. Good evening, Shady Grove.").startswith("Shady Grove online.")
    assert fix("Cady Grove's online. Cady Grove, this is").startswith("Shady Grove online.")
    assert fix("Holy cow, Sturm and Tau copies. We will see you") == "Holy Cross, Germantown copies. We will see you"
    assert fix("We crashed German town copies.") == "Holy Cross Germantown copies."
    assert fix("Great, I'll do this. Suburban L9, this is").endswith("Suburban online, this is")
    assert fix("This is Medic729, Medics 715") == "This is Medic 729, Medic 715"
    for same in ("Shady Grove is online. Good morning", "Holy Cross. Hello, Holy Cross.", "Sure, it's online.",
                 "Holy cow, that hurt. Copies of the ECG", "Holy Cross, Germantown copies."):
        assert fix(same) == same, same
