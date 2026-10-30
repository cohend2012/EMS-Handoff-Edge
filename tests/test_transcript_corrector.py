

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
