"""The call-set scorer: word error rate, burst splitting, alert truth, and the report table."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("score_callsets", ROOT / "scripts" / "corpus" / "score_callsets.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_normalize_makes_number_words_digits_and_drops_punctuation():
    assert sc.normalize("Priority two, 95% on room-air. BP 188/111!") == \
        ["priority", "2", "95", "percent", "on", "room", "air", "bp", "188", "over", "111"]
    assert sc.normalize("Medic seven thirty one") == ["medic", "7", "30", "1"]
    assert sc.normalize("It's the patient's") == ["it's", "the", "patient's"]


def test_wer_counts_substitutions_insertions_and_deletions():
    assert sc.wer("priority 2 chest pain", "priority to chest pain") == (0.25, 1, 4)
    assert sc.wer("priority 2 chest pain", "priority two chest pain") == (0.0, 0, 4), "number words are the digits"
    assert sc.wer("a b c d", "a c d") == (0.25, 1, 4)
    assert sc.wer("a b c d", "a b x c d") == (0.25, 1, 4)
    assert sc.wer("", "") == (0.0, 0, 0) and sc.wer("", "x y")[0] == 1.0


def test_session_transcript_splits_at_burst_headers():
    t = "[Burst 1 · 00:29:33 UTC · feed 4390 · S054_01] 50 words\nShady Grove is online.\n\n[Burst 2 · 00:29:56 UTC · feed 4430 · S055_01] 3 words\nHow old? Sorry.\n"
    assert sc.burst_texts_from_session(t) == ["Shady Grove is online.", "How old? Sorry."]
    assert sc.burst_texts_from_session("plain text") == ["plain text"]
    assert sc.burst_texts_from_session("") == []


def test_alert_truth_maps_the_stated_priority_and_eta():
    gold = {"stated": {"priority": 2, "eta_minutes": 5}}
    t = sc.alert_truth(gold, {"alert_level": "RED", "eta_minutes": 5, "alert_type": "STEMI"})
    assert t["expect_level"] == "YELLOW" and t["level_ok"] is False and t["eta_ok"] is True
    t = sc.alert_truth({"stated": {}}, {"alert_level": "RED", "eta_minutes": None})
    assert t["level_ok"] is None and t["eta_ok"] is None, "nothing stated, nothing judged"


def test_scores_from_saved_summaries_and_renders_markdown(tmp_path):
    callsets = tmp_path / "callsets"; (callsets / "CSX").mkdir(parents=True)
    (callsets / "CSX" / "callset_info.json").write_text(json.dumps({"callset_id": "CSX", "units": ["731"], "hospital": ["x"], "total_words": 4,
                                                                    "bursts": [{"segment_id": "S1"}, {"segment_id": "S2"}]}))
    (callsets / "CSX" / "S1.txt").write_text("priority to chest pain")
    ref = tmp_path / "reference" / "CSX"; ref.mkdir(parents=True)
    (ref / "S1.ref.txt").write_text("priority 2 chest pain")
    (ref / "S1.candidates.json").write_text(json.dumps({"_audio_s": 3, "groq": "priority 2 chest pain", "vosk": "priority teach has pain", "bad": "<error: x>"}))
    gold = tmp_path / "gold"; gold.mkdir()
    (gold / "CSX.json").write_text(json.dumps({"case": "CSX", "stated": {"priority": 2, "eta_minutes": 5}, "fields": [
        {"path": "alert_level", "kind": "enum", "expect": "YELLOW", "weight": 3},
        {"path": "eta_minutes", "kind": "int", "expect": 5, "weight": 2},
        {"path": "history.allergies", "kind": "unknown", "weight": 3}]}))
    rec = tmp_path / "rec" / "2026-10-25_abc1234"; rec.mkdir(parents=True)
    (rec / "CSX.summary.json").write_text(json.dumps({"commit": "abc1234", "provider": "groq", "model": "m", "session": {
        "transcript": "[Burst 1 · 00:00:00 UTC · feed 1] 4 words\npriority to chest pain\n\n[Burst 2 · 00:00:10 UTC · feed 1] 1 words\nok\n",
        "report": {"alert_type": "MEDICAL", "alert_level": "RED", "eta_minutes": "5 min", "history": {"allergies": None}}}}))
    md = tmp_path / "out.md"
    sc.main(["CSX", "--callsets-dir", str(callsets), "--reference-dir", str(tmp_path / "reference"), "--gold-dir", str(gold),
             "--summaries", str(tmp_path / "rec" / "*"), "--md", str(md)])
    text = md.read_text()
    assert "| groq | 0% | - | 0% (0/4) |" in text, "burst 2 has no reference yet: a dash"
    assert "| stage3 | 25% | - |" in text and "| app abc1234 | 25% | - |" in text
    assert "bad" not in text, "a failed candidate is left out"
    assert "RED (YELLOW) **miss**" in text and "5 (5) |" in text
    assert "alert_level wrong (RED)" in text
