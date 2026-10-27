"""The generated radio reports: deterministic, self-consistent with their gold, scoreable by benchmark.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "corpus"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen_scripts", ROOT / "scripts" / "corpus" / "gen_scripts.py")
bench = _load("benchmark", ROOT / "scripts" / "benchmark.py")


def test_same_seed_and_index_give_the_same_script():
    a = gen.make_one(3, 11)
    b = gen.make_one(3, 11)
    assert a[1] == b[1] and a[2] == b[2]
    assert gen.make_one(4, 11)[1] != a[1]


def test_every_script_passes_its_own_consistency_check_and_every_gold_field_is_scoreable():
    seen = set()
    for i in range(1, 121):
        sid, text, gold, sc = gen.make_one(2, i)        # make_one raises if the gold names a fact the text lacks
        assert gold["case"] == sid and gold["fields"]
        for f in gold["fields"]:
            bucket, _ = bench.check(f, {})               # every kind is one benchmark knows; an empty report is null everywhere
            assert bucket in ("omission", "abstained")
        seen.add(sc.family)
    assert {"stemi", "stroke", "trauma", "sepsis", "arrest", "pediatric", "obstetric", "endocrine", "medical", "psychiatric", "fragment"} <= seen


def test_unspoken_facts_demand_null_and_spoken_ones_are_in_the_text():
    etas = {"int": 0, "unknown": 0}
    for i in range(1, 61):
        sid, text, gold, sc = gen.make_one(5, i)
        by_path = {}
        for f in gold["fields"]:
            by_path.setdefault(f["path"], []).append(f)
        eta = by_path["eta_minutes"][0]
        etas[eta["kind"]] += 1
        if eta["kind"] == "int":
            assert f"{eta['expect']} minutes" in text or f"{eta['expect']} minutes out" in text
        else:
            assert "ETA" not in text
        if not sc.fragment:
            assert by_path["history.allergies"][0]["kind"] == ("contains" if sc.allergies else "unknown")
            assert by_path["history.code_status"][0]["kind"] == ("contains" if sc.code else "unknown")
            gcs = by_path["vital_signs.gcs"][0]
            if gcs["kind"] == "gcs":
                assert (gcs["breakdown"] is not None) == ("eyes" in text or ": " in text.split("GCS")[1].split("\n")[0])
    assert etas["int"] > 0 and etas["unknown"] > 0, "both the spoken and the unspoken ETA must occur"


def test_article_agrees_with_the_age():
    import re
    for i in range(1, 201):
        text = gen.make_one(2, i)[1]
        assert not re.search(r"\b[Aa] (?:8|11|18|8\d)-", text), text
        assert not re.search(r"\b[Aa]n (?:[0-79]|1[0-79]|[2-79]\d)-", text), text


def test_spoken_form_spells_the_abbreviations():
    for i in range(1, 31):
        sid, text, gold, sc = gen.make_one(1, i)
        sp = gen.spoken(text)
        for token in ("SpO2", "%", "/", " IV ", "ETA", "GCS", " mg", " mcg", " mL"):
            assert token not in sp, (token, sp)


def test_generate_writes_the_set_and_refuses_to_overwrite(tmp_path):
    out = tmp_path / "synth_t"
    summary = gen.generate(out, 24, 9)
    assert summary["n"] == 24 and len([p for p in (out / "scripts").glob("SY*.txt") if not p.name.endswith(".spoken.txt")]) == 24
    assert len(list((out / "gold").glob("SY*.json"))) == 24
    sp = json.loads((out / "splits.json").read_text())
    ids = sp["train"] + sp["val"] + sp["test"]
    assert len(ids) == 24 and len(set(ids)) == 24 and sp["test"]
    assert (out / "terms.txt").read_text().strip() and (out / "index.tsv").read_text().count("\n") == 25
    assert gen.main(["--out", str(out), "--n", "5"]) == 2


def test_benchmark_matches_a_generated_gold_by_file_name_or_marker(tmp_path):
    out = tmp_path / "s"
    gen.generate(out, 6, 1)
    golds = {p.stem: json.loads(p.read_text()) for p in (out / "gold").glob("*.json")}
    assert bench.infer_case({}, golds, "SY003.local.summary.json") == "SY003"
    assert bench.infer_case({}, golds, "SY004_cs008_10db.json") == "SY004"
    g = golds["SY002"]
    rep = {"_meta": {"transcript": f"Hello, this is {g['marker']} inbound."}}
    assert bench.infer_case(rep, golds, "run7.json") == "SY002"
    assert bench.infer_case({"_meta": {"transcript": "unit 41 to memorial"}}) == "stemi", "the sample markers still work without --gold"
