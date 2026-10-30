"""Phase 1 trajectory labels: the word alignment and the loop detector, the pure-Python parts of scripts/corpus/trajectories.py."""
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("trajectories", pathlib.Path("scripts/corpus/trajectories.py"))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)


def test_align_words_labels_correct_substituted_inserted_and_reports_deleted_spans():
    ref = "medic 14 priority 1 stroke alert 57 year old male eta 10".split()
    hyp = "medic 14 priority 1 stroke alert 97 year old male thank you very much".split()
    labels, deleted = T.align_words(ref, hyp)
    assert labels[:6] == ["correct"] * 6
    assert labels[6] == "substituted"                       # 57 -> 97
    assert labels[7:10] == ["correct"] * 3
    assert labels[10:].count("inserted") >= 2                # four words where the reference had two: at least two insertions
    labels2, deleted2 = T.align_words(ref, "medic 14 priority 1 stroke alert 57 year old male".split())
    assert labels2 == ["correct"] * 10 and deleted2 == [(10, 11, "eta 10")]   # the dropped tail is one deletion span


def test_align_words_handles_empty_sides():
    assert T.align_words([], ["a", "b"])[0] == ["inserted", "inserted"]
    labels, deleted = T.align_words(["a", "b", "c"], [])
    assert labels == [] and deleted == [(0, 2, "a b c")]


def test_loop_words_marks_runs_of_three_or_more_repeats_only():
    hyp = "priority 1 protocol priority 1 protocol priority 1 protocol eta 5".split()
    loops = T.loop_words(hyp)
    assert loops == set(range(9))                            # three repeats of a 3-gram
    assert T.loop_words("thank you thank you".split()) == set()   # two repeats are not a loop
    assert T.loop_words("the the the the".split()) == set(range(4))
