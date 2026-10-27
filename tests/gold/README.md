# Gold answers for the sample radio reports

One file per sample case (`quick`, `stemi`, `trauma`, `sepsis`). Each lists the
facts a receiving ED must get right, with a clinical weight and how to check it:

| kind | expect | passes when |
|---|---|---|
| `enum` | string | the field equals it (case-insensitive) |
| `int` | number | the field, or the first integer in it, equals it |
| `contains` | list of phrases | the field text contains any phrase |
| `list` | list of alias lists | every alias list is matched by some item; extra items count as hallucinations |
| `unknown` | — | the field is null, empty, or says unknown. Anything else is a hallucination |
| `gcs` | total, optional breakdown | total matches; a breakdown that was never spoken is a hallucination |

Weights: 3 = wrong answer changes treatment (alert type, code status, allergies,
medications, doses, destination); 2 = important (vitals, demographics, activation);
1 = nice to have. `scripts/benchmark.py` scores saved reports against these.
