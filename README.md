# hackathon-prep

Topic-agnostic scaffold for the Veritas AI hackathon (kickoff 2026-11-14,
tracks announced 2026-10-27). Pattern is distilled from `projects/physics-ai`
(read-only reference, not copied) — see `CHECKLIST.md` for the reasoning.

Five stages, each a thin, swappable module with a fixed JSON/npz contract so
they chain without renegotiating interfaces once the real brief lands:

```
pipeline/
  simulate_template.py    acquire or synthesize data -> data/*.npz
  preprocess_template.py  fit/apply feature pipeline, same path train+real
  train_template.py       model-agnostic train loop -> checkpoint, predictions.npz, metrics_summary.json
  validate_template.py    predictions vs independent ground truth -> validation_results.json
  export_template.py      results JSON -> interface/result.html (static, offline)
interface/
  template.html           plain-language results page, reads window.RESULTS_DATA
run_pipeline.py           CLI driver chaining all five stages
```

## Using this once the brief drops

1. Pick the track, swap in real `simulate_template.py` logic for that domain.
2. Everything downstream (train/validate/export) should need little to no
   change if the npz/JSON schemas below are respected.
3. Split work across agents by stage (see CHECKLIST.md's division-of-labor
   section) — one agent per stage, me owning validation/integration.

## Contracts

- `predictions.npz`: `y_true`, `y_pred`, `split` (array of 'train'/'val'/'test')
- `metrics_summary.json`: `{"median_error":, "p95_error":, "worst_error":, "n":}`
- `validation_results.json`: `{"source":, "n_cases":, "median_error":, "p95_error":, "worst_error":, "cases":[{"id","true","pred","error"}], "provenance":{}}`
- `interface/result.html` reads `window.RESULTS_DATA` injected via
  `__RESULTS_JSON__` string substitution (same trick as physics-ai's
  `inject_weights.py`) — no backend needed, works from `file://`.

Run `python run_pipeline.py --help` once stages are filled in for the real
track.
