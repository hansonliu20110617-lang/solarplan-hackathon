# Hackathon ML pipeline checklist

Distilled from `projects/physics-ai` (read-only reference — not copied code, just
the pattern that made real data actually work there). Track/brief unknown until
2026-10-27 / 2026-11-14, so this is topic-agnostic. Fill in the five stage
templates under `pipeline/` once the brief drops.

## The five stages

1. **Acquire/simulate** (`simulate_template.py`) — real data if it exists for the
   track, synthetic otherwise. Every simplifying assumption gets written down in
   the module docstring *before* training starts, not discovered later.
2. **Preprocess** (`preprocess_template.py`) — the exact same function object
   applied to train, val, real-test. Never two code paths that are "supposed to"
   do the same thing — that gap is where domain gap hides.
3. **Train** (`train_template.py`) — size the model to the data you actually
   have, not the data you wish you had. Early stopping on val loss. Save
   predictions + metrics as JSON, not just stdout.
4. **Validate against independent ground truth** (`validate_template.py`) —
   report median / 95th-percentile / worst error, not just one aggregate loss
   number. If there's any real/held-out ground truth at all, test against it
   even if small.
5. **Export/explain** (`export_template.py` + `interface/template.html`) — bake
   results JSON into a static HTML page via string substitution. No backend,
   works offline, demos in two minutes. Provenance line in every generated
   file: which script, which source JSON, "no values recomputed here."

## Hard-won rules (from physics-ai's actual failure/fix history)

- A synthetic-to-real mismatch is usually **physical, not statistical** — look
  for a missing effect in the simulator before tuning the model.
- Split held-out data by **real unit** (event/subject/session), never by
  crop/window, or validation is inflated by near-duplicates.
- A symmetry/invariance argument *before* fitting beats throwing more model at
  a hard regression — constrains the search space correctly instead of hoping.
- Normalize per-channel/per-row, not globally, when channels can differ by
  orders of magnitude — one loud channel silently dominates otherwise.
- "Detected" and "detected and consistent on a second independent test" are
  different claims — report both when you can, it's a stronger result and
  judges notice the distinction.

## Division of labor for the next 48 hours once the brief drops

- Me (orchestrator + one active contributor): read the brief, pick the track,
  assign stage ownership below, and personally own whichever stage is the
  critical path (usually validation — it's where the actual score comes from).
- Spin up one Claude Code agent per pipeline stage via the Agent tool once
  there's real data to point them at (`fork` for anything that needs this
  session's context, fresh `general-purpose` agents for self-contained stages).
- A short critic pass at the end: one agent (or me) checks that stage outputs
  actually chain — train's JSON schema matches what validate expects, matches
  what export expects — before the demo, not during it.
