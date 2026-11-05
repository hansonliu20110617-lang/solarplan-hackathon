"""Topic-agnostic validation scaffold: compare model predictions against an
*independent* ground-truth source, not just the model's own test split.

The hackathon lesson this encodes: a model's reported test-split accuracy and
"did it hold up against ground truth it never touched" are different claims.
A held-out split from the same pipeline can still share subtle biases with
train (same sensor, same labeling process, same synthetic generator). Real
validation means checking against a source the model/training process never
saw at all -- a public dataset, a physical constant, an independently
measured reference value, etc.

Assumptions (replace load_ground_truth's body, keep this list updated):
- Ground truth is one scalar per case, matched to predictions by id (here:
  the row index into predictions.npz).
- load_ground_truth() is the single seam: swap its body for a real external
  source once the track is known; keep returning the same dict shape
  ({"ids": [...], "true_values": ndarray}) so the rest of this file does not
  need to change.
- predictions.npz is produced by a sibling training stage (not this file).
  If it is missing -- e.g. running this scaffold standalone before that
  stage exists -- a trivial toy stand-in is synthesized so the demo still
  runs end to end.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(HERE, "predictions.npz")
RESULTS_PATH = os.path.join(HERE, "validation_results.json")


def _load_predictions(seed=0):
    """Load pipeline/predictions.npz, or synthesize a toy stand-in if the
    sibling training stage hasn't produced it yet (standalone-demo fallback
    only -- the real training stage owns this file otherwise).
    """
    if os.path.exists(PREDICTIONS_PATH):
        data = np.load(PREDICTIONS_PATH, allow_pickle=True)
        return {
            "y_true": np.asarray(data["y_true"], dtype=float),
            "y_pred": np.asarray(data["y_pred"], dtype=float),
            "split": np.asarray(data["split"]),
        }

    rng = np.random.default_rng(seed)
    n = 50
    y_true = rng.uniform(0.0, 10.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.3, size=n)
    split = np.array(["test"] * n)
    return {"y_true": y_true, "y_pred": y_pred, "split": split}


def load_ground_truth() -> dict:
    """PLACEHOLDER -- wire this up to a real external ground-truth source once
    the hackathon track is known (e.g. a held-out real-world reference table,
    a public dataset the model never trained/tuned on, a known physical
    constant). The point of this function: it must come from somewhere the
    training pipeline above never touched.

    For this runnable demo, it fabricates a handful of "externally measured"
    true values by lightly perturbing a few real test-split y_true values --
    standing in for an independent remeasurement of the same cases.

    Returns {"ids": [...], "true_values": ndarray}, where each id indexes
    into the predictions arrays from _load_predictions().
    """
    preds = _load_predictions()
    test_ids = np.where(preds["split"] == "test")[0]
    k = min(5, len(test_ids))
    chosen = test_ids[:k]

    rng = np.random.default_rng(123)  # deliberately not the predictions seed/source
    independent_noise = rng.normal(0.0, 0.15, size=k)
    true_values = preds["y_true"][chosen] + independent_noise

    return {"ids": [int(i) for i in chosen], "true_values": true_values}


def validate(predictions, ground_truth):
    """Compare ground truth to the model's predictions at matching ids."""
    ids = ground_truth["ids"]
    true_values = np.asarray(ground_truth["true_values"], dtype=float)
    pred_values = predictions["y_pred"][ids]

    errors = np.abs(true_values - pred_values)

    cases = [
        {
            "id": ids[i],
            "true": float(true_values[i]),
            "pred": float(pred_values[i]),
            "error": float(errors[i]),
        }
        for i in range(len(ids))
    ]

    return {
        "n_cases": len(ids),
        "median_error": float(np.median(errors)),
        "p95_error": float(np.percentile(errors, 95)),
        "worst_error": float(np.max(errors)),
        "cases": cases,
    }


if __name__ == "__main__":
    predictions = _load_predictions()
    ground_truth = load_ground_truth()
    result = validate(predictions, ground_truth)

    output = {
        "source": (
            "PLACEHOLDER synthetic external reference (load_ground_truth() in "
            "validate_template.py) -- swap for a real independent source once "
            "the track is known"
        ),
        **result,
        "provenance": {
            "predictions_source": "pipeline/predictions.npz",
            "note": "no scientific values recomputed here beyond simple error arithmetic",
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {RESULTS_PATH}")
    print(
        f"n_cases={output['n_cases']}  "
        f"median_error={output['median_error']:.4f}  "
        f"p95_error={output['p95_error']:.4f}  "
        f"worst_error={output['worst_error']:.4f}"
    )
