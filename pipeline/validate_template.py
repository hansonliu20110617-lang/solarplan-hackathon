"""
Validates the solar forecaster against the real, never-trained-on 2024 test
rows -- that's this stage's "independent ground truth": data the training
loop never saw, not fabricated noise (the generic placeholder this file used
to contain). Reports what judges explicitly ask for: a comparison against the
persistence baseline ("tomorrow = today"), and a breakdown by city, since the
whole point of this project is that one city's forecast is far more
trustworthy than the other's.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(HERE, "predictions.npz")
TEST_META_PATH = os.path.join(HERE, "test_meta.npz")
RESULTS_PATH = os.path.join(HERE, "validation_results.json")


def _load_predictions():
    data = np.load(PREDICTIONS_PATH, allow_pickle=True)
    return {
        "y_true": np.asarray(data["y_true"], dtype=float),
        "y_pred": np.asarray(data["y_pred"], dtype=float),
        "split": np.asarray(data["split"]),
    }


def _load_test_meta():
    data = np.load(TEST_META_PATH, allow_pickle=True)
    return {
        "city": np.asarray(data["city"]),
        "date": np.asarray(data["date"]),
        "baseline_pred": np.asarray(data["baseline_pred"], dtype=float),
        "target": np.asarray(data["target"], dtype=float),
        "decision_threshold": np.asarray(data["decision_threshold"], dtype=float),
    }


def _error_stats(errors):
    if len(errors) == 0:
        raise SystemExit(
            "[validate] got an empty error array -- a city filter or split "
            "produced zero test rows, so median/p95/worst are undefined."
        )
    if not np.all(np.isfinite(errors)):
        raise SystemExit(
            "[validate] non-finite value in errors -- a NaN/Inf prediction or "
            "target would otherwise silently write invalid JSON (json.dump "
            "emits non-standard NaN/Infinity tokens) and corrupt every stat "
            "derived from it."
        )
    return {
        # the brief's own suggested project explicitly asks for mean absolute
        # error -- median/p95/worst (this project's usual convention, more
        # robust to the occasional huge storm-front miss) are reported
        # alongside it, not instead of it.
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p95_error": float(np.percentile(errors, 95)),
        "worst_error": float(np.max(errors)),
    }


def _decision_simulation(y_true, y_pred, baseline_pred, threshold):
    """"Enough sun to run a high-draw appliance off solar tonight, or charge
    from the grid instead?" -- the actual decision brief §4.3 asks projects to
    name ("explain what decision your prediction supports"). The oracle
    decision uses the real tomorrow's value; model/baseline decisions use
    their own forecasts. Reports how often each forecast's decision would
    have matched what the homeowner should have actually done, split into
    the two different-cost mistakes: false confidence (told them there'd be
    enough sun; there wasn't -- appliance runs short) and false caution (told
    them there wouldn't be; there was -- a day of free solar wasted)."""
    if not np.all(np.isfinite(threshold)):
        raise SystemExit(
            "[validate] non-finite decision_threshold -- every >= comparison "
            "against it would be False, making every decision look 'correct' "
            "for that row instead of raising an error."
        )
    oracle = y_true >= threshold
    model_decision = y_pred >= threshold
    baseline_decision = baseline_pred >= threshold
    # threshold may be one scalar (per-city call) or a per-row array (pooled
    # call, each row judged against its own city's normal) -- broadcast to
    # the comparison's shape before reducing to one reportable number.
    threshold_for_reporting = float(np.mean(np.broadcast_to(threshold, oracle.shape)))

    def _rates(decision):
        correct = decision == oracle
        false_confidence = decision & ~oracle  # said "enough sun", wasn't
        false_caution = ~decision & oracle  # said "not enough", was
        n = len(oracle)
        return {
            "correct_rate": float(correct.mean()),
            "false_confidence_rate": float(false_confidence.mean()),
            "false_caution_rate": float(false_caution.mean()),
            "n": int(n),
        }

    return {
        "threshold": threshold_for_reporting,
        "oracle_run_off_solar_rate": float(oracle.mean()),
        "model": _rates(model_decision),
        "baseline": _rates(baseline_decision),
    }


def validate(predictions, test_meta):
    # train_template.py concatenates train+val+test in that fixed order when
    # writing predictions.npz, and test_meta.npz was built in the same row
    # order as X_test (prepare_solar_data.py's city loop), so a boolean mask
    # on "test" lines up positionally with test_meta's arrays.
    test_mask = predictions["split"] == "test"
    y_true = predictions["y_true"][test_mask].reshape(-1)
    y_pred = predictions["y_pred"][test_mask].reshape(-1)
    n = len(y_true)
    # test_meta.npz's own arrays are built from flat Python lists in
    # prepare_solar_data.py so they're always 1-D in practice, but a (N, 1)
    # column vector here would silently broadcast against (N,) targets into
    # an (N, N) matrix everywhere below -- check the shape, don't assume it.
    for field in ("baseline_pred", "target", "decision_threshold"):
        if test_meta[field].ndim != 1 or test_meta[field].shape[0] != len(test_meta["city"]):
            raise SystemExit(
                f"[validate] test_meta['{field}'] has shape {test_meta[field].shape}, "
                f"expected a flat array of length {len(test_meta['city'])} -- would "
                "silently broadcast into a wrong-shaped result instead of erroring."
            )
    if n != len(test_meta["city"]) or n != y_pred.shape[0]:
        raise SystemExit(
            f"predictions.npz has {n} test rows but test_meta.npz has "
            f"{len(test_meta['city'])} -- did prepare_solar_data.py run before "
            "the last train run? Re-run the pipeline from prepare_data."
        )
    # Length matching alone can't catch a silent reorder between the two
    # files -- test_meta.npz's own "target" (written straight from the same
    # example that produced y_true) must equal y_true row-for-row, or the
    # per-city/per-case breakdown below would pair the wrong city/date with
    # each prediction without any error.
    if not np.allclose(y_true, test_meta["target"]):
        raise SystemExit(
            "[validate] predictions.npz test rows don't match test_meta.npz's "
            "targets row-for-row -- the two files are out of sync (stale "
            "data.npz/test_meta.npz from different prepare_solar_data.py runs?). "
            "Re-run prepare_solar_data.py then the rest of the pipeline."
        )

    model_errors = np.abs(y_true - y_pred)
    baseline_errors = np.abs(y_true - test_meta["baseline_pred"])

    cities = test_meta["city"]
    thresholds = test_meta["decision_threshold"]
    per_city = {}
    for city in sorted(set(cities.tolist())):
        mask = cities == city
        per_city[city] = {
            "n": int(mask.sum()),
            "model": _error_stats(model_errors[mask]),
            "baseline": _error_stats(baseline_errors[mask]),
            "decision": _decision_simulation(
                y_true[mask], y_pred[mask], test_meta["baseline_pred"][mask], thresholds[mask][0]
            ),
        }

    worst_idx = np.argsort(-model_errors)[:5]
    cases = [
        {
            "id": int(i),
            "city": str(cities[i]),
            "date": str(test_meta["date"][i]),
            "true": float(y_true[i]),
            "pred": float(y_pred[i]),
            "baseline_pred": float(test_meta["baseline_pred"][i]),
            "error": float(model_errors[i]),
        }
        for i in worst_idx
    ]

    model_stats = _error_stats(model_errors)
    baseline_stats = _error_stats(baseline_errors)
    # Pooled decision simulation uses each row's own city threshold (an
    # array, not one shared number) -- comparing Phoenix's abundant sun and
    # Seattle's scarce sun against a single global cutoff wouldn't mean
    # anything; each city is judged against its own normal.
    decision = _decision_simulation(y_true, y_pred, test_meta["baseline_pred"], thresholds)

    return {
        "n_cases": n,
        **model_stats,
        "baseline_mean_error": baseline_stats["mean_error"],
        "baseline_median_error": baseline_stats["median_error"],
        "baseline_p95_error": baseline_stats["p95_error"],
        "baseline_worst_error": baseline_stats["worst_error"],
        "beats_baseline": model_stats["median_error"] < baseline_stats["median_error"],
        "decision": decision,
        "per_city": per_city,
        "cases": cases,
    }


if __name__ == "__main__":
    predictions = _load_predictions()
    test_meta = _load_test_meta()
    result = validate(predictions, test_meta)

    output = {
        "source": (
            "Daily NASA POWER readings for Phoenix AZ, Seattle WA, and Chicago IL "
            "-- held out by calendar year, never touched during training or "
            "early stopping. Units: kWh/m^2/day. Baseline is the persistence "
            "rule (tomorrow's solar = today's solar), computed directly from "
            "the raw data, not from the model."
        ),
        **result,
        "provenance": {
            "predictions_source": "pipeline/predictions.npz",
            "test_meta_source": "pipeline/test_meta.npz (written by prepare_solar_data.py)",
            "note": (
                "baseline_pred is today's actual solar reading for that row, not a "
                "model output. decision_threshold is each city's median TRAINING-period "
                "solar value, so the decision simulation never peeks at test data."
            ),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {RESULTS_PATH}")
    print(
        f"n_cases={output['n_cases']}  "
        f"model_mean={output['mean_error']:.4f}  model_median={output['median_error']:.4f}  "
        f"baseline_mean={output['baseline_mean_error']:.4f}  baseline_median={output['baseline_median_error']:.4f}  "
        f"beats_baseline={output['beats_baseline']}"
    )
    d = output["decision"]
    print(
        f"  decision (pooled): model_correct={d['model']['correct_rate']:.1%}  "
        f"baseline_correct={d['baseline']['correct_rate']:.1%}"
    )
    for city, stats in output["per_city"].items():
        cd = stats["decision"]
        print(
            f"  {city}: n={stats['n']}  model_mean={stats['model']['mean_error']:.4f}  "
            f"baseline_mean={stats['baseline']['mean_error']:.4f}  "
            f"model_decision_correct={cd['model']['correct_rate']:.1%}  "
            f"baseline_decision_correct={cd['baseline']['correct_rate']:.1%}"
        )
