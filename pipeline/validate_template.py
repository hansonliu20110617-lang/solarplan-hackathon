"""
Validates the solar forecaster against the real, never-trained-on test rows
-- data the training loop never saw. The model is trained to predict the
CLEAR-SKY INDEX (actual/theoretical-max), not raw kWh (see
prepare_solar_data.py for why) -- this file reports error in that native
unit, then converts back to kWh using tomorrow_clrsky (known in advance for
any date/location, not a forecast, so this introduces no leakage) for the
decision simulation and for a directly comparable kWh error figure. Also
reports the forecasting literature's standard skill score (1 - RMSE_model /
RMSE_baseline; e.g. Nguyen & Mu"sgens) so this isn't just an invented metric.
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
        "tomorrow_clrsky": np.asarray(data["tomorrow_clrsky"], dtype=float),
        "tomorrow_actual_kwh": np.asarray(data["tomorrow_actual_kwh"], dtype=float),
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
            "target would otherwise silently write invalid JSON and corrupt "
            "every stat derived from it."
        )
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p95_error": float(np.percentile(errors, 95)),
        "worst_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def _skill_score(model_errors, baseline_errors):
    """Standard solar-forecasting-literature skill score: 1 - RMSE_model /
    RMSE_baseline. Positive means the model beats the baseline; 0 means tied;
    negative means the baseline wins. This is the field's own convention,
    not something invented for this project."""
    model_rmse = float(np.sqrt(np.mean(model_errors ** 2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
    return 1.0 - (model_rmse / baseline_rmse) if baseline_rmse > 0 else float("nan")


def _decision_simulation(true_kwh, model_pred_kwh, baseline_pred_kwh, threshold):
    """"Enough sun to run a high-draw appliance off solar tonight, or charge
    from the grid instead?" -- the actual decision brief §4.3 asks projects to
    name. All three arrays are in kWh (converted from clear-sky-index space
    using the known-in-advance clear-sky ceiling) so the threshold comparison
    is physically meaningful. Reports false confidence (said "enough sun",
    wasn't) and false caution (said "not enough", was) separately, since they
    cost differently (see pipeline/estimate_impact.py)."""
    if not np.all(np.isfinite(threshold)):
        raise SystemExit(
            "[validate] non-finite decision_threshold -- every >= comparison "
            "against it would be False, making every decision look 'correct' "
            "for that row instead of raising an error."
        )
    oracle = true_kwh >= threshold
    model_decision = model_pred_kwh >= threshold
    baseline_decision = baseline_pred_kwh >= threshold
    threshold_for_reporting = float(np.mean(np.broadcast_to(threshold, oracle.shape)))

    def _rates(decision):
        correct = decision == oracle
        false_confidence = decision & ~oracle
        false_caution = ~decision & oracle
        return {
            "correct_rate": float(correct.mean()),
            "false_confidence_rate": float(false_confidence.mean()),
            "false_caution_rate": float(false_caution.mean()),
            "n": int(len(oracle)),
        }

    # Reference point raised in review: 85-94% decision accuracy means
    # nothing on its own if simply always guessing the majority class would
    # already score that high in a lopsided climate (e.g. Phoenix is "enough
    # sun" on the vast majority of days). This is the number any reported
    # accuracy actually needs to beat to mean anything.
    oracle_rate = float(oracle.mean())
    majority_class_accuracy = max(oracle_rate, 1.0 - oracle_rate)

    return {
        "threshold": threshold_for_reporting,
        "oracle_run_off_solar_rate": oracle_rate,
        "majority_class_accuracy": majority_class_accuracy,
        "model": _rates(model_decision),
        "baseline": _rates(baseline_decision),
    }


def validate(predictions, test_meta):
    # train_template.py concatenates train+val+test in that fixed order when
    # writing predictions.npz, and test_meta.npz was built in the same row
    # order as X_test (prepare_solar_data.py's city loop), so a boolean mask
    # on "test" lines up positionally with test_meta's arrays.
    test_mask = predictions["split"] == "test"
    y_true_csi = predictions["y_true"][test_mask].reshape(-1)
    y_pred_csi = predictions["y_pred"][test_mask].reshape(-1)
    n = len(y_true_csi)
    for field in ("baseline_pred", "target", "decision_threshold", "tomorrow_clrsky", "tomorrow_actual_kwh"):
        if test_meta[field].ndim != 1 or test_meta[field].shape[0] != len(test_meta["city"]):
            raise SystemExit(
                f"[validate] test_meta['{field}'] has shape {test_meta[field].shape}, "
                f"expected a flat array of length {len(test_meta['city'])} -- would "
                "silently broadcast into a wrong-shaped result instead of erroring."
            )
    if n != len(test_meta["city"]) or n != y_pred_csi.shape[0]:
        raise SystemExit(
            f"predictions.npz has {n} test rows but test_meta.npz has "
            f"{len(test_meta['city'])} -- did prepare_solar_data.py run before "
            "the last train run? Re-run the pipeline from prepare_data."
        )
    # Length matching alone can't catch a silent reorder between the two
    # files -- test_meta.npz's own "target" (CSI, written straight from the
    # same example that produced y_true) must equal y_true row-for-row.
    if not np.allclose(y_true_csi, test_meta["target"]):
        raise SystemExit(
            "[validate] predictions.npz test rows don't match test_meta.npz's "
            "targets row-for-row -- the two files are out of sync (stale "
            "data.npz/test_meta.npz from different prepare_solar_data.py runs?). "
            "Re-run prepare_solar_data.py then the rest of the pipeline."
        )

    # Convert clear-sky-index predictions back to kWh using the known-in-advance
    # clear-sky ceiling for that row's date/location -- this is arithmetic on a
    # deterministic/astronomical quantity, not a second forecast, so it adds no leakage.
    clrsky = test_meta["tomorrow_clrsky"]
    true_kwh = test_meta["tomorrow_actual_kwh"]
    model_pred_kwh = y_pred_csi * clrsky
    baseline_pred_kwh = test_meta["baseline_pred"] * clrsky

    csi_model_errors = np.abs(y_true_csi - y_pred_csi)
    csi_baseline_errors = np.abs(y_true_csi - test_meta["baseline_pred"])
    kwh_model_errors = np.abs(true_kwh - model_pred_kwh)
    kwh_baseline_errors = np.abs(true_kwh - baseline_pred_kwh)

    cities = test_meta["city"]
    thresholds = test_meta["decision_threshold"]
    per_city = {}
    for city in sorted(set(cities.tolist())):
        mask = cities == city
        per_city[city] = {
            "n": int(mask.sum()),
            "model": _error_stats(kwh_model_errors[mask]),
            "baseline": _error_stats(kwh_baseline_errors[mask]),
            "csi_model": _error_stats(csi_model_errors[mask]),
            "csi_baseline": _error_stats(csi_baseline_errors[mask]),
            "skill_score": _skill_score(kwh_model_errors[mask], kwh_baseline_errors[mask]),
            "decision": _decision_simulation(
                true_kwh[mask], model_pred_kwh[mask], baseline_pred_kwh[mask], thresholds[mask][0]
            ),
        }

    worst_idx = np.argsort(-kwh_model_errors)[:5]
    cases = [
        {
            "id": int(i),
            "city": str(cities[i]),
            "date": str(test_meta["date"][i]),
            "true": float(true_kwh[i]),
            "pred": float(model_pred_kwh[i]),
            "baseline_pred": float(baseline_pred_kwh[i]),
            "error": float(kwh_model_errors[i]),
        }
        for i in worst_idx
    ]

    model_stats = _error_stats(kwh_model_errors)
    baseline_stats = _error_stats(kwh_baseline_errors)
    skill_score = _skill_score(kwh_model_errors, kwh_baseline_errors)
    # Pooled decision simulation uses each row's own city threshold (an
    # array, not one shared number) -- comparing Phoenix's abundant sun and
    # Seattle's scarce sun against a single global cutoff wouldn't mean
    # anything; each city is judged against its own normal.
    decision = _decision_simulation(true_kwh, model_pred_kwh, baseline_pred_kwh, thresholds)

    return {
        "n_cases": n,
        **model_stats,
        "baseline_mean_error": baseline_stats["mean_error"],
        "baseline_median_error": baseline_stats["median_error"],
        "baseline_p95_error": baseline_stats["p95_error"],
        "baseline_worst_error": baseline_stats["worst_error"],
        "baseline_rmse": baseline_stats["rmse"],
        "skill_score": skill_score,
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
            "Daily NASA POWER readings, held out by calendar year (and, in "
            "leave-one-city-out runs, by location) -- never touched during "
            "training or early stopping. The model predicts the clear-sky index; "
            "errors below are reported in kWh/m^2/day after converting back with "
            "a training-years-only climatological clear-sky ceiling. The baseline "
            "is 'smart persistence' in clear-sky-index space (tomorrow's clearness "
            "= today's, scaled by each day's own climatological ceiling) -- a "
            "standard, non-trivial baseline in the solar-forecasting literature, "
            "not naive raw-kWh persistence."
        ),
        **result,
        "provenance": {
            "predictions_source": "pipeline/predictions.npz",
            "test_meta_source": "pipeline/test_meta.npz (written by prepare_solar_data.py)",
            "note": (
                "decision_threshold is each city's median TRAINING-period actual kWh, "
                "so the decision simulation never peeks at test data. skill_score is "
                "1 - RMSE_model/RMSE_baseline, the standard solar-forecasting-literature "
                "definition, not an invented metric."
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
        f"skill_score={output['skill_score']:+.4f}  beats_baseline={output['beats_baseline']}"
    )
    d = output["decision"]
    print(
        f"  decision (pooled): model_correct={d['model']['correct_rate']:.1%}  "
        f"baseline_correct={d['baseline']['correct_rate']:.1%}  "
        f"majority_class={d['majority_class_accuracy']:.1%}"
    )
    for city, stats in output["per_city"].items():
        cd = stats["decision"]
        print(
            f"  {city}: n={stats['n']}  model_mean={stats['model']['mean_error']:.4f}  "
            f"baseline_mean={stats['baseline']['mean_error']:.4f}  skill={stats['skill_score']:+.4f}  "
            f"model_decision_correct={cd['model']['correct_rate']:.1%}  "
            f"baseline_decision_correct={cd['baseline']['correct_rate']:.1%}  "
            f"majority_class={cd['majority_class_accuracy']:.1%}"
        )
