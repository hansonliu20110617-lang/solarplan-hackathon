"""
The generalization test almost no hackathon team runs, round two: does
training on MANY locations (not just 2) let the model generalize to a city
it's never seen?

The first version of this experiment (3 cities, train on 2, test on the
3rd) found a hard collapse: held-out decision accuracy fell to a coin flip
or worse, even after switching to clear-sky-index normalization specifically
to reduce location/season confounding. The research behind that change
(Cargan et al., Applied Intelligence 2024 -- a "global" model trained
across 20 UK sites generalized; foundation models trained on many sites
degrade only slightly at unseen sites, while a small model trained on few
sites nearly doubles its error) pointed at a different culprit: not the
units predictions are made in, but how few distinct locations the model
ever saw. This script tests that directly with ten real NASA POWER
locations (desert, coastal, continental, humid subtropical, tropical,
high-altitude, Mediterranean) instead of three.

For a representative, climate-diverse subset of holdout cities (not all
ten, to keep runtime reasonable -- each fold now trains on 9 cities' full
18-year history): train on the other 9, test on the one held out, and
compare against a fresh in-distribution reference trained on all 10 (no
holdout) so "generalization gap" is measured against the same many-city
model family, not the old 3-city one.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

ALL_TEN = "phoenix,seattle,chicago,denver,miami,minneapolis,albuquerque,houston,losangeles,boston"
# Chosen for climate spread, not convenience: desert (phoenix), volatile coastal
# (seattle), tropical (miami), snowy continental (minneapolis).
HOLDOUT_SUBSET = ["phoenix", "seattle", "miami", "minneapolis"]
TRAIN_YEARS = ",".join(str(y) for y in range(2005, 2023))
VAL_YEARS = "2023"
TEST_YEARS = "2024"


def run(cmd):
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)} -- stopping.")
        sys.exit(result.returncode)


def main():
    print("=== in-distribution reference: all 10 cities, no holdout ===", flush=True)
    run([sys.executable, str(PIPELINE / "prepare_solar_data.py"),
         "--train-years", TRAIN_YEARS, "--val-years", VAL_YEARS, "--test-years", TEST_YEARS,
         "--cities", ALL_TEN])
    run([sys.executable, str(PIPELINE / "preprocess_template.py")])
    run([sys.executable, str(PIPELINE / "train_template.py")])
    run([sys.executable, str(PIPELINE / "validate_template.py")])
    in_dist_path = PIPELINE / "validation_results_10city_in_distribution.json"
    shutil.copy(PIPELINE / "validation_results.json", in_dist_path)
    in_dist = json.load(open(in_dist_path))

    loco_results = {}
    for city in HOLDOUT_SUBSET:
        print(f"\n=== leave-one-city-out (9 of 10 cities): holding out {city} ===", flush=True)
        run([sys.executable, str(PIPELINE / "prepare_solar_data.py"),
             "--train-years", TRAIN_YEARS, "--val-years", VAL_YEARS, "--test-years", TEST_YEARS,
             "--cities", ALL_TEN, "--holdout-city", city])
        run([sys.executable, str(PIPELINE / "preprocess_template.py")])
        run([sys.executable, str(PIPELINE / "train_template.py")])
        run([sys.executable, str(PIPELINE / "validate_template.py")])

        dst = PIPELINE / f"validation_results_loco10_{city}.json"
        shutil.copy(PIPELINE / "validation_results.json", dst)
        with open(dst) as f:
            loco_results[city] = json.load(f)

    print("\n=== Leave-one-city-out summary (trained on 9 of 10 cities) ===", flush=True)
    summary = {}
    for city, result in loco_results.items():
        cd = result["per_city"][city]["decision"]
        model_acc = cd["model"]["correct_rate"]
        baseline_acc = cd["baseline"]["correct_rate"]
        in_dist_acc = in_dist["per_city"][city]["decision"]["model"]["correct_rate"]
        gap = in_dist_acc - model_acc
        entry = {
            "loco_model_decision_accuracy": model_acc,
            "loco_baseline_decision_accuracy": baseline_acc,
            "loco_beats_baseline": model_acc > baseline_acc,
            "in_distribution_decision_accuracy": in_dist_acc,
            "generalization_gap": gap,
        }
        summary[city] = entry
        print(f"{city:12s} never trained on this city (9/10 cities used) -- decision "
              f"accuracy {model_acc:.1%} (baseline {baseline_acc:.1%}), vs. {in_dist_acc:.1%} "
              f"when trained with all 10 (gap: {gap:+.1%})")

    out_path = ROOT / "leave_one_city_out_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
