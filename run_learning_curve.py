"""
A properly controlled version of the generalization experiment -- a review
from another model (relayed through Grok Bot) correctly pointed out that
run_leave_one_city_out.py's 3-city vs 10-city comparison changed two things
at once: the number of training cities AND which city was held out. That
confounds "more locations helps" with "a different held-out city happened
to be easier."

This script holds ONE city out (Miami, the city with the real, confirmed
generalization gap -- see the wet/dry season split in the README) and
varies ONLY how many of the other 9 cities train the model: 2, 5, and 9.
At each size, it draws 3 different random subsets (seeded, reproducible)
and trains with 3 different MLP init seeds each, so the result is a mean
and spread, not one run that random chance could have been kind to.

That's 3 sizes x 3 subset-seeds x 3 init-seeds = 27 training runs. Each one
is independent in the training-set-size sense; the subset draws are what
let "how many cities" vary while controlling for "it just happened to be
an easy/hard random subset" -- EXCEPT at size=9, where there are only 9
non-Miami cities total, so all three "draws" are necessarily the identical
city set (confirmed: each init seed gives exactly the same result across
all three nominal draws). Size=9's spread therefore reflects only random
initialization noise, not subset-sampling noise, while sizes 2 and 5
reflect both. This doesn't change the reported means (averaging the same
three values three times over doesn't move the average), but report it
honestly as 3 unique (subset, seed) combinations at size=9, not 9.
"""
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

HOLDOUT_CITY = "miami"
OTHER_CITIES = ["phoenix", "seattle", "chicago", "denver", "minneapolis",
                 "albuquerque", "houston", "losangeles", "boston"]
TRAIN_SIZES = [2, 5, 9]
N_SUBSET_DRAWS = 3
N_INIT_SEEDS = 3
TRAIN_YEARS = ",".join(str(y) for y in range(2005, 2023))
VAL_YEARS = "2023"
TEST_YEARS = "2024"


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        print(f"Command failed: {' '.join(cmd)} -- stopping.")
        sys.exit(result.returncode)
    return result.stdout


def main():
    rng = random.Random(42)
    results = []

    for train_size in TRAIN_SIZES:
        for draw_idx in range(N_SUBSET_DRAWS):
            subset = rng.sample(OTHER_CITIES, train_size)
            cities_arg = ",".join(subset + [HOLDOUT_CITY])
            for init_seed in range(N_INIT_SEEDS):
                label = f"size={train_size} draw={draw_idx} seed={init_seed}"
                print(f"\n=== {label}: training cities = {subset} ===", flush=True)
                run([sys.executable, str(PIPELINE / "prepare_solar_data.py"),
                     "--train-years", TRAIN_YEARS, "--val-years", VAL_YEARS,
                     "--test-years", TEST_YEARS, "--cities", cities_arg,
                     "--holdout-city", HOLDOUT_CITY])
                run([sys.executable, str(PIPELINE / "preprocess_template.py")])
                run([sys.executable, str(PIPELINE / "train_template.py"), "--seed", str(init_seed)])
                run([sys.executable, str(PIPELINE / "validate_template.py")])

                with open(PIPELINE / "validation_results.json") as f:
                    result = json.load(f)
                miami = result["per_city"][HOLDOUT_CITY]
                row = {
                    "train_size": train_size,
                    "draw_idx": draw_idx,
                    "init_seed": init_seed,
                    "training_cities": subset,
                    "model_decision_accuracy": miami["decision"]["model"]["correct_rate"],
                    "baseline_decision_accuracy": miami["decision"]["baseline"]["correct_rate"],
                    "skill_score": miami["skill_score"],
                }
                results.append(row)
                print(f"  -> decision_acc={row['model_decision_accuracy']:.1%} "
                      f"(baseline {row['baseline_decision_accuracy']:.1%}) "
                      f"skill={row['skill_score']:+.4f}")

    print("\n=== Learning curve summary (Miami held out, varying training-city count) ===")
    for train_size in TRAIN_SIZES:
        rows = [r for r in results if r["train_size"] == train_size]
        accs = [r["model_decision_accuracy"] for r in rows]
        skills = [r["skill_score"] for r in rows]
        mean_acc, spread_acc = sum(accs) / len(accs), max(accs) - min(accs)
        mean_skill, spread_skill = sum(skills) / len(skills), max(skills) - min(skills)
        print(f"{train_size} training cities (n={len(rows)} runs): "
              f"decision_acc mean={mean_acc:.1%} range={spread_acc:.1%}, "
              f"skill mean={mean_skill:+.4f} range={spread_skill:.4f}")

    out_path = ROOT / "learning_curve_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
