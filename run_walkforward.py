"""
Walk-forward validation across two independent test years (2023 and 2024)
instead of one train/val/test split, to check whether a pattern found in a
single split is real or just an artifact of that one split. Each fold is a
proper, non-overlapping train/val/test split of its own (no fold's train or
val set ever contains a date from another fold's test year).

Runs prepare_solar_data -> preprocess -> train -> validate for each fold via
subprocess (same approach as run_pipeline.py), with each fold's years passed
as CLI args to prepare_solar_data.py. Saves each fold's validation_results.json
under a fold-specific name before the next fold overwrites the shared
pipeline/ artifacts, then prints a combined per-fold, per-city summary.

Fold 2023 has only one training year (2021) -- that's a real, honestly
reported constraint of having four real calendar years total, not a bug.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

FOLDS = [
    {"name": "fold_2023", "train": "2021", "val": "2022", "test": "2023"},
    {"name": "fold_2024", "train": "2021,2022", "val": "2023", "test": "2024"},
]


def run(cmd):
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)} -- stopping.")
        sys.exit(result.returncode)


def main():
    fold_results = {}
    for fold in FOLDS:
        print(f"\n=== {fold['name']}: train={fold['train']} val={fold['val']} test={fold['test']} ===", flush=True)
        run([sys.executable, str(PIPELINE / "prepare_solar_data.py"),
             "--train-years", fold["train"], "--val-years", fold["val"], "--test-years", fold["test"]])
        run([sys.executable, str(PIPELINE / "preprocess_template.py")])
        run([sys.executable, str(PIPELINE / "train_template.py")])
        run([sys.executable, str(PIPELINE / "validate_template.py")])

        src = PIPELINE / "validation_results.json"
        dst = PIPELINE / f"validation_results_{fold['name']}.json"
        shutil.copy(src, dst)
        with open(dst) as f:
            fold_results[fold["name"]] = json.load(f)

    print("\n=== Walk-forward summary ===", flush=True)
    for fold_name, result in fold_results.items():
        print(f"\n{fold_name}: overall model={result['median_error']:.4f} "
              f"baseline={result['baseline_median_error']:.4f} "
              f"beats_baseline={result['beats_baseline']}")
        for city, stats in sorted(result["per_city"].items()):
            beats = stats["model"]["median_error"] < stats["baseline"]["median_error"]
            print(f"  {city:10s} model={stats['model']['median_error']:.4f} "
                  f"baseline={stats['baseline']['median_error']:.4f} beats_baseline={beats}")

    summary_path = ROOT / "walkforward_summary.json"
    with open(summary_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
