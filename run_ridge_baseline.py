"""
Sanity-check baseline: is a from-scratch MLP actually earning its complexity
here, or would plain linear regression do just as well? Trains closed-form
ridge regression (no sklearn needed -- (X^T X + lambda*I)^-1 X^T y) on the
exact same preprocessed features pipeline/data.npz already holds for the
shipped 3-city submission, picks lambda by validation MSE, then reuses
validate_template.py so the comparison (skill score, decision accuracy) is
computed identically to the MLP's.

Only ever run this against data.npz/test_meta.npz already written by
prepare_solar_data.py for the standard 3-city scope -- it doesn't call
prepare_solar_data.py itself, since it's meant to be an apples-to-apples
comparison against whatever's already there, not a separate experiment.

Overwrites predictions.npz/validation_results.json like train_template.py
would, then restores the MLP's by re-running train_template.py +
validate_template.py at the end -- this script is a diagnostic, not a
replacement for what's actually shipped.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def ridge_fit(X, y, lam):
    n_features = X.shape[1]
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # bias via augmented column
    reg = lam * np.eye(n_features + 1)
    reg[-1, -1] = 0.0  # don't regularize the bias term
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return w


def ridge_predict(X, w):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit(result.returncode)
    return result.stdout


def main():
    data_path = PIPELINE / "data.npz"
    if not data_path.exists():
        sys.exit("pipeline/data.npz not found -- run prepare_solar_data.py "
                  "(and preprocess_template.py) first, same as for the MLP.")

    npz = np.load(data_path)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val, y_val = npz["X_val"], npz["y_val"]
    X_test, y_test = npz["X_test"], npz["y_test"]

    best_lam, best_val_mse, best_w = None, np.inf, None
    print("Ridge lambda sweep (selecting by validation MSE):")
    for lam in LAMBDAS:
        w = ridge_fit(X_train, y_train, lam)
        val_mse = float(np.mean((ridge_predict(X_val, w) - y_val) ** 2))
        print(f"  lambda={lam:>8g}  val_mse={val_mse:.6f}")
        if val_mse < best_val_mse:
            best_lam, best_val_mse, best_w = lam, val_mse, w
    print(f"Selected lambda={best_lam} (val_mse={best_val_mse:.6f})")

    y_true_all, y_pred_all, split_all = [], [], []
    test_pred = None
    for name, (X, y) in {"train": (X_train, y_train), "val": (X_val, y_val),
                          "test": (X_test, y_test)}.items():
        pred = ridge_predict(X, best_w)
        if name == "test":
            test_pred = pred
        y_true_all.append(y)
        y_pred_all.append(pred)
        split_all.append(np.full(len(y), name))

    abs_err = np.abs(y_test - test_pred)
    print(f"Ridge test metrics: median_error={np.median(abs_err):.4f} "
          f"p95_error={np.percentile(abs_err, 95):.4f} worst_error={np.max(abs_err):.4f}")

    np.savez(PIPELINE / "predictions.npz",
             y_true=np.concatenate(y_true_all),
             y_pred=np.concatenate(y_pred_all),
             split=np.concatenate(split_all))

    print("\nRunning validate_template.py against the ridge predictions...")
    run([sys.executable, str(PIPELINE / "validate_template.py")])
    shutil.copy(PIPELINE / "validation_results.json", ROOT / "ridge_validation_results.json")
    with open(ROOT / "ridge_validation_results.json") as f:
        ridge_results = json.load(f)
    print(f"\nRidge pooled: skill_score={ridge_results['skill_score']:+.4f}  "
          f"decision model_correct={ridge_results['decision']['model']['correct_rate']:.1%}  "
          f"baseline_correct={ridge_results['decision']['baseline']['correct_rate']:.1%}")
    for city, row in ridge_results["per_city"].items():
        print(f"  {city}: skill={row['skill_score']:+.4f}  "
              f"decision model={row['decision']['model']['correct_rate']:.1%} "
              f"baseline={row['decision']['baseline']['correct_rate']:.1%}")

    print("\nRestoring the shipped MLP's predictions.npz/validation_results.json "
          "(re-running train_template.py + validate_template.py)...")
    run([sys.executable, str(PIPELINE / "train_template.py")])
    run([sys.executable, str(PIPELINE / "validate_template.py")])
    print(f"Done -- MLP results restored. Ridge comparison saved to "
          f"{ROOT / 'ridge_validation_results.json'}")


if __name__ == "__main__":
    main()
