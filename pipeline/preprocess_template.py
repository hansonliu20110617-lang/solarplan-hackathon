"""Topic-agnostic preprocessing scaffold: fit on train only, apply identically everywhere.

Assumptions (replace fit_preprocess/apply_preprocess together, keep this list updated):
- Features are a dense numeric 2D array (n_samples, n_features), no NaNs,
  no categorical/text columns needing encoding.
- Z-score normalization (per-feature mean/std) is an adequate placeholder
  transform; real data may need log scaling, clipping, imputation, or
  categorical encoding instead.
- Feature count/order is fixed across train/val/test/real data, so params
  fit on train apply unchanged to any later split.
"""

import os

import numpy as np


def fit_preprocess(X_train):
    """Compute preprocessing params from train data only (never from val/test/real data)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant features
    return {"mean": mean, "std": std}


def apply_preprocess(X, params):
    """Apply fitted params to any split. Same function for train/val/test/real
    data on purpose: two separate code paths that are "supposed to do the same
    thing" tend to drift apart and are the usual source of train/real mismatch.
    """
    return (X - params["mean"]) / params["std"]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(here, "data.npz")

    if not os.path.exists(npz_path):
        print(f"No {npz_path} found; run simulate_template.py first.")
    else:
        data = np.load(npz_path)
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]

        params = fit_preprocess(X_train)
        X_train_p = apply_preprocess(X_train, params)
        X_val_p = apply_preprocess(X_val, params)
        X_test_p = apply_preprocess(X_test, params)

        print(f"fit on train: mean={params['mean']}, std={params['std']}")
        print(f"train: {X_train.shape} -> {X_train_p.shape}")
        print(f"val:   {X_val.shape} -> {X_val_p.shape}")
        print(f"test:  {X_test.shape} -> {X_test_p.shape}")
