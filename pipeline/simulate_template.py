"""Topic-agnostic data scaffold: generates train/val/test splits as a toy placeholder.

Assumptions (replace the generator, keep this list updated when you do):
- The task is framed as supervised regression on a fixed-size feature vector;
  if the real track turns out to be classification, images, text, etc., this
  whole generator (not just its parameters) needs to be swapped out.
- Features are i.i.d. continuous values with no missingness, no categorical
  columns, and no time/sequence structure.
- The target is a smooth, low-noise function of the features — real data may
  be noisier, multi-modal, or have outliers that this placeholder never models.
- Train/val/test are drawn from the same distribution (no domain shift); real
  deployment data may violate this.
- generate_dataset() is the single seam: swap its body for real acquisition
  (file load, API pull, sensor read, etc.) once the track is known, and keep
  returning the same dict shape so downstream code does not change.
"""

import json
import os

import numpy as np


def generate_dataset(n_train=2000, n_val=400, n_test=400, seed=0):
    """Return {"train": {"X", "y"}, "val": {...}, "test": {...}, "meta": {...}}.

    PLACEHOLDER: toy synthetic regression, y = sin(x0) + x1**2 + noise.
    Swap this body for real data loading when the hackathon track is known;
    keep the returned dict shape identical so preprocessing/training code
    downstream does not need to change.
    """
    rng = np.random.default_rng(seed)

    def make_split(n):
        X = rng.uniform(-3.0, 3.0, size=(n, 2))
        noise = rng.normal(0.0, 0.1, size=n)
        y = np.sin(X[:, 0]) + X[:, 1] ** 2 + noise
        return X, y

    X_train, y_train = make_split(n_train)
    X_val, y_val = make_split(n_val)
    X_test, y_test = make_split(n_test)

    assumptions = [
        "supervised regression on a fixed-size feature vector (2 features here)",
        "features are i.i.d. continuous, no missingness, no categorical columns",
        "target is a smooth low-noise function of the features (toy: sin(x0) + x1**2 + noise)",
        "train/val/test share one distribution; no domain shift modeled",
    ]

    return {
        "train": {"X": X_train, "y": y_train},
        "val": {"X": X_val, "y": y_val},
        "test": {"X": X_test, "y": y_test},
        "meta": {"assumptions": assumptions, "seed": seed},
    }


if __name__ == "__main__":
    dataset = generate_dataset()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(out_dir, "data.npz")
    meta_path = os.path.join(out_dir, "meta.json")

    # npz can't hold nested dicts/strings cleanly, so meta goes in a sidecar JSON.
    np.savez(
        npz_path,
        X_train=dataset["train"]["X"],
        y_train=dataset["train"]["y"],
        X_val=dataset["val"]["X"],
        y_val=dataset["val"]["y"],
        X_test=dataset["test"]["X"],
        y_test=dataset["test"]["y"],
    )
    with open(meta_path, "w") as f:
        json.dump(dataset["meta"], f, indent=2)

    print(f"Saved {npz_path}")
    print(f"Saved {meta_path}")
    for split in ("train", "val", "test"):
        X, y = dataset[split]["X"], dataset[split]["y"]
        print(f"{split}: X{X.shape} y{y.shape}")
