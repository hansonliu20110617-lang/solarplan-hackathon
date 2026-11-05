"""
train_template.py -- topic-agnostic training loop scaffold (Veritas AI hackathon prep).

Model-agnostic: ships a tiny from-scratch numpy MLP + manual Adam step, no
torch/sklearn required. Data-agnostic: tries to load data.npz written by
simulate_template.py and preprocessing from preprocess_template.py, but
falls back to an inline toy regression dataset so this file always runs
standalone. Swap in a real model/dataset later; keep the loop structure
(early stopping, saved artifacts) as-is.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data: load sibling-produced data.npz if present, else regenerate via
# simulate_template.py, else fall back to a trivial synthetic dataset.
# ---------------------------------------------------------------------------

def _toy_dataset(seed=0, n_train=400, n_val=100, n_test=100, n_features=5):
    rng = np.random.default_rng(seed)

    def make(n):
        X = rng.uniform(-3, 3, size=(n, n_features))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.25 * X[:, 2] ** 2
        y = y + rng.normal(0, 0.1, size=n)
        return X.astype(np.float64), y.astype(np.float64)

    X_train, y_train = make(n_train)
    X_val, y_val = make(n_val)
    X_test, y_test = make(n_test)
    return dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)


def load_data():
    data_path = os.path.join(HERE, "data.npz")
    if os.path.exists(data_path):
        npz = np.load(data_path)
        return {k: npz[k] for k in npz.files}

    try:
        import simulate_template
        if hasattr(simulate_template, "main"):
            simulate_template.main()
        elif hasattr(simulate_template, "simulate"):
            simulate_template.simulate()
        if os.path.exists(data_path):
            npz = np.load(data_path)
            return {k: npz[k] for k in npz.files}
    except Exception:
        pass

    print("[train_template] data.npz / simulate_template.py not usable; "
          "using inline toy dataset.")
    return _toy_dataset()


def preprocess(X_train, X_val, X_test):
    try:
        import preprocess_template
        params = preprocess_template.fit_preprocess(X_train)
        return (preprocess_template.apply_preprocess(X_train, params),
                preprocess_template.apply_preprocess(X_val, params),
                preprocess_template.apply_preprocess(X_test, params))
    except Exception:
        # inline standardization fallback keeps this file runnable alone
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        return ((X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma)


# ---------------------------------------------------------------------------
# Model: small MLP, plain numpy forward/backward + manual Adam. Sized for a
# dataset of a few hundred rows -- two hidden layers of width 32-64, not
# bigger, since a wider net would just overfit a toy-sized dataset.
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, n_in, hidden=(64, 32), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [1]
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self._cache = [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z  # ReLU hidden, linear output
            self._cache.append(a)
        return a.ravel()

    def backward(self, y_true, y_pred):
        n = y_true.shape[0]
        dz = (2.0 / n) * (y_pred - y_true).reshape(-1, 1)
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a_prev = self._cache[i]
            grads_W[i] = a_prev.T @ dz
            grads_b[i] = dz.sum(axis=0)
            if i > 0:
                da_prev = dz @ self.W[i].T
                # cache[i] holds the ReLU'd activation, so (cache[i] > 0) is
                # exactly the ReLU derivative mask for the layer below it.
                relu_mask = (self._cache[i] > 0).astype(a_prev.dtype)
                dz = da_prev * relu_mask
        return grads_W, grads_b

    def step(self, grads_W, grads_b, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        bias1 = 1 - beta1 ** self.t
        bias2 = 1 - beta2 ** self.t
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * grads_W[i] ** 2
            self.W[i] -= lr * (self.mW[i] / bias1) / (np.sqrt(self.vW[i] / bias2) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * grads_b[i] ** 2
            self.b[i] -= lr * (self.mb[i] / bias1) / (np.sqrt(self.vb[i] / bias2) + eps)

    def get_weights(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_weights(self, weights):
        W, b = weights
        self.W = [w.copy() for w in W]
        self.b = [bb.copy() for bb in b]


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Training loop with early stopping on validation loss.
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val, epochs=2000, lr=1e-3, patience=30):
    best_val = np.inf
    best_epoch = -1
    # snapshot the actual weights here, not just the epoch number: training
    # keeps mutating the model after the best point, so recovering "the best
    # model" later would otherwise mean replaying the whole run
    best_weights = model.get_weights()
    wait = 0
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        y_pred_train = model.forward(X_train)
        loss_t = mse(y_train, y_pred_train)
        grads_W, grads_b = model.backward(y_train, y_pred_train)
        model.step(grads_W, grads_b, lr=lr)

        y_pred_val = model.forward(X_val)
        loss_v = mse(y_val, y_pred_val)

        train_loss.append(loss_t)
        val_loss.append(loss_v)

        if loss_v < best_val - 1e-9:
            best_val = loss_v
            best_epoch = epoch
            best_weights = model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.set_weights(best_weights)
    return train_loss, val_loss, best_epoch, best_val


def error_metrics(y_true, y_pred):
    abs_err = np.abs(y_true - y_pred)
    return {
        "median_error": float(np.median(abs_err)),
        "p95_error": float(np.percentile(abs_err, 95)),
        "worst_error": float(np.max(abs_err)),
        "n": int(len(abs_err)),
    }


def main():
    data = load_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    model = MLP(n_in=X_train.shape[1], hidden=(64, 32), seed=0)
    train_loss, val_loss, best_epoch, best_val = train(
        model, X_train, y_train, X_val, y_val, epochs=2000, lr=1e-3, patience=30
    )

    print(f"[train_template] stopped after {len(train_loss)} epochs "
          f"(best epoch {best_epoch}, best val loss {best_val:.6f})")
    print("[train_template] loss curve (every 10th epoch):")
    for epoch in range(0, len(train_loss), 10):
        print(f"  epoch {epoch:4d}  train_loss={train_loss[epoch]:.6f}  "
              f"val_loss={val_loss[epoch]:.6f}")

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    y_true_all, y_pred_all, split_all = [], [], []
    test_pred = None
    for name, (X, y) in splits.items():
        pred = model.forward(X)
        if name == "test":
            test_pred = pred
        y_true_all.append(y)
        y_pred_all.append(pred)
        split_all.append(np.full(len(y), name))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    split_all = np.concatenate(split_all)

    metrics_summary = error_metrics(y_test, test_pred)

    np.savez(os.path.join(HERE, "predictions.npz"),
             y_true=y_true_all, y_pred=y_pred_all, split=split_all)
    with open(os.path.join(HERE, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(HERE, "train_metrics.json"), "w") as f:
        json.dump({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }, f, indent=2)

    print(f"[train_template] test metrics: {metrics_summary}")
    print(f"[train_template] wrote predictions.npz, metrics_summary.json, "
          f"train_metrics.json to {HERE}")


if __name__ == "__main__":
    main()
