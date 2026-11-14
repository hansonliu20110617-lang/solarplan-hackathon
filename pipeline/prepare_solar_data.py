"""
Replaces simulate_template.py for the real track: AI + Climate, next-day solar
forecast, compared across three climates (Phoenix AZ = sunny/stable, Chicago
IL = continental/frontal, Seattle WA = cloudy/volatile). Real NASA POWER
daily data, no API key.

Feature row (today) -> target (tomorrow's solar):
  [solar_today, temp_today, humidity_today, cloud_today,
   is_seattle, is_chicago, doy_sin, doy_cos] -> solar_tomorrow

day-of-year is encoded as sin/cos (not the raw integer) so Dec 31 and Jan 1
are adjacent in feature space instead of 364 apart -- solar output depends
heavily on season, which persistence captures only indirectly through
yesterday's single noisy reading.

Train/val/test years are CLI-configurable (--train-years/--val-years/
--test-years, comma-separated) so run_walkforward.py can run this multiple
times with different fold boundaries without duplicating this file. Defaults
reproduce the original single-fold split: train=2021-2022, val=2023,
test=2024.

Split membership is keyed on the TARGET (tomorrow's) date, not today's -- a
Dec-31 training example's label is Jan-1 of the next year, and keying on
today's date would put that label inside the held-out period (the model's
training loss would then be supervised by a value dated inside
validation/test, a real if small form of the leakage the brief itself warns
judges watch for).

Writes data.npz in the exact schema preprocess_template.py/train_template.py
already expect, so those two stages need zero changes. Also writes
test_meta.npz (city/date/baseline_pred/target/decision_threshold per test
row, same order as X_test) so validate_template.py can report the
persistence baseline, a per-city breakdown, and a decision simulation
("would this forecast have told the homeowner to run the appliance off solar
tonight, and was that the right call?") against a per-city threshold set
from training data only -- NASA POWER's -999 missing-data sentinel (and any
non-finite reading) is dropped, not imputed, since a guessed value here
would corrupt both the feature and the baseline.
"""
import argparse
import csv
import datetime
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

# (name, csv path, [is_seattle, is_chicago] one-hot -- phoenix is the reference city, [0, 0])
CITIES = [
    ("phoenix", os.path.join(DATA_DIR, "phoenix_az.csv"), [0.0, 0.0]),
    ("seattle", os.path.join(DATA_DIR, "seattle_wa.csv"), [1.0, 0.0]),
    ("chicago", os.path.join(DATA_DIR, "chicago_il.csv"), [0.0, 1.0]),
]
MISSING_SENTINEL = -900.0  # NASA POWER uses -999; threshold catches formatting variants


def _read_city_csv(path):
    with open(path, newline="") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("YEAR,MO,DY"))
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for row in reader:
        rows.append({
            "date": f"{int(row['YEAR']):04d}-{int(row['MO']):02d}-{int(row['DY']):02d}",
            "sw": float(row["ALLSKY_SFC_SW_DWN"]),
            "t2m": float(row["T2M"]),
            "rh2m": float(row["RH2M"]),
            "cloud": float(row["CLOUD_AMT"]),
            "year": int(row["YEAR"]),
        })
    rows.sort(key=lambda r: r["date"])  # NASA POWER already returns this sorted; be explicit anyway
    return rows


def _build_examples(rows, city_onehot):
    """One example per (today, tomorrow) pair. Drops the last row of the
    series (no tomorrow), any pair touching a missing/non-finite reading,
    and any pair that isn't actually consecutive days (a gap or duplicate
    date in the source data would otherwise silently create a multi-day or
    same-day "forecast")."""
    examples = []
    skipped_gap = 0
    for i in range(len(rows) - 1):
        today, tomorrow = rows[i], rows[i + 1]
        today_date = datetime.date.fromisoformat(today["date"])
        tomorrow_date = datetime.date.fromisoformat(tomorrow["date"])
        if tomorrow_date - today_date != datetime.timedelta(days=1):
            skipped_gap += 1
            continue
        values = (today["sw"], today["t2m"], today["rh2m"], today["cloud"], tomorrow["sw"])
        if any(not np.isfinite(v) or v <= MISSING_SENTINEL for v in values):
            continue
        doy = today_date.timetuple().tm_yday
        angle = 2 * math.pi * doy / 365.25
        examples.append({
            "features": [
                today["sw"], today["t2m"], today["rh2m"], today["cloud"],
                *city_onehot, math.sin(angle), math.cos(angle),
            ],
            "target": tomorrow["sw"],
            "baseline_pred": today["sw"],  # persistence: "tomorrow = today"
            "date": today["date"],
            "target_year": tomorrow_date.year,  # split membership keys on this, not today's year
        })
    if skipped_gap:
        print(f"[prepare_solar_data] skipped {skipped_gap} non-consecutive-day pair(s)")
    return examples


def _split(examples, years):
    return [e for e in examples if e["target_year"] in years]


def _parse_years(s):
    return {int(y) for y in s.split(",")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-years", default="2021,2022")
    ap.add_argument("--val-years", default="2023")
    ap.add_argument("--test-years", default="2024")
    args = ap.parse_args()
    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    test_years = _parse_years(args.test_years)

    all_examples = {"train": [], "val": [], "test": []}
    test_meta = {"city": [], "date": [], "baseline_pred": [], "target": [], "decision_threshold": []}

    for city_name, path, city_onehot in CITIES:
        rows = _read_city_csv(path)
        examples = _build_examples(rows, city_onehot)
        print(f"[prepare_solar_data] {city_name}: {len(rows)} daily rows -> {len(examples)} usable examples")

        city_splits = {
            split_name: _split(examples, years)
            for split_name, years in (("train", train_years), ("val", val_years), ("test", test_years))
        }
        # "Enough sun to be worth running an appliance off solar" is defined
        # per city from TRAINING data only (the city's typical day), never
        # from test -- using a test-derived threshold would let the decision
        # simulation peek at the exact period it's being judged on.
        decision_threshold = float(np.median([e["target"] for e in city_splits["train"]]))

        for split_name, split_examples in city_splits.items():
            all_examples[split_name].extend(split_examples)
            if split_name == "test":
                test_meta["city"].extend([city_name] * len(split_examples))
                test_meta["date"].extend([e["date"] for e in split_examples])
                test_meta["baseline_pred"].extend([e["baseline_pred"] for e in split_examples])
                test_meta["target"].extend([e["target"] for e in split_examples])
                test_meta["decision_threshold"].extend([decision_threshold] * len(split_examples))

    def to_arrays(split_name):
        exs = all_examples[split_name]
        if not exs:
            raise SystemExit(
                f"[prepare_solar_data] '{split_name}' split is empty -- check "
                "--train-years/--val-years/--test-years against the date range "
                "actually downloaded, rather than writing a malformed (0,) array "
                "downstream."
            )
        X = np.array([e["features"] for e in exs], dtype=np.float64)
        y = np.array([e["target"] for e in exs], dtype=np.float64)
        return X, y

    X_train, y_train = to_arrays("train")
    X_val, y_val = to_arrays("val")
    X_test, y_test = to_arrays("test")

    np.savez(
        os.path.join(HERE, "data.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    np.savez(
        os.path.join(HERE, "test_meta.npz"),
        city=np.array(test_meta["city"]),
        date=np.array(test_meta["date"]),
        baseline_pred=np.array(test_meta["baseline_pred"], dtype=np.float64),
        # kept alongside baseline_pred so validate_template.py can assert its
        # test-split y_true lines up row-for-row with this file, not just
        # matches it in length -- a silent reordering between the two files
        # would otherwise pair the wrong city/date with each prediction.
        target=np.array(test_meta["target"], dtype=np.float64),
        # that city's median training-period solar value -- the "enough sun"
        # line the decision simulation in validate_template.py uses, never
        # derived from test data.
        decision_threshold=np.array(test_meta["decision_threshold"], dtype=np.float64),
    )

    print(f"[prepare_solar_data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"[prepare_solar_data] wrote data.npz, test_meta.npz to {HERE}")


if __name__ == "__main__":
    main()
