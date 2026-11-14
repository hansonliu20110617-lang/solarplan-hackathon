"""
Replaces simulate_template.py for the real track: AI + Climate, next-day solar
forecast, compared across two climates (Phoenix AZ = sunny/predictable,
Seattle WA = cloudy/volatile). Real NASA POWER daily data, no API key.

Feature row (today) -> target (tomorrow's solar):
  [solar_today, temp_today, humidity_today, cloud_today, is_seattle] -> solar_tomorrow

Split by calendar year, never shuffled (brief's rule for time-series data):
  train = 2021-2022, val = 2023, test = 2024. Split membership is keyed on
  the TARGET (tomorrow's) date, not today's -- a Dec-31 training example's
  label is Jan-1 of the next year, and keying on today's date would put that
  label inside the held-out period (the model's training loss would then be
  supervised by a value dated inside validation/test, a real if small form
  of the leakage the brief itself warns judges watch for).
Writes data.npz in the exact schema preprocess_template.py/train_template.py
already expect, so those two stages need zero changes. Also writes
test_meta.npz (city/date/baseline_pred per test row, same order as X_test)
so validate_template.py can report the persistence baseline and a per-city
breakdown -- NASA POWER's -999 missing-data sentinel (and any non-finite
reading) is dropped, not imputed, since a guessed value here would corrupt
both the feature and the baseline.
"""
import csv
import datetime
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

CITIES = [
    ("phoenix", os.path.join(DATA_DIR, "phoenix_az.csv"), 0.0),
    ("seattle", os.path.join(DATA_DIR, "seattle_wa.csv"), 1.0),
]
TRAIN_YEARS = {2021, 2022}
VAL_YEARS = {2023}
TEST_YEARS = {2024}
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


def _build_examples(rows, city_flag):
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
        examples.append({
            "features": [today["sw"], today["t2m"], today["rh2m"], today["cloud"], city_flag],
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


def main():
    all_examples = {"train": [], "val": [], "test": []}
    test_meta = {"city": [], "date": [], "baseline_pred": [], "target": []}

    for city_name, path, city_flag in CITIES:
        rows = _read_city_csv(path)
        examples = _build_examples(rows, city_flag)
        print(f"[prepare_solar_data] {city_name}: {len(rows)} daily rows -> {len(examples)} usable examples")

        for split_name, years in (("train", TRAIN_YEARS), ("val", VAL_YEARS), ("test", TEST_YEARS)):
            split_examples = _split(examples, years)
            all_examples[split_name].extend(split_examples)
            if split_name == "test":
                test_meta["city"].extend([city_name] * len(split_examples))
                test_meta["date"].extend([e["date"] for e in split_examples])
                test_meta["baseline_pred"].extend([e["baseline_pred"] for e in split_examples])
                test_meta["target"].extend([e["target"] for e in split_examples])

    def to_arrays(split_name):
        exs = all_examples[split_name]
        if not exs:
            raise SystemExit(
                f"[prepare_solar_data] '{split_name}' split is empty -- check "
                "TRAIN_YEARS/VAL_YEARS/TEST_YEARS against the date range actually "
                "downloaded, rather than writing a malformed (0,) array downstream."
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
    )

    print(f"[prepare_solar_data] train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"[prepare_solar_data] wrote data.npz, test_meta.npz to {HERE}")


if __name__ == "__main__":
    main()
