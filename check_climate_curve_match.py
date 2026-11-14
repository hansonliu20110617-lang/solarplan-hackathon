"""
Cheap pre-screen for a generalization-swap candidate city: compute each
city's own monthly clear-sky-index curve (training years only) and its
correlation with the holdout city's curve. No training involved -- this is
meant to rule out weak candidates *before* spending a 3-seed training run
on them, the way San Juan and New Orleans were ruled out as Miami swap
candidates (see README, Generalization section) despite San Juan looking
like the more obvious pick by Koppen climate classification.

Usage: python check_climate_curve_match.py miami_fl.csv tampa_fl.csv sanjuan_pr.csv ...
(first city listed is the reference; the rest are scored against it)
"""
import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_YEARS = set(range(2005, 2023))


def monthly_csi(csv_filename):
    rows_by_month = {m: [] for m in range(1, 13)}
    with open(DATA_DIR / csv_filename) as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("YEAR,MO,DY"))
    for row in csv.DictReader(lines[header_idx:]):
        if int(row["YEAR"]) not in TRAIN_YEARS:
            continue
        sw, clr = float(row["ALLSKY_SFC_SW_DWN"]), float(row["CLRSKY_SFC_SW_DWN"])
        if sw < -500 or clr < -500 or clr <= 0:
            continue
        rows_by_month[int(row["MO"])].append(sw / clr)
    return [sum(rows_by_month[m]) / len(rows_by_month[m]) for m in range(1, 13)]


def correlation(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (sa * sb)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_file = sys.argv[1]
    ref_profile = monthly_csi(ref_file)
    print(f"Reference: {ref_file}")
    print("  " + " ".join(f"{v:.2f}" for v in ref_profile))
    for candidate in sys.argv[2:]:
        p = monthly_csi(candidate)
        print(f"{candidate}: corr={correlation(ref_profile, p):+.3f} "
              f"amplitude={max(p) - min(p):.3f} mean={sum(p) / 12:.3f}")
        print("  " + " ".join(f"{v:.2f}" for v in p))


if __name__ == "__main__":
    main()
