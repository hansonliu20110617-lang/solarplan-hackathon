"""
Walk-forward validation across ten independent test years (2015-2024, each
its own full train/val/test split with no overlap between folds' test years)
instead of trusting one split or even two. Two folds was enough to catch one
first-draft claim that wasn't real (see README) -- ten folds is enough to
actually ask, with real statistics, whether a per-city result is a repeatable
pattern or noise.

For test year Y: val = Y-1, train = every year from DATA_START_YEAR up to
(but not including) val -- so each later fold trains on more data, same as
how this would actually be deployed and retrained over time.

Runs prepare_solar_data -> preprocess -> train -> validate for each fold via
subprocess (same approach as run_pipeline.py). Saves each fold's
validation_results.json under a fold-specific name, then runs two
significance tests on each city's per-fold decision-accuracy difference
(model - baseline): a sign test (exact binomial, asks only "how often did
the model win") and a paired t-test (asks "how often, weighted by how much
each fold's win or loss was worth" -- pure-Python Student's-t via the
regularized incomplete beta, no scipy). The t-test matters most exactly
where the sign test alone is inconclusive: a city can have an unremarkable
win/loss record but still show a real, consistent-in-magnitude effect.

Honest caveat on that test: the sign test's p-value is exact only under
independent trials, and these ten folds are not fully independent -- the
expanding training window means fold_2024's training set contains almost
all of fold_2015's, and nearby years' weather has real serial correlation
(multi-year patterns like ENSO, long-term trends). Ten non-overlapping TEST
years is still a much stronger check than one or two, and the p-values below
are a useful, standard way to summarize how one-sided the record is -- but
they should be read as "this record is extremely unlikely to be pure chance"
rather than "formally, p < 0.002 under a fully justified independence
assumption."
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

DATA_START_YEAR = 2005
TEST_YEARS = list(range(2015, 2025))  # 2015..2024 inclusive -- 10 folds


def _fold_years(test_year):
    val_year = test_year - 1
    train_years = list(range(DATA_START_YEAR, val_year))
    return train_years, [val_year], [test_year]


def run(cmd):
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)} -- stopping.")
        sys.exit(result.returncode)


def _reg_incomplete_beta(x, a, b):
    """Regularized incomplete beta function I_x(a, b) via Lentz's continued
    fraction (Numerical Recipes 6.4) -- pure math, no scipy. Needed for the
    Student's-t CDF below."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
               + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_beta) / a
    if x >= (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_incomplete_beta(1.0 - x, b, a)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c
        if abs(1.0 - d * c) < 1e-10:
            break
    return front * (f - 1.0)


def student_t_two_sided_p(t_stat, df):
    """Two-sided p-value for Student's t, via the regularized incomplete
    beta function -- the same quantity scipy.stats.t.sf uses internally."""
    x = df / (df + t_stat * t_stat)
    return _reg_incomplete_beta(x, df / 2.0, 0.5)


def paired_t_test(diffs):
    """Paired t-test on a list of per-fold differences (e.g. model minus
    baseline decision accuracy, one value per independent test year).
    Complements the sign test above: the sign test only asks "how often did
    the model win," throwing away how big each win or loss was -- this uses
    that magnitude too, which matters most exactly when the win count alone
    isn't decisive (e.g. a 7/10 record where the wins are consistently
    bigger than the losses). Same independence caveat as the sign test:
    treats each fold's mean as one independent observation, which is only
    approximately true given the expanding training window and real
    multi-year weather correlation."""
    n = len(diffs)
    if n < 2:
        return {"t_statistic": None, "p_value": 1.0, "df": 0}
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return {"t_statistic": None, "p_value": 0.0 if mean != 0 else 1.0, "df": n - 1}
    t_stat = mean / (std / math.sqrt(n))
    p_value = student_t_two_sided_p(t_stat, n - 1)
    return {"t_statistic": t_stat, "p_value": p_value, "df": n - 1}


def sign_test_p_value(n_favor_a, n_total):
    """Exact two-sided binomial sign test p-value under the null that each
    fold is a coin flip (p=0.5) between "model wins" and "baseline wins"
    (ties excluded from n_total before calling this). No scipy needed --
    this is just the binomial PMF summed over outcomes at least as extreme
    as what was observed, doubled for two-sidedness and capped at 1.0."""
    if n_total == 0:
        return 1.0
    more_extreme = min(n_favor_a, n_total - n_favor_a)
    tail = sum(math.comb(n_total, k) for k in range(0, more_extreme + 1)) / (2 ** n_total)
    return min(1.0, 2 * tail)


def main():
    fold_results = {}
    for test_year in TEST_YEARS:
        train_years, val_years, test_years = _fold_years(test_year)
        fold_name = f"fold_{test_year}"
        train_str = ",".join(str(y) for y in train_years)
        val_str = ",".join(str(y) for y in val_years)
        test_str = ",".join(str(y) for y in test_years)

        print(f"\n=== {fold_name}: train={train_years[0]}-{train_years[-1]} "
              f"val={val_years[0]} test={test_years[0]} ===", flush=True)
        run([sys.executable, str(PIPELINE / "prepare_solar_data.py"),
             "--train-years", train_str, "--val-years", val_str, "--test-years", test_str])
        run([sys.executable, str(PIPELINE / "preprocess_template.py")])
        run([sys.executable, str(PIPELINE / "train_template.py")])
        run([sys.executable, str(PIPELINE / "validate_template.py")])

        src = PIPELINE / "validation_results.json"
        dst = PIPELINE / f"validation_results_{fold_name}.json"
        shutil.copy(src, dst)
        with open(dst) as f:
            fold_results[fold_name] = json.load(f)

    # --- Aggregate: per-city, per-fold decision-accuracy difference (model - baseline) ---
    cities = sorted(next(iter(fold_results.values()))["per_city"].keys())
    print("\n=== Walk-forward summary (10 folds, test years 2015-2024) ===", flush=True)

    significance = {}
    pooled_diffs = [result["baseline_mean_error"] - result["mean_error"] for result in fold_results.values()]
    pooled_wins = sum(1 for d in pooled_diffs if d > 0)
    pooled_losses = sum(1 for d in pooled_diffs if d < 0)
    pooled_decisive = pooled_wins + pooled_losses  # excludes exact ties, same as the per-city test below
    pooled_p = sign_test_p_value(pooled_wins, pooled_decisive)
    pooled_t = paired_t_test(pooled_diffs)
    significance["_pooled_mean_error"] = {
        "per_fold_diff_baseline_minus_model": pooled_diffs,
        "mean_diff": sum(pooled_diffs) / len(pooled_diffs),
        "folds_model_wins": pooled_wins,
        "folds_tied": len(pooled_diffs) - pooled_decisive,
        "sign_test_p_value": pooled_p,
        "paired_t_test": pooled_t,
    }
    print(f"\npooled (all cities): model beat baseline's mean error in "
          f"{pooled_wins}/{len(pooled_diffs)} folds, mean difference "
          f"{sum(pooled_diffs) / len(pooled_diffs):+.4f} kWh/m2/day, sign-test p={pooled_p:.5f}, "
          f"paired t-test p={pooled_t['p_value']:.5f} (t={pooled_t['t_statistic']:.2f}, df={pooled_t['df']})")

    for city in cities:
        diffs = []
        for fold_name, result in fold_results.items():
            cd = result["per_city"][city]["decision"]
            diffs.append(cd["model"]["correct_rate"] - cd["baseline"]["correct_rate"])
        n_model_wins = sum(1 for d in diffs if d > 0)
        n_baseline_wins = sum(1 for d in diffs if d < 0)
        n_ties = len(diffs) - n_model_wins - n_baseline_wins
        n_decisive = n_model_wins + n_baseline_wins
        p_value = sign_test_p_value(n_model_wins, n_decisive)
        t_result = paired_t_test(diffs)
        mean_diff = sum(diffs) / len(diffs)
        significance[city] = {
            "per_fold_diff_model_minus_baseline": diffs,
            "mean_diff": mean_diff,
            "folds_model_wins": n_model_wins,
            "folds_baseline_wins": n_baseline_wins,
            "folds_tied": n_ties,
            "sign_test_p_value": p_value,
            "paired_t_test": t_result,
        }
        print(f"\n{city}: model beat baseline's decision accuracy in "
              f"{n_model_wins}/{len(diffs)} folds (baseline won {n_baseline_wins}, "
              f"tied {n_ties}), mean difference {mean_diff:+.1%}, "
              f"sign-test p={p_value:.4f}, paired t-test p={t_result['p_value']:.4f} "
              f"(t={t_result['t_statistic']:.2f}, df={t_result['df']})")

    summary_path = ROOT / "walkforward_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"folds": fold_results, "significance": significance}, f, indent=2)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
