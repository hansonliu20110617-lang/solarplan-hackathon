"""
Translates the decision-accuracy results (already computed per fold by
validate_template.py) into a dollar estimate, for the "impact" judging
criterion -- "does it tackle a real problem for real people" needs a real
number, not just a percentage.

Stated assumptions (illustrative, not measured -- change these and rerun if
you have better numbers):
  APPLIANCE_KWH_PER_DAY = 10    a representative day's high-draw appliance
                                 use (a laundry load + partial EV charge)
  RETAIL_PRICE_PER_KWH = 0.165  US average residential electricity price,
                                 2024 (EIA, see Sources in README)
  FALSE_CONFIDENCE_MULTIPLIER = 2.0
      A false-confidence mistake (told "enough sun", wasn't) means the
      appliance comes up short and needs an unplanned top-up -- plausibly at
      a worse (peak/backup) rate, not the flat retail rate. A false-caution
      mistake (told "not enough", there was) just means the homeowner drew
      from the grid when they didn't have to, at the normal retail rate.
      2x is a deliberately modest, stated assumption (the brief's own loan
      example uses a 5x asymmetry for a similar "two mistakes cost
      differently" situation) -- tune it and rerun if you disagree with it.

Reads the ten already-saved per-fold validation results from
run_walkforward.py (pipeline/validation_results_fold_*.json) rather than
re-running anything, and reports the average annual $ cost per household
for the model's decisions vs. the baseline's, per city, across all ten
independent years.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

APPLIANCE_KWH_PER_DAY = 10.0
RETAIL_PRICE_PER_KWH = 0.165
FALSE_CONFIDENCE_MULTIPLIER = 2.0


def daily_cost(rates):
    """Expected $/day cost of a decision policy's mistakes, from its
    false_confidence_rate and false_caution_rate (correct decisions cost
    nothing in this simplified model -- the appliance runs as planned)."""
    false_confidence_cost = APPLIANCE_KWH_PER_DAY * RETAIL_PRICE_PER_KWH * FALSE_CONFIDENCE_MULTIPLIER
    false_caution_cost = APPLIANCE_KWH_PER_DAY * RETAIL_PRICE_PER_KWH
    return (rates["false_confidence_rate"] * false_confidence_cost
            + rates["false_caution_rate"] * false_caution_cost)


def main():
    fold_paths = sorted(glob.glob(os.path.join(HERE, "validation_results_fold_*.json")))
    if not fold_paths:
        raise SystemExit(
            "[estimate_impact] no pipeline/validation_results_fold_*.json found -- "
            "run run_walkforward.py first, this script only reads its saved output."
        )

    if len(fold_paths) != 10:
        raise SystemExit(
            f"[estimate_impact] found {len(fold_paths)} fold file(s), expected 10 -- "
            "an interrupted run_walkforward.py or stale leftover files would silently "
            "change this estimate instead of erroring. Re-run run_walkforward.py fully."
        )
    folds = [json.load(open(p)) for p in fold_paths]
    city_sets = [frozenset(f["per_city"].keys()) for f in folds]
    if len(set(city_sets)) != 1 or not city_sets[0]:
        raise SystemExit(
            f"[estimate_impact] folds don't all cover the same nonempty set of cities "
            f"({city_sets}) -- can't average across folds that don't match."
        )
    cities = sorted(city_sets[0])

    print(f"Assumptions: {APPLIANCE_KWH_PER_DAY} kWh/day appliance, "
          f"${RETAIL_PRICE_PER_KWH}/kWh (EIA 2024 US residential average), "
          f"false-confidence mistakes cost {FALSE_CONFIDENCE_MULTIPLIER}x "
          f"(unplanned top-up, plausibly at a worse rate)\n")

    results = {}
    for city in cities:
        model_daily = [daily_cost(f["per_city"][city]["decision"]["model"]) for f in folds]
        baseline_daily = [daily_cost(f["per_city"][city]["decision"]["baseline"]) for f in folds]
        model_annual = sum(model_daily) / len(model_daily) * 365
        baseline_annual = sum(baseline_daily) / len(baseline_daily) * 365
        savings = baseline_annual - model_annual
        results[city] = {
            "model_annual_cost_usd": model_annual,
            "baseline_annual_cost_usd": baseline_annual,
            "annual_savings_usd": savings,
            "n_folds": len(folds),
        }
        print(f"{city:10s} baseline ${baseline_annual:6.2f}/yr  model ${model_annual:6.2f}/yr  "
              f"savings ${savings:+6.2f}/yr/household (avg over {len(folds)} independent years)")

    pooled_savings = sum(r["annual_savings_usd"] for r in results.values()) / len(results)
    print(f"\npooled average across cities: ${pooled_savings:+.2f}/yr/household")
    print(f"illustrative scale-up: a 1,000-household community -> ${pooled_savings * 1000:+,.0f}/yr")

    out_path = os.path.join(HERE, "impact_estimate.json")
    with open(out_path, "w") as f:
        json.dump({
            "assumptions": {
                "appliance_kwh_per_day": APPLIANCE_KWH_PER_DAY,
                "retail_price_per_kwh": RETAIL_PRICE_PER_KWH,
                "false_confidence_multiplier": FALSE_CONFIDENCE_MULTIPLIER,
                "source": "EIA 2024 US average residential electricity price, 16.5 cents/kWh",
            },
            "per_city": results,
            "pooled_avg_savings_per_household_per_year_usd": pooled_savings,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
