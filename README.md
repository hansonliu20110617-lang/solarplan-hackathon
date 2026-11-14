# SolarPlan: a next-day solar forecast, compared across three climates

**Track:** AI + Climate
**Team:** Hanson Liu (solo) — hansonliu20110617@gmail.com — <school — fill in before submitting>

## The problem

Homeowners with rooftop solar and a home battery decide every evening whether
tomorrow's sun will cover a high-draw appliance (laundry, EV charging) on
solar, or whether to top up from the grid overnight instead. Getting this
wrong either wastes grid electricity or drains the battery for nothing — and
how much you can trust "tomorrow will be like today" depends heavily on where
you live.

## What we built

SolarPlan takes today's weather for a chosen city (solar irradiance,
temperature, humidity, cloud cover) and predicts tomorrow's solar energy
output in kWh/m²/day, so a homeowner can decide tonight whether to plan
around solar tomorrow.

## What we added

Starting point: brief §4.3.1, *Next-day solar energy forecast*. We went past
the brief's own suggested extension (compare two climates) in three ways:
a **third, intermediate climate** (Chicago — continental, frontal weather)
instead of just two extremes; a **day-of-year feature** (season, not just
yesterday's reading); and **walk-forward validation across two independent
test years** (2023 and 2024, each its own proper train/val/test split)
instead of trusting a single split — which is what actually exposed that one
of our first-draft findings wasn't real (see Results).

## How it works

**Data:** NASA POWER daily readings for Phoenix AZ (sunny, stable), Seattle
WA (cloudy, volatile), and Chicago IL (continental, frontal), 2021-01-01
through 2024-12-31 (1,461 days each, no missing readings in any city).
Features: today's solar irradiance, temperature, humidity, cloud cover, a
city flag, and day-of-year encoded as sin/cos (so Dec 31 and Jan 1 are
adjacent in feature space, not 364 apart) → target: tomorrow's solar
irradiance. Split by calendar year, never shuffled, keyed on each example's
*target* date so a Dec-31 label (which falls on Jan 1) can't leak into the
next period.

**Validation:** walk-forward across two independent folds, each a full
train/val/test split with no overlap in test years:

| Fold | Train | Val | Test |
|---|---|---|---|
| fold_2023 | 2021 | 2022 | 2023 |
| fold_2024 | 2021–2022 | 2023 | 2024 |

(`run_walkforward.py` runs both; `run_pipeline.py` alone runs just fold_2024
as the single "official" demo run.)

**Model:** a small from-scratch numpy MLP (two hidden layers, 64→32, ReLU,
manual Adam optimizer), early-stopped on validation loss. We also checked a
plain closed-form linear regression as a sanity check on whether the MLP's
extra capacity was actually earning its keep.

## Results

Baseline (persistence — "tomorrow's solar = today's solar") beats our model
on the pooled, all-cities number in both folds. That pooled number hides the
real story, and checking it across *two* independent years instead of one
caught something our first draft got wrong:

| City | fold_2023: model vs. baseline | fold_2024: model vs. baseline | Consistent? |
|---|---|---|---|
| Phoenix (sunny, stable) | 0.370 vs. **0.265** | 0.313 vs. **0.235** | Yes — baseline wins both years |
| Chicago (continental, frontal) | **0.790** vs. 0.871 | **0.793** vs. 0.804 | Yes — model wins both years |
| Seattle (cloudy, volatile) | 0.550 vs. **0.537** | **0.541** vs. 0.601 | **No — flips between years** |

Our first draft (fold_2024 only) reported "the model beats the baseline in
Seattle" as a real finding. Testing a second, independent year shows that
claim doesn't hold — it was this close to a coin flip, not a repeatable
effect. Chicago's win *does* hold up in both years, which is the one honest,
repeatable case of our model adding real value over the naive baseline:
continental climates have enough frontal-system structure in humidity/cloud
cover that the model can exploit beyond "today's reading," while Phoenix's
solar output is stable enough day-to-day that persistence is already close
to optimal and Seattle is just too noisy either way at this sample size. A
plain linear regression doesn't beat the baseline in any city in either
fold, so Chicago's win needs real nonlinearity, not just more features.

**Where it fails:** the single worst error in fold_2024 (4.6 kWh/m²/day) is a
two-day whiplash in Phoenix in May — solar output cratered overnight, then
snapped back up the next day, almost certainly a passing storm front. Both
the model and the baseline miss it equally, because neither has any
forward-looking weather signal — today's data alone can't see tomorrow's
incoming front. (We looked into adding a real next-day weather-forecast
feature to fix exactly this, via Open-Meteo's archive — and found its
"historical forecast" endpoint is stitched from the first few hours of each
model run, so it closely tracks the actual outcome rather than being a true
day-ahead forecast. Using it would have quietly leaked the answer, so we
didn't.)

## What we'd do next

Get real archived day-ahead forecasts (e.g. via a provider with genuine
lead-time data covering 2021–2024) to directly attack the Phoenix storm-front
failure case. Also run a third fold if more NASA POWER years become
available, since two folds already changed our conclusion once — a third
would make the Seattle "coin flip" call more certain either way.

## Sources

- [NASA POWER](https://power.larc.nasa.gov/) — daily solar/weather data for
  Phoenix AZ, Seattle WA, and Chicago IL, 2021–2024
- Veritas AI Hackathon 2026 Project Briefs, §4.3.1 (Next-day solar energy
  forecast) — starting point and the persistence-baseline idea
- numpy — the only dependency; no pandas/sklearn required

---

## Repo structure / how to reproduce

```
data/            raw NASA POWER CSVs for all three cities (committed, so the
                 pipeline runs offline with no re-download)
pipeline/
  prepare_solar_data.py   builds features/target, splits by fold -> data.npz
  preprocess_template.py  z-score normalize (fit on train only)
  train_template.py       from-scratch numpy MLP, early-stopped
  validate_template.py    model vs persistence baseline, per-city breakdown
  export_template.py      bakes results JSON into interface/result.html
interface/
  template.html    static results page (no backend, works from file://)
run_pipeline.py      runs the single "official" fold_2024 demo end to end
run_walkforward.py   runs both folds (2023 and 2024) and prints/saves the
                      combined per-city comparison in the Results table above
```

Run the single official demo with:

```
python run_pipeline.py
```

then open `interface/result.html`. Run the full walk-forward comparison
(both folds) with:

```
python run_walkforward.py
```

No installs beyond numpy (Python 3's standard library handles the CSV
parsing and the NASA POWER download).

The generic five-stage *pattern* (not this project's content) was scaffolded
before kickoff, which the hackathon rules explicitly allow ("you're welcome
to learn tools... beforehand"); the actual solar data, model, and results
above were built during the hackathon window.

Datasets, pretrained models, and AI tools used are listed under Sources
above; no AI-generated code was used beyond standard coding-assistant help
during development (Claude Code).
