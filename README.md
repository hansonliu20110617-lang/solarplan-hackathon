# SolarPlan: a next-day solar forecast, compared across three climates

**Track:** AI + Climate
**Team:** Hanson Liu (solo) — hansonliu20110617@gmail.com — SJS

## The problem

Homeowners with rooftop solar and a home battery decide every evening whether
tomorrow's sun will cover a high-draw appliance (laundry, EV charging) on
solar, or whether to top up from the grid overnight instead. Getting this
wrong either wastes grid electricity or drains the battery for nothing — and
how much you can trust "tomorrow will be like today" depends heavily on where
you live.

## What we built

SolarPlan takes today's weather for a chosen city (solar irradiance,
temperature, humidity, cloud cover, season) and predicts tomorrow's solar
energy output in kWh/m²/day, then compares it against that city's "enough
sun to be worth it" threshold to answer the actual yes/no question the
homeowner has: run the appliance off solar tonight, or charge from the grid?

**A real example (Seattle, 2024-08-13, from the held-out test set):** today's
solar reading was 2.47 kWh/m², below Seattle's threshold of 3.04. The naive
persistence rule ("tomorrow = today") says: not enough sun, charge from the
grid. Our model, using today's full weather picture, predicted 3.74 — above
threshold, "yes, use solar." Tomorrow actually came in at 5.33. The model's
call was right; persistence would have sent the homeowner to the grid on a
genuinely sunny day.

## What we added

Starting point: brief §4.3.1, *Next-day solar energy forecast*. We went past
the brief's own suggested extension (compare two climates) in four ways: a
**third, intermediate climate** (Chicago — continental, frontal weather)
instead of just two extremes; a **day-of-year feature** (season, not just
yesterday's reading); a **decision-accuracy simulation** instead of only
reporting raw forecast error (the brief's own "make it your own" suggestion);
and **walk-forward validation across ten independent test years**, not one
or even two — which is what turned an ambiguous, coin-flip-looking result
from our first two-year draft into an actual, statistically significant
finding (see Results).

## How it works

**Data:** NASA POWER daily readings for Phoenix AZ (sunny, stable), Seattle
WA (cloudy, volatile), and Chicago IL (continental, frontal), 2005-01-01
through 2024-12-31 (20 full years, 7,304 usable daily examples per city, no
missing readings in any city). Features: today's solar irradiance,
temperature, humidity, cloud cover, a city flag, and day-of-year encoded as
sin/cos (so Dec 31 and Jan 1 are adjacent in feature space, not 364 apart) →
target: tomorrow's solar irradiance. Split by calendar year, never shuffled,
keyed on each example's *target* date so a Dec-31 label (which falls on Jan
1) can't leak into the next period.

**Validation:** walk-forward across ten independent folds (test years
2015–2024), each a full train/val/test split with no overlap in test years
and an expanding training window (each fold trains on every year of data
available before it, same as how this would actually be retrained over
time):

| Fold | Train | Val | Test |
|---|---|---|---|
| fold_2015 | 2005–2013 | 2014 | 2015 |
| fold_2016 | 2005–2014 | 2015 | 2016 |
| … | … | … | … |
| fold_2024 | 2005–2022 | 2023 | 2024 |

(`run_walkforward.py` runs all ten folds and a sign test across them;
`run_pipeline.py` alone runs just fold_2024 — the most-informed, best-trained
fold — as the single "official" demo.)

**Model:** a small from-scratch numpy MLP (two hidden layers, 64→32, ReLU,
manual Adam optimizer), early-stopped on validation loss. We also checked a
plain closed-form linear regression as a sanity check on whether the MLP's
extra capacity was actually earning its keep.

## Results

**The decision that actually matters, not just the raw error.** The brief
asks what decision a forecast supports (§4.3) and suggests measuring decision
quality directly (§4.3.1 "make it your own"). We define that decision
concretely: each city has an "enough sun to run the appliance off solar
tonight" threshold, set from that city's own *training-period* median (never
test data), and we check how often each forecast's yes/no call matches what
the homeowner should actually have done (the oracle: did tomorrow really
have enough sun?).

Checking this across one year, or even two, isn't enough to call a pattern
real — so we ran it across **ten independent walk-forward folds** (test
years 2015–2024) and a sign test on each one (an exact binomial test: out of
10 independent yes/no outcomes — did the model's decision accuracy beat the
baseline's that year? — how likely is a record this lopsided by pure chance):

| City | Folds model wins | Mean advantage | Sign-test p-value | Real effect? |
|---|---|---|---|---|
| **Chicago** | **10 / 10** | **+4.4 pts** | **0.0020** | Yes — essentially certain |
| **Seattle** | **10 / 10** | **+3.2 pts** | **0.0020** | Yes — essentially certain |
| Phoenix | 8 / 10 | +1.3 pts | 0.0391 | Yes, but smaller and noisier |

(p < 0.05 is the usual "probably not chance" line; p = 0.002 means a record
this one-sided would happen by pure luck roughly 1 time in 500 — *if* the ten
years were fully independent trials. They're not quite: the expanding
training window means later folds' training data overlaps earlier folds',
and nearby years' weather has real multi-year correlation (ENSO-like
patterns, long-term trends). So read these p-values as "this record is
extremely unlikely to be pure chance," not as a textbook-exact guarantee —
ten non-overlapping test years is still a much stronger check than the one
or two we started with.) Our *first*
draft, run on just two folds, had reported Seattle's result as "too close to
call" and Phoenix as "baseline wins both years" — both wrong, because two
data points can't tell a real small effect apart from noise. Ten folds can.

**The same pattern holds on raw mean absolute error** (what the brief's
suggested project literally asks for), checked the same way: our model beat
the baseline's pooled mean error in **10 out of 10** independent years
(mean advantage +0.144 kWh/m²/day, sign-test p = 0.00195). Every single
held-out year, for twenty years of real weather, our model's average miss
was smaller than just assuming "tomorrow = today."

**Median** error is the one place the picture stays genuinely mixed —
Phoenix's median-error comparison flips between folds depending on the year
(e.g. fold_2024: model 0.31 vs. baseline 0.24 kWh/m²/day, baseline wins; but
mean error for that same fold has the model winning, 0.43 vs. 0.51). That's
not a contradiction: median is about the *typical* day, mean is pulled by
the *rare large misses*. Our model is sometimes slightly worse on ordinary
calm Phoenix days but consistently better on the rare volatile ones — which
is exactly why the decision-accuracy and mean-error numbers above (the ones
that actually matter for "would this forecast have cost the homeowner
money") are the honest headline, not the median.

A plain linear regression doesn't beat the baseline's mean error in any
city in any fold we checked, so these wins need real nonlinearity, not just
more features or more data alone.

**Where it fails:** the single worst error in the official demo fold (4.7
kWh/m²/day) is a storm-front whiplash in Phoenix — solar output cratered
overnight, then swung back the next day. Both the model and the baseline
miss events like this equally, because neither has any forward-looking
weather signal — today's data alone can't see tomorrow's incoming front.
(We looked into adding a real next-day weather-forecast feature to fix
exactly this, via Open-Meteo's archive — and found its "historical forecast"
endpoint is stitched from the first few hours of each model run, so it
closely tracks the actual outcome rather than being a true day-ahead
forecast. Using it would have quietly leaked the answer, so we didn't.)

## Impact: what the decision-accuracy gain is actually worth

A percentage point of decision accuracy doesn't mean much until it's priced.
Using the false-confidence/false-caution rates already measured across all
ten walk-forward years, plus two real, cited numbers — EIA's 2024 US average
residential electricity price (16.5¢/kWh) and a stated, deliberately modest
assumption that a false-confidence mistake (appliance comes up short, needs
an unplanned top-up) costs 2x a false-caution one (just missed free solar) —
`pipeline/estimate_impact.py` turns our decision-accuracy numbers into a
dollar figure per household per year:

| City | Baseline cost/yr | Model cost/yr | Savings/household/yr |
|---|---|---|---|
| Chicago | $184.99 | $157.79 | **+$27.21** |
| Seattle | $134.05 | $113.60 | **+$20.45** |
| Phoenix | $79.63 | $70.23 | **+$9.40** |

Pooled average: **+$19.02/household/year** — modest for one house, but for
an illustrative 1,000-household community, that's **~$19,000/year** in
avoided unnecessary grid draw and avoided shortfall top-ups, from switching
one decision rule to another at zero marginal hardware cost. This is a
stated, labeled *estimate* built on two assumptions you can disagree with
(appliance size, the 2x cost multiplier) — change `APPLIANCE_KWH_PER_DAY` or
`FALSE_CONFIDENCE_MULTIPLIER` in the script and rerun it for your own
numbers; the decision-accuracy data underneath it is real and already
validated across ten independent years.

## What we'd do next

Get real archived day-ahead forecasts (e.g. via a provider with genuine
multi-day lead-time data covering back to 2005) to directly attack the
storm-front failure case — that's the one scenario the ten-fold validation
can't fix, since no feature here can see tomorrow's incoming front. Also
push the data further back than 2005 if NASA POWER's coverage allows, to see
whether Phoenix's smaller, noisier effect (p = 0.039, not p = 0.002 like the
other two cities) firms up or fades with even more independent years.

## Sources

- [NASA POWER](https://power.larc.nasa.gov/) — daily solar/weather data for
  Phoenix AZ, Seattle WA, and Chicago IL, 2005–2024 (20 years)
- [EIA](https://www.eia.gov/energyexplained/electricity/prices-and-factors-affecting-prices.php) —
  2024 US average residential electricity price (16.5¢/kWh), used in the
  Impact section's dollar estimate
- Veritas AI Hackathon 2026 Project Briefs, §4.3.1 (Next-day solar energy
  forecast) — starting point, the persistence-baseline idea, and the
  decision-accuracy "make it your own" suggestion
- numpy — the only dependency; no pandas/sklearn required
- Claude Code (Claude Sonnet 5) — coding assistant used throughout
  development
- OpenAI (`gpt-6-astra`) and xAI (`grok-4`) APIs — used during development
  for automated code review (`tools/peer_review.py`), which caught a real
  year-boundary data-leakage bug and a silent-misalignment risk, both fixed
  before this submission; not used to generate the project's results

---

## Repo structure / how to reproduce

```
data/            raw NASA POWER CSVs for all three cities, 2005-2024
                 (committed, so the pipeline runs offline with no re-download)
pipeline/
  prepare_solar_data.py   builds features/target, splits by fold -> data.npz
  preprocess_template.py  z-score normalize (fit on train only)
  train_template.py       from-scratch numpy MLP, early-stopped
  validate_template.py    model vs baseline (mean/median/p95/worst error),
                           per-city breakdown, and the decision simulation
  estimate_impact.py       turns the walk-forward decision stats into the
                           $/household/year estimate in Impact, above
  export_template.py      bakes results JSON into interface/result.html
interface/
  template.html    static results page (no backend, works from file://)
run_pipeline.py      runs the single "official" fold_2024 demo end to end
                      (trained on every year 2005-2022, the most-informed fold)
run_walkforward.py   runs all ten folds (test years 2015-2024) and a sign
                      test across them -- produces the statistics in Results
```

Run the single official demo with:

```
python run_pipeline.py
```

then open `interface/result.html`. Run the full ten-fold walk-forward
comparison with real significance testing (takes a few minutes: ten full
train/val/test cycles):

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
