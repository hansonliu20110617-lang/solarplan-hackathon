# SolarPlan: a next-day solar forecast, and what it takes to make one that generalizes

**Track:** AI + Climate
**Team:** Hanson Liu (solo) — hansonliu20110617@gmail.com — SJS

**A note on scope, to avoid a confusion we hit ourselves:** this project has
two parts that use different numbers of cities on purpose. The **core
submission** (walk-forward validation, decision accuracy, the dollar impact
estimate) uses **3 cities** — Phoenix, Seattle, Chicago — and is the main
result. A **separate generalization study** (does this approach work for a
city the model never saw at all?) uses **10 cities**, because that question
specifically needs many training locations to test properly. Every number
below says which scope it's from.

## The problem

Homeowners with rooftop solar and a home battery decide every evening
whether tomorrow will be sunny enough to run a high-draw appliance (laundry,
EV charging) during the day on solar, or whether to run it overnight on the
grid instead (or top up the battery from the grid tonight, just in case).
Solar panels don't generate at night, so this is a decision about tomorrow,
made tonight — and how much you can trust "tomorrow will be like today"
depends heavily on where you live.

## What we built

SolarPlan takes today's weather for a chosen city (clear-sky index,
temperature, humidity, cloud cover, season) and predicts tomorrow's clear-sky
index, converts that back to an expected kWh/m²/day figure, and compares it
against that city's own historical "normal" day to answer the actual yes/no
question the homeowner has: wait for solar tomorrow, or run it on the grid
tonight instead?

**A real example (Seattle, 2024-08-13, from the held-out test set):** today's
reading converts to 2.45 kWh/m², below Seattle's threshold of 3.04. The
baseline — "smart persistence" (tomorrow's clearness = today's, see *How it
works*) — says: not enough sun coming, use the grid. Our model, using
today's full weather picture, predicted 3.43 — above threshold, "wait for
solar." Tomorrow actually came in at 5.33. The model's call was right; the
baseline would have sent the homeowner to the grid on a genuinely sunny day.

## What we added

Starting point: brief §4.3.1, *Next-day solar energy forecast*. Beyond the
brief's own suggested extension (compare two climates), this went through
several real rounds of research, failure, and correction — not a straight
line to a clean result:

1. **A third climate** (Chicago) and a **decision-accuracy simulation**
   instead of only reporting raw error — directly answering the brief's own
   "make it your own" suggestion (§3, §4.3.1).
2. **Walk-forward validation across ten independent test years** (2015–2024),
   not one or two, which is what caught a first-draft claim that didn't
   actually hold up (see *Results*).
3. **Clear-sky index normalization** (standard in the solar-forecasting
   literature), adopted specifically to test whether a city the model never
   trained on at all could still be forecast. It couldn't, at first — see
   *Generalization*.
4. **Two real bugs caught by multi-model peer review and fixed, not
   glossed over:** NASA's clear-sky parameter isn't a pure astronomical
   constant (it reflects real retrieved atmospheric state — verified
   empirically, see `prepare_solar_data.py`), so using tomorrow's raw value
   was quietly leaking real information; and a day-of-year climatology
   silently mis-keys dates after Feb 29 in leap years. Both fixed; every
   number below is post-fix.
5. **A properly controlled generalization experiment** (not just "2 cities
   bad, 10 good") that isolates training-city-count from which city was held
   out, with multiple random subsets and random seeds — because an earlier,
   confounded version of this claim was correctly challenged in review.

## How it works

**Data:** NASA POWER daily readings, 2005–2024 (20 years), for the 3 core
cities (Phoenix AZ – sunny/stable, Seattle WA – cloudy/volatile, Chicago IL –
continental/frontal), 7 more for the generalization study (Denver, Miami,
Minneapolis, Albuquerque, Houston, Los Angeles, Boston — chosen for climate
diversity: desert, tropical, snowy continental, high-altitude, humid
subtropical, Mediterranean, coastal), and 5 added later specifically to
test and fix the generalization gaps found (Honolulu, Tampa, Portland used
in actual training runs; San Juan and New Orleans screened by curve
correlation only, never trained on — see *Generalization*).

**Clear-sky index, not raw kWh.** The model predicts tomorrow's actual solar
÷ a *climatological* clear-sky ceiling for that location and calendar day —
the ceiling is each (month, day)'s average across **training years only**,
never the real value for that specific future day. That distinction matters:
checking the raw data shows NASA's clear-sky parameter genuinely varies with
real atmospheric conditions (Phoenix's June 15 value ranges 7.81–9.26
kWh/m² across 20 different years) — it is not a fixed astronomical number.
Using tomorrow's *actual* clear-sky value would have quietly leaked real
information about tomorrow's atmosphere; the training-years-only climatology
has no such problem, since it's a fixed number computable before any
forecast is made. (Caveat worth stating plainly: NASA POWER's humidity and
cloud-cover inputs are themselves reanalysis products, reconstructed after
the fact with a real publication delay — a true same-day operational system
would need near-real-time observations instead, not this retrospective data.)

**Baseline: "smart persistence," not naive persistence.** The baseline
predicts tomorrow's clear-sky index equals today's, then converts back
through tomorrow's own climatological ceiling — so it already accounts for
the slow seasonal change in the ceiling itself, unlike naively assuming
tomorrow's raw kWh equals today's raw kWh. This is the standard,
non-trivial literature baseline, not a weak strawman.

**Features:** today's clear-sky index, temperature, humidity, cloud cover,
latitude, longitude (continuous, not a one-hot city flag — see
*Generalization*), and day-of-year as sin/cos. **Model:** a small
from-scratch numpy MLP (64→32 hidden, ReLU, manual Adam), early-stopped on
validation loss. A closed-form ridge-regression baseline on the same
features also beats smart persistence (see the sanity check in *Results*)
— so the features themselves carry most of the signal, and the MLP's
nonlinearity adds a real but modest further edge, not the entire win.

**Validation:** walk-forward across ten independent folds (test years
2015–2024, 3-city scope), each a full train/val/test split with an
expanding training window (every fold trains on every year available before
it — train=2005–2013/val=2014/test=2015 for the first fold, up to
train=2005–2022/val=2023/test=2024 for the last). `run_walkforward.py` runs
all ten and a sign test across them; `run_pipeline.py` alone runs just the
last, most-informed fold as the single "official" demo.

## Results (3-city core scope)

**Skill score — the clean headline.** The solar-forecasting literature's
standard metric (1 − RMSE_model/RMSE_baseline; positive means the model
wins) was positive in **10 out of 10 independent years, for all three
cities**, sign-test p = 0.00195 each:

| City | Folds model wins | Mean skill | Sign-test p |
|---|---|---|---|
| Chicago | 10 / 10 | +0.229 | 0.00195 |
| Seattle | 10 / 10 | +0.169 | 0.00195 |
| Phoenix | 10 / 10 | +0.165 | 0.00195 |
| Pooled | 10 / 10 | +0.202 | 0.00195 |

**Decision accuracy is honest, but messier.** Translated into the actual
yes/no decision (same ten folds):

| City | Folds model wins | Mean advantage | Sign-test p | Paired t-test p | Real effect? |
|---|---|---|---|---|---|
| Seattle | 10 / 10 | +2.8 pts | 0.0020 | <0.0001 (t=7.44, df=9) | Yes |
| Chicago | 9 / 10 | +3.9 pts | 0.0215 | 0.0003 (t=5.80, df=9) | Yes |
| Phoenix | 7 / 10 | +1.0 pt | 0.3438 | **0.0349** (t=2.48, df=9) | Yes — sign test alone missed this |

Phoenix is the case this second test earns its place for. The sign test only
asks "how often did the model win," and a 7/10 record isn't distinguishable
from chance on its own — but the paired t-test asks the sharper question of
how large each win or loss was, and on that question Phoenix's wins are
consistently bigger than its losses (mean +1.0 pt, t=2.48), which clears
p<0.05. That's the same information the skill-score table already pointed
to (Phoenix's skill is positive in all 10 folds); the t-test is what lets
decision accuracy agree with it instead of looking contradictory.
**Caveat on all three tests:** a 10/10 sign-test record always produces the
same minimum possible p-value (0.00195) regardless of margin — it confirms
*direction* reliably, not *size*, which is exactly the gap the t-test fills.
Both tests assume the ten years are independent trials, which is only
approximately true (overlapping expanding training windows; real multi-year
weather correlation like ENSO) — read these p-values as "extremely unlikely
to be pure chance," not a textbook-exact guarantee.

**The official single demo** (trained on 2005–2022, tested on 2024, the
same numbers `run_pipeline.py` reproduces):

| City | Model mean error | Baseline mean error | Skill | Decision accuracy (model / baseline / always-guess-majority) |
|---|---|---|---|---|
| Chicago | 0.991 | 1.202 | +0.260 | 85.0% / 80.6% / 56.6% |
| Phoenix | 0.459 | 0.518 | +0.189 | 94.3% / 91.8% / 51.4% |
| Seattle | 0.745 | 0.897 | +0.174 | 86.3% / 83.3% / 51.4% |

The last column matters: because our decision threshold is each city's own
historical *median*, "enough sun" is close to a 50/50 split by construction
— the always-guess-majority accuracy is ~51–57%, nowhere near the 85–94% our
model and baseline both reach. That was a real check worth running (raised
in review): a lopsided climate could otherwise make high accuracy numbers
meaningless on their own.

**Where it fails:** the worst single error is a storm-front whiplash in
Phoenix — solar cratered overnight, then swung back the next day. Neither
the model nor the baseline can see that coming, since nothing here has a
forward-looking weather signal. (We checked whether Open-Meteo's archive
could supply one — it can't without leaking: its "historical forecast"
endpoint is stitched from the first few hours of each model run, so it
tracks the actual outcome rather than being a genuine day-ahead forecast.)

**Sanity check: is the MLP actually earning its complexity?** A from-scratch
neural net is easy to oversell, so `run_ridge_baseline.py` swaps in
closed-form ridge regression (same preprocessed features, lambda picked by
validation MSE) and runs it through the identical validation pipeline:

| | Pooled skill | Pooled decision accuracy |
|---|---|---|
| Ridge regression | +0.215 | 88.3% (baseline 85.2%) |
| MLP (shipped) | +0.223 | 88.5% (baseline 85.2%) |

Ridge gets most of the way there on its own — the clear-sky normalization
and smart-persistence baseline are doing more of the real work than the
network's nonlinearity. The MLP's edge is real but modest (and in Seattle,
ridge actually ties it: +0.177 vs +0.174). Worth knowing either way: it
means the gains in this project aren't an MLP-specific artifact, and it's
an honest answer to "why not just use linear regression here" instead of
assuming the fancier model needs no justification.

## Impact: what the decision-accuracy gain is actually worth

Using the false-confidence/false-caution rates measured across all ten
walk-forward years, plus two cited numbers — EIA's 2024 US average
residential electricity price (16.5¢/kWh) and a stated, modest assumption
that a false-confidence mistake (appliance comes up short, needs an
unplanned top-up) costs 2× a false-caution one (just missed free solar) —
`pipeline/estimate_impact.py` turns the decision-accuracy numbers into a
dollar figure:

| City | Baseline cost/yr | Model cost/yr | Savings/household/yr |
|---|---|---|---|
| Chicago | $184.99 | $158.29 | **+$26.70** |
| Seattle | $133.39 | $115.91 | **+$17.48** |
| Phoenix | $80.12 | $71.39 | **+$8.74** |

Pooled average **+$17.64/household/year** — modest per house, but for an
illustrative 1,000-household community, **~$17,600/year**, from switching
decision rules at zero marginal hardware cost. This is a labeled *estimate*
on two stated, disagreeable assumptions (appliance size, the 2× multiplier);
change them in the script and rerun for your own numbers. The
decision-accuracy data underneath is real.

## Generalization: does this work for a city the model never saw?

This is the part of the project that didn't go in a straight line, and the
honesty of that process is the point, not a detour from it.

**First attempt (3-city scope): catastrophic failure.** Training on 2 of the
3 cities and testing on the one left out, decision accuracy collapsed to a
coin flip or worse (44–55%) — even switching to clear-sky-index
normalization specifically to reduce location/season confounding didn't
fix it. Continuous latitude/longitude doesn't actually behave as a
continuous, generalizable feature when the model has only ever seen 2–3
distinct values of it — it still effectively memorizes points.

**A properly controlled second attempt.** A review of this project
(independently, via Grok) correctly flagged that our first "fix" (3 → 10
cities) changed two things at once: how many cities trained the model, and
which city got held out — so "more locations helps" wasn't actually
isolated. `run_learning_curve.py` fixes that: **one city held out throughout
(Miami)**, varying only how many of the other 9 cities train the model (2,
5, 9), with 3 random subsets and 3 random init seeds at each size (27 runs
total):

| Training cities | Mean decision accuracy | Spread across runs | Mean skill |
|---|---|---|---|
| 2 | 47.3% | ±14.8 pts (unstable) | −14.0 (catastrophic) |
| 5 | 68.4% | ±23.5 pts (still unstable) | −0.98 |
| 9 | 81.5% | ±3.3 pts (stable) | −0.086 |

(At 9, there are only 9 non-Miami cities total, so all three "draws" are
necessarily the same set — that row's spread reflects only random-seed
noise, not subset noise, unlike the 2 and 5 rows. Doesn't change the means.)

More training locations monotonically and substantially improve both
accuracy *and* stability. This matches published results (a global model
trained across 20 UK sites generalized consistently; a small model trained
on few sites nearly doubled its error on an unseen site — Cargan et al.,
*Applied Intelligence* 2024).

**The honest remainder.** Even at 9 training cities, Miami's skill score
stays slightly negative (−0.086) — it doesn't fully close. Splitting
Miami's held-out result by season (using the already-collected 9-city-trained
run, no retraining needed) shows why:

| Season | Decision accuracy (model / baseline) | Skill |
|---|---|---|
| Wet (May–Oct) | 72.8% / 79.9% | −0.089 |
| Dry (Nov–Apr) | 86.8% / 89.0% | −0.019 |

Miami has an inverted seasonal cycle (tropical, Köppen Am: clear skies in
*winter*, cloudy wet season in *summer*) — the opposite of every other
training city's Northern-temperate pattern. The day-of-year feature learns
"summer = clearer" from the other 9 cities, which is backwards for Miami,
and the gap concentrates exactly in Miami's wet season as that mechanism
predicts (confirmed directly: checking each candidate city's own monthly
clear-sky curve, Houston — already in training — actually follows the
*standard* summer-peak pattern, not Miami's; Honolulu does too, despite
being tropical. Tampa's peninsula-driven spring-peak/summer-dip shape is
the real match).

**Two follow-up tests, both run, both honestly reported — one helped a
little, one made things much worse.**

*Swapping in a seasonally-matched city.* Replacing Boston with Tampa (same
training-set size, same holdout) roughly halved the mean skill deficit
(−0.086 → −0.039 across 3 seeds) but was noisy (−0.071, −0.072, +0.027) and
didn't move decision accuracy (~81% either way) — suggestive, not a clean
fix with just one matched city. Before spending more training runs chasing
a better match, two more candidates were screened the cheap way — compare
monthly CSI *curve correlation* with Miami's, no training involved — and
both ruled out: San Juan, PR (actually tropical by Köppen classification,
which made it the obvious next guess) correlates at −0.045 with Miami's
curve, and New Orleans, LA at −0.055, both far worse than Tampa's 0.614.
Climate-label intuition ("more tropical" should mean "more similar") isn't
what predicts a good swap candidate here — curve correlation is, and by
that measure Tampa isn't an arbitrary choice, it's a genuine standout among
the plausible options checked so far.

*Reframing the target as an anomaly* (predicting the deviation from Miami's
own monthly climatological CSI, instead of the absolute CSI level — the
top-ranked fix from the same review, on the reasoning that the seasonal
shape should come from the data, not be learned from 9 other cities' shapes)
made things **substantially worse**, not better, confirmed across 3 seeds
(same 9 training cities, same holdout each time):

| | seed 0 | seed 1 | seed 2 | Mean |
|---|---|---|---|---|
| Absolute (shipped) | −0.057 | −0.059 | −0.142 | **−0.086** |
| Anomaly | −0.209 | −0.802 | −0.215 | **−0.409** |

This is the opposite of what the review predicted, and it's reported as-is
rather than re-run until it looks better. Anomaly mode is also visibly less
stable across seeds (one run collapsed to −0.80), not just worse on average.
A plausible reason: anomaly framing throws away *today's* absolute CSI level
as a feature, and for a city whose day-to-day variance is itself unusually
large (Miami's wet season), that level carries real information the
climatology-relative framing doesn't recover elsewhere in the feature set.
This is implemented and reproducible (`--target-mode anomaly`), kept for
exactly that reason, but not what's shipped.

*Replacing lat/lon with explicit climate-descriptor features* (each city's
own 24-value monthly clear-sky-index + humidity profile, computed from
training years only — the literature-suggested fix for exactly this
problem) made Miami's gap **dramatically worse**, not better: skill
−4.08/−0.42/−0.70 across seeds, versus lat/lon's −0.086. A second review
(Grok) offered a sharper explanation than our first guess ("memorization"):
9 training cities' profiles span only 9 points in a 24-dimensional space, and
a ReLU network extrapolates roughly linearly outside the span those 9 points
jointly cover — almost any new city lands outside that span even when none
of its individual coordinates look extreme on their own. We checked that
directly: Miami's z-scores against the 9 training cities' distribution are
*not* extreme marginal outliers (max |z| = 1.07, mean |z| = 0.64), which
rules out "bad scaling" and is consistent with the joint-coverage
explanation instead. This is kept in `prepare_solar_data.py`
(`--location-feature climate_profile`) rather than deleted, so the negative
result stays reproducible. Lat/lon remains what's actually shipped, because
it's what performed best across every experiment in this project, not by
default.

**Extending the same controlled test beyond Miami.** Miami was the first
city run through `run_learning_curve.py`; the same 27-run (3 sizes × 3
subset draws × 3 init seeds) protocol was then run holding out Phoenix,
Seattle, and Minneapolis in turn, training on the other 9 of the original
10 cities each time:

| Holdout city | 2 cities: mean skill | 5 cities: mean skill | 9 cities: mean skill |
|---|---|---|---|
| Miami | −14.0 | −0.98 | −0.086 |
| Phoenix | −4.81 | −0.21 | **+0.13** |
| Seattle | −3.49 | −2.01 | −0.58 |
| Minneapolis | −1.07 | −0.02 | **+0.15** |

Two of three new holdout cities *flip to a positive skill* at 9 training
cities — the model actually beats smart persistence on a city it never
trained on, which is the strongest generalization result in this project.
Seattle is the exception, and it fails for a different, equally checkable
reason than Miami's: its seasonal *timing* is completely normal (CSI peaks
in August, same as every other Northern-temperate city here), but its
*magnitude* is an outlier — the lowest mean clear-sky index of any city in
the dataset (0.634, versus 0.70–0.91 everywhere else) and the largest
seasonal swing (0.278). That's the signature of a persistently overcast
maritime climate, and nothing else in the training set looks like it. Same
underlying story as Miami, different mechanism: the gap tracks how well the
held-out city's climate is represented among the training cities, not just
how many training cities there are.

**That hypothesis predicts its own fix, and the fix worked cleanly.** The
same swap test run for Miami/Tampa, run for Seattle: fetched Portland, OR
(confirmed via its own monthly CSI curve — amplitude 0.312, mean 0.657,
peak in July — to be the same Pacific-Northwest maritime pattern as
Seattle, not assumed), swapped it in for Boston in Seattle's 9-city
training set, same holdout, 3 seeds:

| | seed 0 | seed 1 | seed 2 | Mean |
|---|---|---|---|---|
| Boston (original) | −0.280 | +0.073 | −1.519 | −0.575 |
| Portland (swap) | +0.111 | +0.144 | +0.162 | **+0.139** |

This is a full flip to positive skill, and — unlike the Tampa swap, which
was noisy and inconsistent in sign across seeds — all three Portland seeds
land positive and close together. The cleanest, most decisive result in
this project's whole generalization investigation: once a training set
actually contains a city with a representative climate, Seattle stops
being a generalization failure at all.

## What we'd do next

Get genuine archived day-ahead forecasts (with real, verified lead time,
unlike Open-Meteo's archive) to attack the storm-front failure case, which
nothing in this project currently can see coming. On the generalization
side: Miami's gap is still only partially closed (Tampa helped some,
noisily, and it's the best of the three candidates screened by curve
correlation so far — San Juan and New Orleans were both ruled out cheaply
before training on them). No candidate found yet matches Tampa the way
Portland matched Seattle; a wider, more systematic search over Gulf/
Caribbean coastal cities, still screened by curve correlation first, could
still find one. We also added a paired t-test alongside the sign test for the
walk-forward comparison (it's what caught Phoenix's real-but-easy-to-miss
effect — see Results); a further step would be a true per-day
Diebold-Mariano test with autocorrelation-robust (HAC) variance on the
daily loss series within each fold, which is a strictly finer-grained
question than "is each fold's year-level mean distinguishable from chance,"
and matters more the more serial correlation there is within a test year.

## Sources

- [NASA POWER](https://power.larc.nasa.gov/) — daily solar/weather data,
  15 US/PR cities, 2005–2024 (13 used in actual training runs; San Juan and
  New Orleans fetched only for the curve-correlation screening check)
- [EIA](https://www.eia.gov/energyexplained/electricity/prices-and-factors-affecting-prices.php) —
  2024 US average residential electricity price (16.5¢/kWh)
- Veritas AI Hackathon 2026 Project Briefs, §4.3.1 — starting point, the
  persistence-baseline idea, and the decision-accuracy "make it your own"
  suggestion
- Cargan et al., *Applied Intelligence* 2024 — global multi-site solar
  forecasting models generalize more consistently than site-specific ones
  trained on few locations; cited for the generalization discussion above
- numpy — the only dependency; no pandas/sklearn required
- Claude Code (Claude Sonnet 5) — coding assistant used throughout
  development
- OpenAI (`gpt-6-astra`), Google Gemini (`gemini-pro-latest`), and xAI
  (`grok-4`) APIs — used during development for automated code review
  (`tools/peer_review.py`), which caught several real bugs: a year-boundary
  data-leakage bug, a silent test/prediction misalignment risk, the
  clear-sky leakage bug, and the leap-year climatology bug. Not used to
  generate the project's results.
- Grok (the standalone app, relayed manually, not the API) — an independent
  methodology review that correctly identified the confounded
  generalization comparison (fixed via `run_learning_curve.py`), the
  "tonight" framing error in the decision story (fixed throughout this
  document), the missing majority-class reference point, and the
  operational-realism caveat about NASA POWER's reanalysis delay — all
  addressed above, not just noted.

---

## Repo structure / how to reproduce

```
data/              raw NASA POWER CSVs, 15 cities, 2005-2024 (committed, so
                   the pipeline runs offline with no re-download)
pipeline/
  prepare_solar_data.py    builds features/target (clear-sky index), splits
                           by fold, supports --cities and --holdout-city
  preprocess_template.py   z-score normalize (fit on train only)
  train_template.py        from-scratch numpy MLP, early-stopped, --seed
  validate_template.py     model vs. smart-persistence baseline (mean/
                           median/p95/worst/skill), decision simulation
  estimate_impact.py       turns decision stats into the $/household/year
                           estimate in Impact, above
  export_template.py       bakes results JSON into interface/result.html
interface/
  template.html    static results page (no backend, works from file://)
run_pipeline.py            runs the single "official" fold_2024 demo
                           (3-city scope, trained on 2005-2022)
run_walkforward.py         runs all ten 3-city folds (2015-2024) + sign
                           test and paired t-test -- produces the Results
                           tables above
run_ridge_baseline.py      swaps closed-form ridge regression into the
                           shipped pipeline as a sanity check against the
                           MLP -- produces the ridge-vs-MLP table above
check_climate_curve_match.py  cheap pre-screen for a swap candidate's
                           monthly CSI curve correlation with the holdout
                           city's, before spending a training run on it
run_leave_one_city_out.py  the first (confounded) 10-city generalization
                           check -- superseded by run_learning_curve.py
                           below, kept for the record
run_learning_curve.py      the properly controlled generalization check,
                           parameterized by --holdout-city (run for Miami,
                           Phoenix, Seattle, Minneapolis: 2/5/9 training
                           cities, 27 runs each) -- produces the
                           Generalization tables above
```

Run the official 3-city demo with:

```
pip install -r requirements.txt   # just numpy, pinned
python run_pipeline.py
```

then open `interface/result.html` — a dark, static page (no backend, works
from `file://`) with a "Try a Day" explorer: pick any city and any real
held-out test date and see the naive baseline's call, SolarPlan's call, and
what actually happened, side by side. Run the full ten-fold walk-forward
(a few minutes, ten full train/val/test cycles):

```
python run_walkforward.py
python pipeline/estimate_impact.py   # optional: $/household and kg CO2/household
                                      # per year, shown on result.html if present
```

Run the controlled generalization study (longer — 27 training runs per
holdout city):

```
python run_learning_curve.py --holdout-city miami    # or phoenix/seattle/minneapolis
```

No installs beyond numpy, pinned in `requirements.txt` (Python's standard
library handles CSV parsing and the NASA POWER download).

The generic five-stage *pattern* (not this project's content) was scaffolded
before kickoff, which the hackathon rules explicitly allow ("you're welcome
to learn tools... beforehand"); the actual solar data, model, and results
above were built during the hackathon window.
