# SolarPlan: a next-day solar forecast, compared across two climates

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

Starting point: brief §4.3.1, *Next-day solar energy forecast*. We added the
brief's own suggested extension — comparing how predictable solar output is
in two very different climates — using four years (2021–2024) of real NASA
POWER daily data for Phoenix, AZ (sunny, stable) and Seattle, WA (cloudy,
volatile), with a persistence-baseline comparison and a per-city breakdown
instead of one pooled number.

## How it works

**Data:** NASA POWER daily readings, Phoenix + Seattle, 2021-01-01 through
2024-12-31 (1,461 days each, no missing readings in either city). Features:
today's solar irradiance, temperature, humidity, cloud cover, and a city
flag → target: tomorrow's solar irradiance. Split by calendar year, never
shuffled: train 2021–2022 (1,460 rows), val 2023 (730 rows), test 2024 (730
rows, held out — never touched during training or early stopping).

**Model:** a small from-scratch numpy MLP (two hidden layers, 64→32, ReLU,
manual Adam optimizer), early-stopped on validation loss. We also checked a
plain closed-form linear regression as a sanity check on whether the MLP's
extra capacity was actually earning its keep.

## Results

Baseline (persistence — "tomorrow's solar = today's solar"): median error
**0.377** kWh/m²/day, pooled across both cities. Our model: **0.442** pooled —
loses to the baseline overall, but that pooled number hides the real story:

| City | Baseline median error | Our model median error | Model wins? |
|---|---|---|---|
| Phoenix (sunny, stable) | 0.235 | 0.319 | No |
| Seattle (cloudy, volatile) | 0.601 | 0.597 | Yes |

In Phoenix, solar output barely changes day to day, so "today ≈ tomorrow" is
already close to optimal — a model can't easily improve on that. In Seattle,
day-to-day solar is genuinely harder to guess, and our model modestly beats
persistence there. A plain linear regression does **not** beat the baseline
in either city (pooled median 0.501), which tells us the Seattle improvement
needs real nonlinearity (cloud/humidity interaction effects), not just extra
features — so it's not just noise from a model with more parameters.

**Where it fails:** the single worst error (4.6 kWh/m²/day) is a two-day
whiplash in Phoenix in May 2024 — solar output cratered overnight, then
snapped back up the next day, almost certainly a passing storm front. Both
the model and the baseline miss it equally, because neither has any
forward-looking weather signal — today's data alone can't see tomorrow's
incoming front.

(Split membership is keyed on each example's *target* date, not its feature
date, so a Dec-31 training example's label — which falls on Jan 1 of the
next year — can't leak into the validation/test period it's dated inside.)

## What we'd do next

Add a real forward-looking feature (e.g. next-day forecast data from
Open-Meteo) instead of relying only on today's readings — that should help
specifically with the abrupt-transition failures we found. Also test a third,
intermediate climate to see whether the Phoenix/Seattle pattern (stable
climate favors the baseline, volatile climate favors the model) generalizes.

## Sources

- [NASA POWER](https://power.larc.nasa.gov/) — daily solar/weather data for
  Phoenix, AZ and Seattle, WA, 2021–2024
- Veritas AI Hackathon 2026 Project Briefs, §4.3.1 (Next-day solar energy
  forecast) — starting point and the persistence-baseline idea
- numpy — the only dependency; no pandas/sklearn required

---

## Repo structure / how to reproduce

```
data/            raw NASA POWER CSVs for both cities (committed, so the
                 pipeline runs offline with no re-download)
pipeline/
  prepare_solar_data.py   builds features/target, splits by year -> data.npz
  preprocess_template.py  z-score normalize (fit on train only)
  train_template.py       from-scratch numpy MLP, early-stopped
  validate_template.py    model vs persistence baseline, per-city breakdown
  export_template.py      bakes results JSON into interface/result.html
interface/
  template.html    static results page (no backend, works from file://)
run_pipeline.py     runs all five stages in order
```

Run the whole thing with:

```
python run_pipeline.py
```

then open `interface/result.html`. No installs beyond numpy (Python 3's
standard library handles the CSV parsing and the NASA POWER download).

The generic five-stage *pattern* (not this project's content) was scaffolded
before kickoff, which the hackathon rules explicitly allow ("you're welcome
to learn tools... beforehand"); the actual solar data, model, and results
above were built during the hackathon window.

Datasets, pretrained models, and AI tools used are listed under Sources
above; no AI-generated code was used beyond standard coding-assistant help
during development (Claude Code).
