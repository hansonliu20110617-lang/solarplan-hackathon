"""
Replaces simulate_template.py for the real track: AI + Climate, next-day solar
forecast, compared across three climates (Phoenix AZ = sunny/stable, Chicago
IL = continental/frontal, Seattle WA = cloudy/volatile). Real NASA POWER
daily data, no API key.

Predicts the CLEAR-SKY INDEX (actual solar / typical clear-sky maximum for
that location and day), not raw kWh/m^2 -- this is standard practice in the
solar-forecasting literature (e.g. Lauret et al. 2022, Solar), and matters
here for a second reason: raw kWh/m^2 is heavily confounded by latitude and
season, which a model trained on only 2-3 distinct locations cannot
actually learn as a *continuous* function of (see
run_leave_one_city_out.py's result -- a lat/lon model badly failed to
generalize to an unseen city).

The "clear-sky maximum" is a CLIMATOLOGICAL figure -- each location's own
multi-year (training-years-only) average of NASA POWER's CLRSKY_SFC_SW_DWN
for that calendar day, not that specific day's raw value. This distinction
matters and was wrong in an earlier version of this file: CLRSKY_SFC_SW_DWN
is not a pure astronomical constant -- checking the raw data shows real
day-to-day variation (e.g. Phoenix's June 15 clear-sky value ranged from
7.81 to 9.26 kWh/m^2 across 20 different years), because it depends on
retrieved atmospheric state (aerosols, water vapor), not sun angle alone.
Using *tomorrow's own raw* CLRSKY value to convert a CSI prediction back to
kWh would therefore have quietly leaked real information about tomorrow's
actual atmosphere. The climatological (many-year average, training-years
only) version has no such problem: it's a fixed, pre-computable number for
any given calendar day, derivable before any forecast is made.

Feature row (today) -> target (tomorrow's), default --location-feature=latlon:
  [csi_today, temp_today, humidity_today, cloud_today,
   latitude, longitude, doy_sin, doy_cos] -> csi_tomorrow

Location is continuous latitude/longitude by default, not a one-hot city
flag -- a one-hot would let the model memorize fixed slots, useless (and
misleading) for a city it's never seen. We also tried --location-feature
climate_profile (each city's own 24-value monthly clear-sky-index and
humidity profile, computed from training years only, instead of lat/lon),
specifically hoping it would fix the generalization gap documented in
run_leave_one_city_out.py / run_learning_curve.py -- it made Miami's gap
dramatically WORSE instead (skill -4.08/-0.42/-0.70 across seeds vs.
lat/lon's -0.086). Kept in this file (not deleted) so that negative result
stays reproducible; the likely mechanism is that 24 raw per-city constants
are a higher-dimensional "fingerprint" than lat/lon's 2 numbers, giving the
model more room to memorize the training cities instead of learning
something transferable -- more expressive location features don't
automatically fix small-sample spatial generalization; they can make the
underlying memorization risk worse. day-of-year is sin/cos encoded so
Dec 31 and Jan 1 are adjacent
in feature space, not 364 apart.

Train/val/test years are CLI-configurable (--train-years/--val-years/
--test-years, comma-separated) so run_walkforward.py can run this multiple
times with different fold boundaries without duplicating this file. The
three year sets must be disjoint and in non-decreasing chronological order
(train before val before test) -- an overlap would leak evaluation rows
into training; a reversed order would (bizarrely) train on data from after
the period being evaluated. Both are rejected with a clear error rather
than silently producing a misleading score.

--holdout-city NAME removes that city's rows from train/val entirely (it
only appears in test), for run_leave_one_city_out.py's generalization
check -- it must also be present in --cities, or the flag would silently
do nothing (no row anywhere to remove). The held-out city's decision
threshold and clear-sky climatology are still computed from its own
train-year data -- a homeowner's own long-run sense of their local weather
and typical clear-sky days, not something the forecasting model needs to
learn, so this isn't leakage into the thing actually being tested
(next-day forecast accuracy).

Split membership is keyed on the TARGET (tomorrow's) date, not today's -- a
Dec-31 training example's label is Jan-1 of the next year, and keying on
today's date would put that label inside the held-out period.

Writes data.npz in the exact schema preprocess_template.py/train_template.py
already expect (X/y now in clear-sky-index units for y), so those two
stages need zero changes. Also writes test_meta.npz (city/date/
baseline_pred/target/decision_threshold/tomorrow_clrsky/tomorrow_actual_kwh
per test row, same order as X_test) so validate_template.py can convert CSI
predictions back to kWh and run the decision simulation in real-world
units -- NASA POWER's -999 missing-data sentinel (and any non-finite
reading) is dropped, not imputed.
"""
import argparse
import csv
import datetime
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

# (name, csv path, latitude, longitude) -- lat/lon kept for reference/
# documentation only, not fed to the model (see module docstring). Cities
# chosen for climate diversity (desert, coastal, continental, humid
# subtropical, tropical, high-altitude, Mediterranean) so a
# leave-one-city-out test has enough distinct training locations to learn
# something that generalizes, plus Tampa and Honolulu added specifically to
# test the seasonal-mechanism hypothesis behind Miami's generalization gap.
CITIES = [
    ("phoenix", os.path.join(DATA_DIR, "phoenix_az.csv"), 33.4484, -112.0740),
    ("seattle", os.path.join(DATA_DIR, "seattle_wa.csv"), 47.6062, -122.3321),
    ("chicago", os.path.join(DATA_DIR, "chicago_il.csv"), 41.8781, -87.6298),
    ("denver", os.path.join(DATA_DIR, "denver_co.csv"), 39.7392, -104.9903),
    ("miami", os.path.join(DATA_DIR, "miami_fl.csv"), 25.7617, -80.1918),
    ("minneapolis", os.path.join(DATA_DIR, "minneapolis_mn.csv"), 44.9778, -93.2650),
    ("albuquerque", os.path.join(DATA_DIR, "albuquerque_nm.csv"), 35.0844, -106.6504),
    ("houston", os.path.join(DATA_DIR, "houston_tx.csv"), 29.7601, -95.3701),
    ("losangeles", os.path.join(DATA_DIR, "losangeles_ca.csv"), 34.0522, -118.2437),
    ("boston", os.path.join(DATA_DIR, "boston_ma.csv"), 42.3601, -71.0589),
    # Checked empirically (not assumed): Honolulu's monthly clear-sky-index
    # curve actually matches the STANDARD pattern (clearest in summer) like
    # every other training city, not Miami's -- so it's a bad test of the
    # seasonal-mechanism hypothesis, kept here only for the record.
    ("honolulu", os.path.join(DATA_DIR, "honolulu_hi.csv"), 21.3099, -157.8583),
    # Tampa shares Miami's actual shape (spring peak, summer dip from
    # Florida-peninsula sea-breeze convection), confirmed by comparing their
    # monthly CSI curves directly -- the real test city for that hypothesis.
    ("tampa", os.path.join(DATA_DIR, "tampa_fl.csv"), 27.9506, -82.4572),
    # Portland is Seattle's analogue to Tampa/Miami: confirmed by comparing
    # monthly CSI curves directly, it's the same Pacific-Northwest maritime
    # pattern (low mean CSI, large seasonal amplitude) that makes Seattle a
    # generalization outlier -- a test city for the Seattle swap experiment.
    ("portland", os.path.join(DATA_DIR, "portland_or.csv"), 45.5152, -122.6784),
]
MISSING_SENTINEL = -900.0  # NASA POWER uses -999; threshold catches formatting variants
# CSI can run slightly above 1.0 (cloud-enhancement effects can briefly exceed the
# theoretical clear-sky value) or be driven huge by a near-zero clear-sky denominator
# in edge cases; clip to a generous but bounded range so one bad ratio can't dominate
# every error statistic downstream.
CSI_CLIP = (0.0, 1.5)


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
            "clrsky": float(row["CLRSKY_SFC_SW_DWN"]),
            "t2m": float(row["T2M"]),
            "rh2m": float(row["RH2M"]),
            "cloud": float(row["CLOUD_AMT"]),
            "year": int(row["YEAR"]),
        })
    rows.sort(key=lambda r: r["date"])  # NASA POWER already returns this sorted; be explicit anyway
    return rows


def _climatological_clrsky(rows, train_years):
    """Per-calendar-day average CLRSKY_SFC_SW_DWN, computed ONLY from
    train_years rows -- this is the "typical clear-sky day" figure used as
    the CSI denominator everywhere (train/val/test alike), so it never
    depends on the specific day being converted. Missing/non-finite
    readings are excluded from the average, not treated as zero.

    Keyed by (month, day), not day-of-year: tm_yday shifts by one for every
    date after Feb 29 in a leap year (day 60 is Feb 29 in a leap year but
    Mar 1 otherwise), so day-of-year silently averages together different
    calendar dates across leap/non-leap years -- (month, day) has no such
    problem. Feb 29 itself only has training data in years when a leap year
    falls inside train_years; see _clrsky_lookup for what happens when it
    doesn't (falling back to the overall annual average would badly distort
    a winter day's value -- caught by review: Seattle's annual mean clear
    sky is much higher than its February value)."""
    by_date = {}
    for row in rows:
        if row["year"] not in train_years:
            continue
        clrsky = row["clrsky"]
        if not np.isfinite(clrsky) or clrsky <= MISSING_SENTINEL or clrsky <= 0:
            continue
        d = datetime.date.fromisoformat(row["date"])
        by_date.setdefault((d.month, d.day), []).append(clrsky)
    if not by_date:
        raise SystemExit(
            "[prepare_solar_data] no usable CLRSKY_SFC_SW_DWN readings in the training "
            "years -- can't build a clear-sky climatology to normalize against."
        )
    overall_fallback = float(np.mean([v for vals in by_date.values() for v in vals]))
    return {md: float(np.mean(vals)) for md, vals in by_date.items()}, overall_fallback


def _clrsky_lookup(clim_clrsky, month, day, fallback):
    """Look up a (month, day) climatology value. Feb 29 specifically falls
    back to a Feb-28/Mar-1 neighbor average instead of the global annual
    mean, which would badly distort a winter day's value -- the ONLY
    calendar date expected to ever be legitimately missing from a
    multi-year training window (every other date appears every single
    year). Peer review correctly caught an earlier version of this that
    applied the Feb 28/Mar 1 fallback to ANY missing date, which would have
    wrongly given a wintertime denominator to some other season's date in
    the (very unlikely) event its own readings were entirely missing --
    that general case still falls back to the honest annual average."""
    if (month, day) in clim_clrsky:
        return clim_clrsky[(month, day)]
    if (month, day) != (2, 29):
        return fallback
    neighbors = [v for v in (clim_clrsky.get((2, 28)), clim_clrsky.get((3, 1))) if v is not None]
    return float(np.mean(neighbors)) if neighbors else fallback


def _climatological_monthly_profile(rows, train_years, clim_clrsky, clim_fallback):
    """Each location's own 12-value average clear-sky-index-by-month and
    12-value average-humidity-by-month, computed ONLY from train_years --
    the climate descriptor that replaces lat/lon (see module docstring).
    CSI here reuses the same per-(month,day) climatological ceiling as
    _build_examples, so this profile is on the same scale as the daily CSI
    feature/target everywhere else. Returns a flat 24-value list, months
    1-12 CSI then 1-12 humidity, with any month missing training data
    (shouldn't happen for a multi-year training window) filled from the
    overall training-period average for that variable."""
    csi_by_month = {m: [] for m in range(1, 13)}
    rh_by_month = {m: [] for m in range(1, 13)}
    for row in rows:
        if row["year"] not in train_years:
            continue
        d = datetime.date.fromisoformat(row["date"])
        # CSI and humidity are accumulated independently -- a day with a
        # missing solar reading but a valid humidity reading (or vice versa)
        # should still contribute to whichever profile it can. Peer review
        # caught an earlier version that skipped humidity too whenever sw
        # was missing, needlessly discarding valid data (didn't affect this
        # project's actual numbers since none of our cities had missing
        # readings, but wrong in general).
        if np.isfinite(row["sw"]) and row["sw"] > MISSING_SENTINEL:
            ceiling = _clrsky_lookup(clim_clrsky, d.month, d.day, clim_fallback)
            csi_by_month[d.month].append(np.clip(row["sw"] / ceiling, *CSI_CLIP))
        if np.isfinite(row["rh2m"]) and row["rh2m"] > MISSING_SENTINEL:
            rh_by_month[d.month].append(row["rh2m"])

    all_csi = [v for vals in csi_by_month.values() for v in vals]
    all_rh = [v for vals in rh_by_month.values() for v in vals]
    if not all_csi or not all_rh:
        raise SystemExit(
            "[prepare_solar_data] no usable readings in the training years -- "
            "can't build a monthly climate-descriptor profile."
        )
    csi_fallback = float(np.mean(all_csi))
    rh_fallback = float(np.mean(all_rh))
    csi_profile = [float(np.mean(csi_by_month[m])) if csi_by_month[m] else csi_fallback for m in range(1, 13)]
    rh_profile = [float(np.mean(rh_by_month[m])) if rh_by_month[m] else rh_fallback for m in range(1, 13)]
    return csi_profile + rh_profile


def _build_examples(rows, location_feature, clim_clrsky, clim_fallback, monthly_csi=None):
    """One example per (today, tomorrow) pair. Drops the last row of the
    series (no tomorrow), any pair touching a missing/non-finite reading,
    and any pair that isn't actually consecutive days. clim_clrsky is the
    per-calendar-day CLIMATOLOGICAL clear-sky ceiling (see
    _climatological_clrsky) -- using it instead of that day's own raw
    CLRSKY value is what makes the kWh conversion in validate_template.py
    leak-free.

    monthly_csi (12 values, or None): if given, both the "today" feature
    and the target become ANOMALIES -- CSI minus that city's own monthly
    climatological CSI -- instead of absolute CSI (see --target-mode in
    main()). Rationale (from review, confirmed by checking the data): the
    climate-descriptor location feature made an unseen city's forecasts
    much worse, not better, because 9 training cities can't span 24
    feature dimensions -- almost any new city falls outside the training
    points' span even when no single coordinate looks like an outlier.
    Reframing the target as a deviation from each city's OWN climatology
    (a direct lookup, not something the network has to learn or
    extrapolate) sidesteps that: the network only ever has to learn
    weather-driven departures from normal, which should behave similarly
    in any climate, rather than the climate's absolute level itself."""
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

        today_ceiling = _clrsky_lookup(clim_clrsky, today_date.month, today_date.day, clim_fallback)
        tomorrow_ceiling = _clrsky_lookup(clim_clrsky, tomorrow_date.month, tomorrow_date.day, clim_fallback)

        today_csi = np.clip(today["sw"] / today_ceiling, *CSI_CLIP)
        tomorrow_csi = np.clip(tomorrow["sw"] / tomorrow_ceiling, *CSI_CLIP)

        if monthly_csi is not None:
            today_offset = monthly_csi[today_date.month - 1]
            tomorrow_offset = monthly_csi[tomorrow_date.month - 1]
            today_value = today_csi - today_offset  # anomaly: today's CSI feature
            target_value = tomorrow_csi - tomorrow_offset  # anomaly: the target
            baseline_value = today_value  # persistence in anomaly-space
        else:
            today_offset = tomorrow_offset = 0.0  # no-op reconstruction offset in absolute mode
            today_value = today_csi
            target_value = tomorrow_csi
            baseline_value = today_csi

        # The sin/cos seasonal feature can stay day-of-year-based: a leap
        # year shifts its phase by at most 1/365 of a full cycle, a
        # negligible wobble for a smooth cyclical feature -- unlike the
        # climatology above, there's no discrete "wrong bucket" failure mode.
        angle = 2 * math.pi * today_date.timetuple().tm_yday / 365.25
        examples.append({
            "features": [
                float(today_value), today["t2m"], today["rh2m"], today["cloud"],
                *location_feature, math.sin(angle), math.cos(angle),
            ],
            "target": float(target_value),
            "baseline_pred": float(baseline_value),
            "date": today["date"],
            "target_year": tomorrow_date.year,  # split membership keys on this, not today's year
            "tomorrow_clrsky": tomorrow_ceiling,  # climatological -- pre-computable, not a forecast
            # added back to a predicted/true anomaly to reconstruct absolute
            # CSI before converting to kWh -- 0.0 in absolute mode (no-op)
            "tomorrow_csi_offset": float(tomorrow_offset),
            "tomorrow_actual_kwh": tomorrow["sw"],
        })
    if skipped_gap:
        print(f"[prepare_solar_data] skipped {skipped_gap} non-consecutive-day pair(s)")
    return examples


def _split(examples, years):
    return [e for e in examples if e["target_year"] in years]


def _parse_years(s):
    return {int(y) for y in s.split(",")}


def _check_year_sets_disjoint_and_ordered(train_years, val_years, test_years):
    if train_years & val_years or train_years & test_years or val_years & test_years:
        raise SystemExit(
            "[prepare_solar_data] --train-years/--val-years/--test-years overlap -- "
            "that would put identical examples in both training and evaluation, "
            "invalidating the evaluation metrics."
        )
    # Every year in train must precede every year in val, which must precede
    # every year in test -- comparing only the closest edges (max(train) vs
    # min(val)) missed the case of val years that are disjoint from but
    # *after* test years (e.g. train={2000}, val={2003}, test={2001,2002}).
    if train_years and val_years and max(train_years) >= min(val_years):
        raise SystemExit(
            "[prepare_solar_data] train years aren't entirely before val years -- "
            "a reversed order would train on data from after the validation period."
        )
    if val_years and test_years and max(val_years) >= min(test_years):
        raise SystemExit(
            "[prepare_solar_data] val years aren't entirely before test years -- "
            "a reversed order would let model selection see data from after the "
            "period being evaluated."
        )
    if train_years and test_years and max(train_years) >= min(test_years):
        raise SystemExit(
            "[prepare_solar_data] train years aren't entirely before test years -- "
            "a reversed order would train on data from after the period being evaluated."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-years", default="2005,2006,2007,2008,2009,2010,2011,2012,2013,"
                                              "2014,2015,2016,2017,2018,2019,2020,2021,2022")
    ap.add_argument("--val-years", default="2023")
    ap.add_argument("--test-years", default="2024")
    ap.add_argument("--holdout-city", default=None, choices=[c[0] for c in CITIES],
                     help="remove this city's rows from train/val entirely -- it only "
                          "appears in test, for a true unseen-location generalization check")
    ap.add_argument("--cities", default="phoenix,seattle,chicago",
                     help="comma-separated subset of CITIES to use. Default is the "
                          "original three (the main submission's scope, already "
                          "validated extensively) -- run_leave_one_city_out.py passes "
                          "all ten for the many-location generalization experiment, "
                          "without changing what the main pipeline/walk-forward use.")
    ap.add_argument("--location-feature", default="latlon", choices=["latlon", "climate_profile"],
                     help="latlon (default, shipped): continuous latitude/longitude. "
                          "climate_profile: each city's own 24-value monthly CSI+humidity "
                          "profile instead -- tried specifically to fix Miami's "
                          "generalization gap, found empirically to make it dramatically "
                          "worse (skill -4.08/-0.42/-0.70 across seeds vs. latlon's -0.086) "
                          "-- kept here, not deleted, so that negative result stays "
                          "reproducible. Likely cause: 24 raw per-city constants are a "
                          "higher-dimensional fingerprint than lat/lon's 2 numbers, giving "
                          "the model more room to memorize training cities instead of "
                          "learning something transferable.")
    ap.add_argument("--target-mode", default="absolute", choices=["absolute", "anomaly"],
                     help="absolute (default, shipped): the 'today' feature and target are "
                          "raw clear-sky index. anomaly: both become CSI minus that city's "
                          "own monthly climatological CSI (computed from training years "
                          "only) -- tried specifically because the climate_profile location "
                          "feature failed by making the network extrapolate a learned "
                          "function past only 9 training points; this instead hands the "
                          "seasonal level to the network as a direct, non-learned lookup, "
                          "so it only ever has to learn weather-driven departures from it.")
    args = ap.parse_args()
    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    test_years = _parse_years(args.test_years)
    _check_year_sets_disjoint_and_ordered(train_years, val_years, test_years)

    active_names = set(args.cities.split(","))
    unknown = active_names - {c[0] for c in CITIES}
    if unknown:
        raise SystemExit(f"[prepare_solar_data] --cities has unknown city name(s): {unknown}")
    if args.holdout_city and args.holdout_city not in active_names:
        raise SystemExit(
            f"[prepare_solar_data] --holdout-city {args.holdout_city} isn't in --cities "
            f"({sorted(active_names)}) -- there'd be no row anywhere to remove, so this "
            "would silently run an ordinary (non-holdout) evaluation instead of the "
            "unseen-city check it looks like it's doing."
        )
    active_cities = [c for c in CITIES if c[0] in active_names]

    all_examples = {"train": [], "val": [], "test": []}
    test_meta = {"city": [], "date": [], "baseline_pred": [], "target": [], "decision_threshold": [],
                 "tomorrow_clrsky": [], "tomorrow_actual_kwh": [], "tomorrow_csi_offset": []}

    for city_name, path, lat, lon in active_cities:
        rows = _read_city_csv(path)
        clim_clrsky, clim_fallback = _climatological_clrsky(rows, train_years)
        # Computed from THIS city's own training-year data regardless of
        # --holdout-city status -- same reasoning as decision_threshold
        # above: a direct lookup of a location's own known history, not
        # something the forecasting model needs to have trained on.
        need_profile = args.location_feature == "climate_profile" or args.target_mode == "anomaly"
        profile = _climatological_monthly_profile(rows, train_years, clim_clrsky, clim_fallback) if need_profile else None
        location_feature = profile if args.location_feature == "climate_profile" else [lat, lon]
        monthly_csi = profile[:12] if args.target_mode == "anomaly" else None
        examples = _build_examples(rows, location_feature, clim_clrsky, clim_fallback, monthly_csi=monthly_csi)
        print(f"[prepare_solar_data] {city_name}: {len(rows)} daily rows -> {len(examples)} usable examples")

        city_splits = {
            split_name: _split(examples, years)
            for split_name, years in (("train", train_years), ("val", val_years), ("test", test_years))
        }
        # "Enough sun to be worth running an appliance off solar" is a real kWh
        # threshold (an appliance doesn't care about clear-sky fraction), set
        # from this city's own TRAINING-period actual kWh, never from test --
        # and never from a different city, even in --holdout-city mode (a
        # resident already knows their own area's weather without the model
        # needing to have trained on it).
        train_kwh = [e["tomorrow_actual_kwh"] for e in city_splits["train"]]
        if not train_kwh:
            raise SystemExit(
                f"[prepare_solar_data] {city_name} has zero usable training-period "
                "examples -- can't compute its decision threshold from an empty set "
                "(check --train-years actually overlaps this city's data)."
            )
        decision_threshold = float(np.median(train_kwh))

        is_holdout = city_name == args.holdout_city
        if is_holdout and not city_splits["test"]:
            raise SystemExit(
                f"[prepare_solar_data] --holdout-city {city_name} has zero usable "
                "test-period examples -- with other cities still providing test rows, "
                "this would silently run without ever actually evaluating the held-out "
                "city, defeating the point of the holdout."
            )
        for split_name, split_examples in city_splits.items():
            if is_holdout and split_name in ("train", "val"):
                continue  # this city contributes no training/validation signal at all
            all_examples[split_name].extend(split_examples)
            if split_name == "test":
                test_meta["city"].extend([city_name] * len(split_examples))
                test_meta["date"].extend([e["date"] for e in split_examples])
                test_meta["baseline_pred"].extend([e["baseline_pred"] for e in split_examples])
                test_meta["target"].extend([e["target"] for e in split_examples])
                test_meta["decision_threshold"].extend([decision_threshold] * len(split_examples))
                test_meta["tomorrow_clrsky"].extend([e["tomorrow_clrsky"] for e in split_examples])
                test_meta["tomorrow_actual_kwh"].extend([e["tomorrow_actual_kwh"] for e in split_examples])
                test_meta["tomorrow_csi_offset"].extend([e["tomorrow_csi_offset"] for e in split_examples])

    def to_arrays(split_name):
        exs = all_examples[split_name]
        if not exs:
            raise SystemExit(
                f"[prepare_solar_data] '{split_name}' split is empty -- check "
                "--train-years/--val-years/--test-years (and --holdout-city, which "
                "removes a whole city from train/val) against the date range actually "
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
        # matches it in length.
        target=np.array(test_meta["target"], dtype=np.float64),
        decision_threshold=np.array(test_meta["decision_threshold"], dtype=np.float64),
        # known in advance for any date/location -- not a forecast -- so
        # converting a CSI prediction back to kWh with this introduces no leakage.
        tomorrow_clrsky=np.array(test_meta["tomorrow_clrsky"], dtype=np.float64),
        tomorrow_actual_kwh=np.array(test_meta["tomorrow_actual_kwh"], dtype=np.float64),
        # added back to a predicted/true anomaly to reconstruct absolute CSI
        # before converting to kWh -- all zeros in --target-mode absolute (no-op).
        tomorrow_csi_offset=np.array(test_meta["tomorrow_csi_offset"], dtype=np.float64),
    )

    print(f"[prepare_solar_data] train={X_train.shape} val={X_val.shape} test={X_test.shape}"
          + (f"  (holdout city: {args.holdout_city})" if args.holdout_city else ""))
    print(f"[prepare_solar_data] wrote data.npz, test_meta.npz to {HERE}")


if __name__ == "__main__":
    main()
