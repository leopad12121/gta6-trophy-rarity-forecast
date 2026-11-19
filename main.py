#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Reads every `<id>.in` case in the input directory and writes one
`<id>.out` forecast per case: for each trophy, five quantiles of its earn
rate (percent of tracked players) at the case's target, at levels
0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Model, per trophy, on the logit scale. A launch-week adoption curve
saturates, so the log-rate is fitted with a saturating exponential
`y(t) = L + c * exp(-t / tau)` over the case's own history only
(`t` = hours since the first read; tau on a log-spaced grid of 48 values
from 0.2 to 40 days; at each tau the (L, c) fit is closed-form least
squares of y on u = exp(-t / tau); the tau with the least squared error
wins). The forecast centre is the fitted value at the last read plus
the fitted move to the target, shrunk toward the cross-trophy mean move
(weight distinct / (distinct + 2), distinct = number of distinct rates
the trophy showed) and capped at 2 logit units. The spread widens with
the horizon and with the size of the move. A trophy whose rate never
moved (or with fewer than two usable reads) is not fitted at all: it
reproduces the house persistence baseline exactly.

Python 3.13, standard library only. No network, no randomness, no clock:
the same input bytes always produce the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime

# Standard-normal quantiles at the five levels, as the house baseline
# states them.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Logit bounds: the rates the page prints are clamped to [0.01, 99.99]
# percent before taking the logit.
LOGIT_MIN = math.log(0.0001 / 0.9999)
LOGIT_MAX = -LOGIT_MIN


def logit_pct(rate):
    """logit of an earn rate given in percent, clamped to [0.01, 99.99]."""
    r = min(99.99, max(0.01, rate))
    p = r / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(y):
    """Percent (0..100) for a logit value, overflow-safe."""
    if y >= 0.0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half-up, as Math.round(x * 10000) / 10000.

    Whole numbers come back as ints, so 1.1 stays 1.1, not 1.1000.
    """
    r = math.floor(x * 10000.0 + 0.5) / 10000.0
    if r == int(r):
        return int(r)
    return r


def parse_time(text):
    """Parse an ISO 8601 timestamp, or None when it does not parse."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def read_times(case):
    """Hours since the first read for every history read, and the target.

    Falls back to slot order (0, 1, 2, ...) when a timestamp does not
    parse, so a case with broken clocks still forecasts.
    """
    history = case["history"]
    stamps = [parse_time(read.get("at")) for read in history]
    target = parse_time(case.get("target"))
    if any(s is None for s in stamps) or target is None:
        times = [float(i) for i in range(len(history))]
        horizon_hours = float(case.get("horizon_minutes", 0)) / 60.0
        return times, times[-1] + horizon_hours if times else horizon_hours
    first = stamps[0]
    times = [(s - first).total_seconds() / 3600.0 for s in stamps]
    t_target = (target - first).total_seconds() / 3600.0
    return times, t_target


def ols_on_exp(ts, ys, tau_hours):
    """Least-squares fit of y = L + c * exp(-t / tau); None if degenerate.

    Returns (L, c, sse). Closed form: with u = exp(-t / tau), the slope
    is Cov(u, y) / Var(u) and the intercept is mean(y) - c * mean(u).
    """
    us = [math.exp(-t / tau_hours) for t in ts]
    n = len(ts)
    mean_u = sum(us) / n
    mean_y = sum(ys) / n
    var_u = sum((u - mean_u) ** 2 for u in us)
    if var_u <= 0.0:
        return None
    cov = sum((u - mean_u) * (y - mean_y) for u, y in zip(us, ys))
    c = cov / var_u
    intercept = mean_y - c * mean_u
    sse = sum((y - (intercept + c * u)) ** 2 for u, y in zip(us, ys))
    return intercept, c, sse


def fit_curve(ts, ys):
    """Best saturating-exponential fit over the tau grid, or None.

    The grid is 48 tau values log-spaced from 0.2 to 40 days.
    """
    best = None
    for i in range(48):
        tau_days = 0.2 * (200.0 ** (i / 47.0))
        got = ols_on_exp(ts, ys, tau_days * 24.0)
        if got is None:
            continue
        intercept, c, sse = got
        if best is None or sse < best[0]:
            best = (sse, intercept, c, tau_days * 24.0)
    if best is None:
        return None
    _, intercept, c, tau_hours = best
    return intercept, c, tau_hours


def forecast_case(case):
    """Five quantiles per trophy for one case."""
    history = case["history"]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    times, t_target = read_times(case)
    t_last = times[-1] if times else 0.0
    n_trophies = len(case["trophies"])

    # First pass: fit every trophy that moved, collecting raw moves.
    moves = []
    per_trophy = []
    for index in range(n_trophies):
        usable = []
        for read, t in zip(history, times):
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                usable.append((t, float(rates[index])))
        if len(usable) < 2 or len({rate for _, rate in usable}) < 2:
            per_trophy.append((None, usable))
            continue
        ts = [t for t, _ in usable]
        ys = [logit_pct(rate) for _, rate in usable]
        fitted = fit_curve(ts, ys)
        if fitted is None:
            per_trophy.append((None, usable))
            continue
        intercept, c, tau_hours = fitted
        at_last = intercept + c * math.exp(-t_last / tau_hours)
        at_target = intercept + c * math.exp(-t_target / tau_hours)
        move = at_target - at_last
        moves.append(move)
        per_trophy.append(((at_last, move, len({rate for _, rate in usable})), usable))

    mean_move = sum(moves) / len(moves) if moves else 0.0

    rows = []
    for index, (fit, usable) in enumerate(per_trophy):
        if fit is None:
            # Flat trophy (or too little history): persistence baseline.
            if usable:
                centre = logit_pct(usable[-1][1])
            else:
                centre = 0.0
            sigma = 0.5 * math.sqrt(horizon_days)
        else:
            at_last, move, distinct = fit
            weight = distinct / (distinct + 2.0)
            shrunk = weight * move + (1.0 - weight) * mean_move
            shrunk = max(-2.0, min(2.0, shrunk))
            centre = at_last + shrunk
            centre = max(LOGIT_MIN, min(LOGIT_MAX, centre))
            sigma = math.sqrt(
                (0.40 * math.sqrt(horizon_days)) ** 2
                + (0.20 * abs(shrunk)) ** 2
            )
            sigma = max(sigma, 0.12)
        quantiles = [round4(pct_from_logit(centre + z * sigma)) for z in Z]
        rows.append({"id": case["trophies"][index]["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        # Name the output after the input file, not the case's id: the
        # grader hands each run one <id>.in file and reads <id>.out.
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
