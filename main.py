"""Forecast GTA VI trophy earn rates one week after launch.

Entry for make.fun bounty gta6-trophy-rarity-forecast-01.

What it does
------------
Reads every ``<id>.in`` case in the input directory and writes
``<id>.out`` to the output directory::

    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}

Each case gives the Exophase trophy page for Grand Theft Auto VI on PS5
as the house recorded it: every trophy's earn rate (percent of the
players Exophase tracks who hold it) at each read before the case's
checkpoint, plus the players-tracked count.  The target is the earn rate
at the case's ``target`` (the final case: 2026-11-26T00:00:00Z, a week
after launch; the practice cases: 90 minutes after the checkpoint).
The score is the mean pinball loss on the logit scale, so the quantiles
reported here are the ones the model really believes.

Model (fitted only on the case's own history -- no data from after the
checkpoint is used anywhere)
---------------------------------------------------------------
Work on the logit scale, ``y = logit(rate)``.  A trophy's earn rate
grows as players work through the game, but the growth slows: early
trophies are picked up fast, late ones crawl, and the tracked player
base itself keeps growing with newcomers who hold nothing yet, which
dilutes every rate.  A straight line in time -- or in log time --
would overshoot a week out, because a launch-week adoption curve
saturates, so each trophy's logit rate is fitted with a saturating
curve:

    y(t) = L + c * exp(-t / tau)

fitted by a grid over tau (0.2..40 days, log-spaced) with the (L, c)
fit closed-form OLS at each tau.  The centre is the fitted value
extrapolated to the target, with the extrapolated move shrunk toward
the cross-trophy mean move -- trophies move together, and one
trophy's noisy fit is not trusted on its own -- and capped at 2 logit
units either way:

    centre = fitted(t_last) + shrink(fitted(t_target) - fitted(t_last))

where the shrinkage weight grows with the number of distinct rates
the trophy actually showed (a trophy whose rate never moved gets
weight 0 and reproduces the house persistence baseline exactly).

The spread starts a little tighter than the house baseline's
random-walk width (0.5 * sqrt(horizon in days)) -- the saturating fit
carries real information the random walk ignores -- and widens with
how far the centre had to extrapolate, since a long extrapolation is
genuinely less certain:

    sigma = sqrt((0.40 * sqrt(days))^2 + (0.20 * |move|)^2)

with a 0.12 floor.  (A sweep over synthetic launch weeks picked this
width over both the wider and the tighter settings: it beats the
house baseline in every scenario while keeping a margin for the real
week behaving differently from the synthetic ones.)  Quantiles are centre + z_tau * sigma on the logit
scale, mapped back to percent and rounded to 4 decimals (half up, as
the starter rounds them).

The program is Python 3.13, standard library only, no network, no
randomness, deterministic: the same input bytes give the same output
bytes.  Runtime is milliseconds; the run-time tie-break costs nothing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a model.py shipped beside this file (there is none; this
# entry is one file, but the line keeps the starter's contract).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# House baseline spread, in logit units per square-root day.
BASE_SIGMA = 0.5
# Slightly tighter base once a trend is fitted (the trend carries real
# information the random walk ignores), widened by extrapolation.
FIT_SIGMA = 0.40
DRIFT_SIGMA = 0.20
SIGMA_FLOOR = 0.12
# Hours added inside the log so the first read stays finite.
LOG_OFFSET_HOURS = 6.0
# No extrapolated move is trusted beyond this many logit units.
MAX_MOVE = 2.0


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100 - p))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back
    as an int, so json.dumps writes 100 where JavaScript writes 100
    (not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(value):
    """ISO 8601 to an aware datetime; None when unparsable."""
    if not isinstance(value, str):
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def ols(xs, ys):
    """Ordinary least squares of ys on xs.  Returns (intercept, slope,
    r_squared, sse) or None when xs has no spread."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return intercept, slope, r2, sse


def saturating_fit(ts, ys):
    """Fit y(t) = L + c * exp(-t / tau) by a grid over tau (log-spaced
    0.2..40 days); for each tau the (L, c) fit is closed-form OLS of y
    on u = exp(-t / tau).  The curve saturates at L, so -- unlike a
    line in log time -- it cannot run away past the trophy's eventual
    rate when the horizon is a week out.  Returns (L, c, tau_hours,
    resid_std) or None when the points carry no usable shape."""
    best = None
    n = len(ts)
    for k in range(48):
        tau = 4.8 * (200.0 ** (k / 47.0))  # 4.8 h .. 960 h, log-spaced
        us = [math.exp(-t / tau) for t in ts]
        result = ols(us, ys)
        if result is None:
            continue
        intercept, slope, _r2, sse = result
        if best is None or sse < best[0]:
            best = (sse, intercept, slope, tau)
    if best is None:
        return None
    sse, intercept, slope, tau = best
    return intercept, slope, tau, math.sqrt(sse / n)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0

    # Hours since the first read, per read.  Fall back to slot order
    # when a timestamp does not parse.
    first = parse_time(history[0].get("at")) or parse_time(history[0].get("slot"))
    times = []
    for read in history:
        dt = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if dt is not None and first is not None:
            times.append(max(0.0, (dt - first).total_seconds() / 3600.0))
        else:
            times.append(float(len(times)))
    x_last = math.log(times[-1] + LOG_OFFSET_HOURS)
    target_dt = parse_time(case.get("target"))
    if target_dt is not None and first is not None:
        target_hours = max(times[-1], (target_dt - first).total_seconds() / 3600.0)
    else:
        target_hours = times[-1] + case["horizon_minutes"] / 60.0
    x_target = math.log(target_hours + LOG_OFFSET_HOURS)

    trophies = case["trophies"]
    # Per-trophy fit: (fitted logit at last read, raw move, weight,
    # residual std).  A trophy whose rate never moved gets no fit and
    # reproduces the house persistence baseline exactly.
    fits = []
    for i in range(len(trophies)):
        points = []
        for j, read in enumerate(history):
            rate = read["rates"][i] if i < len(read.get("rates") or []) else None
            if rate is None:
                continue
            points.append((times[j], logit(rate)))
        if len(points) < 2:
            fits.append(None)
            continue
        distinct = len({round(y, 9) for _, y in points})
        if distinct < 2:
            fits.append(None)  # flat: persistence, exactly the baseline
            continue
        ts = [p[0] for p in points]
        ys = [p[1] for p in points]
        result = saturating_fit(ts, ys)
        if result is None:
            fits.append(None)
            continue
        level, c, tau, resid = result
        u_last = math.exp(-times[-1] / tau)
        u_target = math.exp(-target_hours / tau)
        anchor = level + c * u_last
        move = c * (u_target - u_last)
        weight = distinct / (distinct + 2.0)
        fits.append((anchor, move, weight, resid))

    # Cross-trophy mean move: the shared drift every trophy is shrunk
    # toward, so one trophy's noisy fit cannot run away on its own.
    moves = [f[1] for f in fits if f is not None]
    mean_move = sum(moves) / len(moves) if moves else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        fit = fits[i]
        if fit is None:
            last_rate = None
            for read in reversed(history):
                rate = read["rates"][i] if i < len(read.get("rates") or []) else None
                if rate is not None:
                    last_rate = rate
                    break
            centre = logit(last_rate if last_rate is not None else 0)
            sigma = BASE_SIGMA * math.sqrt(horizon_days)
            move = 0.0
        else:
            anchor, raw_move, weight, resid = fit
            move = weight * raw_move + (1.0 - weight) * mean_move
            move = max(-MAX_MOVE, min(MAX_MOVE, move))
            centre = anchor + move
            centre = max(logit(0.01), min(logit(99.99), centre))
            sigma = math.sqrt(
                (FIT_SIGMA * math.sqrt(horizon_days)) ** 2
                + (DRIFT_SIGMA * abs(move)) ** 2
            )
        sigma = max(SIGMA_FLOOR, sigma)
        qs = []
        prev = None
        for z in Z:
            q = round4(100 / (1 + math.exp(-(centre + z * sigma))))
            q = min(100, max(0, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
