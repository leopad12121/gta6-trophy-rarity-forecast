"""GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.

Reads every ``<id>.in`` case in the input directory and writes
``<id>.out`` with one row per trophy::

    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}

Each case is the Exophase trophy page for Grand Theft Auto VI on PS5 as
the house recorded it: each trophy's earn rate (percent of tracked
players holding it) at every read before the case checkpoint, plus the
target time a week after launch. The score is mean pinball loss on the
logit scale.

Method
------
Work on the logit scale, ``y = logit(rate)``. Launch-week adoption
saturates — early trophies are picked up fast, late ones crawl, and the
tracked base keeps growing with newcomers who hold nothing yet, which
dilutes every rate — so a line in time or in log time would overshoot a
week out. Each trophy's logit rate is therefore fitted with a saturating
curve ``y(t) = L + c * exp(-t / tau)``, using only that case's own
history. The fit scans tau over 48 log-spaced values from 0.2 to 40
days; at each tau the (L, c) pair is closed-form least squares of y on
u = exp(-t / tau), and the tau with the smallest residual sum of squares
wins. The centre is the fitted value at the last read plus the fitted
move to the target, shrunk toward the cross-trophy mean move (trophies
move together; one trophy's noisy fit is not trusted alone) with weight
distinct / (distinct + 2), where distinct is the number of distinct
rates the trophy actually showed, and capped at 2 logit units either
way. A trophy whose rate never moved, or with fewer than two usable
reads, gets no fit at all: centre = logit(last rate) and spread
0.5 * sqrt(horizon in days), which reproduces the house persistence
baseline exactly on flat data.

Spread: ``sigma = sqrt((0.40 * sqrt(days))^2 + (0.20 * |move|)^2)``,
floored at 0.12 — a little tighter than the house random walk because
the fitted trend carries real information, widened by how far the
centre had to extrapolate. Quantiles are centre + z * sigma on the
logit scale, mapped back to percent and rounded to 4 decimals, half up
(as JavaScript's Math.round(x * 10000) / 10000 does it; whole numbers
are written as ints).

Python 3.13, standard library only, no network, no randomness,
deterministic: the same input bytes give the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an
# `import model` finds a model.py shipped beside this file (there is
# none; this entry is one file, but the line keeps the starter's
# contract).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.645, -0.674, 0.0, 0.674, 1.645]

PERSIST_SIGMA = 0.5     # house baseline spread per sqrt(day)
FIT_SIGMA = 0.40        # base spread once a trend is fitted
DRIFT_SIGMA = 0.20      # extra spread per logit unit of extrapolation
SIGMA_FLOOR = 0.12
MAX_MOVE = 2.0          # no extrapolated move trusted beyond this
TAU_MIN_HOURS = 4.8     # 0.2 days
TAU_MAX_HOURS = 960.0   # 40 days
TAU_GRID = 48


def clamp_rate(pct):
    """Keep a percent inside (0, 100) so its logit stays finite."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y):
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. Whole numbers come back
    as ints, so json writes 100 where JavaScript writes 100."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(value):
    """ISO 8601 string to an aware datetime; None when unparsable."""
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


def least_squares(xs, ys):
    """Least squares of ys on xs. Returns (intercept, slope, sse) or
    None when xs has no spread."""
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
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, sse


def fit_saturating(ts, ys):
    """Fit y(t) = L + c * exp(-t / tau) by scanning tau on a log grid;
    at each tau the (L, c) fit is closed-form least squares of y on
    u = exp(-t / tau). Returns (L, c, tau_hours) for the tau with the
    least residual sum of squares, or None when nothing fits."""
    best = None
    for k in range(TAU_GRID):
        tau = TAU_MIN_HOURS * ((TAU_MAX_HOURS / TAU_MIN_HOURS) ** (k / (TAU_GRID - 1)))
        us = [math.exp(-t / tau) for t in ts]
        result = least_squares(us, ys)
        if result is None:
            continue
        intercept, slope, sse = result
        if best is None or sse < best[0]:
            best = (sse, intercept, slope, tau)
    if best is None:
        return None
    _sse, level, c, tau = best
    return level, c, tau


def rate_at(reads, idx):
    """The trophy's rate at one read, or None when missing/null."""
    rates = reads.get("rates") or []
    if idx >= len(rates):
        return None
    return rates[idx]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0

    # Hours since the first read, per read. Fall back to slot order
    # when a timestamp does not parse.
    first = parse_time(history[0].get("at")) or parse_time(history[0].get("slot"))
    times = []
    for read in history:
        dt = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if dt is not None and first is not None:
            times.append(max(0.0, (dt - first).total_seconds() / 3600.0))
        else:
            times.append(float(len(times)))
    target_dt = parse_time(case.get("target"))
    if target_dt is not None and first is not None:
        target_hours = max(times[-1], (target_dt - first).total_seconds() / 3600.0)
    else:
        target_hours = times[-1] + case["horizon_minutes"] / 60.0

    trophies = case["trophies"]

    # Per-trophy fit: (fitted logit at last read, raw move to target,
    # shrinkage weight). A trophy whose rate never moved, or with
    # fewer than two usable reads, gets no fit and reproduces the
    # house persistence baseline exactly.
    fits = []
    for i in range(len(trophies)):
        points = []
        for j, read in enumerate(history):
            rate = rate_at(read, i)
            if rate is None:
                continue
            points.append((times[j], logit(rate)))
        if len(points) < 2:
            fits.append(None)
            continue
        distinct = len({round(y, 9) for _, y in points})
        if distinct < 2:
            fits.append(None)
            continue
        ts = [p[0] for p in points]
        ys = [p[1] for p in points]
        result = fit_saturating(ts, ys)
        if result is None:
            fits.append(None)
            continue
        level, c, tau = result
        anchor = level + c * math.exp(-times[-1] / tau)
        move = c * (math.exp(-target_hours / tau) - math.exp(-times[-1] / tau))
        weight = distinct / (distinct + 2.0)
        fits.append((anchor, move, weight))

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
                rate = rate_at(read, i)
                if rate is not None:
                    last_rate = rate
                    break
            centre = logit(last_rate if last_rate is not None else 0)
            sigma = PERSIST_SIGMA * math.sqrt(horizon_days)
            move = 0.0
        else:
            anchor, raw_move, weight = fit
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
        for z in Z_VALUES:
            q = round4(inv_logit(centre + z * sigma))
            q = min(100, max(0, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
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
