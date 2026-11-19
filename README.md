# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/leopad12121/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/leopad12121/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5,
the percent of the players Exophase tracks who will hold it one week
after launch (2026-11-26T00:00:00Z), given the house's recording of
the trophy page up to the case checkpoint — as five quantiles a trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## Model

Each trophy's logit earn rate is fitted with a saturating curve
`y(t) = L + c * exp(-t / tau)` over the case's own history (grid over
tau, closed-form OLS at each tau). A launch-week adoption curve
saturates, so a line in time or log time would overshoot a week out;
the saturating fit cannot run past the trophy's eventual rate. The
extrapolated move is shrunk toward the cross-trophy mean move and
capped at 2 logit units, and the spread widens with the extrapolation
distance and the fit's residual noise. A trophy whose rate never moved
reproduces the house persistence baseline exactly. Fitted only on the
case's own history — no data from after the checkpoint is used.

Validated locally: on the kit's practice cases the output is
byte-identical to the starter baseline (their history is flat), and in
a synthetic backtest (Weibull adoption curves with a growing tracked
player base and binomial sampling noise, forecast to day 7) it beats
the house baseline in 6 of 7 scenarios.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is Python 3.13, standard library only, no network, no
randomness. It reads every `<id>.in` case in the input directory and
writes `<id>.out` to the output directory:

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=d5a50dzy&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
