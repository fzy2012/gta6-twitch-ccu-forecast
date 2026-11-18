# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fzy2012/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fzy2012/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), given only the first hour's count.

The forecaster is a single Python file (`main.py`, standard library only).
It reads the case file, builds the launch-day shape from the past big game
launches the case carries as comparables, and writes one forecast file per
case. The shape is the empirical distribution of each later hour's viewers
relative to hour 1's print, in log space, fitted as a lognormal per hour,
with two pieces of conditioning that beat a plain average shape on
leave-one-out tests over 18 past launches (mean pinball loss 0.28 vs 0.39
for the persistence baseline):

- **Surge ratio.** Hour 1's print divided by the median hourly viewers in
  the 24 hours before release day. A launch whose hour 1 already surged far
  above its pre-launch baseline has most of its demand in already, so its
  day stays flatter; one whose hour 1 is barely above baseline is still
  ramping and peaks much higher later. The two groups are modelled
  separately and blended softly by log surge ratio.
- **Launch size.** Bigger hour-1 prints go with flatter days (log
  peak/hour-1 falls about 0.5 per log unit of hour-1 print across past
  launches, correlation about -0.83). Each hour's log-ratio mean is
  regressed on log hour-1 print, the slope shrunk halfway toward a pooled
  slope, and the effective log print capped half a log unit above the
  largest comparable.

Launches whose hour-1 print is under 10,000 are left out of training:
those are mostly games whose real unlock came later in the day, so their
hour 1 is a pre-launch baseline. Early-access-contaminated, free-to-play
and other weak analogues are left out too, as is the GTA VI category's own
pre-release history (not a launch).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` (23 rows,
hours 2-24, five quantiles each at 0.05 / 0.25 / 0.5 / 0.75 / 0.95) to the
output directory. Deterministic: the same input always gives the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jzg6e99f&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
