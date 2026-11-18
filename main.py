"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun program/v1).

Model: empirical launch-day shapes from past big game launches.
The case file carries the comparables (past launches' hourly Twitch counts),
hour 1's print, and the pre-launch hourly history. For each comparable launch
we form the ratio of each later hour's viewers to its hour-1 print, in log
space, and fit a small lognormal per hour. Two pieces of conditioning make
the forecast sharper than a plain average shape:

1. Surge ratio b = hour-1 print / median hourly viewers in the 24 h before
   the release day. A launch whose hour 1 already surged far above its
   pre-launch baseline (b high) has most of its demand in already, so its
   day stays flatter; one whose hour 1 is barely above baseline (b low) is
   still ramping and peaks much higher later. The two groups are modelled
   separately and blended softly by log b, so there is no cliff at the cut.
2. Launch size: bigger hour-1 prints go with flatter days (log peak/hour-1
   falls about 0.5 per log unit of hour-1 print across past launches,
   corr about -0.83). Each hour's log-ratio mean is regressed on log
   hour-1 print, the slope shrunk halfway toward a pooled slope over all
   hours, and the effective log print is capped half a log unit above the
   largest comparable, so an unprecedented launch does not extrapolate
   without bound.

Launches whose hour-1 print is tiny (under 10,000) are left out of the
training set: those are mostly games whose real unlock came later in the
day, so their hour 1 is a pre-launch baseline and their ratios to it are
meaningless for a rolling-midnight launch. Early-access-contaminated,
free-to-play and other weak analogues are left out too, as is the GTA VI
category's own pre-release history (not a launch).

Everything is standard library only, deterministic (no randomness, fixed
iteration order), and runs in well under a second.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. This entry is a single file, but
# the line keeps the layout honest if one is ever added.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

MIN_HOUR1 = 10000        # a launch hour worth training on
B_CUT = 5.0              # surge-ratio split between the two shape groups
B_BLEND_WIDTH = 1.5      # softness of the blend, in log units
SIGMA_FLOOR = 0.10       # never claim to know an hour better than this
CAP_EXTRA = 0.5          # max log extrapolation past the largest comparable
SLOPE_MIN = -1.0         # size slopes are clipped to [SLOPE_MIN, 0]


def parse_time(s):
    """ISO 8601 UTC timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_of(comp):
    """Tier tag from the comparable's unlock note, or '' when unmarked."""
    unlock = comp.get("unlock") or ""
    for tag in ("early-access-contaminated", "weak", "target-history",
                "comparable"):
        if "Tier: " + tag in unlock:
            return tag
    return ""


def day_series(comp):
    """{hour 1..24 of the release day: viewers} from the hourly points."""
    day_start = parse_time(comp["day_start"])
    out = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_time(pt["from"])
            viewers = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and viewers is not None:
            out[h] = viewers
    return out


def pre_baseline(comp):
    """Median hourly viewers in the 24 h before the release day."""
    day_start = parse_time(comp["day_start"])
    vals = []
    for pt in comp.get("hourly") or []:
        try:
            t = parse_time(pt["from"])
            viewers = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        dh = (t - day_start).total_seconds() / 3600.0
        if -25 <= dh < 0 and viewers is not None:
            vals.append(viewers)
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def case_baseline(case):
    """Median of history_hourly in the 24 h before the case's day_start."""
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    vals = []
    for pt in case.get("history_hourly") or []:
        try:
            t = parse_time(pt["from"])
            viewers = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        dh = (t - day_start).total_seconds() / 3600.0
        if -25 <= dh < 0 and viewers is not None:
            vals.append(viewers)
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def mean(values):
    return sum(values) / len(values)


def pstdev(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def fit_group(p1, group):
    """Per-hour lognormal of the ratios, with the size regression.

    group: list of (log_hour1, {hour: viewers}) training launches.
    Returns {hour: (mu, sigma)} for hours 2..24 in log-ratio space.
    """
    lp = math.log(max(p1, 1))
    # Pooled size slope over every hour and launch in the group.
    xs, ys = [], []
    for logp1, series in group:
        for h in range(2, 25):
            if 1 in series and h in series and series[1] > 0:
                xs.append(logp1)
                ys.append(math.log(max(series[h] / series[1], 1e-6)))
    if len(xs) < 4:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((a - mx) ** 2 for a in xs) / len(xs)
    pooled = 0.0
    if vx > 0:
        pooled = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs) / vx
    pooled = max(SLOPE_MIN, min(0.0, pooled))
    lp_eff = min(lp, max(xs) + CAP_EXTRA)

    out = {}
    for h in range(2, 25):
        pts = []
        for logp1, series in group:
            if 1 in series and h in series and series[1] > 0:
                pts.append((logp1, math.log(max(series[h] / series[1], 1e-6))))
        if len(pts) < 3:
            # Too little data for this hour: fall back to the pooled shape.
            mu = my + pooled * (lp_eff - mx)
            sd = SIGMA_FLOOR
            out[h] = (mu, sd)
            continue
        xs2 = [a for a, _ in pts]
        ys2 = [b for _, b in pts]
        mx2, my2 = mean(xs2), mean(ys2)
        vx2 = sum((a - mx2) ** 2 for a in xs2) / len(xs2)
        slope_h = 0.0
        if vx2 > 0:
            slope_h = (sum((a - mx2) * (b - my2) for a, b in pts)
                       / len(pts) / vx2)
        slope_h = max(SLOPE_MIN, min(0.0, slope_h))
        slope = 0.5 * slope_h + 0.5 * pooled
        mu = my2 + slope * (lp_eff - mx2)
        resid = [b - (my2 + slope * (a - mx2)) for a, b in pts]
        sd = max(pstdev(resid) if len(resid) > 1 else 0.3, SIGMA_FLOOR)
        out[h] = (mu, sd)
    return out


def group_quantiles(p1, fit):
    """{hour: [q05, q25, q50, q75, q95]} from a fitted group."""
    out = {}
    for h, (mu, sd) in fit.items():
        out[h] = [p1 * math.exp(mu + z * sd) for z in Z]
    return out


def forecast_case(case):
    """Forecast hours 2..24 for one case dict."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the latest history point.
        hist = case.get("history_hourly") or case.get("history") or []
        p1 = 1
        for pt in hist:
            if pt.get("viewers"):
                p1 = max(1, pt["viewers"])
    hours = [row["hour"] for row in case.get("forecast") or []
             if row.get("hour") not in (None, 1)]
    if not hours:
        hours = list(range(2, 25))

    # Training launches from the case's comparables.
    train = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) in ("early-access-contaminated", "weak",
                             "target-history"):
            continue
        if "Grand Theft Auto VI" in (comp.get("name") or ""):
            continue
        series = day_series(comp)
        if 1 not in series or series[1] < MIN_HOUR1:
            continue
        train.append((math.log(series[1]), series, pre_baseline(comp)))

    fit_all = fit_group(p1, [(lp, s) for lp, s, _ in train]) if train else None
    if fit_all is None:
        # No usable comparables: house-style persistence fallback.
        return {"hours": [{"hour": h,
                           "q": [round(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 2)
                                 for z in Z]}
                          for h in hours]}

    base = case_baseline(case)
    b = (p1 / base) if base and base > 0 else None

    hi = [(lp, s) for lp, s, bl in train
          if bl and bl > 0 and s[1] / bl >= B_CUT]
    lo = [(lp, s) for lp, s, bl in train
          if bl and bl > 0 and s[1] / bl < B_CUT]
    if len(hi) >= 4 and len(lo) >= 4 and b is not None:
        fit_hi = fit_group(p1, hi)
        fit_lo = fit_group(p1, lo)
        if fit_hi is not None and fit_lo is not None:
            q_hi = group_quantiles(p1, fit_hi)
            q_lo = group_quantiles(p1, fit_lo)
            x = (math.log(b) - math.log(B_CUT)) / B_BLEND_WIDTH
            w = 1.0 / (1.0 + math.exp(-x)) if abs(x) < 30 else (1.0 if x > 0 else 0.0)
            rows = []
            for h in hours:
                qh = [math.exp(w * math.log(max(q_hi[h][i], 1e-9))
                               + (1 - w) * math.log(max(q_lo[h][i], 1e-9)))
                      for i in range(5)]
                rows.append({"hour": h, "q": [round(v, 2) for v in qh]})
            return {"hours": rows}

    quantiles = group_quantiles(p1, fit_all)
    return {"hours": [{"hour": h, "q": [round(v, 2) for v in quantiles[h]]}
                      for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
