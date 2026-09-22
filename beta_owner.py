#!/usr/bin/env python3
"""beta_owner.py

Beta repair pipeline with explicit transition prior for round-to-round gap.

This version treats the round transition as a known prior window:
    9900 ms <= transition_gap_ms <= 10100 ms

This is not a claim of absolute truth, but a practical prior to stabilize the
model around the server-side round boundary while leaving client-side receive
noise handled separately.

It keeps output clean:
- CSV with the original columns plus repair columns only if needed in report
- Markdown output with exactly the original 8 columns, no flags
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from test_kalibrasi import (
    REQUIRED_COLUMNS,
    format_clock_ms,
    infer_clock_offset,
    naive_clock_ms_near_epoch,
    parse_epoch_series,
    read_input,
    robust_line_fit,
    robust_location,
    write_markdown_like_source,
)

TRANSITION_MIN_MS = 9900.0
TRANSITION_MAX_MS = 10100.0
DEFAULT_K_GAP = 0.0830
DEFAULT_BASE_GAP_MS = 10050.0


def num(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def fit_gap_model(df, epoch):
    xs, ys = [], []
    for i in range(len(df) - 1):
        e0, e1 = epoch.iloc[i], epoch.iloc[i + 1]
        m = num(df["Result Gr"].iloc[i])
        if pd.isna(e0) or pd.isna(e1) or not np.isfinite(m) or m <= 0:
            continue
        gap = float(e1 - e0)
        if gap <= 0:
            continue
        xs.append(math.log(m))
        ys.append(gap)

    if len(xs) < 30:
        return DEFAULT_K_GAP, DEFAULT_BASE_GAP_MS, 500.0, len(xs), 0

    fit = robust_line_fit(np.asarray(xs), np.asarray(ys))
    if fit is None:
        return DEFAULT_K_GAP, DEFAULT_BASE_GAP_MS, 500.0, len(xs), 0

    intercept, slope, _, _, sigma, total, used = fit
    if not np.isfinite(slope) or slope <= 0:
        return DEFAULT_K_GAP, DEFAULT_BASE_GAP_MS, 500.0, total, used

    k_gap = 1000.0 / slope
    base_gap = float(intercept)
    if not np.isfinite(k_gap) or k_gap <= 0:
        k_gap = DEFAULT_K_GAP
    if not np.isfinite(base_gap):
        base_gap = DEFAULT_BASE_GAP_MS
    # Beta prior: transition gap should be around 10s, not arbitrary.
    base_gap = float(np.clip(base_gap, TRANSITION_MIN_MS, TRANSITION_MAX_MS))
    return float(k_gap), float(base_gap), float(max(sigma, 1.0)), total, used


def missing_runs(epoch):
    missing = epoch.isna().to_numpy()
    out = []
    i = 0
    n = len(missing)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < n and missing[i]:
            i += 1
        out.append((start, i - 1))
    return out


def fill_epoch_blocks(df, epoch_original, k_gap, base_gap):
    epoch_work = epoch_original.copy().astype(float)
    steps = np.maximum(
        base_gap + (1000.0 / k_gap) * np.log(np.where(
            pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float) > 0,
            pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float),
            1.0,
        )),
        1000.0,
    )

    for start, end in missing_runs(epoch_original):
        left = start - 1 if start > 0 and not pd.isna(epoch_original.iloc[start - 1]) else None
        right = end + 1 if end + 1 < len(df) and not pd.isna(epoch_original.iloc[end + 1]) else None
        if left is not None and right is not None:
            p, q = left, right
            raw_steps = steps[p:q]
            cumulative = np.concatenate(([0.0], np.cumsum(raw_steps)))
            endpoint_error = float(epoch_original.iloc[q] - epoch_original.iloc[p] - cumulative[-1])
            total_steps = q - p
            if total_steps <= 0:
                continue
            for j, row in enumerate(range(start, end + 1), start=1):
                epoch_work.iloc[row] = float(epoch_original.iloc[p]) + cumulative[j] + endpoint_error * j / total_steps
        else:
            if left is not None:
                for row in range(start, end + 1):
                    epoch_work.iloc[row] = float(epoch_work.iloc[row - 1] + steps[row - 1])
            elif right is not None:
                for row in range(end, start - 1, -1):
                    epoch_work.iloc[row] = float(epoch_work.iloc[row + 1] - steps[row])
    return epoch_work


def fit_flight_model(df, epoch, clock_offset, dealing_low, dealing_high):
    xs, ys = [], []
    for i in range(len(df)):
        e = epoch.iloc[i]
        mult = num(df["Result Gr"].iloc[i])
        deal_s = df["Ts Start Dealing"].iloc[i]
        gr_s = df["Ts Gr"].iloc[i]
        if pd.isna(e) or not np.isfinite(mult) or mult <= 0:
            continue
        if str(deal_s).strip() in {"", "-"} or str(gr_s).strip() in {"", "-"}:
            continue
        d = naive_clock_ms_near_epoch(deal_s, e)
        g = naive_clock_ms_near_epoch(gr_s, e)
        if pd.isna(d) or pd.isna(g):
            continue
        d -= clock_offset
        g -= clock_offset
        if not (dealing_low <= d - e <= dealing_high):
            continue
        flight = g - d
        if 0 < flight < 20 * 60 * 1000:
            xs.append(math.log(mult))
            ys.append(flight)
    if len(xs) < 30:
        return 250.0, 3200.0, 5000.0, len(xs), 0
    fit = robust_line_fit(np.asarray(xs), np.asarray(ys))
    if fit is None:
        return 250.0, 3200.0, 5000.0, len(xs), 0
    intercept, slope, _, _, sigma, total, used = fit
    if not np.isfinite(slope) or slope <= 0:
        return 250.0, 3200.0, 5000.0, total, used
    return float(max(intercept, 1.0)), float(slope), float(max(sigma, 50.0)), total, used


def calibrate_dealing(df, epoch, clock_offset):
    vals = []
    for i, e in enumerate(epoch):
        s = df["Ts Start Dealing"].iloc[i]
        if pd.isna(e) or str(s).strip() in {"", "-"}:
            continue
        t = naive_clock_ms_near_epoch(s, e)
        if pd.isna(t):
            continue
        vals.append(t - clock_offset - e)
    if len(vals) < 30:
        return 7000.0, 500.0, 6000.0, 8500.0, len(vals)
    med, _, sigma, lo, hi, _ = robust_location(np.asarray(vals), z=3.5)
    lo = max(6000.0, lo)
    hi = min(8500.0, hi)
    return float(med), float(max(sigma, 500.0)), float(lo), float(hi), len(vals)


def repair(df):
    epochs = parse_epoch_series(df["Ts Epoch Starttime"])
    clock_offset, _, _, _ = infer_clock_offset(df, epochs)
    deal_med, deal_sigma, deal_low, deal_high, _ = calibrate_dealing(df, epochs, clock_offset)
    k_gap, base_gap, gap_sigma, gap_total, gap_used = fit_gap_model(df, epochs)
    epoch_work = fill_epoch_blocks(df, epochs, k_gap, base_gap)
    flight_a, flight_b, flight_sigma, flight_total, flight_used = fit_flight_model(
        df, epoch_work, clock_offset, deal_low, deal_high
    )

    n = len(df)
    deal_abs = np.full(n, np.nan)
    gr_abs = np.full(n, np.nan)

    for i in range(n):
        e = float(epoch_work.iloc[i])
        if pd.isna(e):
            continue

        raw_d = df["Ts Start Dealing"].iloc[i]
        if str(raw_d).strip() not in {"", "-"}:
            d_val = naive_clock_ms_near_epoch(raw_d, e)
            if not pd.isna(d_val):
                d_abs = d_val - clock_offset
                if deal_low <= d_abs - e <= deal_high:
                    deal_abs[i] = d_abs

        if not np.isfinite(deal_abs[i]):
            deal_abs[i] = e + deal_med

        mult = num(df["Result Gr"].iloc[i])
        raw_g = df["Ts Gr"].iloc[i]
        raw_g_abs = np.nan
        if str(raw_g).strip() not in {"", "-"}:
            t = naive_clock_ms_near_epoch(raw_g, e)
            if not pd.isna(t):
                raw_g_abs = t - clock_offset

        if np.isfinite(raw_g_abs) and raw_g_abs > deal_abs[i]:
            if i + 1 < n:
                next_e = float(epoch_work.iloc[i + 1])
                post_gap = next_e - raw_g_abs
                if 0 < post_gap <= 6000.0:
                    gr_abs[i] = raw_g_abs
                    continue

        if np.isfinite(mult) and mult > 0:
            pred = deal_abs[i] + flight_a + flight_b * math.log(mult)
        else:
            pred = deal_abs[i] + 2500.0

        if i + 1 < n:
            next_e = float(epoch_work.iloc[i + 1])
            if np.isfinite(next_e):
                backward_est = next_e - 2400.0
                if backward_est > deal_abs[i]:
                    pred = min(pred, backward_est)
                    if np.isfinite(raw_g_abs):
                        if abs(raw_g_abs - backward_est) <= max(5000.0, 3.0 * flight_sigma):
                            gr_abs[i] = raw_g_abs
                            continue
                    gr_abs[i] = backward_est
                    continue

        gr_abs[i] = pred

    out = df.copy()
    out["Ts Epoch Starttime"] = [str(int(round(float(x)))) if np.isfinite(x) else "-" for x in epoch_work]
    out["Ts Starttime (Bukan Epoch)"] = [format_clock_ms(x, clock_offset, False) for x in epoch_work]
    out["Ts Start Dealing"] = [format_clock_ms(x, clock_offset, True) for x in deal_abs]
    out["Ts Gr"] = [format_clock_ms(x, clock_offset, True) for x in gr_abs]
    out = out[REQUIRED_COLUMNS]

    report = {
        "beta_owner": True,
        "input_rows": int(len(df)),
        "clock_offset_ms": float(clock_offset),
        "dealing_median_ms": float(deal_med),
        "dealing_sigma_ms": float(deal_sigma),
        "dealing_valid_band_ms": [float(deal_low), float(deal_high)],
        "k_gap": float(k_gap),
        "base_gap_ms": float(base_gap),
        "gap_sigma_ms": float(gap_sigma),
        "flight_intercept_ms": float(flight_a),
        "flight_slope_ms_per_ln_mult": float(flight_b),
        "flight_sigma_ms": float(flight_sigma),
        "transition_prior_ms": [TRANSITION_MIN_MS, TRANSITION_MAX_MS],
        "notes": [
            "Beta prior transition gap fixed to around 10s with jitter tolerance.",
            "Epoch server values remain the anchor.",
            "One-sided gap is extrapolated to keep the output complete.",
            "Backward estimate from next Epoch is preferred over forward flight for impossible Gr.",
        ],
    }
    return out, report


def main():
    parser = argparse.ArgumentParser(description="Beta owner repair pipeline with explicit 10s transition prior.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    df = read_input(args.input)
    repaired, report = repair(df)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    repaired.to_csv(args.csv_output, index=False)
    write_markdown_like_source(repaired, args.input, args.md_output)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV    : {args.csv_output}")
    print(f"MD     : {args.md_output}")
    print(f"Report : {args.report}")
    print(json.dumps({"k_gap": report["k_gap"], "base_gap_ms": report["base_gap_ms"], "transition_prior_ms": report["transition_prior_ms"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
