#!/usr/bin/env python3
"""Practical timestamp repair for the current dataset.

This is a conservative working version, not a proof of server truth.
It produces a clean Markdown file with exactly the original eight columns.

Rules:
- Existing server Epoch is never changed.
- Internal missing Epoch blocks are filled once using both real anchors.
- Leading/trailing Epoch gaps are extrapolated only to keep the dataset full.
- Dealing is kept when it is structurally valid; otherwise it is repaired.
- Gr is kept when it is structurally valid; clearly late/early values are
  replaced with a model estimate.
- K_GAP uses consistent units: all gaps are milliseconds and the logarithmic
  term is ``1000 / K_GAP * ln(multiplier)``.

Example:
    python repair_ts_current.py raw1.md \
      --md-output repaired.md --csv-output repaired.csv --report report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from test_kalibrasi import (
    DEALING_SANITY_MAX_MS,
    DEALING_SANITY_MIN_MS,
    DEALING_MAD_Z,
    MIN_MODEL_SAMPLES,
    POST_GAP_MAX_SANITY_MS,
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

DEFAULT_K_GAP = 0.0830
DEFAULT_BASE_GAP_MS = 10050.0
DEFAULT_DEALING_MS = 7000.0
DEFAULT_POST_GAP_MS = 2400.0


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def multiplier_steps(df: pd.DataFrame, k_gap: float, base_gap_ms: float):
    m = pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float)
    m = np.where(np.isfinite(m) & (m > 0), m, 1.0)
    # k_gap is per second; base_gap_ms and result are milliseconds.
    return np.maximum(base_gap_ms + (1000.0 / k_gap) * np.log(m), 1000.0)


def fit_gap_model(df, epochs):
    x, y = [], []
    for i in range(len(df) - 1):
        a, b = epochs.iloc[i], epochs.iloc[i + 1]
        m = num(df["Result Gr"].iloc[i])
        if pd.isna(a) or pd.isna(b) or not np.isfinite(m) or m <= 0:
            continue
        gap = float(b - a)
        if gap > 0:
            x.append(math.log(m))
            y.append(gap)
    if len(x) < MIN_MODEL_SAMPLES:
        return DEFAULT_K_GAP, DEFAULT_BASE_GAP_MS, 500.0, len(x), 0
    fit = robust_line_fit(np.asarray(x), np.asarray(y))
    if fit is None or not np.isfinite(fit[1]) or fit[1] <= 0:
        return DEFAULT_K_GAP, DEFAULT_BASE_GAP_MS, 500.0, len(x), 0
    intercept, slope, _, _, sigma, total, used = fit
    return 1000.0 / slope, max(float(intercept), 1000.0), max(float(sigma), 1.0), total, used


def fill_epoch_blocks(df, original, k_gap, base_gap_ms):
    out = original.copy().astype(float)
    steps = multiplier_steps(df, k_gap, base_gap_ms)
    missing = original.isna().to_numpy()
    details = []
    n = len(out)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < n and missing[i]:
            i += 1
        end = i - 1
        left = start - 1 if start > 0 and not pd.isna(original.iloc[start - 1]) else None
        right = i if i < n and not pd.isna(original.iloc[i]) else None
        count = end - start + 1

        if left is not None and right is not None:
            # Calculate the whole block from original anchors, then distribute
            # the endpoint residual. Never chain synthetic values as anchors.
            p, q = left, right
            raw_steps = steps[p:q]
            cumulative = np.concatenate(([0.0], np.cumsum(raw_steps)))
            correction = float(original.iloc[q] - original.iloc[p] - cumulative[-1])
            for j, row in enumerate(range(start, end + 1), 1):
                out.iloc[row] = float(original.iloc[p] + cumulative[j] + correction * j / (q - p))
            details.append({"start": start, "end": end, "type": "internal", "endpoint_correction_ms": correction})
        else:
            # One-sided estimates are written for a complete test file, but
            # are explicitly reported as extrapolation rather than server data.
            if left is not None:
                for row in range(start, end + 1):
                    out.iloc[row] = float(out.iloc[row - 1] + steps[row - 1])
            elif right is not None:
                for row in range(end, start - 1, -1):
                    out.iloc[row] = float(out.iloc[row + 1] - steps[row])
            details.append({"start": start, "end": end, "type": "one_sided"})
    return out, details


def fit_flight_model(df, epochs, clock_offset, deal_low, deal_high):
    x, y = [], []
    for i in range(len(df)):
        e = epochs.iloc[i]
        m = num(df["Result Gr"].iloc[i])
        ds, gs = df["Ts Start Dealing"].iloc[i], df["Ts Gr"].iloc[i]
        if pd.isna(e) or not np.isfinite(m) or m <= 0 or str(ds).strip() in {"", "-"} or str(gs).strip() in {"", "-"}:
            continue
        d = naive_clock_ms_near_epoch(ds, e)
        g = naive_clock_ms_near_epoch(gs, e)
        if pd.isna(d) or pd.isna(g):
            continue
        d -= clock_offset
        g -= clock_offset
        if not (deal_low <= d - e <= deal_high):
            continue
        flight = g - d
        if 0 < flight < 20 * 60 * 1000:
            x.append(math.log(m))
            y.append(flight)
    if len(x) < MIN_MODEL_SAMPLES:
        return 250.0, 3200.0, 5000.0, len(x), 0
    fit = robust_line_fit(np.asarray(x), np.asarray(y))
    if fit is None or not np.isfinite(fit[1]) or fit[1] <= 0:
        return 250.0, 3200.0, 5000.0, len(x), 0
    intercept, slope, _, _, sigma, total, used = fit
    return max(float(intercept), 1.0), float(slope), max(float(sigma), 50.0), total, used


def repair(input_df):
    epochs = parse_epoch_series(input_df["Ts Epoch Starttime"])
    clock_offset, _, _, clock_diag = infer_clock_offset(input_df, epochs)
    dealing = []
    for e, s in zip(epochs, input_df["Ts Start Dealing"]):
        if pd.isna(e) or str(s).strip() in {"", "-"}:
            continue
        t = naive_clock_ms_near_epoch(s, e)
        if not pd.isna(t):
            dealing.append(t - clock_offset - e)
    if len(dealing) >= MIN_MODEL_SAMPLES:
        dmed, _, dsigma, _, _, _ = robust_location(dealing, z=DEALING_MAD_Z)
        deal_low = max(DEALING_SANITY_MIN_MS, dmed - DEALING_MAD_Z * max(dsigma, 500.0))
        deal_high = min(DEALING_SANITY_MAX_MS, dmed + DEALING_MAD_Z * max(dsigma, 500.0))
    else:
        dmed, dsigma, deal_low, deal_high = DEFAULT_DEALING_MS, 500.0, 6000.0, 8500.0

    k_gap, base_gap, gap_sigma, gap_total, gap_used = fit_gap_model(input_df, epochs)
    epoch_work, gap_details = fill_epoch_blocks(input_df, epochs, k_gap, base_gap)
    flight_a, flight_b, flight_sigma, flight_total, flight_used = fit_flight_model(
        input_df, epochs, clock_offset, deal_low, deal_high
    )

    n = len(input_df)
    deal_abs = np.full(n, np.nan)
    gr_abs = np.full(n, np.nan)
    raw_gr_abs = np.full(n, np.nan)

    for i in range(n):
        e = float(epoch_work.iloc[i])
        raw_d = input_df["Ts Start Dealing"].iloc[i]
        if str(raw_d).strip() not in {"", "-"}:
            t = naive_clock_ms_near_epoch(raw_d, e)
            if not pd.isna(t) and deal_low <= t - clock_offset - e <= deal_high:
                deal_abs[i] = t - clock_offset
        if not np.isfinite(deal_abs[i]):
            deal_abs[i] = e + dmed

        raw_g = input_df["Ts Gr"].iloc[i]
        if str(raw_g).strip() not in {"", "-"}:
            t = naive_clock_ms_near_epoch(raw_g, e)
            if not pd.isna(t):
                raw_gr_abs[i] = t - clock_offset

        m = num(input_df["Result Gr"].iloc[i])
        if np.isfinite(m) and m > 0:
            predicted = deal_abs[i] + flight_a + flight_b * math.log(m)
        else:
            predicted = deal_abs[i] + max(flight_a, 250.0)

        # Preserve measured Gr unless it is structurally impossible or clearly
        # a receive-time outlier. A valid long flight is not rejected merely
        # because it differs from the global multiplier model.
        keep = np.isfinite(raw_gr_abs[i]) and raw_gr_abs[i] > deal_abs[i]
        if keep and i + 1 < n:
            next_e = float(epoch_work.iloc[i + 1])
            post = next_e - raw_gr_abs[i]
            keep = 0 < post <= POST_GAP_MAX_SANITY_MS
        gr_abs[i] = raw_gr_abs[i] if keep else predicted
        if gr_abs[i] <= deal_abs[i]:
            gr_abs[i] = predicted

    out = input_df.copy()
    out["Ts Epoch Starttime"] = [str(int(round(x))) for x in epoch_work]
    out["Ts Starttime (Bukan Epoch)"] = [format_clock_ms(x, clock_offset, False) for x in epoch_work]
    out["Ts Start Dealing"] = [format_clock_ms(x, clock_offset, True) for x in deal_abs]
    out["Ts Gr"] = [format_clock_ms(x, clock_offset, True) for x in gr_abs]
    out = out[REQUIRED_COLUMNS]

    report = {
        "rows": len(out),
        "server_epoch_rows": int(epochs.notna().sum()),
        "filled_epoch_rows": int((epochs.isna() & epoch_work.notna()).sum()),
        "raw_gr_kept": int(np.isfinite(raw_gr_abs).sum() - sum(
            np.isfinite(raw_gr_abs[i]) and gr_abs[i] != raw_gr_abs[i] for i in range(n)
        )),
        "model": {
            "clock_offset_ms": float(clock_offset),
            "dealing_median_ms": float(dmed),
            "dealing_sigma_ms": float(dsigma),
            "dealing_band_ms": [float(deal_low), float(deal_high)],
            "k_gap_per_second": float(k_gap),
            "base_gap_ms": float(base_gap),
            "gap_sigma_ms": float(gap_sigma),
            "flight_intercept_ms": float(flight_a),
            "flight_slope_ms_per_ln_multiplier": float(flight_b),
            "flight_sigma_ms": float(flight_sigma),
            "flight_fit_total": int(flight_total),
            "flight_fit_used": int(flight_used),
            "gap_fit_total": int(gap_total),
            "gap_fit_used": int(gap_used),
        },
        "clock_diagnostics": clock_diag,
        "epoch_blocks": gap_details,
        "notes": [
            "Existing server Epoch values are preserved.",
            "Internal gaps use a two-anchor bridge calculated as one block.",
            "One-sided Epoch gaps are extrapolations, not server ground truth.",
            "Valid structural Gr observations are preserved; missing/impossible/late ones use the model.",
            "Markdown output contains only the original eight columns.",
        ],
    }
    return out, report


def main():
    p = argparse.ArgumentParser(description="Practical timestamp repair with clean MD output")
    p.add_argument("input", type=Path)
    p.add_argument("--md-output", type=Path, required=True)
    p.add_argument("--csv-output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    df = read_input(args.input)
    repaired, report = repair(df)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_like_source(repaired, args.input, args.md_output)
    repaired.to_csv(args.csv_output, index=False)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MD     : {args.md_output}")
    print(f"CSV    : {args.csv_output}")
    print(f"Report : {args.report}")


if __name__ == "__main__":
    main()
