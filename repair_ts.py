#!/usr/bin/env python3
"""Repair timestamp crash-game dari Epoch server dan multiplier.

Output Markdown sengaja hanya mempertahankan 8 kolom sumber, sehingga dapat
langsung dipakai untuk menguji script lain. Informasi audit hanya ditulis ke
JSON report; tidak ada flag/keterangan di file Markdown.

Prinsip:
- Epoch yang berasal dari server adalah anchor utama.
- Timestamp penerimaan WebSocket tidak dipercaya mentah-mentah.
- Flight dikalibrasi dengan regresi robust: flight = intercept + slope*ln(mult).
  Intercept penting untuk multiplier 1.00; model lama memaksa flight=0.
- Timestamp Gr final dibuat dari event model, bukan dari waktu tag diterima.
- Gap Epoch internal dan edge gap diisi dengan model interval round.
- Endpoint server yang tersedia tetap dipertahankan persis.

Ini adalah estimasi event-time. Tanpa timestamp event dari server untuk setiap
round, tidak ada metode yang dapat membuktikan setiap baris benar secara absolut.
Script ini memilih hasil yang paling konsisten dengan anchor server dan semua
constraint waktu yang tersedia.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from test_kalibrasi import (
    DEALING_MAD_Z,
    DEALING_SANITY_MAX_MS,
    DEALING_SANITY_MIN_MS,
    MIN_MODEL_SAMPLES,
    POST_GAP_FALLBACK_MS,
    TARGET_TOL_MS,
    backfill_internal_epochs,
    build_report,
    calibrate_dealing,
    format_clock_ms,
    infer_clock_offset,
    naive_clock_ms_near_epoch,
    parse_epoch_series,
    read_input,
    robust_line_fit,
    robust_location,
    write_markdown_like_source,
)


def fit_event_flight(df, epoch, clock_offset_ms):
    """Fit flight duration robustly, including a non-zero 1.00 intercept."""
    x, y = [], []
    for i in range(len(df)):
        e = epoch.iloc[i]
        mult = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
        deal_s = df["Ts Start Dealing"].iloc[i]
        gr_s = df["Ts Gr"].iloc[i]
        if pd.isna(e) or pd.isna(mult) or mult <= 0:
            continue
        if str(deal_s).strip() in {"", "-"} or str(gr_s).strip() in {"", "-"}:
            continue
        d = naive_clock_ms_near_epoch(deal_s, e)
        g = naive_clock_ms_near_epoch(gr_s, e)
        if pd.isna(d) or pd.isna(g):
            continue
        d -= clock_offset_ms
        g -= clock_offset_ms
        deal_delay = d - e
        flight = g - d
        if not (DEALING_SANITY_MIN_MS <= deal_delay <= DEALING_SANITY_MAX_MS):
            continue
        if not (0 < flight < 20 * 60 * 1000):
            continue
        x.append(math.log(max(float(mult), 1.0)))
        y.append(float(flight))

    if len(x) < MIN_MODEL_SAMPLES:
        return 250.0, 1000.0, len(x), 0

    fit = robust_line_fit(np.asarray(x), np.asarray(y))
    if fit is None:
        return 250.0, 1000.0, len(x), 0

    intercept, slope, _, _, sigma, total, used = fit
    # A negative slope is physically impossible here. A tiny/non-finite slope
    # is also rejected because it would make multiplier ordering meaningless.
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope < 0:
        return 250.0, 1000.0, total, used

    return float(max(intercept, 0.0)), float(max(slope, 0.0)), total, used


def fill_all_epoch_gaps(df, epoch, k_gap, base_gap):
    """Fill internal, leading, and trailing Epoch gaps.

    Existing server Epoch values are never changed. Edge gaps are extrapolated
    only because the requested output must be complete; they are based on game
    multipliers and the nearest server anchor.
    """
    out = epoch.copy().astype(float)
    n = len(out)
    multipliers = pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float)
    steps = base_gap + np.log(np.where(np.isfinite(multipliers) & (multipliers > 0), multipliers, 1.0)) / k_gap
    steps = np.maximum(steps, 1000.0)

    anchors = np.flatnonzero(np.isfinite(out.to_numpy(float)))
    if len(anchors) == 0:
        raise ValueError("Tidak ada Epoch server sebagai anchor.")

    # Forward-fill every row after the first anchor, using transition p -> p+1.
    first = int(anchors[0])
    for i in range(first + 1, n):
        if not np.isfinite(out.iloc[i]):
            out.iloc[i] = out.iloc[i - 1] + steps[i - 1]

    # Backward-fill every row before the first anchor.
    for i in range(first - 1, -1, -1):
        if not np.isfinite(out.iloc[i]):
            out.iloc[i] = out.iloc[i + 1] - steps[i]

    # For internal gaps, use the existing bridge logic so both server anchors
    # are preserved exactly. This is the strongest reconstruction available.
    bridged, _, uncertainty, details = backfill_internal_epochs(df, epoch, type("Cal", (), {"base_gap_ms": base_gap, "k_gap": k_gap, "gap_sigma_ms": 500.0})())
    for i in range(n):
        if not pd.isna(bridged.iloc[i]):
            out.iloc[i] = bridged.iloc[i]

    return out, uncertainty, details


def repair_event_times(df, epoch_original, epoch_work, clock_offset, dealing_median, dealing_low, dealing_high, flight_intercept, flight_slope, flight_sigma):
    n = len(df)
    deal = np.full(n, np.nan)
    gr = np.full(n, np.nan)
    multipliers = pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float)

    for i in range(n):
        e = float(epoch_work.iloc[i])
        d = df["Ts Start Dealing"].iloc[i]
        if str(d).strip() not in {"", "-"}:
            parsed = naive_clock_ms_near_epoch(d, e) - clock_offset
            if np.isfinite(parsed) and dealing_low <= parsed - e <= dealing_high:
                deal[i] = parsed
        if not np.isfinite(deal[i]):
            deal[i] = e + dealing_median

        # Always use event-time model for Gr. This deliberately removes receive
        # delay from reconnect/queueing instead of preserving a late tag.
        m = multipliers[i]
        if np.isfinite(m) and m > 0:
            flight = flight_intercept + flight_slope * math.log(max(float(m), 1.0))
            gr[i] = deal[i] + max(flight, 0.0)

    # Enforce physical ordering without touching server Epoch values.
    for i in range(n):
        if gr[i] <= deal[i]:
            gr[i] = deal[i] + max(flight_intercept, 1.0)

    return deal, gr


def main():
    p = argparse.ArgumentParser(description="Repair timestamp dan isi seluruh kolom waktu.")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True, help="Output CSV lengkap.")
    p.add_argument("--write-md", type=Path, required=True, help="Output Markdown 8 kolom tanpa flag.")
    p.add_argument("--report", type=Path, required=True, help="JSON audit report.")
    args = p.parse_args()

    df = read_input(args.input)
    epoch_original = parse_epoch_series(df["Ts Epoch Starttime"])
    clock_offset, rounding_bias, start_mad, start_diag = infer_clock_offset(df, epoch_original)
    deal_med, deal_sigma, deal_low, deal_high, deal_samples = calibrate_dealing(df, epoch_original, clock_offset)

    # Calibrate Epoch interval from the original server anchors. Reuse the
    # existing robust gap model, then fill every missing region.
    from test_kalibrasi import calibrate_gap_model
    k_gap, base_gap, gap_sigma, gap_samples, gap_used = calibrate_gap_model(df, epoch_original)
    epoch_work, epoch_uncertainty, epoch_details = fill_all_epoch_gaps(df, epoch_original, k_gap, base_gap)

    intercept, slope, flight_samples, flight_used = fit_event_flight(df, epoch_original, clock_offset)
    flight_sigma = 1000.0
    if flight_samples >= MIN_MODEL_SAMPLES:
        # Estimate robust residual scale for the report.
        vals = []
        for i in range(len(df)):
            m = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
            e = epoch_original.iloc[i]
            if pd.isna(m) or pd.isna(e) or m <= 0:
                continue
            d = df["Ts Start Dealing"].iloc[i]
            g = df["Ts Gr"].iloc[i]
            if str(d).strip() in {"", "-"} or str(g).strip() in {"", "-"}:
                continue
            da = naive_clock_ms_near_epoch(d, e) - clock_offset
            ga = naive_clock_ms_near_epoch(g, e) - clock_offset
            if np.isfinite(da) and np.isfinite(ga):
                pred = intercept + slope * math.log(max(float(m), 1.0))
                vals.append((ga - da) - pred)
        if vals:
            _, _, flight_sigma, _, _, _ = robust_location(vals)
            flight_sigma = max(float(flight_sigma), 50.0)

    deal_abs, gr_abs = repair_event_times(
        df, epoch_original, epoch_work, clock_offset, deal_med, deal_low, deal_high, intercept, slope, flight_sigma
    )

    out = df.copy()
    out["Ts Epoch Starttime"] = [str(int(round(v))) if np.isfinite(v) else "-" for v in epoch_work]
    out["Ts Starttime (Bukan Epoch)"] = [format_clock_ms(v, clock_offset, False) for v in epoch_work]
    out["Ts Start Dealing"] = [format_clock_ms(v, clock_offset, True) for v in deal_abs]
    out["Ts Gr"] = [format_clock_ms(v, clock_offset, True) for v in gr_abs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.write_md.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    write_markdown_like_source(out, args.input, args.write_md)

    report = {
        "input_file": str(args.input),
        "rows": len(df),
        "model": {
            "clock_offset_ms": clock_offset,
            "dealing_median_ms": deal_med,
            "dealing_sigma_ms": deal_sigma,
            "k_gap": k_gap,
            "base_gap_ms": base_gap,
            "gap_sigma_ms": gap_sigma,
            "flight_intercept_ms": intercept,
            "flight_slope_ms_per_ln_multiplier": slope,
            "flight_sigma_ms": flight_sigma,
            "flight_samples": flight_samples,
            "flight_samples_used": flight_used,
        },
        "server_epoch_rows": int(epoch_original.notna().sum()),
        "filled_epoch_rows": int((epoch_original.isna() & epoch_work.notna()).sum()),
        "epoch_backfill_details": epoch_details,
        "design_notes": [
            "Markdown output selalu tepat 8 kolom sumber tanpa flag atau keterangan tambahan.",
            "Ts Gr dihitung dari event model, bukan dipertahankan dari waktu tag WebSocket diterima.",
            "Model flight memakai intercept non-zero agar multiplier 1.00 tidak menghasilkan flight nol.",
            "Epoch server yang tersedia dipertahankan sebagai anchor; gap lainnya diekstrapolasi/interpolasi.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CSV    : {args.output}")
    print(f"MD     : {args.write_md}")
    print(f"Report : {args.report}")


if __name__ == "__main__":
    main()
