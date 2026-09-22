#!/usr/bin/env python3
"""test_kalibrasi4.py

Threshold-based repair pipeline using per-segment calibration.

Rules implemented:
- Epoch -> Starttime (non-Epoch): center from raw valid data, keep if within +/-100 ms
- Starttime -> Start Dealing: center from raw valid data, keep if within +/-200 ms
- Gr -> Next Epoch: center from raw valid data, keep if within +/-200 ms
- Gr -> Next Start Dealing: fixed center 9900 ms with +/-300 ms tolerance
- Start Dealing -> Gr: dynamic, do not apply fixed threshold; only repair must-haves

This is a beta script intended for repair and testing, not a proof of absolute server truth.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def is_floatable(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def parse_clock_to_ms(value: str):
    value = (value or "").strip()
    if not value or value in {"-", "nan", "NaN"}:
        return None
    if "." in value:
        hms, ms_part = value.split(".", 1)
        ms = int((ms_part + "000")[:3])
    else:
        hms = value
        ms = 0
    parts = hms.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = map(int, parts)
    except Exception:
        return None
    return ((h * 3600 + m * 60 + s) * 1000) + ms


def parse_epoch_value(value):
    value = (value or "").strip()
    if not value or value in {"-", "nan", "NaN"}:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def epoch_to_dt(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)


def dt_from_day_and_clock(day: datetime.date, clock_ms: int):
    base = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    return base + timedelta(milliseconds=clock_ms)


def approx_median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return statistics.median(vals)


def read_markdown_table(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("-"):
                continue
            if s.startswith("No |"):
                continue
            if "|" not in s:
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 8:
                continue
            try:
                idx = int(parts[0])
            except Exception:
                continue
            # Keep the original column order from markdown files.
            row = {
                "index": idx,
                "epoch": parts[1],
                "starttime": parts[2],
                "start_dealing": parts[3],
                "gr": parts[4],
                "result_gr": parts[5],
                "id_game": parts[6],
                "status": parts[7],
            }
            rows.append(row)
    return rows


def repair_rounds(rows):
    # Tunable thresholds (millisecond-level)
    THRESH_EPOCH_TO_START = 100.0
    THRESH_START_TO_DEAL = 200.0
    THRESH_GR_TO_NEXT_EPOCH = 200.0
    GR_TO_NEXT_DEALING_CENTER = 9900.0
    GR_TO_NEXT_DEALING_TOL = 300.0

    # Prepare raw values
    repaired = []
    for r in rows:
        repaired.append({
            **r,
            "epoch_ms": parse_epoch_value(r["epoch"]),
            "start_ms": parse_clock_to_ms(r["starttime"]),
            "deal_ms": parse_clock_to_ms(r["start_dealing"]),
            "gr_ms": parse_clock_to_ms(r["gr"]),
        })

    # Compute raw residuals for calibration using valid rows only.
    epoch_to_start_deltas = []
    start_to_deal_deltas = []
    gr_to_next_epoch_deltas = []

    n = len(repaired)
    for i, r in enumerate(repaired):
        epoch_ms = r["epoch_ms"]
        if epoch_ms is None:
            continue
        dt_epoch = epoch_to_dt(epoch_ms)

        start_ms = r["start_ms"]
        if start_ms is not None:
            start_dt = dt_from_day_and_clock(dt_epoch.date(), start_ms)
            if (start_dt - dt_epoch).total_seconds() > 12 * 3600:
                start_dt -= timedelta(days=1)
            elif (start_dt - dt_epoch).total_seconds() < -12 * 3600:
                start_dt += timedelta(days=1)
            delta = (start_dt - dt_epoch).total_seconds() * 1000.0
            # Keep only values near the expected ~-0.5s due to wall-clock offset.
            if abs(delta) < 2000.0:
                epoch_to_start_deltas.append(delta)

        deal_ms = r["deal_ms"]
        if start_ms is not None and deal_ms is not None:
            start_dt = dt_from_day_and_clock(dt_epoch.date(), start_ms)
            if (start_dt - dt_epoch).total_seconds() > 12 * 3600:
                start_dt -= timedelta(days=1)
            elif (start_dt - dt_epoch).total_seconds() < -12 * 3600:
                start_dt += timedelta(days=1)
            deal_dt = dt_from_day_and_clock(dt_epoch.date(), deal_ms)
            if (deal_dt - dt_epoch).total_seconds() > 12 * 3600:
                deal_dt -= timedelta(days=1)
            elif (deal_dt - dt_epoch).total_seconds() < -12 * 3600:
                deal_dt += timedelta(days=1)
            # Prefer a realistic range for dealing delay.
            delta = (deal_dt - start_dt).total_seconds() * 1000.0
            if 5000.0 <= delta <= 12000.0:
                start_to_deal_deltas.append(delta)

    epoch_to_start_center = approx_median(epoch_to_start_deltas) or -500.0
    start_to_deal_center = approx_median(start_to_deal_deltas) or 7500.0

    # Build the repaired timestamps in one pass.
    # We do not mutate the original raw data in-place; we produce a separate final structure.
    final_rows = []
    for i, r in enumerate(repaired):
        epoch_ms = r["epoch_ms"]
        if epoch_ms is None:
            epoch_ms = None
        dt_epoch = epoch_to_dt(epoch_ms) if epoch_ms is not None else None

        # Determine starttime_abs for this row.
        if epoch_ms is not None:
            start_raw = r["start_ms"]
            if start_raw is None:
                start_abs = dt_epoch + timedelta(milliseconds=epoch_to_start_center)
            else:
                start_dt = dt_from_day_and_clock(dt_epoch.date(), start_raw)
                if (start_dt - dt_epoch).total_seconds() > 12 * 3600:
                    start_dt -= timedelta(days=1)
                elif (start_dt - dt_epoch).total_seconds() < -12 * 3600:
                    start_dt += timedelta(days=1)
                delta = (start_dt - dt_epoch).total_seconds() * 1000.0
                if abs(delta - epoch_to_start_center) <= THRESH_EPOCH_TO_START:
                    start_abs = start_dt
                else:
                    start_abs = dt_epoch + timedelta(milliseconds=epoch_to_start_center)
        else:
            start_abs = None

        # Determine dealing_abs for this row.
        if start_abs is not None:
            deal_raw = r["deal_ms"]
            if deal_raw is None:
                deal_abs = start_abs + timedelta(milliseconds=start_to_deal_center)
            else:
                deal_dt = dt_from_day_and_clock(start_abs.date(), deal_raw)
                # Use same-day alignment logic around the starttime.
                if (deal_dt - start_abs).total_seconds() > 12 * 3600:
                    deal_dt -= timedelta(days=1)
                elif (deal_dt - start_abs).total_seconds() < -12 * 3600:
                    deal_dt += timedelta(days=1)
                delta = (deal_dt - start_abs).total_seconds() * 1000.0
                if abs(delta - start_to_deal_center) <= THRESH_START_TO_DEAL:
                    deal_abs = deal_dt
                else:
                    deal_abs = start_abs + timedelta(milliseconds=start_to_deal_center)
        else:
            deal_abs = None

        # Determine Gr for this row.
        if i + 1 < n:
            next_row = repaired[i + 1]
            next_epoch_ms = next_row["epoch_ms"]
            next_epoch_dt = epoch_to_dt(next_epoch_ms) if next_epoch_ms is not None else None
            next_start_deal_raw = next_row["deal_ms"]
            next_start_deal_abs = None
            if next_epoch_dt is not None and next_start_deal_raw is not None:
                next_start_deal_abs = dt_from_day_and_clock(next_epoch_dt.date(), next_start_deal_raw)
                # Align across midnight if needed.
                if (next_start_deal_abs - next_epoch_dt).total_seconds() > 12 * 3600:
                    next_start_deal_abs -= timedelta(days=1)
                elif (next_start_deal_abs - next_epoch_dt).total_seconds() < -12 * 3600:
                    next_start_deal_abs += timedelta(days=1)

            gr_raw = r["gr_ms"]
            if gr_raw is not None:
                gr_dt = dt_from_day_and_clock(start_abs.date(), gr_raw) if start_abs is not None else None
                if gr_dt is not None and start_abs is not None:
                    # fix cross-day alignment
                    if (gr_dt - start_abs).total_seconds() > 12 * 3600:
                        gr_dt -= timedelta(days=1)
                    elif (gr_dt - start_abs).total_seconds() < -12 * 3600:
                        gr_dt += timedelta(days=1)
                    if next_epoch_dt is not None and next_start_deal_abs is not None:
                        if gr_dt > start_abs and gr_dt < next_epoch_dt:
                            gr_to_next_epoch = (next_epoch_dt - gr_dt).total_seconds() * 1000.0
                            if abs(gr_to_next_epoch - 2400.0) <= THRESH_GR_TO_NEXT_EPOCH:
                                gr_abs = gr_dt
                            else:
                                if next_start_deal_abs is not None:
                                    candidate = next_start_deal_abs - timedelta(milliseconds=GR_TO_NEXT_DEALING_CENTER)
                                    if start_abs < candidate < next_epoch_dt:
                                        gr_abs = candidate
                                    else:
                                        gr_abs = gr_dt
                                else:
                                    gr_abs = gr_dt
                        else:
                            if next_start_deal_abs is not None:
                                candidate = next_start_deal_abs - timedelta(milliseconds=GR_TO_NEXT_DEALING_CENTER)
                                if start_abs < candidate < next_epoch_dt:
                                    gr_abs = candidate
                                else:
                                    gr_abs = start_abs + timedelta(milliseconds=7500.0)
                            else:
                                gr_abs = start_abs + timedelta(milliseconds=7500.0)
                    else:
                        # no next epoch -> fallback to start + 7500ms if possible
                        gr_abs = start_abs + timedelta(milliseconds=7500.0)
                else:
                    gr_abs = None
            else:
                gr_abs = None

            if gr_abs is None and next_start_deal_abs is not None:
                gr_abs = next_start_deal_abs - timedelta(milliseconds=GR_TO_NEXT_DEALING_CENTER)

            if gr_abs is not None and start_abs is not None and deal_abs is not None:
                if gr_abs <= deal_abs:
                    # Must be after Start Dealing.
                    if next_start_deal_abs is not None:
                        gr_abs = next_start_deal_abs - timedelta(milliseconds=GR_TO_NEXT_DEALING_CENTER)
                    else:
                        gr_abs = deal_abs + timedelta(milliseconds=2500.0)

            if gr_abs is not None and next_epoch_dt is not None and gr_abs >= next_epoch_dt:
                if next_start_deal_abs is not None:
                    gr_abs = next_start_deal_abs - timedelta(milliseconds=GR_TO_NEXT_DEALING_CENTER)
                else:
                    gr_abs = next_epoch_dt - timedelta(milliseconds=2400.0)

        else:
            gr_abs = None

        # Final formatting
        format_epoch = str(int(epoch_ms)) if epoch_ms is not None else "-"
        if start_abs is not None:
            st = start_abs.strftime("%H:%M:%S.%f")[:-3]
        else:
            st = "-"
        if deal_abs is not None:
            deal = deal_abs.strftime("%H:%M:%S.%f")[:-3]
        else:
            deal = "-"
        if gr_abs is not None:
            gr = gr_abs.strftime("%H:%M:%S.%f")[:-3]
        else:
            gr = "-"

        final_rows.append({
            "index": r["index"],
            "epoch": format_epoch,
            "starttime": st,
            "start_dealing": deal,
            "gr": gr,
            "result_gr": r["result_gr"],
            "id_game": r["id_game"],
            "status": r["status"],
            "__raw_epoch": epoch_ms,
            "__raw_starttime": r["start_ms"],
            "__raw_deal": r["deal_ms"],
            "__raw_gr": r["gr_ms"],
        })

    return final_rows, {
        "epoch_to_start_center_ms": epoch_to_start_center,
        "start_to_deal_center_ms": start_to_deal_center,
        "thresh_epoch_to_start_ms": THRESH_EPOCH_TO_START,
        "thresh_start_to_deal_ms": THRESH_START_TO_DEAL,
        "thresh_gr_to_next_epoch_ms": THRESH_GR_TO_NEXT_EPOCH,
        "gr_to_next_dealing_center_ms": GR_TO_NEXT_DEALING_CENTER,
        "gr_to_next_dealing_tol_ms": GR_TO_NEXT_DEALING_TOL,
    }


def write_markdown(rows, out_path: Path):
    header = ["No", "Ts Epoch Starttime", "Ts Starttime (Bukan Epoch)", "Ts Start Dealing", "Ts Gr", "Result Gr", "Id Game", "Status (Live/History)"]
    lines = [
        "# Hasil Repair Timestamp (test_kalibrasi4)",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for idx, r in enumerate(rows, start=1):
        row = [
            str(idx),
            str(r["epoch"]),
            str(r["starttime"]),
            str(r["start_dealing"]),
            str(r["gr"]),
            str(r["result_gr"]),
            str(r["id_game"]),
            str(r["status"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Repair timestamp table using per-segment thresholds.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = read_markdown_table(args.input)
    repaired_rows, metrics = repair_rounds(rows)

    csv_out = args.csv_output or args.input.with_name(args.input.stem + "_repaired.csv")
    md_out = args.md_output or args.input.with_name(args.input.stem + "_repaired.md")
    report_out = args.report or args.input.with_name(args.input.stem + "_repair_report.json")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV preserving useful columns only.
    csv_fields = ["index", "epoch", "starttime", "start_dealing", "gr", "result_gr", "id_game", "status"]
    with csv_out.open("w", encoding="utf-8") as f:
        f.write(",".join(csv_fields) + "\n")
        for r in repaired_rows:
            vals = [
                str(r["index"]),
                str(r["epoch"]),
                str(r["starttime"]),
                str(r["start_dealing"]),
                str(r["gr"]),
                str(r["result_gr"]),
                str(r["id_game"]),
                str(r["status"]),
            ]
            f.write(",".join(vals) + "\n")

    write_markdown(repaired_rows, md_out)
    report_out.write_text(json.dumps({"rows": len(repaired_rows), "metrics": metrics}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV   : {csv_out}")
    print(f"MD    : {md_out}")
    print(f"JSON  : {report_out}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
