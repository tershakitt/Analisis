#!/usr/bin/env python3
"""kalibrasi5.py

Threshold-based repair pipeline with a strict lock on the first 50 rows.

Design:
- Rows 0..49 are protected and never repaired.
- All calibration is computed from rows 50+ only.
- Fixed per-segment thresholds:
    Epoch -> Starttime            +/- 100 ms
    Starttime -> Start Dealing    +/- 200 ms
    Gr -> Next Epoch              +/- 200 ms
    Gr -> Next Start Dealing      9900 +/- 300 ms  (center 9900)
- Start Dealing -> Gr remains dynamic; the transition prior is used only to repair
  impossible or missing Gr values.
- K_FLIGHT exists only as fallback/validator and is not used to overwrite valid raw data.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOCKED_ROWS = 50

EPOCH_TO_START_CENTER = None
START_TO_DEAL_CENTER = None
GR_TO_NEXT_EPOCH_CENTER = None

THRESH_EPOCH_TO_START_MS = 100.0
THRESH_START_TO_DEAL_MS = 200.0
THRESH_GR_TO_NEXT_EPOCH_MS = 200.0
TRANSITION_CENTER_MS = 9900.0
TRANSITION_TOL_MS = 300.0


def parse_clock_to_ms(value: object):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "NaN"}:
        return None
    if "." in text:
        hms, frac = text.split(".", 1)
        ms = int((frac + "000")[:3])
    else:
        hms = text
        ms = 0
    try:
        h, m, s = map(int, hms.split(":"))
    except Exception:
        return None
    return ((h * 3600 + m * 60 + s) * 1000) + ms


def parse_epoch_to_ms(value: object):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "NaN"}:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def epoch_to_dt(epoch_ms: float):
    return datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc)


def align_day_clock(day_date, clock_ms: int):
    base = datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc)
    dt = base + timedelta(milliseconds=clock_ms)
    return dt


def stable_median(vals):
    arr = [float(v) for v in vals if v is not None and math.isfinite(v)]
    if not arr:
        return None
    return float(statistics.median(arr))


def read_markdown_table(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("No |") or s.startswith("---"):
                continue
            if "|" not in s:
                continue
            parts = [p.strip() for p in s.split("|")]
            if len(parts) < 8:
                continue
            try:
                int(parts[0])
            except Exception:
                continue
            rows.append({
                "index": int(parts[0]),
                "epoch": parts[1],
                "starttime": parts[2],
                "start_dealing": parts[3],
                "gr": parts[4],
                "result_gr": parts[5],
                "id_game": parts[6],
                "status": parts[7],
            })
    return rows


def build_abs_from_row(row, row_index, calibration):
    epoch_ms = parse_epoch_to_ms(row["epoch"])
    start_ms = parse_clock_to_ms(row["starttime"])
    deal_ms = parse_clock_to_ms(row["start_dealing"])
    gr_ms = parse_clock_to_ms(row["gr"])

    abs_times = {
        "epoch": epoch_ms,
        "start": None,
        "deal": None,
        "gr": None,
    }

    if epoch_ms is not None:
        epoch_dt = epoch_to_dt(epoch_ms)
        if start_ms is not None:
            dt = align_day_clock(epoch_dt.date(), start_ms)
            # adjust across midnight if needed
            if (dt - epoch_dt).total_seconds() > 12 * 3600:
                dt -= timedelta(days=1)
            elif (dt - epoch_dt).total_seconds() < -12 * 3600:
                dt += timedelta(days=1)
            abs_times["start"] = dt.timestamp() * 1000.0

        if deal_ms is not None:
            dt = align_day_clock(epoch_dt.date(), deal_ms)
            if (dt - epoch_dt).total_seconds() > 12 * 3600:
                dt -= timedelta(days=1)
            elif (dt - epoch_dt).total_seconds() < -12 * 3600:
                dt += timedelta(days=1)
            abs_times["deal"] = dt.timestamp() * 1000.0

        if gr_ms is not None:
            dt = align_day_clock(epoch_dt.date(), gr_ms)
            if (dt - epoch_dt).total_seconds() > 12 * 3600:
                dt -= timedelta(days=1)
            elif (dt - epoch_dt).total_seconds() < -12 * 3600:
                dt += timedelta(days=1)
            abs_times["gr"] = dt.timestamp() * 1000.0

    return abs_times


def format_abs_to_clock(abs_ms: float, with_ms: bool = True):
    if abs_ms is None or abs_ms == float("nan") or not math.isfinite(abs_ms):
        return "-"
    sec_ms = int(math.floor(abs_ms / 1000.0))
    milli = int(round(abs_ms - sec_ms * 1000.0))
    if milli >= 1000:
        sec_ms += milli // 1000
        milli = milli % 1000
    elif milli < 0:
        sec_ms -= 1
        milli = 1000 + milli
    dt = datetime.fromtimestamp(sec_ms, tz=timezone.utc)
    if with_ms:
        return f"{dt:%H:%M:%S}.{milli:03d}"
    return f"{dt:%H:%M:%S}"


def calibrate_centers(rows):
    # Only rows 50+ participate in calibration.
    epoch_to_start = []
    start_to_deal = []
    gr_to_next_epoch = []
    valid_count = 0

    n = len(rows)
    for i, row in enumerate(rows):
        if i < LOCKED_ROWS:
            continue
        abs_times = build_abs_from_row(row, i, {})
        epoch_ms = abs_times["epoch"]
        if epoch_ms is None:
            continue

        start_ms = abs_times["start"]
        deal_ms = abs_times["deal"]
        gr_ms = abs_times["gr"]

        if start_ms is not None:
            delta = start_ms - epoch_ms
            if abs(delta) < 2000.0:
                epoch_to_start.append(delta)

        if start_ms is not None and deal_ms is not None:
            delay = deal_ms - start_ms
            if 5000.0 <= delay <= 12000.0:
                start_to_deal.append(delay)

        if gr_ms is not None and i + 1 < n:
            next_row = rows[i + 1]
            next_epoch_ms = parse_epoch_to_ms(next_row["epoch"])
            if next_epoch_ms is not None:
                post_gap = next_epoch_ms - gr_ms
                if 0 < post_gap <= 6000.0:
                    gr_to_next_epoch.append(post_gap)

        valid_count += 1

    epoch_center = stable_median(epoch_to_start)
    deal_center = stable_median(start_to_deal)
    post_center = stable_median(gr_to_next_epoch)

    if epoch_center is None:
        epoch_center = -500.0
    if deal_center is None:
        deal_center = 7500.0
    if post_center is None:
        post_center = 2400.0

    return {
        "epoch_to_start_center_ms": epoch_center,
        "start_to_deal_center_ms": deal_center,
        "gr_to_next_epoch_center_ms": post_center,
        "gr_to_next_dealing_center_ms": TRANSITION_CENTER_MS,
    }


def repair_rows(rows):
    calib = calibrate_centers(rows)
    epoch_to_start_center = calib["epoch_to_start_center_ms"]
    start_to_deal_center = calib["start_to_deal_center_ms"]
    gr_to_next_epoch_center = calib["gr_to_next_epoch_center_ms"]

    repaired = []
    n = len(rows)

    for idx, row in enumerate(rows):
        out = dict(row)
        epoch_ms = parse_epoch_to_ms(row["epoch"])

        # Keep first 50 rows untouched.
        if idx < LOCKED_ROWS:
            repaired.append(out)
            continue

        # Build absolute times for current row, from raw data if possible.
        abs_times = build_abs_from_row(row, idx, calib)
        epoch_abs = abs_times["epoch"]
        start_abs = abs_times["start"]
        deal_abs = abs_times["deal"]
        gr_abs = abs_times["gr"]

        # Epoch is anchor; if missing we may leave it empty.
        if epoch_abs is not None:
            # Starttime
            if start_abs is not None:
                delta = start_abs - epoch_abs
                if abs(delta - epoch_to_start_center) <= THRESH_EPOCH_TO_START_MS:
                    start_value = format_abs_to_clock(start_abs, with_ms=False)
                else:
                    start_value = format_abs_to_clock(epoch_abs + epoch_to_start_center, with_ms=False)
            else:
                start_value = format_abs_to_clock(epoch_abs + epoch_to_start_center, with_ms=False)

            # Start Dealing
            if deal_abs is not None:
                delta = deal_abs - start_abs if start_abs is not None else deal_abs - epoch_abs
                center = start_to_deal_center
                if start_abs is not None and abs(delta - center) <= THRESH_START_TO_DEAL_MS:
                    deal_value = format_abs_to_clock(deal_abs, with_ms=True)
                else:
                    deal_value = format_abs_to_clock(epoch_abs + center, with_ms=True)
            else:
                deal_value = format_abs_to_clock(epoch_abs + start_to_deal_center, with_ms=True)
        else:
            start_value = "-"
            deal_value = "-"

        # Gr repair using transition prior and structural constraints.
        if epoch_abs is not None:
            if gr_abs is not None and start_abs is not None and deal_abs is not None:
                if gr_abs > deal_abs:
                    if idx + 1 < n:
                        next_epoch_ms = parse_epoch_to_ms(rows[idx + 1]["epoch"])
                        if next_epoch_ms is not None:
                            next_deal_raw = parse_clock_to_ms(rows[idx + 1]["start_dealing"])
                            next_deal_abs = None
                            if next_deal_raw is not None:
                                next_epoch_dt = epoch_to_dt(next_epoch_ms)
                                dt = align_day_clock(next_epoch_dt.date(), next_deal_raw)
                                if (dt - next_epoch_dt).total_seconds() > 12 * 3600:
                                    dt -= timedelta(days=1)
                                elif (dt - next_epoch_dt).total_seconds() < -12 * 3600:
                                    dt += timedelta(days=1)
                                next_deal_abs = dt.timestamp() * 1000.0

                            if next_deal_abs is not None:
                                transition = next_deal_abs - gr_abs
                                if 9600.0 <= transition <= 10200.0:
                                    gr_value = format_abs_to_clock(gr_abs, with_ms=True)
                                else:
                                    candidate = next_deal_abs - TRANSITION_CENTER_MS
                                    if deal_abs < candidate < next_epoch_ms:
                                        gr_value = format_abs_to_clock(candidate, with_ms=True)
                                    else:
                                        gr_value = format_abs_to_clock(gr_abs, with_ms=True)
                            else:
                                gr_value = format_abs_to_clock(gr_abs, with_ms=True)
                        else:
                            gr_value = format_abs_to_clock(gr_abs, with_ms=True)
                    else:
                        gr_value = format_abs_to_clock(gr_abs, with_ms=True)
                else:
                    # Must be after dealing, otherwise repair.
                    if idx + 1 < n:
                        next_epoch_ms = parse_epoch_to_ms(rows[idx + 1]["epoch"])
                        next_epoch_dt = epoch_to_dt(next_epoch_ms) if next_epoch_ms is not None else None
                        if next_epoch_dt is not None:
                            next_deal_raw = parse_clock_to_ms(rows[idx + 1]["start_dealing"])
                            if next_deal_raw is not None:
                                dt = align_day_clock(next_epoch_dt.date(), next_deal_raw)
                                if (dt - next_epoch_dt).total_seconds() > 12 * 3600:
                                    dt -= timedelta(days=1)
                                elif (dt - next_epoch_dt).total_seconds() < -12 * 3600:
                                    dt += timedelta(days=1)
                                next_deal_abs = dt.timestamp() * 1000.0
                                candidate = next_deal_abs - TRANSITION_CENTER_MS
                                if deal_abs < candidate < next_epoch_ms:
                                    gr_value = format_abs_to_clock(candidate, with_ms=True)
                                else:
                                    gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True)
                            else:
                                gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True)
                        else:
                            gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True)
                    else:
                        gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True)
            else:
                # Missing or structurally invalid Gr.
                if idx + 1 < n:
                    next_epoch_ms = parse_epoch_to_ms(rows[idx + 1]["epoch"])
                    next_deal_raw = parse_clock_to_ms(rows[idx + 1]["start_dealing"])
                    if next_epoch_ms is not None and next_deal_raw is not None:
                        next_epoch_dt = epoch_to_dt(next_epoch_ms)
                        dt = align_day_clock(next_epoch_dt.date(), next_deal_raw)
                        if (dt - next_epoch_dt).total_seconds() > 12 * 3600:
                            dt -= timedelta(days=1)
                        elif (dt - next_epoch_dt).total_seconds() < -12 * 3600:
                            dt += timedelta(days=1)
                        next_deal_abs = dt.timestamp() * 1000.0
                        candidate = next_deal_abs - TRANSITION_CENTER_MS
                        if deal_abs is not None and next_deal_abs is not None:
                            if deal_abs < candidate < next_epoch_ms:
                                gr_value = format_abs_to_clock(candidate, with_ms=True)
                            else:
                                gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True)
                        else:
                            gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True) if deal_abs is not None else "-"
                    else:
                        gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True) if deal_abs is not None else "-"
                else:
                    gr_value = format_abs_to_clock(deal_abs + 7500.0, with_ms=True) if deal_abs is not None else "-"
        else:
            gr_value = "-"

        out["Ts Epoch Starttime"] = str(epoch_ms) if epoch_ms is not None else "-"
        out["Ts Starttime (Bukan Epoch)"] = start_value
        out["Ts Start Dealing"] = deal_value
        out["Ts Gr"] = gr_value

        repaired.append(out)

    return repaired, calib


def write_markdown(rows, output_path: Path):
    cols = [
        "No",
        "Ts Epoch Starttime",
        "Ts Starttime (Bukan Epoch)",
        "Ts Start Dealing",
        "Ts Gr",
        "Result Gr",
        "Id Game",
        "Status (Live/History)",
    ]
    lines = [
        "# Hasil Repair Timestamp (kalibrasi5)",
        "",
        " | ".join(cols),
        " | ".join(["---"] * len(cols)),
    ]
    for j, row in enumerate(rows, start=1):
        vals = [
            str(j),
            str(row.get("Ts Epoch Starttime", "-")),
            str(row.get("Ts Starttime (Bukan Epoch)", "-")),
            str(row.get("Ts Start Dealing", "-")),
            str(row.get("Ts Gr", "-")),
            str(row.get("Result Gr", "-")),
            str(row.get("Id Game", "-")),
            str(row.get("Status (Live/History)", "-")),
        ]
        lines.append(" | ".join(vals))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="kalibrasi5: threshold repair with locked first 50 rows")
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = read_markdown_table(args.input)
    repaired_rows, calib = repair_rows(rows)

    csv_out = args.csv_output or args.input.with_name(args.input.stem + "_repaired.csv")
    md_out = args.md_output or args.input.with_name(args.input.stem + "_repaired.md")
    report_out = args.report or args.input.with_name(args.input.stem + "_repair_report.json")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    csv_cols = [
        "Ts Epoch Starttime",
        "Ts Starttime (Bukan Epoch)",
        "Ts Start Dealing",
        "Ts Gr",
        "Result Gr",
        "Id Game",
        "Status (Live/History)",
    ]
    with csv_out.open("w", encoding="utf-8") as f:
        f.write("No," + ",".join(csv_cols) + "\n")
        for j, row in enumerate(repaired_rows, start=1):
            values = [
                str(j),
                str(row.get("Ts Epoch Starttime", "-")),
                str(row.get("Ts Starttime (Bukan Epoch)", "-")),
                str(row.get("Ts Start Dealing", "-")),
                str(row.get("Ts Gr", "-")),
                str(row.get("Result Gr", "-")),
                str(row.get("Id Game", "-")),
                str(row.get("Status (Live/History)", "-")),
            ]
            f.write(",".join(values) + "\n")

    write_markdown(repaired_rows, md_out)
    report_out.write_text(json.dumps({
        "locked_rows": LOCKED_ROWS,
        "thresholds": {
            "epoch_to_start_ms": THRESH_EPOCH_TO_START_MS,
            "start_to_deal_ms": THRESH_START_TO_DEAL_MS,
            "gr_to_next_epoch_ms": THRESH_GR_TO_NEXT_EPOCH_MS,
            "gr_to_next_dealing_center_ms": TRANSITION_CENTER_MS,
            "gr_to_next_dealing_tol_ms": TRANSITION_TOL_MS,
        },
        "calibration": calib,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV   : {csv_out}")
    print(f"MD    : {md_out}")
    print(f"JSON  : {report_out}")
    print(json.dumps(calib, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
