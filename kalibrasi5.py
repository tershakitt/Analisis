#!/usr/bin/env python3
"""kalibrasi5.py

Per-segment threshold repair with locked first 50 rows.

Rules implemented:
- Rows 0..49 are preserved as-is.
- Epoch values missing internally are backfilled with a simple anchor bridge.
- Calibration centers are computed from rows 50+ only.
- Thresholds:
    Epoch -> Starttime            +/- 100 ms
    Starttime -> Start Dealing    +/- 200 ms
    Gr -> Next Epoch              +/- 200 ms
    Gr -> Next Start Dealing      9900 +/- 300 ms
- Start Dealing -> Gr remains dynamic and is repaired only when impossible.
- K_FLIGHT is not used as the main repair rule; it is left as a fallback/validator concept.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOCKED_ROWS = 50
THRESH_EPOCH_TO_START_MS = 100.0
THRESH_START_TO_DEAL_MS = 200.0
THRESH_GR_TO_NEXT_EPOCH_MS = 200.0
TRANSITION_CENTER_MS = 9900.0
TRANSITION_TOL_MS = 300.0
OUTPUT_COLUMNS = ["No", "Ts Epoch Starttime", "Ts Starttime (Bukan Epoch)", "Ts Start Dealing", "Ts Gr", "Result Gr", "Id Game", "Status (Live/History)"]


def parse_epoch_ms(value):
    if value is None or str(value).strip() in {"", "-", "nan", "NaN"}:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_clock_ms(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "NaN"}:
        return None
    if "." in text:
        hms, frac = text.split(".", 1)
        ms = int((frac + "000")[:3])
    else:
        hms, ms = text, 0
    try:
        h, m, s = map(int, hms.split(":"))
    except Exception:
        return None
    return (h * 3600 + m * 60 + s) * 1000 + ms


def epoch_to_dt(epoch_ms):
    return datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc)


def clock_abs(clock_ms, reference_epoch):
    if clock_ms is None or reference_epoch is None:
        return None
    epoch_dt = epoch_to_dt(reference_epoch)
    dt = datetime.combine(epoch_dt.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(milliseconds=clock_ms)
    diff = (dt - epoch_dt).total_seconds()
    if diff > 12 * 3600:
        dt -= timedelta(days=1)
    elif diff < -12 * 3600:
        dt += timedelta(days=1)
    return dt.timestamp() * 1000.0


def format_clock(abs_ms, with_ms=True):
    if abs_ms is None or not math.isfinite(abs_ms):
        return "-"
    sec_ms = int(math.floor(abs_ms / 1000.0))
    milli = int(round(abs_ms - sec_ms * 1000.0))
    if milli >= 1000:
        sec_ms += milli // 1000
        milli %= 1000
    elif milli < 0:
        sec_ms -= 1
        milli += 1000
    dt = datetime.fromtimestamp(sec_ms, tz=timezone.utc)
    return f"{dt:%H:%M:%S}.{milli:03d}" if with_ms else f"{dt:%H:%M:%S}"


def median(values, fallback):
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    return float(statistics.median(vals)) if vals else fallback


def read_markdown_table(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("No |") or s.startswith("---") or "|" not in s:
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) < 8:
            continue
        try:
            no = int(parts[0])
        except Exception:
            continue
        rows.append(dict(zip(OUTPUT_COLUMNS, [no] + parts[1:8])))
    return rows


def fill_missing_epochs(rows):
    epochs = [parse_epoch_ms(r["Ts Epoch Starttime"]) for r in rows]
    n = len(epochs)
    i = 0
    while i < n:
        if epochs[i] is not None:
            i += 1
            continue
        start = i
        while i < n and epochs[i] is None:
            i += 1
        end = i - 1
        left = start - 1 if start > 0 and epochs[start - 1] is not None else None
        right = end + 1 if end + 1 < n and epochs[end + 1] is not None else None
        if left is not None and right is not None:
            step = (epochs[right] - epochs[left]) / (right - left)
            for j in range(start, end + 1):
                epochs[j] = epochs[left] + step * (j - left)
        elif left is not None:
            for j in range(start, end + 1):
                epochs[j] = epochs[j - 1] + 2400.0
        elif right is not None:
            for j in range(end, start - 1, -1):
                epochs[j] = epochs[j + 1] - 2400.0
    return epochs


def calibrate(rows, epochs):
    epoch_start = []
    start_deal = []
    gr_next_epoch = []
    for i in range(LOCKED_ROWS, len(rows)):
        e = epochs[i]
        if e is None:
            continue
        s = clock_abs(parse_clock_ms(rows[i]["Ts Starttime (Bukan Epoch)"]), e)
        d = clock_abs(parse_clock_ms(rows[i]["Ts Start Dealing"]), e)
        g = clock_abs(parse_clock_ms(rows[i]["Ts Gr"]), e)
        if s is not None and abs(s - e) < 2000:
            epoch_start.append(s - e)
        if s is not None and d is not None and 5000 <= d - s <= 12000:
            start_deal.append(d - s)
        if g is not None and i + 1 < len(rows) and epochs[i + 1] is not None:
            post = epochs[i + 1] - g
            if 0 < post <= 6000:
                gr_next_epoch.append(post)
    return {
        "epoch_to_start_center_ms": median(epoch_start, -500.0),
        "start_to_deal_center_ms": median(start_deal, 7500.0),
        "gr_to_next_epoch_center_ms": median(gr_next_epoch, 2400.0),
        "gr_to_next_dealing_center_ms": TRANSITION_CENTER_MS,
    }


def repair(rows):
    epochs = fill_missing_epochs(rows)
    cal = calibrate(rows, epochs)
    output = []
    n = len(rows)
    for i, raw in enumerate(rows):
        out = dict(raw)
        if i < LOCKED_ROWS:
            output.append(out)
            continue
        e = epochs[i]
        if e is None:
            output.append(out)
            continue

        s = clock_abs(parse_clock_ms(raw["Ts Starttime (Bukan Epoch)"]), e)
        d = clock_abs(parse_clock_ms(raw["Ts Start Dealing"]), e)
        g = clock_abs(parse_clock_ms(raw["Ts Gr"]), e)

        if s is None or abs((s - e) - cal["epoch_to_start_center_ms"]) > THRESH_EPOCH_TO_START_MS:
            s = e + cal["epoch_to_start_center_ms"]
        if d is None or s is None or abs((d - s) - cal["start_to_deal_center_ms"]) > THRESH_START_TO_DEAL_MS:
            d = e + cal["start_to_deal_center_ms"]

        # Prefer next repaired dealing where possible; otherwise use next Epoch/post-gap.
        next_e = epochs[i + 1] if i + 1 < n else None
        next_d = None
        if i + 1 < n and epochs[i + 1] is not None:
            next_d_raw = clock_abs(parse_clock_ms(rows[i + 1]["Ts Start Dealing"]), epochs[i + 1])
            if next_d_raw is not None and abs((next_d_raw - epochs[i + 1]) - cal["start_to_deal_center_ms"]) <= THRESH_START_TO_DEAL_MS:
                next_d = next_d_raw
            else:
                next_d = epochs[i + 1] + cal["start_to_deal_center_ms"]

        valid_g = g is not None and d is not None and g > d
        if valid_g and next_d is not None:
            transition = next_d - g
            if not (TRANSITION_CENTER_MS - TRANSITION_TOL_MS <= transition <= TRANSITION_CENTER_MS + TRANSITION_TOL_MS):
                valid_g = False
        if valid_g and next_e is not None:
            post = next_e - g
            if post <= 0 or abs(post - cal["gr_to_next_epoch_center_ms"]) > THRESH_GR_TO_NEXT_EPOCH_MS:
                # An outlying post-gap is repaired from the stronger next-dealing anchor first.
                valid_g = False

        if not valid_g:
            if next_d is not None:
                candidate = next_d - TRANSITION_CENTER_MS
                if candidate > d and (next_e is None or candidate < next_e):
                    g = candidate
            if g is None or g <= d or (next_e is not None and g >= next_e):
                if next_e is not None:
                    candidate = next_e - cal["gr_to_next_epoch_center_ms"]
                    if candidate > d and candidate < next_e:
                        g = candidate
                if g is None or g <= d:
                    g = d + 7500.0 if d is not None else None

        out["Ts Epoch Starttime"] = str(int(round(e)))
        out["Ts Starttime (Bukan Epoch)"] = format_clock(s, False)
        out["Ts Start Dealing"] = format_clock(d, True)
        out["Ts Gr"] = format_clock(g, True)
        output.append(out)
    return output, epochs, cal


def write_markdown(rows, path):
    lines = ["# Hasil Repair Timestamp (kalibrasi5)", "", "| " + " | ".join(OUTPUT_COLUMNS) + " |", "| " + " | ".join(["---"] * 8) + " |"]
    for i, row in enumerate(rows, 1):
        vals = [str(i)] + [str(row.get(c, "-")) for c in OUTPUT_COLUMNS[1:]]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="kalibrasi5 threshold repair")
    p.add_argument("input", type=Path)
    p.add_argument("--csv-output", type=Path, default=None)
    p.add_argument("--md-output", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    rows = read_markdown_table(args.input)
    repaired, epochs, cal = repair(rows)
    csv_path = args.csv_output or args.input.with_name(args.input.stem + "_repaired.csv")
    md_path = args.md_output or args.input.with_name(args.input.stem + "_repaired.md")
    report_path = args.report or args.input.with_name(args.input.stem + "_repair_report.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(OUTPUT_COLUMNS) + "\n")
        for i, row in enumerate(repaired, 1):
            f.write(",".join([str(i)] + [str(row.get(c, "-")) for c in OUTPUT_COLUMNS[1:]]) + "\n")
    write_markdown(repaired, md_path)
    report_path.write_text(json.dumps({"locked_rows": LOCKED_ROWS, "thresholds": {"epoch_to_start_ms": THRESH_EPOCH_TO_START_MS, "start_to_deal_ms": THRESH_START_TO_DEAL_MS, "gr_to_next_epoch_ms": THRESH_GR_TO_NEXT_EPOCH_MS, "gr_to_next_dealing_center_ms": TRANSITION_CENTER_MS, "gr_to_next_dealing_tolerance_ms": TRANSITION_TOL_MS}, "calibration": cal, "filled_epoch_rows": sum(parse_epoch_ms(rows[i]["Ts Epoch Starttime"]) is None and epochs[i] is not None for i in range(len(rows)))}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CSV   : {csv_path}")
    print(f"MD    : {md_path}")
    print(f"JSON  : {report_path}")


if __name__ == "__main__":
    main()
