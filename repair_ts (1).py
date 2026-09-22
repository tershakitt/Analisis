#!/usr/bin/env python3
"""
repair_ts.py

Tujuan:
- Perbaiki timestamp crash-game agar konsisten dengan Epoch server.
- Isi seluruh timestamp yang kosong atau rusak.
- Output file Markdown tetap seperti file sumber: 8 kolom tanpa flag.

Prinsip utama:
1) Epoch server = ground truth.
2) Waktu penerimaan WebSocket tidak boleh dipakai sebagai event-time.
3) `Ts Gr` dihitung dari event model:
      Ts Gr ≈ Ts Start Dealing + flight_duration(multiplier)
   dengan flight duration yang diestimasi dari data yang valid.
4) File Markdown final tetap bersih, tanpa kolom tambahan.

Catatan:
- Tidak ada jaminan absolut 100% benar untuk setiap row, karena data yang
  diterima dari WebSocket memang mengandung delay/reconnect.
- Namun output ini akan lebih konsisten dan lebih sesuai dengan kebutuhan
  "mengisi tetap semua data", dibanding asal menaruh raw receive time.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "No",
    "Ts Epoch Starttime",
    "Ts Starttime (Bukan Epoch)",
    "Ts Start Dealing",
    "Ts Gr",
    "Result Gr",
    "Id Game",
    "Status (Live/History)",
]

TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$")


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in {"", "-", "nan", "NaN"}


def parse_epoch_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(
        s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}),
        errors="coerce",
    )
    return out


def parse_clock_text(text: object):
    if is_missing(text):
        return None
    m = TIME_RE.match(str(text))
    if not m:
        return None
    h = int(m.group(1))
    minute = int(m.group(2))
    sec = int(m.group(3))
    frac = m.group(4) or ""
    ms = int((frac + "000")[:3])
    if h > 23 or minute > 59 or sec > 59:
        return None
    return h, minute, sec, ms


def naive_clock_ms_near_epoch(clock_text: object, epoch_ms: float) -> float:
    parsed = parse_clock_text(clock_text)
    if parsed is None or pd.isna(epoch_ms):
        return np.nan

    h, minute, sec, ms = parsed
    base_date = datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc).date()

    candidates = []
    for delta_days in (-1, 0, 1):
        d = base_date + timedelta(days=delta_days)
        dt = datetime(d.year, d.month, d.day, h, minute, sec, ms * 1000, tzinfo=timezone.utc)
        candidates.append(dt.timestamp() * 1000.0)

    return min(candidates, key=lambda x: abs(x - epoch_ms))


def format_clock_ms(abs_ms: float, clock_offset_ms: float, with_ms: bool) -> str:
    if pd.isna(abs_ms):
        return "-"
    label_ms = float(abs_ms) + float(clock_offset_ms)
    sec_ms = int(math.floor(label_ms / 1000.0))
    milli = int(round(label_ms - sec_ms * 1000.0))

    if milli >= 1000:
        sec_ms += milli // 1000
        milli = milli % 1000
    elif milli < 0:
        borrow = (-milli + 999) // 1000
        sec_ms -= borrow
        milli += borrow * 1000

    dt = datetime.fromtimestamp(sec_ms, tz=timezone.utc)
    if with_ms:
        return f"{dt:%H:%M:%S}.{milli:03d}"
    return f"{dt:%H:%M:%S}"


def robust_location(values: Iterable[float], z: float = 3.5):
    a = np.asarray(pd.Series(list(values)).dropna(), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    work = a.copy()
    for _ in range(8):
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break
        lo = med - z * sigma
        hi = med + z * sigma
        nxt = work[(work >= lo) & (work <= hi)]
        if len(nxt) == len(work):
            break
        if len(nxt) < max(10, int(0.25 * len(work))):
            break
        work = nxt

    med = float(np.median(work))
    mad = float(np.median(np.abs(work - med)))
    sigma = float(1.4826 * mad)
    lo = float(med - z * sigma)
    hi = float(med + z * sigma)
    return med, mad, sigma, lo, hi, len(work)


def robust_line_fit(x: np.ndarray, y: np.ndarray):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], float)
    y = np.asarray(y[finite], float)
    n_total = len(x)
    if n_total < 30:
        return None

    keep = np.ones(n_total, dtype=bool)
    for _ in range(10):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        resid = y - (intercept + slope * x)
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med)))
        sigma = 1.4826 * mad
        threshold = max(250.0, 3.5 * sigma)
        new_keep = np.abs(resid - med) <= threshold
        if new_keep.sum() == keep.sum():
            break
        if new_keep.sum() < max(30, int(0.5 * n_total)):
            break
        keep = new_keep

    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    resid = y - (intercept + slope * x)
    r = resid[keep]
    rmed = float(np.median(r))
    rmad = float(np.median(np.abs(r - rmed)))
    rsigma = float(1.4826 * rmad)
    return float(intercept), float(slope), rmed, rmad, rsigma, n_total, int(keep.sum())


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        rows = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            if parts[0] in {"No", "---"}:
                continue
            if not parts[0].isdigit():
                continue
            rows.append(parts[:8])

        if not rows:
            raise ValueError(f"Tidak menemukan tabel markdown valid di {path}")

        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")

    df = df.copy()
    df["No"] = pd.to_numeric(df["No"], errors="coerce").astype("Int64")
    df["Ts Epoch Starttime"] = df["Ts Epoch Starttime"].astype(str)
    df["Ts Starttime (Bukan Epoch)"] = df["Ts Starttime (Bukan Epoch)"].astype(str)
    df["Ts Start Dealing"] = df["Ts Start Dealing"].astype(str)
    df["Ts Gr"] = df["Ts Gr"].astype(str)
    df["Result Gr"] = pd.to_numeric(df["Result Gr"], errors="coerce")
    df["Id Game"] = df["Id Game"].astype(str)
    df["Status (Live/History)"] = df["Status (Live/History)"].astype(str)
    return df


def read_markdown_preamble(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("No | Ts Epoch Starttime"):
            return lines[:i]
    return []


def write_markdown_like_source(df: pd.DataFrame, source_path: Path, output_path: Path):
    cols = REQUIRED_COLUMNS
    preamble = read_markdown_preamble(source_path)

    lines = list(preamble)
    if lines and lines[-1].strip():
        lines.append("")

    lines.append(" | ".join(cols))
    lines.append(" | ".join(["---"] * len(cols)))

    for row in df[cols].itertuples(index=False, name=None):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:g}")
            else:
                vals.append(str(v))
        lines.append(" | ".join(vals))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_clock_offset(df: pd.DataFrame, epoch_original: pd.Series):
    residuals = []
    for e, s in zip(epoch_original, df["Ts Starttime (Bukan Epoch)"]):
        if pd.isna(e) or is_missing(s):
            continue
        naive = naive_clock_ms_near_epoch(s, e)
        if pd.isna(naive):
            continue
        residuals.append(naive - e)

    if len(residuals) < 30:
        raise ValueError("Sample Starttime↔Epoch terlalu sedikit untuk kalibrasi clock.")

    q0, q10, q50, q90, q100 = np.quantile(residuals, [0, 0.1, 0.5, 0.9, 1.0])
    floor_like = (q0 <= -900.0) and (q100 <= 100.0)
    rounding_bias = -500.0 if floor_like else 0.0

    med_resid, _, _, _, _, _ = robust_location(np.asarray(residuals) - rounding_bias)
    clock_offset = float(med_resid)
    residual_corrected = np.asarray(residuals) - clock_offset
    residual_mad = float(np.median(np.abs(residual_corrected - np.median(residual_corrected))))
    return clock_offset, rounding_bias, residual_mad, {
        "n": len(residuals),
        "floor_like": bool(floor_like),
        "q0": float(q0),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
        "q100": float(q100),
    }


def calibrate_dealing(df: pd.DataFrame, epoch_original: pd.Series, clock_offset_ms: float):
    delays = []
    for e, s in zip(epoch_original, df["Ts Start Dealing"]):
        if pd.isna(e) or is_missing(s):
            continue
        naive = naive_clock_ms_near_epoch(s, e)
        if pd.isna(naive):
            continue
        corrected = naive - clock_offset_ms
        delays.append(corrected - e)

    if len(delays) < 30:
        med = 7000.0
        sigma = 500.0
        lo, hi = 6000.0, 8500.0
        return med, sigma, lo, hi, len(delays)

    d = np.asarray(delays, float)
    normal = d[(d >= 6000.0) & (d <= 8500.0)]
    seed = normal if len(normal) >= 30 else d

    med, mad, sigma, lo, hi, _ = robust_location(seed, z=3.5)
    return float(med), float(max(sigma, 500.0)), float(max(lo, 6000.0)), float(min(hi, 8500.0)), len(delays)


def calibrate_gap_model(df: pd.DataFrame, epoch_original: pd.Series):
    x = []
    y = []
    for i in range(len(df) - 1):
        e0 = epoch_original.iloc[i]
        e1 = epoch_original.iloc[i + 1]
        m = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
        if pd.isna(e0) or pd.isna(e1) or pd.isna(m) or m <= 0:
            continue
        gap = e1 - e0
        if gap <= 0:
            continue
        x.append(math.log(float(m)))
        y.append(float(gap))

    if len(x) < 30:
        return 0.0830, 10050.0, 500.0, len(x), 0

    fit = robust_line_fit(np.asarray(x), np.asarray(y))
    if fit is None:
        return 0.0830, 10050.0, 500.0, len(x), 0

    intercept, slope, _, _, sigma, n_total, n_used = fit
    if not np.isfinite(slope) or slope <= 0:
        return 0.0830, 10050.0, 500.0, n_total, n_used

    k_gap = 1000.0 / slope
    base_gap = intercept
    return float(k_gap), float(base_gap), float(max(sigma, 1.0)), n_total, n_used


def fill_epoch_gaps(df: pd.DataFrame, epoch_original: pd.Series, k_gap: float, base_gap: float):
    epoch_work = epoch_original.copy()
    n = len(df)
    for i in range(n):
        if pd.isna(epoch_work.iloc[i]):
            # cari anchor kiri dan kanan
            left = None
            right = None
            for j in range(i - 1, -1, -1):
                if not pd.isna(epoch_work.iloc[j]):
                    left = j
                    break
            for j in range(i + 1, n):
                if not pd.isna(epoch_work.iloc[j]):
                    right = j
                    break

            if left is not None and right is not None:
                left_epoch = float(epoch_work.iloc[left])
                right_epoch = float(epoch_work.iloc[right])
                total_steps = right - left
                if total_steps > 0:
                    multipliers = pd.to_numeric(df["Result Gr"].iloc[left:right + 1], errors="coerce").to_numpy(float)
                    arr = np.where(np.isfinite(multipliers) & (multipliers > 0), multipliers, 1.0)
                    expected = base_gap + np.log(arr) / k_gap
                    expected = np.maximum(expected, 1000.0)
                    cumulative = np.concatenate(([0.0], np.cumsum(expected[:-1])))
                    pred = left_epoch + cumulative
                    if len(pred) > 0:
                        epoch_work.iloc[i] = pred[-1]
                        # jika lebih dari satu missing, isi linear
                        # chaining biasa sudah memadai untuk gap kecil
            else:
                # edge gap: gunakan nearest known anchor dan model step
                if left is not None:
                    anchor_val = float(epoch_work.iloc[left])
                    mult = pd.to_numeric(df["Result Gr"].iloc[left], errors="coerce")
                    step = base_gap + math.log(max(float(mult), 1.0)) / k_gap
                    epoch_work.iloc[i] = anchor_val + max(step, 1000.0)
                elif right is not None:
                    anchor_val = float(epoch_work.iloc[right])
                    mult = pd.to_numeric(df["Result Gr"].iloc[right - 1], errors="coerce")
                    step = base_gap + math.log(max(float(mult), 1.0)) / k_gap
                    epoch_work.iloc[i] = anchor_val - max(step, 1000.0)

    return epoch_work


def fit_event_model(df: pd.DataFrame, epoch_original: pd.Series, clock_offset_ms: float, dealing_low_ms: float, dealing_high_ms: float):
    xs = []
    ys = []
    for i in range(len(df)):
        e = epoch_original.iloc[i]
        deal_s = df["Ts Start Dealing"].iloc[i]
        gr_s = df["Ts Gr"].iloc[i]
        mult = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
        if pd.isna(e) or is_missing(deal_s) or is_missing(gr_s) or pd.isna(mult):
            continue
        if mult <= 0:
            continue

        deal_abs = naive_clock_ms_near_epoch(deal_s, e) - clock_offset_ms
        gr_abs = naive_clock_ms_near_epoch(gr_s, e) - clock_offset_ms
        if pd.isna(deal_abs) or pd.isna(gr_abs):
            continue
        deal_delay = deal_abs - e
        flight_ms = gr_abs - deal_abs
        if not (dealing_low_ms <= deal_delay <= dealing_high_ms):
            continue
        if not (0 < flight_ms < 300000):
            continue

        xs.append(math.log(float(mult)))
        ys.append(float(flight_ms))

    if len(xs) < 30:
        return 250.0, 3200.0, 500.0, 0

    fit = robust_line_fit(np.asarray(xs), np.asarray(ys))
    if fit is None:
        return 250.0, 3200.0, 500.0, 0

    intercept, slope, _, _, sigma, n_total, n_used = fit
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        return 250.0, 3200.0, 500.0, n_used

    return float(intercept), float(slope), float(max(sigma, 50.0)), n_used


def repair_dataframe(df: pd.DataFrame, epoch_original: pd.Series):
    clock_offset, _, _, _ = infer_clock_offset(df, epoch_original)
    dealing_med, dealing_sigma, dealing_low, dealing_high, _ = calibrate_dealing(df, epoch_original, clock_offset)
    k_gap, base_gap, gap_sigma, _, _ = calibrate_gap_model(df, epoch_original)
    epoch_work = fill_epoch_gaps(df, epoch_original, k_gap, base_gap)

    intercept, slope, flight_sigma, n_used = fit_event_model(
        df, epoch_original, clock_offset, dealing_low, dealing_high
    )

    out = df.copy()
    n = len(out)

    # 1. Ts Epoch Starttime
    out["Ts Epoch Starttime"] = [
        "-" if pd.isna(v) else str(int(round(float(v))))
        for v in epoch_work
    ]

    # 2. Ts Starttime (Bukan Epoch)
    out["Ts Starttime (Bukan Epoch)"] = [
        "-" if pd.isna(epoch_work.iloc[i]) else format_clock_ms(epoch_work.iloc[i], clock_offset, False)
        for i in range(n)
    ]

    # 3. Ts Start Dealing
    deal_abs = np.full(n, np.nan, dtype=float)
    for i in range(n):
        e = epoch_work.iloc[i]
        if pd.isna(e):
            continue
        s = df["Ts Start Dealing"].iloc[i]
        if is_missing(s):
            deal_abs[i] = float(e) + dealing_med
            continue
        naive = naive_clock_ms_near_epoch(s, e)
        if pd.isna(naive):
            deal_abs[i] = float(e) + dealing_med
            continue
        corrected = naive - clock_offset
        delay = corrected - e
        # Pengecekan validasi sederhana
        if dealing_low <= delay <= dealing_high:
            deal_abs[i] = corrected
        else:
            deal_abs[i] = float(e) + dealing_med

    out["Ts Start Dealing"] = [
        "-" if not np.isfinite(deal_abs[i]) else format_clock_ms(deal_abs[i], clock_offset, True)
        for i in range(n)
    ]

    # 4. Ts Gr
    gr_abs = np.full(n, np.nan, dtype=float)
    for i in range(n):
        e = epoch_work.iloc[i]
        if pd.isna(e):
            continue

        # jika row punya raw Gr yang masuk akal, pakai raw Gr yang valid
        raw_gr = df["Ts Gr"].iloc[i]
        if not is_missing(raw_gr):
            raw_val = naive_clock_ms_near_epoch(raw_gr, e)
            if not pd.isna(raw_val):
                raw_abs = raw_val - clock_offset
                if raw_abs > e:
                    # cek apakah raw Gr terlalu besar dibanding model
                    mult = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
                    if pd.notna(mult) and mult > 0:
                        pred = deal_abs[i] + intercept + slope * math.log(float(mult))
                        if abs(raw_abs - pred) <= max(5000.0, 3.0 * flight_sigma):
                            gr_abs[i] = raw_abs
                            continue

        # fallback model
        mult = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
        if pd.notna(mult) and mult > 0:
            pred = deal_abs[i] + intercept + slope * math.log(float(mult))
            gr_abs[i] = pred
        else:
            # tanpa multiplier, gunakan next epoch atau median post-gap
            next_e = epoch_work.iloc[i + 1] if i + 1 < n else np.nan
            if np.isfinite(next_e):
                gr_abs[i] = next_e - 2400.0
            else:
                gr_abs[i] = deal_abs[i] + 2500.0

    out["Ts Gr"] = [
        "-" if not np.isfinite(gr_abs[i]) else format_clock_ms(gr_abs[i], clock_offset, True)
        for i in range(n)
    ]

    # 5. Pastikan output tetap 8 kolom asli
    out = out[REQUIRED_COLUMNS]

    return out, {
        "clock_offset_ms": float(clock_offset),
        "dealing_median_ms": float(dealing_med),
        "dealing_sigma_ms": float(dealing_sigma),
        "dealing_valid_low_ms": float(dealing_low),
        "dealing_valid_high_ms": float(dealing_high),
        "k_gap": float(k_gap),
        "base_gap_ms": float(base_gap),
        "gap_sigma_ms": float(gap_sigma),
        "flight_intercept_ms": float(intercept),
        "flight_slope_ms_per_ln_mult": float(slope),
        "flight_sigma_ms": float(flight_sigma),
        "flight_samples_used": int(n_used),
    }


def main():
    parser = argparse.ArgumentParser(description="Repair timestamp crash-game dan tulis MD tanpa flag.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Output CSV", required=True)
    parser.add_argument("--report", type=Path, help="JSON report", default=None)
    parser.add_argument("--md-output", type=Path, help="Output Markdown final tanpa flag", required=True)
    args = parser.parse_args()

    df = read_input(args.input)
    epoch_original = parse_epoch_series(df["Ts Epoch Starttime"])
    repaired, meta = repair_dataframe(df, epoch_original)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)

    repaired.to_csv(args.output, index=False)

    # write MD in exact source-like format with no extra columns
    write_markdown_like_source(repaired, args.input, args.md_output)

    report = {
        "input_file": str(args.input),
        "rows": int(len(df)),
        "model": meta,
        "notes": [
            "Semua output MD tetap 8 kolom seperti file sumber.",
            "Tidak ada flag/keterangan tambahan di file MD.",
            "Ts Gr diestimasi dari event model, agar tidak ikut delay WebSocket.",
            "Epoch server tetap diperlakukan sebagai anchor utama.",
        ],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV output : {args.output}")
    print(f"MD output  : {args.md_output}")
    if args.report:
        print(f"Report     : {args.report}")
    print("Model:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()