#!/usr/bin/env python3
"""
repair_ts.py

Perbaikan timestamp untuk tabel crash-game dengan Epoch server sebagai anchor.

Input:
  Markdown table seperti master_merged.md / raw1.md
  atau CSV dengan nama kolom yang sama.

Output default:
  CSV hasil repair
  JSON ringkasan kalibrasi + daftar gap Epoch yang tidak bisa diisi

Contoh:
  python repair_ts.py master_merged.md -o repaired_master.csv
  python repair_ts.py raw1.md -o repaired_raw1.csv
  python repair_ts.py master_merged.md -o repaired_master.csv --report master_report.json
  python repair_ts.py master_merged.md -o repaired_master.csv --write-md repaired_master.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# 1) KONFIGURASI
# ============================================================================
# Semua angka di sini sengaja dibuat eksplisit supaya gampang diaudit oleh
# LLM/manusia lain. Nilai fallback hanya dipakai bila data input tidak cukup
# untuk mengkalibrasi parameter dari file itu sendiri.
#
# TARGET_TOL_MS = 500:
#   Ini BUKAN klaim bahwa seluruh timestamp bisa akurat <500 ms. Ini adalah
#   batas error target yang dipakai untuk memberi label confidence dan menolak
#   backfill Epoch yang terlalu panjang. Dari pengujian yang diberikan,
#   satu K_FLIGHT tidak lolos 500 ms di seluruh range multiplier, terutama
#   multiplier besar.
TARGET_TOL_MS = 500

# Ts Starttime hanya punya presisi detik. Pada data contoh, residual terhadap
# Epoch membentuk rentang kira-kira [-999, 0] ms, yang konsisten dengan
# floor/truncate ke detik. Median quantization error-nya ~ -500 ms.
# Karena itu offset clock diestimasi sebagai:
#   median(start_clock - epoch) - (-500 ms)
# Dengan cara ini kita tidak salah menganggap -500 ms sebagai clock skew.
STARTTIME_ROUNDING_BIAS_MS = -500
STARTTIME_RESIDUAL_TOL_MS = 1500

# Sanity range ini berasal dari pola dealing normal pada file yang diberikan:
# sekitar 7.0--7.1 detik, dengan robust bound sekitar 6.0--8.5 detik.
# Estimator utama tetap data-driven; range ini hanya pagar pengaman agar
# reconnect/disconnect/AFK tidak ikut dijadikan pusat kalibrasi.
DEALING_SANITY_MIN_MS = 6000
DEALING_SANITY_MAX_MS = 8500
DEALING_MAD_Z = 3.5

# Fallback dari hasil pengujian yang diberikan user:
# K_FLIGHT dipakai untuk flight murni, BUKAN untuk Epoch->Epoch gap.
K_FLIGHT_FALLBACK = 0.0781

# K_GAP dipakai hanya untuk hubungan:
#   Epoch[n+1] - Epoch[n] ~= BASE_GAP + ln(mult[n]) / K_GAP
# Ini berbeda fase dari K_FLIGHT.
K_GAP_FALLBACK = 0.0830
BASE_GAP_FALLBACK_MS = 10050.0

# Post-crash gap = Epoch[n+1] - TsGr[n].
# Tidak ada ground truth absolut lain, jadi nilai ini dikalibrasi robust dari
# pasangan yang valid. 2400 ms hanya fallback bila sample terlalu sedikit.
POST_GAP_FALLBACK_MS = 2400.0
POST_GAP_MAX_SANITY_MS = 6000.0

# Leading/trailing block tidak dapat diisi secara absolut karena hanya
# memiliki satu/nihil Epoch anchor. Internal gap TIDAK dibatasi panjangnya:
# semuanya diisi untuk kebutuhan dataset uji. Target 500 ms dipakai sebagai
# metrik evaluasi, bukan sebagai syarat untuk melakukan backfill.
MAX_EPOCH_BACKFILL_ROUNDS_CAP = 15  # retained only for backward compatibility

# Jumlah sample minimum agar estimator lokal dianggap cukup stabil.
MIN_MODEL_SAMPLES = 30

# Binning residual flight. Binning dipakai sebagai koreksi kecil terhadap
# formula log, bukan mengganti definisi K_FLIGHT.
FLIGHT_LOG_BINS = np.log(
    np.array([1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 10000.0])
)
FLIGHT_BIN_MIN_SAMPLES = 20

# Single-K diketahui memburuk pada multiplier besar. Karena itu pengisian
# berbasis formula tanpa Epoch berikutnya tidak boleh diberi label HIGH
# confidence pada range ini.
HIGH_MULT_LOW_CONFIDENCE = 50.0


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


@dataclass
class Calibration:
    clock_offset_ms: float
    starttime_rounding_bias_ms: float
    starttime_residual_mad_ms: float
    dealing_median_ms: float
    dealing_sigma_ms: float
    dealing_low_ms: float
    dealing_high_ms: float
    k_flight: float
    flight_samples: int
    post_gap_median_ms: float
    post_gap_sigma_ms: float
    post_gap_samples: int
    k_gap: float
    base_gap_ms: float
    gap_sigma_ms: float
    gap_samples: int
    gap_samples_used: int


# ============================================================================
# 2) UTILITAS PARSING
# ============================================================================

TIME_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$"
)


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() in {"", "-"}


def parse_epoch_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(
        s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}),
        errors="coerce",
    )
    return out


def parse_clock_text(text: object) -> tuple[int, int, int, int] | None:
    if is_missing(text):
        return None
    m = TIME_RE.match(str(text))
    if not m:
        return None
    h = int(m.group(1))
    minute = int(m.group(2))
    sec = int(m.group(3))
    frac = (m.group(4) or "")
    ms = int((frac + "000")[:3])
    if h > 23 or minute > 59 or sec > 59:
        return None
    return h, minute, sec, ms


def naive_clock_ms_near_epoch(clock_text: object, epoch_ms: float) -> float:
    """
    Anggap string jam sebagai 'clock label' tanpa timezone.
    Coba tanggal -1/0/+1 hari, lalu pilih yang paling dekat ke Epoch.

    Ini sengaja tidak meng-hardcode WITA/UTC. Clock offset termasuk timezone
    ditangkap sebagai parameter kalibrasi. Ini mencegah parsing timezone yang
    salah menghasilkan delay palsu ~8 jam.
    """
    parsed = parse_clock_text(clock_text)
    if parsed is None or pd.isna(epoch_ms):
        return np.nan

    h, minute, sec, ms = parsed
    base = datetime.fromtimestamp(float(epoch_ms) / 1000.0, tz=timezone.utc).date()

    candidates = []
    for delta_days in (-1, 0, 1):
        d = base + timedelta(days=delta_days)
        dt = datetime(
            d.year, d.month, d.day, h, minute, sec, ms * 1000, tzinfo=timezone.utc
        )
        candidates.append(dt.timestamp() * 1000.0)

    return min(candidates, key=lambda x: abs(x - epoch_ms))


def format_clock_ms(abs_ms: float, clock_offset_ms: float, with_ms: bool) -> str:
    """
    Kembalikan timestamp dalam 'clock space' yang sama seperti input.

    Internal time disimpan sebagai UTC-like absolute ms.
    Agar output tetap berada pada zona/jam tampilan sumber, tambahkan kembali
    clock_offset yang telah dikalibrasi sebelum diformat.
    """
    if pd.isna(abs_ms):
        return "-"
    label_ms = float(abs_ms) + float(clock_offset_ms)
    sec_ms = int(math.floor(label_ms / 1000.0))
    milli = int(round(label_ms - sec_ms * 1000.0))

    # Normalisasi carry.
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


def robust_location(values: Iterable[float], z: float = 3.5, max_iter: int = 8):
    """
    Robust center memakai median + MAD iteratif.
    Tidak sensitif terhadap delay reconnect/AFK yang jauh dari cluster normal.
    """
    a = np.asarray(pd.Series(list(values)).dropna(), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0

    work = a.copy()
    for _ in range(max_iter):
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
    """
    Fit y = a + b*x secara robust dengan iteratively reweighted trimming.
    Return a, b, residual median, MAD, sigma, n_total, n_used.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], float)
    y = np.asarray(y[finite], float)
    n_total = len(x)
    if n_total < MIN_MODEL_SAMPLES:
        return None

    keep = np.ones(n_total, dtype=bool)
    a = b = np.nan
    for _ in range(10):
        a, b = np.polyfit(x[keep], y[keep], 1)[1], np.polyfit(x[keep], y[keep], 1)[0]
        resid = y - (a + b * x)
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med)))
        sigma = 1.4826 * mad
        threshold = max(250.0, 3.5 * sigma)
        new_keep = np.abs(resid - med) <= threshold
        if new_keep.sum() == keep.sum():
            break
        if new_keep.sum() < max(MIN_MODEL_SAMPLES, int(0.5 * n_total)):
            break
        keep = new_keep

    a = float(a)
    b = float(b)
    resid = y - (a + b * x)
    r = resid[keep]
    rmed = float(np.median(r))
    rmad = float(np.median(np.abs(r - rmed)))
    rsigma = float(1.4826 * rmad)
    return a, b, rmed, rmad, rsigma, n_total, int(keep.sum())


# ============================================================================
# 3) INPUT / OUTPUT
# ============================================================================

def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv"}:
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
            raise ValueError(f"Tidak menemukan markdown table valid di {path}")

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
    """
    Ambil seluruh isi sebelum header tabel dari file Markdown asli.
    Dengan ini output .md mempertahankan format/header file sumber.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("No | Ts Epoch Starttime"):
            return lines[:i]
    return []


def write_markdown_like_source(
    df: pd.DataFrame,
    source_path: Path,
    output_path: Path,
):
    """
    Output sengaja hanya memakai 8 kolom asli:
      No
      Ts Epoch Starttime
      Ts Starttime (Bukan Epoch)
      Ts Start Dealing
      Ts Gr
      Result Gr
      Id Game
      Status (Live/History)

    Tidak memasukkan Raw..., confidence, uncertainty, atau audit columns.
    Audit detail tetap tersedia melalui --report JSON.
    """
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

def write_markdown(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = [
        " | ".join(cols),
        " | ".join(["---"] * len(cols)),
    ]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:g}")
            else:
                vals.append(str(v))
        lines.append(" | ".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# 4) KALIBRASI
# ============================================================================

def infer_clock_offset(df: pd.DataFrame, epoch_original: pd.Series):
    residuals = []
    for e, s in zip(epoch_original, df["Ts Starttime (Bukan Epoch)"]):
        if pd.isna(e) or is_missing(s):
            continue
        naive = naive_clock_ms_near_epoch(s, e)
        if pd.isna(naive):
            continue
        residuals.append(naive - e)

    if len(residuals) < MIN_MODEL_SAMPLES:
        raise ValueError(
            "Sample Starttime↔Epoch terlalu sedikit untuk kalibrasi clock."
        )

    q0, q10, q50, q90, q100 = np.quantile(residuals, [0, 0.1, 0.5, 0.9, 1.0])

    # Deteksi pola floor-to-second secara sederhana.
    # Pada data contoh upper edge mendekati 0 ms dan lower edge mendekati -1000 ms.
    floor_like = (q0 <= -900.0) and (q100 <= 100.0)
    rounding_bias = STARTTIME_ROUNDING_BIAS_MS if floor_like else 0.0

    med_resid, mad_resid, _, _, _, _ = robust_location(
        np.asarray(residuals) - rounding_bias
    )
    clock_offset = float(med_resid)

    residual_corrected = np.asarray(residuals) - clock_offset
    residual_mad = float(
        np.median(np.abs(residual_corrected - np.median(residual_corrected)))
    )

    return (
        clock_offset,
        rounding_bias,
        residual_mad,
        {
            "n": len(residuals),
            "floor_like": bool(floor_like),
            "q0": float(q0),
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "q100": float(q100),
        },
    )


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

    if len(delays) < MIN_MODEL_SAMPLES:
        med = DEALING_SANITY_MIN_MS + 0.5 * (
            DEALING_SANITY_MAX_MS - DEALING_SANITY_MIN_MS
        )
        sigma = 500.0
        lo, hi = DEALING_SANITY_MIN_MS, DEALING_SANITY_MAX_MS
        return med, sigma, lo, hi, len(delays)

    d = np.asarray(delays, float)

    # Gunakan cluster sanity 6.0--8.5 s bila cukup banyak sample.
    normal_subset = d[(d >= DEALING_SANITY_MIN_MS) & (d <= DEALING_SANITY_MAX_MS)]
    seed = normal_subset if len(normal_subset) >= MIN_MODEL_SAMPLES else d

    med, mad, sigma, _, _, _ = robust_location(seed, z=DEALING_MAD_Z)

    # Bound akhir = robust bound + hard sanity fence.
    robust_lo = med - DEALING_MAD_Z * sigma
    robust_hi = med + DEALING_MAD_Z * sigma
    lo = max(DEALING_SANITY_MIN_MS, robust_lo)
    hi = min(DEALING_SANITY_MAX_MS, robust_hi)

    if lo >= hi:
        lo, hi = DEALING_SANITY_MIN_MS, DEALING_SANITY_MAX_MS

    return float(med), float(sigma), float(lo), float(hi), len(delays)


def calibrate_post_gap(
    df: pd.DataFrame,
    epoch_original: pd.Series,
    clock_offset_ms: float,
):
    vals = []
    n = len(df)
    for i in range(n - 1):
        e_next = epoch_original.iloc[i + 1]
        e_cur = epoch_original.iloc[i]
        s = df["Ts Gr"].iloc[i]
        if pd.isna(e_cur) or pd.isna(e_next) or is_missing(s):
            continue

        naive = naive_clock_ms_near_epoch(s, e_cur)
        if pd.isna(naive):
            continue
        gr_abs = naive - clock_offset_ms
        post = e_next - gr_abs
        if 0 < post <= POST_GAP_MAX_SANITY_MS:
            vals.append(post)

    if len(vals) < MIN_MODEL_SAMPLES:
        return POST_GAP_FALLBACK_MS, 500.0, len(vals)

    med, mad, sigma, _, _, _ = robust_location(vals, z=DEALING_MAD_Z)
    return float(med), float(sigma), len(vals)


def calibrate_gap_model(df: pd.DataFrame, epoch_original: pd.Series):
    x = []
    y = []
    n = len(df)

    for i in range(n - 1):
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

    if len(x) < MIN_MODEL_SAMPLES:
        return (
            K_GAP_FALLBACK,
            BASE_GAP_FALLBACK_MS,
            500.0,
            len(x),
            0,
        )

    fit = robust_line_fit(np.asarray(x), np.asarray(y))
    if fit is None:
        return (
            K_GAP_FALLBACK,
            BASE_GAP_FALLBACK_MS,
            500.0,
            len(x),
            0,
        )

    a, b, _, _, sigma, n_total, n_used = fit
    if b <= 0 or not np.isfinite(b):
        return (
            K_GAP_FALLBACK,
            BASE_GAP_FALLBACK_MS,
            500.0,
            n_total,
            n_used,
        )

    # y(ms) = BASE(ms) + [1000 / K(s^-1)] * ln(mult).
    # Jadi slope b(ms) = 1000 / K. Simpan K dalam satuan per-detik agar
    # konsisten dengan definisi K_GAP ~= 0.083 yang dipakai di luar script.
    k_gap = 1000.0 / b
    base_gap = a
    if not np.isfinite(k_gap) or not np.isfinite(base_gap):
        k_gap = K_GAP_FALLBACK
        base_gap = BASE_GAP_FALLBACK_MS

    return float(k_gap), float(base_gap), float(max(sigma, 1.0)), n_total, n_used


def calibrate_flight(
    df: pd.DataFrame,
    epoch_original: pd.Series,
    clock_offset_ms: float,
    dealing_low_ms: float,
    dealing_high_ms: float,
):
    mults = []
    flights = []

    for e, deal_s, gr_s, m in zip(
        epoch_original,
        df["Ts Start Dealing"],
        df["Ts Gr"],
        df["Result Gr"],
    ):
        if pd.isna(e) or is_missing(deal_s) or is_missing(gr_s) or pd.isna(m):
            continue
        m = float(m)
        if m <= 1.0:
            continue

        deal_naive = naive_clock_ms_near_epoch(deal_s, e)
        gr_naive = naive_clock_ms_near_epoch(gr_s, e)
        if pd.isna(deal_naive) or pd.isna(gr_naive):
            continue

        deal_abs = deal_naive - clock_offset_ms
        gr_abs = gr_naive - clock_offset_ms
        deal_delay = deal_abs - e
        flight = gr_abs - deal_abs

        if not (0 < flight < 300000):
            continue
        if not (dealing_low_ms <= deal_delay <= dealing_high_ms):
            continue

        mults.append(m)
        flights.append(flight)

    mults = np.asarray(mults, dtype=float)
    flights = np.asarray(flights, dtype=float)

    if len(mults) < MIN_MODEL_SAMPLES:
        return (
            K_FLIGHT_FALLBACK,
            5000.0,
            {},
            len(mults),
        )

    # K_FLIGHT didefinisikan per detik: ln(mult) / flight_seconds.
    per_row_k = np.log(mults) / (flights / 1000.0)
    per_row_k = per_row_k[np.isfinite(per_row_k) & (per_row_k > 0)]
    if len(per_row_k) >= MIN_MODEL_SAMPLES:
        k_flight = float(np.median(per_row_k))
    else:
        k_flight = K_FLIGHT_FALLBACK

    # Residual correction terhadap formula log.
    pred = np.log(mults) / k_flight * 1000.0
    residual = flights - pred

    corrections = {}
    for lo, hi in zip(FLIGHT_LOG_BINS[:-1], FLIGHT_LOG_BINS[1:]):
        mask = (np.log(mults) >= lo) & (np.log(mults) < hi)
        if mask.sum() < FLIGHT_BIN_MIN_SAMPLES:
            continue
        r = residual[mask]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma = float(max(1.4826 * mad, 50.0))
        corrections[(float(lo), float(hi))] = {
            "median_ms": med,
            "sigma_ms": sigma,
            "n": int(mask.sum()),
        }

    global_med = float(np.median(residual))
    global_mad = float(np.median(np.abs(residual - global_med)))
    global_sigma = float(max(1.4826 * global_mad, 500.0))

    corrections["_global"] = {
        "median_ms": global_med,
        "sigma_ms": global_sigma,
        "n": int(len(residual)),
    }

    return k_flight, global_sigma, corrections, len(mults)


def make_calibration(df: pd.DataFrame, epoch_original: pd.Series) -> tuple[Calibration, dict]:
    (
        clock_offset_ms,
        rounding_bias,
        start_mad,
        start_diag,
    ) = infer_clock_offset(df, epoch_original)

    (
        dealing_median,
        dealing_sigma,
        dealing_low,
        dealing_high,
        dealing_samples,
    ) = calibrate_dealing(df, epoch_original, clock_offset_ms)

    post_median, post_sigma, post_samples = calibrate_post_gap(
        df, epoch_original, clock_offset_ms
    )

    (
        k_gap,
        base_gap,
        gap_sigma,
        gap_samples,
        gap_used,
    ) = calibrate_gap_model(df, epoch_original)

    (
        k_flight,
        flight_sigma,
        flight_corrections,
        flight_samples,
    ) = calibrate_flight(
        df,
        epoch_original,
        clock_offset_ms,
        dealing_low,
        dealing_high,
    )

    cal = Calibration(
        clock_offset_ms=clock_offset_ms,
        starttime_rounding_bias_ms=rounding_bias,
        starttime_residual_mad_ms=start_mad,
        dealing_median_ms=dealing_median,
        dealing_sigma_ms=dealing_sigma,
        dealing_low_ms=dealing_low,
        dealing_high_ms=dealing_high,
        k_flight=k_flight,
        flight_samples=flight_samples,
        post_gap_median_ms=post_median,
        post_gap_sigma_ms=post_sigma,
        post_gap_samples=post_samples,
        k_gap=k_gap,
        base_gap_ms=base_gap,
        gap_sigma_ms=gap_sigma,
        gap_samples=gap_samples,
        gap_samples_used=gap_used,
    )

    return cal, {
        "starttime": start_diag,
        "flight_corrections": flight_corrections,
        "flight_global_sigma_ms": flight_sigma,
    }


# ============================================================================
# 5) REKONSTRUKSI EPOCH
# ============================================================================

def get_missing_runs(epoch_original: pd.Series):
    missing = epoch_original.isna().to_numpy()
    runs = []
    i = 0
    n = len(missing)

    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        runs.append((i, j - 1, j - i))
        i = j

    return runs


def backfill_internal_epochs(
    df: pd.DataFrame,
    epoch_original: pd.Series,
    cal: Calibration,
):
    """
    Isi SEMUA internal Epoch gap.

    Penting:
      - Leading/trailing gap tetap tidak dapat direkonstruksi secara absolut
        karena hanya punya satu/nihil server-Epoch anchor.
      - Internal gap selalu diisi walaupun panjangnya > batas 500 ms.
      - TARGET_TOL_MS hanya dipakai untuk mengukur/menandai ketidakpastian,
        bukan untuk membiarkan data bolong.

    Metode:
      1. Prediksi tiap step dengan K_GAP:
           gap ~= BASE_GAP + ln(multiplier) / K_GAP
      2. Bridge dari Epoch kiri ke Epoch kanan dengan koreksi linear.
         Dengan ini dua endpoint yang benar-benar berasal dari server Epoch
         tetap dipertahankan persis.
      3. Ketidakpastian internal dilaporkan terpisah; output final tetap
         memakai format file awal sehingga tidak menambah kolom audit.
    """
    epoch_work = epoch_original.copy()
    flags = [[] for _ in range(len(df))]
    uncertainty = np.full(len(df), np.nan, dtype=float)
    details = []

    for start, end, count in get_missing_runs(epoch_original):
        # Leading/trailing block memang tidak mempunyai dua anchor.
        if start == 0 or end == len(df) - 1:
            reason = (
                "LEADING_OR_TRAILING_EPOCH_GAP: tidak ada dua server-Epoch "
                "anchor; tidak diisi karena absolute time tidak teridentifikasi."
            )
            for i in range(start, end + 1):
                flags[i].append(reason)
            details.append({
                "start_index": start,
                "end_index": end,
                "count": count,
                "action": "unresolved",
                "reason": reason,
            })
            continue

        p = start - 1
        q = end + 1

        left_epoch = epoch_original.iloc[p]
        right_epoch = epoch_original.iloc[q]

        if pd.isna(left_epoch) or pd.isna(right_epoch):
            reason = "EPOCH_ANCHOR_MISSING: internal gap kehilangan anchor."
            for i in range(start, end + 1):
                flags[i].append(reason)
            details.append({
                "start_index": start,
                "end_index": end,
                "count": count,
                "action": "unresolved",
                "reason": reason,
            })
            continue

        # Multiplier untuk setiap transisi p->p+1 ... q-1->q.
        multipliers = pd.to_numeric(
            df["Result Gr"].iloc[p:q], errors="coerce"
        ).to_numpy(dtype=float)

        # Jika multiplier rusak/missing, gunakan median log-gap dari model
        # sebagai fallback untuk step tersebut. Ini hanya untuk membuat
        # rangkaian tetap terisi; endpoint kanan tetap mengikat hasil akhir.
        log_mult = np.log(np.where(
            np.isfinite(multipliers) & (multipliers > 0),
            multipliers,
            1.0
        ))

        steps = cal.base_gap_ms + log_mult / cal.k_gap

        # Guard numerik saja. Tidak ada lagi guard "terlalu panjang".
        steps = np.maximum(steps, 1000.0)

        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        raw_pred = float(left_epoch) + cumulative

        # Bridge endpoint:
        # raw_pred[-1] belum tentu sama dengan right_epoch karena model adalah
        # estimasi. Koreksi linear mendistribusikan selisih ke seluruh gap.
        endpoint_correction = float(right_epoch) - raw_pred[-1]
        n_steps = q - p

        # Besarnya koreksi endpoint adalah indikator langsung bahwa model
        # K_GAP tidak mampu menjelaskan interval tersebut dengan akurat.
        # Tetap diisi karena tujuan dataset ini adalah menghilangkan hole.
        model_error_abs = abs(endpoint_correction)

        for j, row_idx in enumerate(range(p + 1, q), start=1):
            bridge = endpoint_correction * (j / n_steps)
            predicted = raw_pred[j] + bridge
            epoch_work.iloc[row_idx] = predicted

            # Ini bukan confidence statistik. Ini hanya uncertainty proxy
            # untuk audit: gabungan residual model dan posisi terhadap anchor.
            position_factor = min(j, n_steps - j) / max(n_steps, 1)
            unc = cal.gap_sigma_ms * math.sqrt(max(j, 1))
            unc += model_error_abs * position_factor

            uncertainty[row_idx] = float(unc)

            if unc <= TARGET_TOL_MS:
                quality = "WITHIN_500MS_PROXY"
            else:
                quality = "OVER_500MS_PROXY"

            flags[row_idx].append(
                f"EPOCH_BACKFILL_INTERNAL({count}_ROWS);{quality};"
                f"uncertainty_proxy≈{unc:.0f}ms"
            )

        details.append({
            "start_index": start,
            "end_index": end,
            "count": count,
            "action": "backfilled",
            "endpoint_correction_ms": endpoint_correction,
            "endpoint_correction_abs_ms": model_error_abs,
            "target_tolerance_ms": TARGET_TOL_MS,
            "note": (
                "Gap tetap diisi walaupun uncertainty proxy > target. "
                "Target 500 ms adalah evaluasi akurasi, bukan fill gate."
            ),
        })

    return epoch_work, flags, uncertainty, details


# ============================================================================
# 6) FLIGHT MODEL
# ============================================================================

def flight_prediction(
    mult: float,
    k_flight: float,
    flight_corrections: dict,
) -> tuple[float, float]:
    if not np.isfinite(mult) or mult <= 1.0:
        return 0.0, 1000.0

    x = math.log(mult)
    base = x / k_flight * 1000.0

    correction = None
    sigma = flight_corrections.get("_global", {}).get("sigma_ms", 5000.0)

    for key, val in flight_corrections.items():
        if key == "_global":
            continue
        lo, hi = key
        if lo <= x < hi:
            correction = val["median_ms"]
            sigma = val["sigma_ms"]
            break

    if correction is None:
        correction = flight_corrections.get("_global", {}).get("median_ms", 0.0)

    return float(base + correction), float(sigma)


# ============================================================================
# 7) REPAIR PER KOLOM
# ============================================================================

def parse_all_absolute_times(
    df: pd.DataFrame,
    epoch_reference: pd.Series,
    clock_offset_ms: float,
):
    out = {
        "start_abs": np.full(len(df), np.nan, dtype=float),
        "deal_abs": np.full(len(df), np.nan, dtype=float),
        "gr_abs": np.full(len(df), np.nan, dtype=float),
    }

    mapping = {
        "start_abs": "Ts Starttime (Bukan Epoch)",
        "deal_abs": "Ts Start Dealing",
        "gr_abs": "Ts Gr",
    }

    for i in range(len(df)):
        ref = epoch_reference.iloc[i]
        if pd.isna(ref):
            continue
        for dest, src in mapping.items():
            if is_missing(df[src].iloc[i]):
                continue
            naive = naive_clock_ms_near_epoch(df[src].iloc[i], ref)
            if pd.isna(naive):
                continue
            out[dest][i] = naive - clock_offset_ms

    return out


def repair_rows(
    df: pd.DataFrame,
    epoch_original: pd.Series,
    epoch_work: pd.Series,
    cal: Calibration,
    cal_extra: dict,
    epoch_flags: list[list[str]],
    epoch_uncertainty: np.ndarray,
):
    n = len(df)

    times = parse_all_absolute_times(
        df,
        epoch_work,
        cal.clock_offset_ms,
    )
    start_abs = times["start_abs"]
    deal_abs = times["deal_abs"]
    gr_abs = times["gr_abs"]

    flight_corrections = cal_extra["flight_corrections"]

    repair_reason = [[] for _ in range(n)]
    repair_confidence = ["ORIGINAL" for _ in range(n)]

    for i in range(n):
        repair_reason[i].extend(epoch_flags[i])

    # ------------------------
    # Starttime
    # ------------------------
    for i in range(n):
        e = epoch_work.iloc[i]
        if pd.isna(e):
            repair_reason[i].append("STARTTIME_UNRESOLVED_NO_EPOCH")
            continue

        original_start = start_abs[i]
        if np.isfinite(original_start):
            residual = original_start - e
            # Karena input starttime presisi detik, residual normal berada
            # sekitar [-1000, 0] setelah clock calibration.
            if not (-STARTTIME_RESIDUAL_TOL_MS <= residual <= STARTTIME_RESIDUAL_TOL_MS):
                repair_reason[i].append(
                    f"STARTTIME_REPAIRED_CLOCK_RESIDUAL({residual:.0f}ms)"
                )
                repair_confidence[i] = "HIGH"
            else:
                # Tetap normalisasi dari Epoch agar konsisten.
                pass

    # ------------------------
    # Dealing
    # ------------------------
    for i in range(n):
        e = epoch_work.iloc[i]
        if pd.isna(e):
            repair_reason[i].append("DEALING_UNRESOLVED_NO_EPOCH")
            continue

        current = deal_abs[i]
        valid = np.isfinite(current)
        if valid:
            delay = current - e
            if not (
                cal.dealing_low_ms <= delay <= cal.dealing_high_ms
            ):
                valid = False
                repair_reason[i].append(
                    f"DEALING_DELAY_OUTLIER({delay:.0f}ms)"
                )

        next_e = epoch_work.iloc[i + 1] if i + 1 < n else np.nan
        if valid and np.isfinite(next_e) and current >= next_e:
            valid = False
            repair_reason[i].append("DEALING_AFTER_NEXT_EPOCH")

        if not valid:
            deal_abs[i] = e + cal.dealing_median_ms
            repair_reason[i].append(
                f"DEALING_REPAIRED_FROM_EPOCH(+{cal.dealing_median_ms:.0f}ms)"
            )
            if repair_confidence[i] == "ORIGINAL":
                repair_confidence[i] = (
                    "HIGH" if cal.dealing_sigma_ms <= TARGET_TOL_MS else "MEDIUM"
                )

    # ------------------------
    # Gr
    # ------------------------
    for i in range(n):
        e = epoch_work.iloc[i]
        mult = pd.to_numeric(df["Result Gr"].iloc[i], errors="coerce")
        current = gr_abs[i]
        next_e = epoch_work.iloc[i + 1] if i + 1 < n else np.nan
        prev_gr = gr_abs[i - 1] if i > 0 else np.nan

        valid = np.isfinite(current)

        if valid and np.isfinite(deal_abs[i]) and current <= deal_abs[i]:
            valid = False
            repair_reason[i].append("GR_NOT_AFTER_DEALING")

        if valid and np.isfinite(prev_gr) and current <= prev_gr:
            # Ini menangkap tag yang arrive bersamaan/tidak monotonic.
            valid = False
            repair_reason[i].append("GR_NON_INCREASING_OR_COLLISION")

        if valid and np.isfinite(next_e):
            post = next_e - current
            if post <= 0:
                valid = False
                repair_reason[i].append("GR_AFTER_NEXT_EPOCH")
            elif post > max(
                POST_GAP_MAX_SANITY_MS,
                cal.post_gap_median_ms + DEALING_MAD_Z * cal.post_gap_sigma_ms,
            ):
                # Post gap yang terlalu panjang sering menunjukkan timestamp
                # receive terlambat. Kita punya server-Epoch berikutnya,
                # sehingga backward repair lebih kuat daripada formula flight.
                valid = False
                repair_reason[i].append(
                    f"GR_POST_GAP_OUTLIER({post:.0f}ms)"
                )

        if not valid:
            estimate = np.nan
            method = None
            confidence = "LOW"

            # Prioritas 1: server Epoch berikutnya -> backward anchor.
            if np.isfinite(next_e):
                estimate = next_e - cal.post_gap_median_ms
                method = "GR_BACKWARD_FROM_NEXT_EPOCH"
                confidence = (
                    "HIGH"
                    if cal.post_gap_sigma_ms <= TARGET_TOL_MS
                    else "MEDIUM"
                )

                # Jangan membuat Gr sebelum dealing.
                if np.isfinite(deal_abs[i]) and estimate <= deal_abs[i]:
                    estimate = np.nan
                    method = None

            # Prioritas 2: dealing + flight model.
            if not np.isfinite(estimate) and np.isfinite(deal_abs[i]):
                if np.isfinite(mult) and mult > 1.0:
                    flight_ms, flight_sigma = flight_prediction(
                        float(mult), cal.k_flight, flight_corrections
                    )
                    estimate = deal_abs[i] + flight_ms
                    method = "GR_FORWARD_DEALING_PLUS_FLIGHT_MODEL"
                    confidence = (
                        "HIGH"
                        if flight_sigma <= TARGET_TOL_MS
                        and float(mult) < HIGH_MULT_LOW_CONFIDENCE
                        else "LOW"
                    )

            if np.isfinite(estimate):
                gr_abs[i] = estimate
                repair_reason[i].append(method)
                repair_confidence[i] = confidence

    # ------------------------
    # Starttime/Dealing/Gr final absolute -> source clock labels
    # ------------------------
    out = df.copy()

    # Simpan raw sebagai audit trail.
    out.insert(
        out.columns.get_loc("Ts Epoch Starttime"),
        "Raw Ts Epoch Starttime",
        df["Ts Epoch Starttime"].astype(str),
    )
    raw_start_col_pos = out.columns.get_loc("Ts Starttime (Bukan Epoch)")
    out.insert(
        raw_start_col_pos,
        "Raw Ts Starttime (Bukan Epoch)",
        df["Ts Starttime (Bukan Epoch)"].astype(str),
    )
    raw_deal_pos = out.columns.get_loc("Ts Start Dealing")
    out.insert(
        raw_deal_pos,
        "Raw Ts Start Dealing",
        df["Ts Start Dealing"].astype(str),
    )
    raw_gr_pos = out.columns.get_loc("Ts Gr")
    out.insert(
        raw_gr_pos,
        "Raw Ts Gr",
        df["Ts Gr"].astype(str),
    )

    # Epoch diperbaiki hanya pada internal gap yang lolos guard.
    out["Ts Epoch Starttime"] = [
        "-" if pd.isna(v) else str(int(round(float(v))))
        for v in epoch_work
    ]

    out["Ts Starttime (Bukan Epoch)"] = [
        "-"
        if pd.isna(epoch_work.iloc[i])
        else format_clock_ms(epoch_work.iloc[i], cal.clock_offset_ms, False)
        for i in range(n)
    ]

    out["Ts Start Dealing"] = [
        "-"
        if not np.isfinite(deal_abs[i])
        else format_clock_ms(deal_abs[i], cal.clock_offset_ms, True)
        for i in range(n)
    ]

    out["Ts Gr"] = [
        "-"
        if not np.isfinite(gr_abs[i])
        else format_clock_ms(gr_abs[i], cal.clock_offset_ms, True)
        for i in range(n)
    ]

    out["Ts Repair Reason"] = [
        "; ".join(dict.fromkeys(x)) if x else ""
        for x in repair_reason
    ]
    out["Ts Repair Confidence"] = repair_confidence
    out["Epoch Repair Uncertainty ms"] = [
        round(float(x), 1) if np.isfinite(x) else ""
        for x in epoch_uncertainty
    ]

    # Confidence row-level dinaikkan menjadi MEDIUM bila ada perubahan Epoch
    # tapi tidak ada confidence assignment sebelumnya.
    for i in range(n):
        if "EPOCH_BACKFILL_INTERNAL" in out.at[i, "Ts Repair Reason"]:
            if out.at[i, "Ts Repair Confidence"] == "ORIGINAL":
                u = epoch_uncertainty[i]
                out.at[i, "Ts Repair Confidence"] = (
                    "HIGH" if np.isfinite(u) and u <= TARGET_TOL_MS else "MEDIUM"
                )

    return out


# ============================================================================
# 8) REPORT + CLI
# ============================================================================

def build_report(
    input_path: Path,
    df_original: pd.DataFrame,
    repaired: pd.DataFrame,
    cal: Calibration,
    cal_extra: dict,
    epoch_details: list[dict],
):
    raw_cols = [
        "Ts Epoch Starttime",
        "Ts Starttime (Bukan Epoch)",
        "Ts Start Dealing",
        "Ts Gr",
    ]

    changed_counts = {}
    for col in raw_cols:
        if col.startswith("Ts "):
            raw_col = {
                "Ts Epoch Starttime": "Raw Ts Epoch Starttime",
                "Ts Starttime (Bukan Epoch)": "Raw Ts Starttime (Bukan Epoch)",
                "Ts Start Dealing": "Raw Ts Start Dealing",
                "Ts Gr": "Raw Ts Gr",
            }[col]
            changed_counts[col] = int(
                (repaired[col].astype(str) != repaired[raw_col].astype(str)).sum()
            )

    unresolved = [
        d for d in epoch_details if d.get("action") == "unresolved"
    ]

    return {
        "input_file": str(input_path),
        "rows": int(len(df_original)),
        "unique_game_ids": int(df_original["Id Game"].nunique()),
        "calibration": asdict(cal),
        "starttime_diagnostics": cal_extra["starttime"],
        "flight_corrections": {
            ("global" if key == "_global" else f"{key[0]:.6f}:{key[1]:.6f}"): val
            for key, val in cal_extra["flight_corrections"].items()
        },
        "epoch_backfill_details": epoch_details,
        "unresolved_epoch_blocks": unresolved,
        "changed_counts": changed_counts,
        "design_notes": [
            "Epoch server diperlakukan sebagai ground truth.",
            "K_GAP hanya dipakai untuk Epoch[n+1]-Epoch[n].",
            "K_FLIGHT hanya dipakai untuk flight dari Dealing ke Gr.",
            "Leading/trailing History tanpa dua Epoch anchor tidak diisi.",
            "Semua internal Epoch gap diisi; target 500 ms hanya menjadi metrik uncertainty, bukan fill gate.",
            "Gr yang memiliki next Epoch anchor diprioritaskan dari arah belakang (next Epoch - post-gap median).",
            "Multiplier besar tidak diberi HIGH confidence dari single-K flight model.",
            "Ts Starttime dinormalisasi dari Epoch karena presisinya hanya detik.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Repair timestamp crash-game berbasis server Epoch + robust calibration."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output. Default: <input_stem>_repaired.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report. Default: <input_stem>_repair_report.json",
    )
    parser.add_argument(
        "--write-md",
        type=Path,
        default=None,
        help="Optional repaired markdown table.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=None,
        help="Write final output in the exact 8-column Markdown format of the source.",
    )
    args = parser.parse_args()

    df = read_input(args.input)
    epoch_original = parse_epoch_series(df["Ts Epoch Starttime"])

    cal, cal_extra = make_calibration(df, epoch_original)

    epoch_work, epoch_flags, epoch_uncertainty, epoch_details = backfill_internal_epochs(
        df,
        epoch_original,
        cal,
    )

    repaired = repair_rows(
        df,
        epoch_original,
        epoch_work,
        cal,
        cal_extra,
        epoch_flags,
        epoch_uncertainty,
    )

    output = args.output or args.input.with_name(
        args.input.stem + "_repaired.csv"
    )
    report_path = args.report or args.input.with_name(
        args.input.stem + "_repair_report.json"
    )

    repaired.to_csv(output, index=False)
    report = build_report(
        args.input,
        df,
        repaired,
        cal,
        cal_extra,
        epoch_details,
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.write_md:
        write_markdown_like_source(df=repaired, source_path=args.input, output_path=args.write_md)

    if args.md_output:
        write_markdown_like_source(df=repaired, source_path=args.input, output_path=args.md_output)

    # Ringkasan console untuk audit cepat.
    print("=" * 72)
    print("TIMESTAMP REPAIR")
    print("=" * 72)
    print(f"Input                     : {args.input}")
    print(f"Output CSV                : {output}")
    print(f"Report JSON               : {report_path}")
    if args.write_md:
        print(f"Output Markdown           : {args.write_md}")
    print()
    print("CALIBRATION")
    print(f"clock_offset_ms           : {cal.clock_offset_ms:.1f}")
    print(f"start rounding bias ms    : {cal.starttime_rounding_bias_ms:.1f}")
    print(f"dealing median ms         : {cal.dealing_median_ms:.1f}")
    print(f"dealing sigma ms          : {cal.dealing_sigma_ms:.1f}")
    print(f"dealing valid band ms     : {cal.dealing_low_ms:.1f} .. {cal.dealing_high_ms:.1f}")
    print(f"K_FLIGHT                  : {cal.k_flight:.8f}")
    print(f"K_GAP                     : {cal.k_gap:.8f}")
    print(f"BASE_GAP ms               : {cal.base_gap_ms:.1f}")
    print(f"gap sigma ms              : {cal.gap_sigma_ms:.1f}")
    print(f"post-gap median ms        : {cal.post_gap_median_ms:.1f}")
    print(f"post-gap sigma ms         : {cal.post_gap_sigma_ms:.1f}")
    print(f"internal Epoch gaps       : SEMUA internal gap diisi")
    print(f"target akurasi             : {TARGET_TOL_MS} ms (audit, bukan fill gate)")
    print()
    print("CHANGES")
    for col, count in report["changed_counts"].items():
        print(f"{col:<28}: {count}")
    print()
    unresolved = report["unresolved_epoch_blocks"]
    print(f"unresolved Epoch blocks   : {len(unresolved)}")
    for block in unresolved:
        print(
            f"  - rows {block['start_index']}..{block['end_index']} "
            f"({block['count']} rounds): {block['reason']}"
        )
    print()
    print("Catatan: angka confidence adalah guard audit, bukan jaminan statistik.")
    print("=" * 72)


if __name__ == "__main__":
    main()
