#!/usr/bin/env python3
"""Strict timestamp repair dengan batas toleransi eksplisit.

Script ini memakai engine pada ``test_kalibrasi.py``, tetapi menerapkan aturan
strict: hasil estimasi yang uncertainty-nya lebih besar dari 500 ms tidak
dipresentasikan sebagai timestamp yang akurat. Nilai tersebut dikosongkan pada
kolom timestamp hasil repair dan tetap dicatat di kolom audit.

Contoh:
    python repair_ts_strict.py raw1.md -o repaired.csv --report report.json

Catatan penting:
- Ts Epoch yang sudah ada diperlakukan sebagai server ground truth.
- Epoch internal hanya dipertahankan jika uncertainty proxy <= tolerance.
- Leading/trailing gap dan hasil estimasi di luar tolerance tidak dipaksa diisi.
- Timestamp penerimaan WebSocket tidak dapat diubah menjadi event time absolut
  tanpa server timestamp atau pengukuran latency tambahan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse parser, calibration, and model from the existing implementation.
from test_kalibrasi import (
    TARGET_TOL_MS,
    build_report,
    make_calibration,
    parse_epoch_series,
    read_input,
    repair_rows,
    backfill_internal_epochs,
)


def apply_strict_gate(repaired, epoch_original, epoch_work, uncertainty, tolerance_ms):
    """Mark and hide estimates that cannot be defended within the tolerance."""
    out = repaired.copy()
    n = len(out)
    quality = []

    timestamp_columns = [
        "Ts Epoch Starttime",
        "Ts Starttime (Bukan Epoch)",
        "Ts Start Dealing",
        "Ts Gr",
    ]

    for i in range(n):
        reasons = str(out.at[i, "Ts Repair Reason"] or "")
        original_epoch = epoch_original.iloc[i]
        estimated_epoch = epoch_work.iloc[i]
        unc = uncertainty[i]

        if np.isfinite(original_epoch):
            row_quality = "SERVER_EPOCH"
        elif np.isfinite(estimated_epoch) and np.isfinite(unc) and unc <= tolerance_ms:
            row_quality = "REPAIRED_WITHIN_TOLERANCE"
        elif np.isfinite(estimated_epoch):
            row_quality = "ESTIMATED_OVER_TOLERANCE"
        else:
            row_quality = "UNRESOLVED"

        # Any forward flight-model estimate is not a server timestamp. It may be
        # retained only when the model itself reports <= tolerance.
        if "GR_FORWARD_DEALING_PLUS_FLIGHT_MODEL" in reasons:
            if out.at[i, "Ts Repair Confidence"] != "HIGH":
                row_quality = "ESTIMATED_OVER_TOLERANCE"

        if row_quality in {"ESTIMATED_OVER_TOLERANCE", "UNRESOLVED"}:
            for column in timestamp_columns:
                if column in out.columns:
                    out.at[i, column] = "-"
            out.at[i, "Ts Repair Reason"] = (
                reasons + "; STRICT_GATE_REJECTED_OVER_500MS"
            ).strip("; ")

        quality.append(row_quality)

    out["Ts Strict Quality"] = quality
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict WebSocket timestamp repair berbasis server Epoch."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=TARGET_TOL_MS,
        help="Batas maksimum uncertainty hasil estimasi, default 500 ms.",
    )
    parser.add_argument(
        "--write-md",
        type=Path,
        default=None,
        help="Opsional: tulis Markdown 8 kolom seperti input.",
    )
    args = parser.parse_args()

    if args.tolerance_ms <= 0:
        parser.error("--tolerance-ms harus lebih besar dari nol")

    df = read_input(args.input)
    epoch_original = parse_epoch_series(df["Ts Epoch Starttime"])

    cal, cal_extra = make_calibration(df, epoch_original)
    epoch_work, epoch_flags, epoch_uncertainty, epoch_details = (
        backfill_internal_epochs(df, epoch_original, cal)
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
    repaired = apply_strict_gate(
        repaired,
        epoch_original,
        epoch_work,
        epoch_uncertainty,
        args.tolerance_ms,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(args.output, index=False)

    report = build_report(
        args.input,
        df,
        repaired,
        cal,
        cal_extra,
        epoch_details,
    )
    report["strict_mode"] = True
    report["strict_tolerance_ms"] = args.tolerance_ms
    report["strict_quality_counts"] = {
        str(k): int(v)
        for k, v in repaired["Ts Strict Quality"].value_counts().items()
    }
    report["design_notes"].extend([
        "Strict gate mengosongkan estimasi dengan uncertainty proxy di atas tolerance.",
        "Timestamp hasil penerimaan WebSocket tidak dianggap server event time tanpa anchor server.",
    ])
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.write_md:
        # Import lazily so the main engine remains the single Markdown writer.
        from test_kalibrasi import write_markdown_like_source

        write_markdown_like_source(repaired, args.input, args.write_md)

    print(f"Output : {args.output}")
    print(f"Report : {args.report}")
    print(f"Tolerance : {args.tolerance_ms:.0f} ms")
    print(json.dumps(report["strict_quality_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
