#!/usr/bin/env python3
"""Beta timestamp repair pipeline.

Nama sengaja memakai ``test``: ini baseline beta untuk diuji di environment,
bukan klaim bahwa seluruh timestamp client adalah ground truth.

Perbedaan utama dari test_kalibrasi.py:
- Unit K_GAP konsisten: base_gap_ms + (1000 / K_GAP) * ln(mult).
- Internal Epoch gap dihitung satu blok dari dua Epoch server asli.
- Existing server Epoch tidak pernah diubah.
- Edge gap boleh diekstrapolasi agar file MD lengkap.
- Ts Gr yang mustahil memakai backward estimate dari Epoch berikutnya melalui
  repair_rows(); flight model hanya fallback.
- Markdown output tetap delapan kolom asli tanpa flag.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from test_kalibrasi import (
    Calibration,
    REQUIRED_COLUMNS,
    backfill_internal_epochs,
    format_clock_ms,
    make_calibration,
    parse_epoch_series,
    read_input,
    repair_rows,
    write_markdown_like_source,
)


# ---------------------------------------------------------------------------
# Epoch reconstruction with explicit units and deterministic block bridging
# ---------------------------------------------------------------------------

def _steps_for_rows(df: pd.DataFrame, cal: Calibration) -> np.ndarray:
    mult = pd.to_numeric(df["Result Gr"], errors="coerce").to_numpy(float)
    mult = np.where(np.isfinite(mult) & (mult > 0), mult, 1.0)
    if not np.isfinite(cal.k_gap) or cal.k_gap <= 0:
        raise ValueError("K_GAP tidak valid")
    # K_GAP is s^-1; all timestamps here are milliseconds.
    steps = cal.base_gap_ms + (1000.0 / cal.k_gap) * np.log(mult)
    return np.maximum(steps, 1000.0)


def _missing_runs(epoch: pd.Series):
    missing = epoch.isna().to_numpy()
    runs = []
    i = 0
    while i < len(missing):
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < len(missing) and missing[i]:
            i += 1
        runs.append((start, i - 1))
    return runs


def backfill_epochs_beta(df: pd.DataFrame, epoch_original: pd.Series, cal: Calibration):
    """Fill every missing Epoch while preserving real anchors.

    Internal blocks use both original anchors and are solved in one pass.
    Leading/trailing blocks use one-sided extrapolation solely so the output
    dataset is complete; they are not server-ground-truth values.
    """
    out = epoch_original.copy().astype(float)
    steps = _steps_for_rows(df, cal)
    uncertainty = np.full(len(df), np.nan, dtype=float)
    details = []

    for start, end in _missing_runs(epoch_original):
        left = start - 1 if start > 0 and not pd.isna(epoch_original.iloc[start - 1]) else None
        right = end + 1 if end + 1 < len(df) and not pd.isna(epoch_original.iloc[end + 1]) else None
        count = end - start + 1

        if left is not None and right is not None:
            # transitions left -> right are steps[left:right]
            raw_steps = steps[left:right]
            cumulative = np.concatenate(([0.0], np.cumsum(raw_steps)))
            endpoint_error = float(epoch_original.iloc[right] - epoch_original.iloc[left] - cumulative[-1])
            total_steps = right - left

            for j, row in enumerate(range(start, end + 1), start=1):
                out.iloc[row] = (
                    float(epoch_original.iloc[left])
                    + float(cumulative[j])
                    + endpoint_error * j / total_steps
                )
                uncertainty[row] = abs(endpoint_error) * min(j, total_steps - j) / max(total_steps, 1)

            details.append({
                "start_index": start,
                "end_index": end,
                "count": count,
                "type": "internal_two_anchor_bridge",
                "endpoint_correction_ms": endpoint_error,
            })
            continue

        # One-sided extrapolation for a complete beta test file.
        if left is not None:
            for row in range(start, end + 1):
                out.iloc[row] = float(out.iloc[row - 1]) + steps[row - 1]
            direction = "forward_from_left_anchor"
        elif right is not None:
            for row in range(end, start - 1, -1):
                out.iloc[row] = float(out.iloc[row + 1]) - steps[row]
            direction = "backward_from_right_anchor"
        else:
            # No anchor anywhere: absolute time is unknowable; leave empty.
            direction = "unresolved_no_anchor"

        details.append({
            "start_index": start,
            "end_index": end,
            "count": count,
            "type": "one_sided_extrapolation" if direction != "unresolved_no_anchor" else direction,
        })

    return out, uncertainty, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beta repair timestamp berbasis Epoch server + two-anchor bridge."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--write-md", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    args = parser.parse_args()

    df = read_input(args.input)
    epoch_original = parse_epoch_series(df["Ts Epoch Starttime"])
    cal, cal_extra = make_calibration(df, epoch_original)

    epoch_work, epoch_uncertainty, epoch_details = backfill_epochs_beta(
        df, epoch_original, cal
    )

    # repair_rows keeps structurally valid measured Gr, repairs impossible or
    # missing Gr, and prioritizes next-Epoch backward repair before flight model.
    repaired = repair_rows(
        df=df,
        epoch_original=epoch_original,
        epoch_work=epoch_work,
        cal=cal,
        cal_extra=cal_extra,
        epoch_flags=[[] for _ in range(len(df))],
        epoch_uncertainty=epoch_uncertainty,
    )

    output = args.output or args.input.with_name(args.input.stem + "_repaired.csv")
    report_path = args.report or args.input.with_name(args.input.stem + "_repair_report.json")
    md_path = args.md_output or args.write_md

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output, index=False)

    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        # This deliberately writes only REQUIRED_COLUMNS; audit columns remain
        # in CSV/report and never leak into the clean Markdown test fixture.
        write_markdown_like_source(repaired, args.input, md_path)

    report = {
        "beta": True,
        "input_file": str(args.input),
        "rows": int(len(df)),
        "server_epoch_rows": int(epoch_original.notna().sum()),
        "filled_epoch_rows": int((epoch_original.isna() & epoch_work.notna()).sum()),
        "epoch_blocks_beta": epoch_details,
        "calibration": cal.__dict__,
        "calibration_extra": cal_extra,
        "notes": [
            "Server Epoch yang sudah tersedia dipertahankan.",
            "Internal gap dihitung sekaligus dari dua anchor Epoch asli.",
            "K_GAP memakai milidetik: base_gap_ms + (1000/K_GAP)*ln(mult).",
            "Ts Gr yang mustahil menggunakan next Epoch backward estimate sebelum flight fallback.",
            "Leading/trailing Epoch adalah extrapolation beta, bukan ground truth server.",
            "Markdown output hanya berisi delapan kolom sumber.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV    : {output}")
    print(f"Report : {report_path}")
    if md_path is not None:
        print(f"MD     : {md_path}")


if __name__ == "__main__":
    main()
