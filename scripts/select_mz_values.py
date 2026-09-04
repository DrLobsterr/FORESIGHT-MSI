#!/usr/bin/env python3
"""Rank candidate ions from imzML files by signal quality."""

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser


def calculate_scores(
    imzml_path: Path,
    top_k: int = 200,
    sample_pixels: int = 500,
    seed: int = 42,
    mz_stride: int = 10,
    tolerance: float = 0.02,
) -> Optional[pd.DataFrame]:
    parser = ImzMLParser(str(imzml_path))
    coordinates = list(parser.coordinates)
    rng = random.Random(seed)
    indices = rng.sample(range(len(coordinates)), min(sample_pixels, len(coordinates)))
    spectra = [parser.getspectrum(index) for index in indices]
    if not spectra:
        return None

    reference_mz = spectra[0][0]
    mz_arrays = [spectrum[0] for spectrum in spectra]
    intensity_arrays = [spectrum[1] for spectrum in spectra]
    rows = []
    for mz_index in range(0, len(reference_mz), mz_stride):
        mz_value = reference_mz[mz_index]
        values = []
        for mzs, intensities in zip(mz_arrays, intensity_arrays):
            index = int(np.searchsorted(mzs, mz_value))
            candidates = []
            if index < len(mzs):
                candidates.append((mzs[index], intensities[index]))
            if index > 0:
                candidates.append((mzs[index - 1], intensities[index - 1]))
            if candidates:
                nearest_mz, intensity = min(
                    candidates, key=lambda item: abs(item[0] - mz_value)
                )
                if abs(nearest_mz - mz_value) <= tolerance:
                    values.append(intensity)
        if len(values) < 10:
            continue
        values = np.asarray(values)
        mean_intensity = float(np.mean(values))
        coverage = float(np.mean(values > mean_intensity * 0.10))
        background = float(np.percentile(values, 10))
        snr = mean_intensity / (background + 1e-8)
        rows.append(
            {
                "mz_value": float(mz_value),
                "quality_score": mean_intensity * coverage * np.log1p(snr),
                "mean_intensity": mean_intensity,
                "coverage": coverage,
                "sn_ratio": snr,
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("quality_score", ascending=False).head(top_k)
    return frame.sort_values("mz_value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("imzml", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--sample-pixels", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.imzml:
        scores = calculate_scores(
            path, args.top_k, args.sample_pixels, args.seed
        )
        if scores is None:
            print("No candidates: {}".format(path))
            continue
        output = args.output_dir / "{}_selected_ions.csv".format(path.stem)
        scores.to_csv(output, index=False, float_format="%.6f")
        print(output)


if __name__ == "__main__":
    main()
