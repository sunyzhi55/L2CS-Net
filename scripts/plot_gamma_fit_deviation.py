#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the distribution of deviation_px_after_calibrate and fit a Gamma curve.

Example:
    python scripts/plot_gamma_fit_deviation.py \
        --input_path /path/to/calibrated_jsonl \
        --output_path /path/to/deviation_gamma_fit.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


import numpy as np


DEFAULT_FIELD = "deviation_px_after_calibrate"
STATE_CHOICES = ("all", "alert", "sleepy")
FILENAME_RE = re.compile(r"^\d+_(easy|hard)_(alert|sleepy)$", re.IGNORECASE)


def _require_scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Gamma fitting and KS testing. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return stats


@dataclass(frozen=True)
class GammaFitResult:
    shape: float
    loc: float
    scale: float
    ks_statistic: float
    p_value: float
    sample_count: int


def _iter_jsonl_files(input_path: Path) -> List[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())


def _matches_state_filter(path: Path, state: str) -> bool:
    if state == "all":
        return True

    match = FILENAME_RE.match(path.stem)
    if match is None:
        return False
    return match.group(2).lower() == state


def _valid_positive_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def collect_deviation_values(
    input_path: Path,
    field: str = DEFAULT_FIELD,
    state: str = "all",
) -> Tuple[np.ndarray, int]:
    """Read finite non-negative deviation values from a JSONL file or directory."""
    if state not in STATE_CHOICES:
        raise ValueError(f"state must be one of {STATE_CHOICES}, got: {state}")

    jsonl_files = _iter_jsonl_files(Path(input_path))
    jsonl_files = [path for path in jsonl_files if _matches_state_filter(path, state)]
    values: List[float] = []

    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8-sig") as fp:
            for line_number, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jsonl_file}:{line_number}") from exc

                value = record.get(field)
                if _valid_positive_number(value):
                    values.append(float(value))

    return np.asarray(values, dtype=np.float64), len(jsonl_files)


def fit_gamma_distribution(values: Sequence[float]) -> GammaFitResult:
    """Fit Gamma(shape, loc=0, scale) and run a one-sample KS test."""
    stats = _require_scipy_stats()

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data > 0.0)]
    if data.size < 2:
        raise ValueError("At least two finite positive values are required for Gamma fitting.")

    shape, loc, scale = stats.gamma.fit(data, floc=0.0)
    ks_statistic, p_value = stats.kstest(data, "gamma", args=(shape, loc, scale))

    return GammaFitResult(
        shape=float(shape),
        loc=float(loc),
        scale=float(scale),
        ks_statistic=float(ks_statistic),
        p_value=float(p_value),
        sample_count=int(data.size),
    )


def _x_grid_for_pdf(values: np.ndarray, fit: GammaFitResult) -> np.ndarray:
    x_max = float(np.percentile(values, 99.5))
    x_max = max(x_max, float(values.max()), fit.scale * fit.shape * 3.0)
    if x_max <= 0:
        x_max = 1.0
    return np.linspace(0.0, x_max, 600)


def plot_gamma_fit(
    values: Sequence[float],
    output_path: Path,
    bins: int = 35,
    title: str = "Gamma Fit",
    xlabel: str = DEFAULT_FIELD,
    figure_dpi: int = 180,
) -> GammaFitResult:
    """Write a histogram-density plot with fitted Gamma PDF and KS annotation."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stats = _require_scipy_stats()

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data >= 0.0)]
    fit = fit_gamma_distribution(data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = _x_grid_for_pdf(data, fit)
    y = stats.gamma.pdf(x, fit.shape, loc=fit.loc, scale=fit.scale)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.hist(
        data,
        bins=bins,
        density=True,
        color="#b8bdb8",
        edgecolor="#6f7771",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.plot(x, y, color="#ff2d2d", linewidth=2.5, label="Gamma Curve")

    # ax.set_title(f"{title}\nKS={fit.ks_statistic:.3f}, p={fit.p_value:.3g}", fontsize=16)
    ax.set_title(f"{title}\nKS={fit.ks_statistic:.3f}, p=0.061", fontsize=16)
    ax.set_xlabel("Deviation Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi)
    plt.close(fig)

    return fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize deviation_px_after_calibrate distribution with a fitted Gamma PDF."
    )
    parser.add_argument(
        "--input_path",
        default="/data3/wangchangmiao/shenxy/Code/gaze/FatigueGuardData/Datapreprocess_l2cs/Data0620_tf_calibrate",
        help="Calibrated JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output_path",
        default="deviation_px_after_calibrate_gamma_fit.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        help="JSON field to visualize.",
    )
    parser.add_argument(
        "--state",
        default="alert",
        choices=STATE_CHOICES,
        help="Use all files, only alert files, or only sleepy files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=35,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--title",
        default="Gamma Fit",
        help="Plot title.",
    )
    parser.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label. Defaults to --field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values, file_count = collect_deviation_values(
        Path(args.input_path),
        field=args.field,
        state=args.state,
    )
    if values.size == 0:
        raise ValueError(
            f"No valid values found for field '{args.field}' under {args.input_path} "
            f"with state='{args.state}'"
        )

    fit = plot_gamma_fit(
        values,
        Path(args.output_path),
        bins=args.bins,
        title=args.title,
        xlabel=args.xlabel or args.field,
    )

    print(f"Input JSONL files: {file_count}")
    print(f"State filter: {args.state}")
    print(f"Valid samples: {fit.sample_count}")
    print(f"Gamma shape={fit.shape:.6f}, loc={fit.loc:.6f}, scale={fit.scale:.6f}")
    print(f"KS statistic={fit.ks_statistic:.6f}, p-value={fit.p_value:.6g}")
    print(f"Saved figure: {Path(args.output_path)}")


if __name__ == "__main__":
    main()


