#!/usr/bin/env python3
"""Plot wall-clock compute time for the PD/PD-FS ablation runs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffmsr")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TimeSpec:
    experiment: str
    label: str
    value: int
    stage1_log: str | None
    stage2_log: str
    test_log: str


LATENT_SPECS = [
    TimeSpec(
        "latent",
        "64",
        64,
        "logs/diffmsr_prior64_stage1_100k.log",
        "logs/diffmsr_prior64_stage2_100k.log",
        "logs/test_pd_fs_test_prior64_100k.log",
    ),
    TimeSpec(
        "latent",
        "128",
        128,
        "logs/diffmsr_prior128_stage1_100k.log",
        "logs/diffmsr_prior128_stage2_100k.log",
        "logs/test_pd_fs_test_prior128_100k.log",
    ),
    TimeSpec(
        "latent",
        "256",
        256,
        "logs/diffmsr_prior256_stage1_100k.log",
        "logs/diffmsr_prior256_stage2_100k.log",
        "logs/test_pd_fs_test_prior256_100k.log",
    ),
    TimeSpec(
        "latent",
        "512",
        512,
        "logs/diffmsr_prior512_stage1_100k.log",
        "logs/diffmsr_prior512_stage2_100k.log",
        "logs/test_pd_fs_test_prior512_100k.log",
    ),
]


TIMESTEP_SPECS = [
    TimeSpec(
        "timesteps",
        "1",
        1,
        None,
        "logs/diffmsr_steps1_100k_gpu3.log",
        "logs/test_pd_fs_test_steps1_100k.log",
    ),
    TimeSpec(
        "timesteps",
        "2",
        2,
        None,
        "logs/diffmsr_steps2_100k_gpu3.log",
        "logs/test_pd_fs_test_steps2_100k.log",
    ),
    TimeSpec(
        "timesteps",
        "4",
        4,
        None,
        "logs/diffmsr_steps4_100k_gpu3.log",
        "logs/test_pd_fs_test_steps4_100k.log",
    ),
    TimeSpec(
        "timesteps",
        "6",
        6,
        None,
        "logs/diffmsr_steps6_100k_gpu3.log",
        "logs/test_pd_fs_test_steps6_100k.log",
    ),
    TimeSpec(
        "timesteps",
        "8",
        8,
        None,
        "logs/diffmsr_steps8_100k_gpu3.log",
        "logs/test_pd_fs_test_steps8_100k.log",
    ),
]


def read_text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8", errors="replace")


def parse_hms(raw: str) -> float:
    hours, minutes, seconds = [int(part) for part in raw.split(":")]
    return hours + minutes / 60.0 + seconds / 3600.0


def parse_train_hours(path: str | None) -> float:
    if path is None:
        return 0.0
    text = read_text(path)
    matches = re.findall(r"Time consumed:\s*(\d+:\d+:\d+)", text)
    if not matches:
        raise ValueError(f"No training duration found in {path}")
    return parse_hms(matches[-1])


def parse_host(path: str | None) -> str:
    if path is None:
        return ""
    match = re.search(r"^Host:\s*(.+)$", read_text(path), flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_wall_time(raw: str) -> datetime:
    raw = re.sub(r"\s+", " ", raw.strip())
    raw = re.sub(r"(\w{3} \w{3}) (\d) ", r"\1 0\2 ", raw)
    formats = [
        "%a %b %d %H:%M:%S IDT %Y",
        "%a %b %d %I:%M:%S %p IDT %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse timestamp: {raw}")


def parse_test_hours(path: str) -> float:
    text = read_text(path)
    started = re.search(r"^Started:\s*(.+)$", text, flags=re.MULTILINE)
    finished = re.search(r"^Finished:\s*(.+)$", text, flags=re.MULTILINE)
    if not started or not finished:
        raise ValueError(f"Could not find Started/Finished in {path}")
    delta = parse_wall_time(finished.group(1)) - parse_wall_time(started.group(1))
    return delta.total_seconds() / 3600.0


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in [*LATENT_SPECS, *TIMESTEP_SPECS]:
        stage1_hours = parse_train_hours(spec.stage1_log)
        stage2_hours = parse_train_hours(spec.stage2_log)
        test_hours = parse_test_hours(spec.test_log)
        rows.append(
            {
                "experiment": spec.experiment,
                "label": spec.label,
                "value": spec.value,
                "stage1_hours": round(stage1_hours, 4),
                "stage2_hours": round(stage2_hours, 4),
                "train_hours": round(stage1_hours + stage2_hours, 4),
                "test_hours": round(test_hours, 4),
                "total_hours": round(stage1_hours + stage2_hours + test_hours, 4),
                "stage1_host": parse_host(spec.stage1_log),
                "stage2_host": parse_host(spec.stage2_log),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


def plot_latent(rows: list[dict[str, object]], out_path: Path) -> None:
    data = [row for row in rows if row["experiment"] == "latent"]
    labels = [str(row["label"]) for row in data]
    stage1 = np.asarray([float(row["stage1_hours"]) for row in data])
    stage2 = np.asarray([float(row["stage2_hours"]) for row in data])
    test = np.asarray([float(row["test_hours"]) for row in data])
    total = stage1 + stage2 + test

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, stage1, label="Stage 1 train", color="#74a9cf")
    ax.bar(x, stage2, bottom=stage1, label="Stage 2 train", color="#2b6cb0")
    ax.bar(x, test, bottom=stage1 + stage2, label="Full valid test", color="#fdae6b")
    for idx, value in enumerate(total):
        ax.text(idx, value + 0.12, f"{value:.1f}h", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("prior_dim")
    ax.set_ylabel("wall-clock hours")
    ax.set_title("Latent Size Ablation Compute Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5,
        -0.2,
        "Wall-clock time; prior256 was run on a different host, so compare with caution.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def plot_timesteps(rows: list[dict[str, object]], out_path: Path) -> None:
    data = [row for row in rows if row["experiment"] == "timesteps"]
    labels = [str(row["label"]) for row in data]
    stage2 = np.asarray([float(row["stage2_hours"]) for row in data])
    test = np.asarray([float(row["test_hours"]) for row in data])
    total = stage2 + test

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, stage2, label="Stage 2 train", color="#2b6cb0")
    ax.bar(x, test, bottom=stage2, label="Full valid test", color="#fdae6b")
    for idx, value in enumerate(total):
        ax.text(idx, value + 0.08, f"{value:.1f}h", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("diffusion denoising steps")
    ax.set_ylabel("wall-clock hours")
    ax.set_title("Denoising Step Ablation Compute Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5,
        -0.2,
        "Stage 1 is shared for this ablation and is not included.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_outputs/pd_fs_ablation", help="Output directory.")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    write_csv(rows, out_dir / "compute_time_summary.csv")
    plot_latent(rows, out_dir / "compute_time_latent.png")
    plot_timesteps(rows, out_dir / "compute_time_timesteps.png")


if __name__ == "__main__":
    main()
