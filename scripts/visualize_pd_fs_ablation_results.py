#!/usr/bin/env python3
"""Plot PD/PD-FS ablation metrics and render side-by-side examples.

The script uses the standalone full-test outputs under results/:

  results/pd_fs_test_prior64_100k/
  results/pd_fs_test_prior128_100k/
  results/pd_fs_test_prior512_100k/
  results/pd_fs_test_steps1_100k/
  ...

It creates:
  - PSNR/SSIM plots for latent-size and diffusion-step ablations.
  - CSV summaries.
  - Side-by-side image grids for selected slices.

Example selection is based on the slices where model reconstructions differ
most, measured by the per-slice PSNR spread across the models in each ablation.
The result .mat files are large, so they are loaded one model at a time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffmsr")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.data.data_util import paired_paths_from_folder  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    experiment: str
    label: str
    value: int
    run_name: str


LATENT_MODELS = [
    ModelSpec("latent", "64", 64, "pd_fs_test_prior64_100k"),
    ModelSpec("latent", "128", 128, "pd_fs_test_prior128_100k"),
    ModelSpec("latent", "256", 256, "pd_fs_test_prior256_100k"),
    ModelSpec("latent", "512", 512, "pd_fs_test_prior512_100k"),
]

STEP_MODELS = [
    ModelSpec("timesteps", "1", 1, "pd_fs_test_steps1_100k"),
    ModelSpec("timesteps", "2", 2, "pd_fs_test_steps2_100k"),
    ModelSpec("timesteps", "4", 4, "pd_fs_test_steps4_100k"),
    ModelSpec("timesteps", "6", 6, "pd_fs_test_steps6_100k"),
    ModelSpec("timesteps", "8", 8, "pd_fs_test_steps8_100k"),
]

BASELINE_MODEL = ModelSpec("baseline", "baseline 256", 256, "pd_fs_test_baseline_prior256_500k")


def result_mat_path(run_name: str) -> Path:
    paths = sorted((PROJECT_ROOT / "results" / run_name / "visualization").glob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No result .mat found for {run_name}")
    if len(paths) > 1:
        print(f"Warning: multiple .mat files for {run_name}; using {paths[-1]}")
    return paths[-1]


def has_result_mat(run_name: str) -> bool:
    return any((PROJECT_ROOT / "results" / run_name / "visualization").glob("*.mat"))


def latest_log_path(run_name: str) -> Path | None:
    candidates = sorted((PROJECT_ROOT / "logs").glob(f"test_{run_name}.log"))
    candidates.extend(sorted((PROJECT_ROOT / "results" / run_name).glob(f"test_{run_name}_*.log")))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_test_metrics(run_name: str) -> tuple[float | None, float | None]:
    log_path = latest_log_path(run_name)
    if log_path is None:
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    psnr_matches = re.findall(r"# psnr:\s*([0-9.]+)", text)
    ssim_matches = re.findall(r"# ssim:\s*([0-9.]+)", text)
    psnr = float(psnr_matches[-1]) if psnr_matches else None
    ssim = float(ssim_matches[-1]) if ssim_matches else None
    return psnr, ssim


def write_metric_csv(out_dir: Path, rows: list[dict[str, object]]) -> None:
    out_path = out_dir / "metrics_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "label", "value", "run_name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


def collect_metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in [*LATENT_MODELS, *STEP_MODELS, BASELINE_MODEL]:
        psnr, ssim = parse_test_metrics(spec.run_name)
        rows.append(
            {
                "experiment": spec.experiment,
                "label": spec.label,
                "value": spec.value,
                "run_name": spec.run_name,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
    return rows


def plot_experiment_metrics(
    out_dir: Path,
    title: str,
    xlabel: str,
    specs: list[ModelSpec],
    rows: list[dict[str, object]],
    out_name: str,
    include_baseline_reference: bool = False,
) -> None:
    row_by_run = {row["run_name"]: row for row in rows}
    available_specs = [
        spec
        for spec in specs
        if row_by_run.get(spec.run_name, {}).get("psnr") is not None
        and row_by_run.get(spec.run_name, {}).get("ssim") is not None
    ]
    missing = [spec.label for spec in specs if spec not in available_specs]
    if missing:
        print(f"Skipping missing metrics for {title}: {', '.join(missing)}")
    if not available_specs:
        print(f"No metrics available for {title}; skipping plot.")
        return

    xs = [spec.value for spec in available_specs]
    labels = [spec.label for spec in available_specs]
    psnr = [float(row_by_run[spec.run_name]["psnr"]) for spec in available_specs]
    ssim = [float(row_by_run[spec.run_name]["ssim"]) for spec in available_specs]

    fig, ax_ssim = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax_psnr = ax_ssim.twinx()

    ssim_line = ax_ssim.plot(xs, ssim, marker="s", linewidth=2.2, color="#2ca25f", label="SSIM")
    psnr_line = ax_psnr.plot(xs, psnr, marker="o", linewidth=2.2, color="#2b6cb0", label="PSNR")

    ax_ssim.set_title(title)
    ax_ssim.set_xlabel(xlabel)
    ax_ssim.set_ylabel("SSIM", color="#2ca25f")
    ax_psnr.set_ylabel("PSNR", color="#2b6cb0")
    ax_ssim.tick_params(axis="y", labelcolor="#2ca25f")
    ax_psnr.tick_params(axis="y", labelcolor="#2b6cb0")
    ax_ssim.set_xticks(xs)
    ax_ssim.set_xticklabels(labels)
    ax_ssim.grid(True, alpha=0.3)

    legend_items = [*ssim_line, *psnr_line]
    if include_baseline_reference:
        baseline = row_by_run.get(BASELINE_MODEL.run_name, {})
        baseline_ssim = baseline.get("ssim")
        baseline_psnr = baseline.get("psnr")
        if baseline_ssim is not None:
            legend_items.append(
                ax_ssim.axhline(
                    float(baseline_ssim),
                    linestyle="--",
                    linewidth=1.2,
                    color="#2ca25f",
                    alpha=0.45,
                    label="SSIM baseline 256/500k",
                )
            )
        if baseline_psnr is not None:
            legend_items.append(
                ax_psnr.axhline(
                    float(baseline_psnr),
                    linestyle="--",
                    linewidth=1.2,
                    color="#2b6cb0",
                    alpha=0.45,
                    label="PSNR baseline 256/500k",
                )
            )

    ax_ssim.legend(legend_items, [item.get_label() for item in legend_items], loc="best", fontsize=8)
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def magnitude(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    elif arr.ndim == 3 and arr.shape[0] == 2:
        arr = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        arr = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    return arr.astype(np.float32, copy=False)


def load_mat_variable(path: Path, variable: str) -> np.ndarray:
    print(f"Loading {variable} from {path}")
    return sio.loadmat(path, variable_names=[variable])[variable].astype(np.float32, copy=False)


def safe_psnr_from_mse(mse: np.ndarray, data_range: float) -> np.ndarray:
    mse = np.maximum(mse, 1e-12)
    return 20.0 * np.log10(max(data_range, 1e-6)) - 10.0 * np.log10(mse)


def ordered_valid_paths() -> list[Path]:
    valid_dir = PROJECT_ROOT / "mri_data_complex" / "mc_knee_pd_fs" / "valid"
    paths = paired_paths_from_folder([str(valid_dir), str(valid_dir)], ["lq", "gt"], "{}")
    return [Path(item["gt_path"]) for item in paths]


def find_index_for_example(valid_paths: list[Path], example: str) -> int:
    example_path = Path(example)
    for idx, path in enumerate(valid_paths):
        if path == example_path or path.name == example_path.name:
            return idx
    raise ValueError(f"Example not found in valid set: {example}")


def parse_slice_number(path: Path) -> int | None:
    match = re.search(r"(?:^|_)s(\d+)(?:_|\.mat$)", path.name)
    if not match:
        return None
    return int(match.group(1))


def slice_group_key(path: Path) -> str:
    return re.sub(r"(?:^|_)s\d+(?=_|\.mat$)", "_s###", path.name)


def middle_slice_indices(valid_paths: list[Path], middle_fraction: float) -> set[int]:
    if middle_fraction >= 0.999:
        return set(range(len(valid_paths)))
    if middle_fraction <= 0:
        raise ValueError("--middle-slice-fraction must be positive")

    grouped: dict[str, list[tuple[int, int]]] = {}
    unparsed: list[int] = []
    for idx, path in enumerate(valid_paths):
        slice_number = parse_slice_number(path)
        if slice_number is None:
            unparsed.append(idx)
            continue
        grouped.setdefault(slice_group_key(path), []).append((slice_number, idx))

    keep: set[int] = set(unparsed)
    for items in grouped.values():
        items = sorted(items)
        n_slices = len(items)
        if n_slices <= 2:
            keep.update(idx for _, idx in items)
            continue

        keep_count = max(1, int(round(n_slices * middle_fraction)))
        keep_count = min(keep_count, n_slices)
        start = max(0, (n_slices - keep_count) // 2)
        stop = start + keep_count
        keep.update(idx for _, idx in items[start:stop])

    if not keep:
        return set(range(len(valid_paths)))
    return keep


def select_interesting_indices(
    specs: list[ModelSpec],
    top_n: int,
    forced_indices: list[int],
    out_csv: Path,
    candidate_indices: set[int] | None = None,
    skip_top: int = 0,
) -> list[int]:
    if top_n <= 0 and forced_indices:
        return forced_indices

    gt = load_mat_variable(result_mat_path(specs[0].run_name), "gt")
    data_range = float(np.percentile(gt, 99.5) - np.percentile(gt, 0.5))
    psnr_by_model: list[np.ndarray] = []

    for spec in specs:
        recon = load_mat_variable(result_mat_path(spec.run_name), "recon")
        diff = recon - gt
        mse = np.mean(diff * diff, axis=(1, 2), dtype=np.float64)
        psnr_by_model.append(safe_psnr_from_mse(mse, data_range).astype(np.float32))
        del recon, diff, mse
        gc.collect()

    del gt
    gc.collect()

    psnr_matrix = np.stack(psnr_by_model, axis=0)
    spread = psnr_matrix.max(axis=0) - psnr_matrix.min(axis=0)
    order = np.argsort(spread)[::-1]

    selected: list[int] = []
    seen = set()
    for idx in forced_indices:
        if idx not in seen:
            selected.append(int(idx))
            seen.add(int(idx))
    skipped = 0
    for idx in order:
        if len(selected) >= top_n + len(forced_indices):
            break
        if candidate_indices is not None and int(idx) not in candidate_indices:
            continue
        if int(idx) not in seen:
            if skipped < skip_top:
                skipped += 1
                continue
            selected.append(int(idx))
            seen.add(int(idx))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "index", "spread_psnr", *[f"psnr_{spec.label}" for spec in specs]])
        for rank, idx in enumerate(selected, start=1):
            writer.writerow([rank, idx, float(spread[idx]), *[float(row[idx]) for row in psnr_matrix]])
    print(out_csv)
    return selected


def normalize_to_image(arr: np.ndarray, lo: float, hi: float, size: tuple[int, int] | None = None) -> Image.Image:
    arr = magnitude(arr)
    if hi <= lo:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L").convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def draw_label(image: Image.Image, label: str, font_size: int = 16) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    height = font_size + 10
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255), font=font)


def stack_h(images: Iterable[Image.Image], gap: int = 8) -> Image.Image:
    images = list(images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def grid_images(images: Iterable[Image.Image], columns: int, gap: int = 8) -> Image.Image:
    images = list(images)
    if not images:
        raise ValueError("Cannot make an image grid without images.")
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    width = columns * cell_width + gap * (columns - 1)
    height = rows * cell_height + gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), "white")

    for pos, image in enumerate(images):
        row, col = divmod(pos, columns)
        x = col * (cell_width + gap) + (cell_width - image.width) // 2
        y = row * (cell_height + gap) + (cell_height - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def stack_v(images: Iterable[Image.Image], gap: int = 12) -> Image.Image:
    images = list(images)
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def load_selected_recons(specs: list[ModelSpec], indices: list[int]) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for spec in specs:
        recon = load_mat_variable(result_mat_path(spec.run_name), "recon")
        selected[spec.label] = recon[indices].copy()
        del recon
        gc.collect()
    return selected


def render_examples(
    experiment_name: str,
    specs: list[ModelSpec],
    indices: list[int],
    valid_paths: list[Path],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_mat_variable(result_mat_path(specs[0].run_name), "gt")
    selected_gt = gt[indices].copy()
    del gt
    gc.collect()

    recons = load_selected_recons(specs, indices)

    rows: list[Image.Image] = []
    summary_path = out_dir / f"{experiment_name}_examples.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "index", "file"])

        for rank, idx in enumerate(indices, start=1):
            source = sio.loadmat(valid_paths[idx])
            gt_slice = selected_gt[rank - 1]
            lo, hi = np.percentile(gt_slice, [1, 99.5])
            if hi <= lo:
                lo, hi = float(gt_slice.min()), float(gt_slice.max())

            target_size = (gt_slice.shape[1], gt_slice.shape[0])
            panels: list[Image.Image] = []

            lr_key = "T2_64" if "T2_64" in source else "T2_128"
            source_panels = [
                ("PD LR", source[lr_key]),
                ("Fat-sat ref", source["T1"]),
                ("PD GT", gt_slice),
            ]
            for label, arr in source_panels:
                image = normalize_to_image(arr, lo, hi, size=target_size)
                draw_label(image, label)
                panels.append(image)

            for spec in specs:
                recon_slice = recons[spec.label][rank - 1]
                image = normalize_to_image(recon_slice, lo, hi, size=target_size)
                mse = float(np.mean((recon_slice - gt_slice) ** 2))
                dr = float(max(hi - lo, 1e-6))
                psnr = 20 * math.log10(dr) - 10 * math.log10(max(mse, 1e-12))
                draw_label(image, f"{spec.experiment} {spec.label} ({psnr:.1f})", font_size=14)
                panels.append(image)

            grid_cols = 3 if len(panels) <= 6 else 4
            grid = grid_images(panels, columns=grid_cols)
            title = Image.new("RGB", (grid.width, 36), "white")
            draw = ImageDraw.Draw(title)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            draw.text((4, 8), f"rank {rank} | index {idx} | {valid_paths[idx].name}", fill=(0, 0, 0), font=font)
            combined = stack_v([title, grid], gap=0)
            rows.append(combined)

            example_path = out_dir / f"{experiment_name}_rank{rank:02d}_idx{idx:05d}.png"
            combined.save(example_path)
            writer.writerow([rank, idx, valid_paths[idx].name])
            print(example_path)

    contact = stack_v(rows)
    contact_path = out_dir / f"{experiment_name}_contact_sheet.png"
    contact.save(contact_path)
    print(summary_path)
    print(contact_path)


def parse_indices(raw: str | None) -> list[int]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_outputs/pd_fs_ablation", help="Output directory.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of high-difference examples per experiment.")
    parser.add_argument("--skip-top", type=int, default=0, help="Skip this many auto-selected examples before rendering.")
    parser.add_argument("--indices", default="", help="Comma-separated validation indices to force-render.")
    parser.add_argument("--example-file", default="", help="Validation .mat basename/path to force-render.")
    parser.add_argument(
        "--middle-slice-fraction",
        type=float,
        default=0.6,
        help="Fraction of slices to keep from the middle of each scan for example selection. Use 1.0 for all slices.",
    )
    parser.add_argument("--only-plots", action="store_true", help="Only make metric plots and CSVs; skip .mat image grids.")
    parser.add_argument(
        "--experiments",
        default="latent,timesteps",
        help="Comma-separated experiments to render: latent,timesteps.",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metric_rows()
    write_metric_csv(out_dir, rows)
    plot_experiment_metrics(out_dir, "Latent vector size ablation", "prior_dim", LATENT_MODELS, rows, "metrics_latent.png")
    plot_experiment_metrics(
        out_dir,
        "Diffusion denoising step ablation",
        "denoising steps",
        STEP_MODELS,
        rows,
        "metrics_timesteps.png",
    )

    if args.only_plots:
        return

    valid_paths = ordered_valid_paths()
    forced_indices = parse_indices(args.indices)
    if args.example_file:
        forced_indices.append(find_index_for_example(valid_paths, args.example_file))
    candidate_indices = middle_slice_indices(valid_paths, args.middle_slice_fraction)
    print(
        f"Example selection uses {len(candidate_indices)} / {len(valid_paths)} validation slices "
        f"from the middle {args.middle_slice_fraction:.0%} of each scan."
    )

    experiments = {item.strip() for item in args.experiments.split(",") if item.strip()}
    if "latent" in experiments:
        latent_specs = [spec for spec in LATENT_MODELS if has_result_mat(spec.run_name)]
        if len(latent_specs) != len(LATENT_MODELS):
            missing = [spec.label for spec in LATENT_MODELS if not has_result_mat(spec.run_name)]
            print(f"Skipping missing latent image results: {', '.join(missing)}")
        if len(latent_specs) < 2:
            print("Not enough latent result files for example selection; skipping latent examples.")
        else:
            selected = select_interesting_indices(
                latent_specs,
                args.top_n,
                forced_indices,
                out_dir / "selected_latent_examples.csv",
                candidate_indices,
                args.skip_top,
            )
            render_examples("latent", latent_specs, selected, valid_paths, out_dir / "latent_examples")

    if "timesteps" in experiments:
        timestep_specs = [spec for spec in STEP_MODELS if has_result_mat(spec.run_name)]
        if len(timestep_specs) != len(STEP_MODELS):
            missing = [spec.label for spec in STEP_MODELS if not has_result_mat(spec.run_name)]
            print(f"Skipping missing timestep image results: {', '.join(missing)}")
        if len(timestep_specs) < 2:
            print("Not enough timestep result files for example selection; skipping timestep examples.")
        else:
            selected = select_interesting_indices(
                timestep_specs,
                args.top_n,
                forced_indices,
                out_dir / "selected_timestep_examples.csv",
                candidate_indices,
                args.skip_top,
            )
            render_examples("timesteps", timestep_specs, selected, valid_paths, out_dir / "timestep_examples")


if __name__ == "__main__":
    main()
