#!/usr/bin/env python3
"""Prepare PD/PD-FS knee DICOMs for the DiffMSR paired dataset loader.

DiffMSR's MRI loader is hard-coded for MATLAB files containing:

    T2, T2_128, T2_64, T1, T1_128, T1_64

For this dataset we map:

    PD high resolution        -> T2
    downsampled PD            -> T2_128 / T2_64
    PD with fat suppression   -> T1 / T1_128 / T1_64

The downsampling follows the repository's MATLAB demo: normalize each DICOM
slice, transform to centered k-space, crop the k-space center, and transform
back to image space.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import scipy.io as sio


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "knee_mri_clinical_seq_batch2"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[1] / "mri_data_complex" / "mc_knee_pd_fs"
DEFAULT_MASK = Path(__file__).resolve().parents[1] / "complex_data_demo" / "dc_mask" / "lr_4x.mat"


@dataclass
class SeriesInfo:
    study_key: str
    path: Path
    files: List[Path]
    description: str
    scan_options: str
    image_type: str
    sequence_name: str
    orientation: str
    rows: int
    cols: int
    series_number: int
    is_pd: bool
    is_fat_suppressed: bool


@dataclass
class DicomSlice:
    path: Path
    instance_number: int
    slice_position: Optional[float]


def dicom_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def normalized_text(*values: object) -> str:
    text = " ".join(dicom_value(v) for v in values)
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().upper()


def infer_orientation(ds: pydicom.dataset.Dataset) -> str:
    text = normalized_text(
        getattr(ds, "SeriesDescription", ""),
        getattr(ds, "ProtocolName", ""),
    )
    if re.search(r"\b(COR|CORONAL)\b", text):
        return "COR"
    if re.search(r"\b(SAG|SAGITTAL)\b", text):
        return "SAG"
    if re.search(r"\b(AX|AXIAL|TRA|TRANSVERSE)\b", text):
        return "AX"

    orientation = getattr(ds, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) < 6:
        return "UNK"

    row = np.asarray([float(v) for v in orientation[:3]], dtype=np.float32)
    col = np.asarray([float(v) for v in orientation[3:6]], dtype=np.float32)
    normal = np.abs(np.cross(row, col))
    axis = int(np.argmax(normal))
    return ("SAG", "COR", "AX")[axis]


def is_fat_suppressed_text(text: str) -> bool:
    return bool(
        re.search(r"\bFS\b", text)
        or re.search(r"\bFAT\s*SAT\b", text)
        or re.search(r"\bFATSAT\b", text)
        or re.search(r"\bFAT\s*SUPP", text)
        or re.search(r"\bSPAIR\b", text)
        or re.search(r"\bSTIR\b", text)
    )


def is_pd_text(text: str) -> bool:
    return bool(re.search(r"\bPD\b", text) or "PROTON" in text)


def natural_sort_key(path: Path) -> Tuple[int, str]:
    matches = re.findall(r"\d+", path.name)
    return (int(matches[0]) if matches else -1, path.name)


def discover_series(input_root: Path) -> List[SeriesInfo]:
    series_dirs = sorted({p.parent for p in input_root.rglob("*.dcm")})
    series_infos: List[SeriesInfo] = []

    for series_dir in series_dirs:
        files = sorted(series_dir.glob("*.dcm"), key=natural_sort_key)
        if not files:
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
        except Exception as exc:
            print(f"warning: could not read header for {files[0]}: {exc}", file=sys.stderr)
            continue

        description = dicom_value(getattr(ds, "SeriesDescription", ""))
        scan_options = dicom_value(getattr(ds, "ScanOptions", ""))
        image_type = dicom_value(getattr(ds, "ImageType", ""))
        sequence_name = dicom_value(getattr(ds, "SequenceName", ""))
        text = normalized_text(description, scan_options, image_type, sequence_name)

        if not is_pd_text(text):
            continue

        try:
            study_key = str(series_dir.parent.relative_to(input_root))
        except ValueError:
            study_key = series_dir.parent.name

        series_infos.append(
            SeriesInfo(
                study_key=study_key,
                path=series_dir,
                files=files,
                description=description,
                scan_options=scan_options,
                image_type=image_type,
                sequence_name=sequence_name,
                orientation=infer_orientation(ds),
                rows=int(getattr(ds, "Rows", 0) or 0),
                cols=int(getattr(ds, "Columns", 0) or 0),
                series_number=int(getattr(ds, "SeriesNumber", 0) or 0),
                is_pd=True,
                is_fat_suppressed=is_fat_suppressed_text(text),
            )
        )

    return series_infos


def group_by_study(series_infos: Iterable[SeriesInfo]) -> Dict[str, List[SeriesInfo]]:
    grouped: Dict[str, List[SeriesInfo]] = {}
    for series in series_infos:
        grouped.setdefault(series.study_key, []).append(series)
    return grouped


def pair_score(pd_series: SeriesInfo, fs_series: SeriesInfo) -> Tuple[int, int, int, int]:
    slice_delta = abs(len(pd_series.files) - len(fs_series.files))
    area_delta = abs((pd_series.rows * pd_series.cols) - (fs_series.rows * fs_series.cols))
    number_delta = abs(pd_series.series_number - fs_series.series_number)
    # Prefer a target with more source pixels when there are multiple choices.
    negative_pd_area = -(pd_series.rows * pd_series.cols)
    return (slice_delta, area_delta, number_delta, negative_pd_area)


def select_pairs_for_study(series_infos: Sequence[SeriesInfo]) -> List[Tuple[SeriesInfo, SeriesInfo]]:
    by_orientation: Dict[str, List[SeriesInfo]] = {}
    for series in series_infos:
        by_orientation.setdefault(series.orientation, []).append(series)

    pairs: List[Tuple[SeriesInfo, SeriesInfo]] = []
    for _, candidates in sorted(by_orientation.items()):
        pd_candidates = [s for s in candidates if not s.is_fat_suppressed]
        fs_candidates = [s for s in candidates if s.is_fat_suppressed]
        if not pd_candidates or not fs_candidates:
            continue

        best_pair = min(
            ((pd_series, fs_series) for pd_series in pd_candidates for fs_series in fs_candidates),
            key=lambda pair: pair_score(pair[0], pair[1]),
        )
        pairs.append(best_pair)
    return pairs


def fft2c(image: np.ndarray) -> np.ndarray:
    factor = np.sqrt(float(image.shape[0] * image.shape[1]))
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))) / factor


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    factor = np.sqrt(float(kspace.shape[0] * kspace.shape[1]))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace))) * factor


def center_crop_2d(data: np.ndarray, size: int) -> np.ndarray:
    height, width = data.shape[:2]
    if height < size or width < size:
        raise ValueError(f"cannot crop {height}x{width} to {size}x{size}")
    top = (height - size) // 2
    left = (width - size) // 2
    return data[top : top + size, left : left + size]


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    scale = float(np.max(np.abs(image)))
    if scale <= 0:
        raise ValueError("blank image")
    return image / scale


def read_dicom_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)
    if image.ndim != 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    image = image * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.max(image) - image
    return image


def kspace_pyramid(image: np.ndarray, hr_size: int, mid_size: int, lr_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = normalize_image(image).astype(np.complex64)
    kspace = fft2c(image)
    k_hr = center_crop_2d(kspace, hr_size)
    k_mid = center_crop_2d(k_hr, mid_size)
    k_lr = center_crop_2d(k_hr, lr_size)
    return (
        ifft2c(k_hr).astype(np.complex64),
        ifft2c(k_mid).astype(np.complex64),
        ifft2c(k_lr).astype(np.complex64),
    )


def read_slice_metadata(path: Path) -> DicomSlice:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    position = getattr(ds, "ImagePositionPatient", None)
    slice_position = None
    if position is not None and len(position) >= 3:
        try:
            slice_position = float(position[2])
        except (TypeError, ValueError):
            slice_position = None
    if slice_position is None:
        try:
            slice_position = float(getattr(ds, "SliceLocation"))
        except (TypeError, ValueError, AttributeError):
            slice_position = None
    return DicomSlice(path=path, instance_number=instance, slice_position=slice_position)


def sorted_slices(files: Sequence[Path]) -> List[DicomSlice]:
    slices = [read_slice_metadata(path) for path in files]
    return sorted(slices, key=lambda item: (item.instance_number, natural_sort_key(item.path)))


def pair_slices(pd_files: Sequence[Path], fs_files: Sequence[Path]) -> List[Tuple[Path, Path]]:
    pd_slices = sorted_slices(pd_files)
    fs_slices = sorted_slices(fs_files)

    if not pd_slices or not fs_slices:
        return []

    pd_positions = [s.slice_position for s in pd_slices]
    fs_positions = [s.slice_position for s in fs_slices]
    if all(v is not None for v in pd_positions + fs_positions):
        fs_available = list(fs_slices)
        pairs = []
        for pd_slice in pd_slices:
            best_idx = min(
                range(len(fs_available)),
                key=lambda idx: abs(float(pd_slice.slice_position) - float(fs_available[idx].slice_position)),
            )
            fs_slice = fs_available.pop(best_idx)
            pairs.append((pd_slice.path, fs_slice.path))
            if not fs_available:
                break
        return pairs

    n = min(len(pd_slices), len(fs_slices))
    return [(pd_slices[idx].path, fs_slices[idx].path) for idx in range(n)]


def safe_name(text: str) -> str:
    text = text.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def prepare_output_dirs(out_root: Path, overwrite: bool) -> Tuple[Path, Path]:
    train_dir = out_root / "train"
    valid_dir = out_root / "valid"

    for folder in (train_dir, valid_dir):
        folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(folder.glob("*.mat"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{folder} already contains {len(existing)} .mat files. "
                "Use --overwrite to replace only generated .mat files."
            )
        if overwrite:
            for path in existing:
                path.unlink()

    return train_dir, valid_dir


def split_studies(study_keys: Sequence[str], train_ratio: float, seed: int) -> Tuple[set, set]:
    keys = list(study_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    train_count = int(round(len(keys) * train_ratio))
    if len(keys) > 1:
        train_count = min(max(train_count, 1), len(keys) - 1)
    return set(keys[:train_count]), set(keys[train_count:])


def write_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(args: argparse.Namespace) -> Dict[str, object]:
    input_root = args.input_root.resolve()
    out_root = args.out_root.resolve()

    series_infos = discover_series(input_root)
    grouped = group_by_study(series_infos)
    study_pairs = {
        study_key: select_pairs_for_study(study_series)
        for study_key, study_series in grouped.items()
    }
    study_pairs = {key: pairs for key, pairs in study_pairs.items() if pairs}

    study_keys = sorted(study_pairs)
    if args.max_studies is not None:
        study_keys = study_keys[: args.max_studies]
    train_studies, valid_studies = split_studies(study_keys, args.train_ratio, args.seed)

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "out_root": str(out_root),
        "pd_series_seen": sum(1 for s in series_infos if not s.is_fat_suppressed),
        "pd_fs_series_seen": sum(1 for s in series_infos if s.is_fat_suppressed),
        "paired_studies": len(study_keys),
        "train_studies": len(train_studies),
        "valid_studies": len(valid_studies),
        "train_samples": 0,
        "valid_samples": 0,
        "skipped_samples": 0,
        "hr_size": args.hr_size,
        "mid_size": args.mid_size,
        "lr_size": args.lr_size,
    }

    if args.dry_run:
        for study_key in study_keys[:20]:
            print(f"{study_key}: {len(study_pairs[study_key])} pair(s)")
            for pd_series, fs_series in study_pairs[study_key]:
                print(
                    "  "
                    f"{pd_series.orientation}: PD={pd_series.path.name} "
                    f"({pd_series.description}, {len(pd_series.files)} slices, {pd_series.rows}x{pd_series.cols}) "
                    f"REF={fs_series.path.name} "
                    f"({fs_series.description}, {len(fs_series.files)} slices, {fs_series.rows}x{fs_series.cols})"
                )
        print(json.dumps(summary, indent=2))
        return summary

    train_dir, valid_dir = prepare_output_dirs(out_root, args.overwrite)
    manifest_rows: List[Dict[str, object]] = []
    sample_count = 0

    for study_idx, study_key in enumerate(study_keys, start=1):
        split = "train" if study_key in train_studies else "valid"
        target_dir = train_dir if split == "train" else valid_dir

        for pair_idx, (pd_series, fs_series) in enumerate(study_pairs[study_key]):
            slice_pairs = pair_slices(pd_series.files, fs_series.files)
            for slice_idx, (pd_path, fs_path) in enumerate(slice_pairs):
                if args.max_samples is not None and sample_count >= args.max_samples:
                    break
                try:
                    pd_img = read_dicom_image(pd_path)
                    fs_img = read_dicom_image(fs_path)
                    t2, t2_128, t2_64 = kspace_pyramid(pd_img, args.hr_size, args.mid_size, args.lr_size)
                    t1, t1_128, t1_64 = kspace_pyramid(fs_img, args.hr_size, args.mid_size, args.lr_size)
                except Exception as exc:
                    summary["skipped_samples"] = int(summary["skipped_samples"]) + 1
                    print(f"warning: skipped {pd_path} / {fs_path}: {exc}", file=sys.stderr)
                    continue

                study_name = safe_name(study_key)
                sample_name = f"{study_name}_{pd_series.orientation}_p{pair_idx:02d}_s{slice_idx:03d}.mat"
                out_path = target_dir / sample_name
                sio.savemat(
                    out_path,
                    {
                        "T2": t2,
                        "T2_128": t2_128,
                        "T2_64": t2_64,
                        "T1": t1,
                        "T1_128": t1_128,
                        "T1_64": t1_64,
                    },
                    do_compression=args.compress,
                )

                rel_pd = pd_path.relative_to(input_root)
                rel_fs = fs_path.relative_to(input_root)
                manifest_rows.append(
                    {
                        "split": split,
                        "sample": sample_name,
                        "study": study_key,
                        "orientation": pd_series.orientation,
                        "pd_series": pd_series.path.name,
                        "pd_description": pd_series.description,
                        "reference_series": fs_series.path.name,
                        "reference_description": fs_series.description,
                        "pd_dicom": str(rel_pd),
                        "reference_dicom": str(rel_fs),
                        "pd_source_shape": f"{pd_img.shape[0]}x{pd_img.shape[1]}",
                        "reference_source_shape": f"{fs_img.shape[0]}x{fs_img.shape[1]}",
                        "hr_shape": f"{t2.shape[0]}x{t2.shape[1]}",
                        "lr_shape": f"{t2_64.shape[0]}x{t2_64.shape[1]}",
                    }
                )
                summary[f"{split}_samples"] = int(summary[f"{split}_samples"]) + 1
                sample_count += 1

            if args.max_samples is not None and sample_count >= args.max_samples:
                break

        if study_idx % args.progress_every == 0:
            print(f"processed {study_idx}/{len(study_keys)} paired studies")
        if args.max_samples is not None and sample_count >= args.max_samples:
            break

    manifest_path = out_root / "manifest.csv"
    summary_path = out_root / "summary.json"
    write_manifest(manifest_path, manifest_rows)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if args.mask_src is not None:
        mask_src = args.mask_src.resolve()
        if mask_src.exists():
            shutil.copy2(mask_src, out_root / "dc_mask.mat")
        else:
            print(f"warning: mask not found: {mask_src}", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    print(f"wrote manifest: {manifest_path}")
    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def train_ratio_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def optional_path(value: str) -> Optional[Path]:
    if value.strip().lower() in {"", "none", "skip"}:
        return None
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PD and PD fat-suppressed DICOM series into DiffMSR .mat files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root containing subject/study/series DICOM folders.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output dataset root. Train and valid folders are created inside it.")
    parser.add_argument("--train-ratio", type=train_ratio_value, default=0.8, help="Study-level train split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the study-level split.")
    parser.add_argument("--hr-size", type=positive_int, default=256, help="High-resolution output size.")
    parser.add_argument("--mid-size", type=positive_int, default=128, help="Intermediate output size stored as T*_128.")
    parser.add_argument("--lr-size", type=positive_int, default=64, help="Low-resolution output size stored as T*_64.")
    parser.add_argument("--max-studies", type=positive_int, default=None, help="Limit paired studies for a quick test.")
    parser.add_argument("--max-samples", type=positive_int, default=None, help="Limit output samples for a quick test.")
    parser.add_argument("--mask-src", type=optional_path, default=DEFAULT_MASK, help="Mask .mat to copy into the output root. Use --mask-src none to skip.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .mat files in output train/valid folders.")
    parser.add_argument("--compress", action="store_true", help="Use scipy .mat compression.")
    parser.add_argument("--dry-run", action="store_true", help="Print detected pairs without writing .mat files.")
    parser.add_argument("--progress-every", type=positive_int, default=25, help="Print progress every N paired studies.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mid_size > args.hr_size or args.lr_size > args.hr_size:
        parser.error("--mid-size and --lr-size must be <= --hr-size")
    if args.lr_size >= args.mid_size:
        parser.error("--lr-size should be smaller than --mid-size")

    process_dataset(args)


if __name__ == "__main__":
    main()
