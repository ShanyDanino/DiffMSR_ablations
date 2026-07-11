#!/usr/bin/env python3
"""Export a DiffMSR one-slice result .mat as a side-by-side PNG preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont


def magnitude(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value).squeeze()
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    elif arr.ndim == 3 and arr.shape[0] == 2:
        arr = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        arr = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
    return arr.astype(np.float32)


def normalize_panel(arr: np.ndarray) -> Image.Image:
    arr = magnitude(arr)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def resize_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.BICUBIC)


def draw_label(image: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 5), label, fill=(255, 255, 255), font=font)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Result .mat from results/.../visualization")
    parser.add_argument("--input", required=True, help="Input DiffMSR .mat with T2_64, T1, and T2")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    result = sio.loadmat(args.result)
    source = sio.loadmat(args.input)

    panels = [
        ("PD LR input", normalize_panel(source["T2_64"])),
        ("Fat-sat ref", normalize_panel(source["T1"])),
        ("Recon PD SR", normalize_panel(result["recon"])),
        ("PD GT", normalize_panel(result["gt"])),
    ]

    target_size = panels[2][1].size
    padded = []
    for label, image in panels:
        image = resize_to(image, target_size)
        draw_label(image, label)
        padded.append(image)

    gap = 8
    width = target_size[0] * len(padded) + gap * (len(padded) - 1)
    canvas = Image.new("RGB", (width, target_size[1]), (255, 255, 255))
    x = 0
    for image in padded:
        canvas.paste(image, (x, 0))
        x += image.width + gap

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
