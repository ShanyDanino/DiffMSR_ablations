#!/usr/bin/env python3
"""Check that the active Python environment can run DiffMSR."""

import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "PIL",
    "tqdm",
    "scipy",
    "pydicom",
    "timm",
    "yaml",
    "einops",
    "tensorboard",
]

OPTIONAL_MODULES = [
    "facexlib",
    "gfpgan",
]


def main() -> None:
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append((module_name, exc))

    print("python", sys.executable)
    if missing:
        for module_name, exc in missing:
            print(f"MISSING {module_name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    for module_name in OPTIONAL_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"optional {module_name} unavailable: {type(exc).__name__}: {exc}")

    import torch

    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
    print("imports ok")


if __name__ == "__main__":
    main()
