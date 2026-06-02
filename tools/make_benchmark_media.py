#!/usr/bin/env python3
"""Generate deterministic encoded media for runtime benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than zero")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    rng = np.random.default_rng(args.seed)
    image = synthetic_rgb_image(args.image_size, rng)
    args.jpeg_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(
        args.jpeg_output,
        format="JPEG",
        quality=args.jpeg_quality,
        subsampling=0,
        optimize=False,
    )
    print(f"wrote {args.jpeg_output}")


def synthetic_rgb_image(size: int, rng: np.random.Generator) -> np.ndarray:
    axis = np.linspace(0, 255, size, dtype=np.uint8)
    xx, yy = np.meshgrid(axis, axis)
    noise = rng.integers(0, 16, size=(size, size), dtype=np.uint8)
    return np.stack(
        [
            xx,
            yy,
            ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2 + noise).astype(np.uint8),
        ],
        axis=-1,
    )


if __name__ == "__main__":
    main()
