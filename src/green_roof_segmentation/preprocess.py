"""Convert paired aerial and polygon-overlay screenshots into a training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MASK_RGB = np.array([255, 0, 255], dtype=np.uint8)


def preprocess_pair(
    image_path: Path,
    overlay_path: Path,
    target_size: tuple[int, int] = (600, 600),
) -> tuple[np.ndarray, np.ndarray]:
    """Load a paired capture and return an RGB image and positive-class binary mask."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    overlay_bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    if overlay_bgr is None:
        raise ValueError(f"Could not read overlay: {overlay_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != target_size[::-1]:
        image_rgb = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)

    mask = np.all(overlay_rgb == MASK_RGB, axis=-1).astype(np.uint8)
    if mask.shape != target_size[::-1]:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return image_rgb, mask[..., np.newaxis]


def build_dataset(image_dir: Path, overlay_dir: Path, output_path: Path) -> int:
    """Preprocess all filename-matched PNG pairs and write a compressed NPZ file."""
    image_paths = {path.name: path for path in image_dir.glob("*.png")}
    overlay_paths = {path.name: path for path in overlay_dir.glob("*.png")}
    if image_paths.keys() != overlay_paths.keys():
        missing_overlays = sorted(image_paths.keys() - overlay_paths.keys())
        missing_images = sorted(overlay_paths.keys() - image_paths.keys())
        raise ValueError(
            f"Capture pairs do not match; missing overlays={missing_overlays}, "
            f"missing images={missing_images}"
        )
    if not image_paths:
        raise ValueError("No PNG capture pairs found")

    identifiers = sorted(image_paths)
    pairs = [preprocess_pair(image_paths[name], overlay_paths[name]) for name in identifiers]
    images = np.stack([pair[0] for pair in pairs])
    masks = np.stack([pair[1] for pair in pairs])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images, masks=masks, identifiers=identifiers)
    return len(identifiers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="Directory of aerial PNG captures")
    parser.add_argument("overlays", type=Path, help="Directory of matching overlay captures")
    parser.add_argument("output", type=Path, help="Destination NPZ file")
    args = parser.parse_args()
    count = build_dataset(args.images, args.overlays, args.output)
    print(f"Wrote {count} paired samples to {args.output}")


if __name__ == "__main__":
    main()
