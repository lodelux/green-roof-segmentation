from pathlib import Path

import cv2
import numpy as np

from green_roof_segmentation.preprocess import build_dataset, preprocess_pair


def _write_rgb(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_preprocess_pair_marks_magenta_overlay_as_positive(tmp_path: Path) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = image.copy()
    overlay[0, 1] = [255, 0, 255]
    image_path, overlay_path = tmp_path / "image.png", tmp_path / "overlay.png"
    _write_rgb(image_path, image)
    _write_rgb(overlay_path, overlay)

    processed_image, mask = preprocess_pair(image_path, overlay_path, target_size=(2, 2))

    assert processed_image.shape == (2, 2, 3)
    assert mask.shape == (2, 2, 1)
    assert mask[:, :, 0].tolist() == [[0, 1], [0, 0]]


def test_build_dataset_matches_pairs_by_filename(tmp_path: Path) -> None:
    images, overlays = tmp_path / "images", tmp_path / "overlays"
    images.mkdir()
    overlays.mkdir()
    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    sample[1, 1] = [255, 0, 255]
    for name in ("45.1_9.1.png", "45.2_9.2.png"):
        _write_rgb(images / name, np.zeros_like(sample))
        _write_rgb(overlays / name, sample)

    output = tmp_path / "dataset.npz"
    assert build_dataset(images, overlays, output) == 2

    dataset = np.load(output)
    assert dataset["images"].shape == (2, 600, 600, 3)
    assert dataset["masks"].shape == (2, 600, 600, 1)
    assert dataset["identifiers"].tolist() == ["45.1_9.1.png", "45.2_9.2.png"]
