#!/usr/bin/env python3
"""Train one segmentation model with geographically grouped data splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def spatial_groups(identifiers: np.ndarray, cell_size: float = 0.005) -> np.ndarray:
    """Group coordinate-named captures into coarse spatial cells."""
    groups = []
    for identifier in identifiers.astype(str):
        latitude, longitude = map(float, Path(identifier).stem.split("_", maxsplit=1))
        groups.append((int(latitude / cell_size), int(longitude / cell_size)))
    return np.asarray([f"{lat}:{lon}" for lat, lon in groups])


def split_indices(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 70/15/15 train, validation, and test splits by spatial group."""
    from sklearn.model_selection import GroupShuffleSplit

    all_indices = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, remainder = next(outer.split(all_indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    validation_local, test_local = next(inner.split(remainder, groups=groups[remainder]))
    return train, remainder[validation_local], remainder[test_local]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=("efficientnet", "deeplab"), default="deeplab")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

    from green_roof_segmentation.models import build_deeplab, build_efficientnet

    dataset = np.load(args.dataset)
    images, masks, identifiers = dataset["images"], dataset["masks"], dataset["identifiers"]
    train, validation, test = split_indices(spatial_groups(identifiers), args.seed)

    model = build_deeplab() if args.model == "deeplab" else build_efficientnet()
    if args.model == "deeplab":
        images = preprocess_resnet(images.astype(np.float32))

    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
        metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="roof_iou")],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        images[train],
        masks[train],
        validation_data=(images[validation], masks[validation]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
    )
    metrics = model.evaluate(images[test], masks[test], return_dict=True)
    print({key: float(value) for key, value in metrics.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
