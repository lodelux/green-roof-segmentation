"""Normalize GeoJSON coordinates while preserving features and properties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _strip_elevation(value: Any) -> Any:
    """Recursively convert coordinate positions to two dimensions."""
    if not isinstance(value, list):
        return value
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        return value[:2]
    return [_strip_elevation(item) for item in value]


def normalize_geojson(document: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a GeoJSON document with 2D geometry coordinates."""
    normalized = json.loads(json.dumps(document))
    features = normalized.get("features")
    if normalized.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("Expected a GeoJSON FeatureCollection")

    for feature in features:
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        if not isinstance(geometry, dict) or "coordinates" not in geometry:
            raise ValueError("Each non-null geometry must contain coordinates")
        geometry["coordinates"] = _strip_elevation(geometry["coordinates"])
    return normalized


def clean_geojson(input_path: Path, output_path: Path) -> None:
    """Read, normalize, and write a GeoJSON FeatureCollection."""
    with input_path.open(encoding="utf-8") as source:
        document = json.load(source)
    normalized = normalize_geojson(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(normalized, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source GeoJSON FeatureCollection")
    parser.add_argument("output", type=Path, help="Destination for normalized GeoJSON")
    args = parser.parse_args()
    clean_geojson(args.input, args.output)


if __name__ == "__main__":
    main()
