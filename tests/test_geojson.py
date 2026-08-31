from green_roof_segmentation.geojson import normalize_geojson


def test_normalizes_all_polygon_and_multipolygon_coordinates() -> None:
    source = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[9.1, 45.4, 100], [9.2, 45.5, 101]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"id": 2},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[9.3, 45.6, 102], [9.4, 45.7, 103]]]],
                },
            },
        ],
    }

    result = normalize_geojson(source)

    assert result["features"][0]["geometry"]["coordinates"] == [[[9.1, 45.4], [9.2, 45.5]]]
    assert result["features"][1]["geometry"]["coordinates"] == [[[[9.3, 45.6], [9.4, 45.7]]]]
    assert result["features"][1]["properties"] == {"id": 2}
    assert source["features"][0]["geometry"]["coordinates"][0][0] == [9.1, 45.4, 100]


def test_rejects_non_feature_collection() -> None:
    try:
        normalize_geojson({"type": "Polygon", "coordinates": []})
    except ValueError as error:
        assert "FeatureCollection" in str(error)
    else:
        raise AssertionError("Expected invalid GeoJSON to be rejected")
