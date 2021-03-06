import os
import numpy as np
import pytest
import logging
import utils


@pytest.mark.parametrize("geopoint, expected_output", [
    (utils.GeoPoint(latitude=0.0, longitude=0.0), utils.MercatorPoint(x=0.0, y=0.0)),
    (utils.GeoPoint(latitude=+45.0, longitude=0.0), utils.MercatorPoint(x=0.0, y=+0.8813735870195429)),
    (utils.GeoPoint(latitude=+50.0, longitude=0.0), utils.MercatorPoint(x=0.0, y=+1.0106831886830212)),
    (utils.GeoPoint(latitude=-90.0, longitude=0.0), utils.MercatorPoint(x=0.0, y=-np.inf)),
    (utils.GeoPoint(latitude=+90.0, longitude=0.0), utils.MercatorPoint(x=0.0, y=+np.inf)),
])
def test_geo_to_mercator(geopoint, expected_output):
    output = utils.geo_to_mercator(geopoint)
    logging.info(f"{geopoint} -> {output}")
    assert output == expected_output
    assert output.x == geopoint.longitude


@pytest.mark.parametrize("mercator_point, expected_geopoint", [
    (utils.MercatorPoint(x=0.0, y=0.0), utils.GeoPoint(latitude=0.0, longitude=0.0)),
    (utils.MercatorPoint(x=0.0, y=+0.8813735870195429), utils.GeoPoint(latitude=-45.0, longitude=0.0)),
    (utils.MercatorPoint(x=0.0, y=+1.0106831886830212), utils.GeoPoint(latitude=-50.0, longitude=0.0)),
])
def test_mercator_to_geo(mercator_point, expected_geopoint):
    geo_point = mercator_point.to_geo()
    logging.info(f"{mercator_point} -> {geo_point}")
    assert geo_point.isclose(expected_geopoint)


@pytest.mark.parametrize("inp, expected_output", [
    ((utils.GeoPoint(latitude=0.0, longitude=-180.0), 0), utils.ScenePoint(0.0, 0.5)),
    ((utils.GeoPoint(latitude=0.0, longitude=0.0), 0), utils.ScenePoint(0.5, 0.5)),
    ((utils.GeoPoint(latitude=0.0, longitude=+180.0), 0), utils.ScenePoint(1.0, 0.5)),
    ((utils.GeoPoint(latitude=85.0511287798066, longitude=-180), 1), utils.ScenePoint(0.0, 0.0)),
    ((utils.GeoPoint(latitude=0.0, longitude=0.0), 1), utils.ScenePoint(1.0, 1.0)),
])
def test_mapbox_geopoint_to_xy(inp, expected_output):
    geopoint, zoom = inp
    output = utils.mapbox_geopoint_to_xy(geopoint, zoom)
    logging.info(f"{geopoint} zoom={zoom} -> mapbox {output}")
    assert np.isclose(output.x, expected_output.x)
    assert np.isclose(output.y, expected_output.y)


@pytest.mark.parametrize("ix,iy,zoom,style_id,tilesize", [
    (8794, 5614, 14, "satellite-v9", 512),
    (4397, 2807, 13, "satellite-v9", 512),
    (2198, 1403, 12, "satellite-v9", 512),
    (1099, 701, 11, "satellite-v9", 512),
    (8794, 5614, 14, "streets-v11", 512),
    (4397, 2807, 13, "streets-v11", 512),
    (2198, 1403, 12, "streets-v11", 512),
    (1099, 701, 11, "streets-v11", 512),
])
def test_download_static_tiles(ix, iy, zoom, style_id, tilesize):
    tile = utils.MapboxTile(x=ix, y=iy, z=zoom)
    logging.info(f"started with {tile}")
    path = tile.static_tile_path(style_id, tilesize)
    if os.path.isfile(path):
        os.remove(path)
    tile.download_static_tile(style_id, tilesize)
    assert os.path.isfile(path)
