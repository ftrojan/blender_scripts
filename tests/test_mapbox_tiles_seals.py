"""
This test confirms that distances between the tiles are the same as the distances within the tiles.

This test looks at the mean absolute elevation differences at the borders between tiles.

Taking example of mustek001 pipeline, we read all the 13*14=182 elevation tiles.
For each inner tile, we take four neighbour tiles - to the north, south, west and east.
We calculate mean absolute differences (MAD) at the border and compare with the MAD in the
X direction (longitude) and Y direction (latitude) within the inner tile.

We compare the distributions of the MADs and if they are similar, we conclude that
distances between the tiles are the same as the distances within the tiles in the
respective dimensions.

Copy pasted log is below. We can see that:
- south seals, north seals and y diffs are distributed the same
- west seals, east seals and x diffs are distributed the same

 INFO test_mapbox_tiles_seals.py:126 MAD south seals:
count    132.000000
mean       0.342554
std        0.185357
min        0.086328
25%        0.201953
50%        0.315625
75%        0.413379
max        1.185547
Name: south, dtype: float64
 INFO test_mapbox_tiles_seals.py:127 MAD north seals:
count    132.000000
mean       0.335301
std        0.152164
min        0.086328
25%        0.203516
50%        0.316602
75%        0.416406
max        0.972656
Name: north, dtype: float64
 INFO test_mapbox_tiles_seals.py:128 MAD y-dir diffs:
count    132.000000
mean       0.330332
std        0.119180
min        0.132350
25%        0.249003
50%        0.309461
75%        0.391134
max        0.768871
Name: y, dtype: float64
 INFO test_mapbox_tiles_seals.py:129 MAD west seals:
count    132.000000
mean       0.380566
std        0.208652
min        0.060156
25%        0.220508
50%        0.343164
75%        0.506152
max        0.933203
Name: west, dtype: float64
 INFO test_mapbox_tiles_seals.py:130 MAD east seals:
count    132.000000
mean       0.370286
std        0.211080
min        0.056250
25%        0.200684
50%        0.322656
75%        0.501074
max        0.933203
Name: east, dtype: float64
 INFO test_mapbox_tiles_seals.py:131 MAD x-dir diffs:
count    132.000000
mean       0.365090
std        0.144268
min        0.144697
25%        0.236663
50%        0.349018
75%        0.458748
max        0.708592
Name: x, dtype: float64

Filip Trojan 2022-01-12
"""
from typing import NamedTuple
import logging
import io
import requests
import numpy as np
import pandas as pd
from PIL import Image
import utils


logfmt = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=logfmt)


class TileSealsStats(NamedTuple):
    idx: int
    idy: int
    south: float
    north: float
    west: float
    east: float
    x: float
    y: float


def read_elevation_tiles(x_min, x_max, y_min, y_max, zoom) -> dict:
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    num_tiles = nx * ny
    logging.info(f"Reading {nx}*{ny}={num_tiles} tiles")
    tile_id = 0
    tile_data = {}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_id += 1
            tile = utils.MapboxTile(x, y, zoom)
            logging.info(f"{tile_id}/{num_tiles}: tile ul={tile.ul()} br={tile.br()}")
            elevation_array = tile.elevation()
            tile_data[tile.tile_key()] = dict(
                tile=tile,
                elevation=elevation_array,
            )
    return tile_data


def calc_seal_stats(x_min, x_max, y_min, y_max, zoom, tile_data) -> pd.DataFrame():
    # check seals on edges between tiles
    inner_tiles = []
    for x in range(x_min + 1, x_max):
        for y in range(y_min + 1, y_max):
            td = tile_data[f"{x}_{y}_{zoom}"]["elevation"]
            td_north = tile_data[f"{x}_{y + 1}_{zoom}"]["elevation"]
            td_south = tile_data[f"{x}_{y - 1}_{zoom}"]["elevation"]
            td_west = tile_data[f"{x - 1}_{y}_{zoom}"]["elevation"]
            td_east = tile_data[f"{x + 1}_{y}_{zoom}"]["elevation"]
            tile_stats = TileSealsStats(
                idx=x,
                idy=y,
                south=float(np.mean(np.abs(td_south[-1, :] - td[0, :]))),
                north=float(np.mean(np.abs(td_north[0, :] - td[-1, :]))),
                west=float(np.mean(np.abs(td_west[:, -1] - td[:, 0]))),
                east=float(np.mean(np.abs(td_east[:, 0] - td[:, -1]))),
                x=float(np.mean(np.abs(td[:, 0:-2] - td[:, 1:-1]))),
                y=float(np.mean(np.abs(td[0:-2, :] - td[1:-1, :]))),
            )
            inner_tiles.append(tile_stats)
    df = pd.DataFrame(inner_tiles).set_index(keys=["idx", "idy"])
    return df


def test_mapbox_tiles_seals():
    logging.info("started")
    pipeline_id = "mustek001"
    zoom = 15
    inputs = utils.pipeline_inputs(pipeline_id)
    pipeline = dict(
        center=utils.GeoPoint(longitude=inputs["cen_longitude"], latitude=inputs["cen_latitude"]),
        size_meters=utils.SceneVector(x=inputs["size_meters_x"], y=inputs["size_meters_y"]),
    )
    logging.debug(pipeline)
    g2s = utils.GeoSceneTransformer(center=pipeline['center'])
    size = pipeline['size_meters']
    bb_scene = np.array([
        [-0.5 * size.x, -0.5 * size.y],
        [+0.5 * size.x, -0.5 * size.y],
        [+0.5 * size.x, +0.5 * size.y],
        [-0.5 * size.x, +0.5 * size.y],
    ]).T
    bb_geo = g2s.scene_to_geo(bb_scene)
    sw, se, ne, nw = tuple(bb_geo.T)
    lon_min, lat_min = tuple(sw)
    lon_max, lat_max = tuple(ne)
    x_min, x_max, y_min, y_max = utils.bbox_to_mapbox_xy(lon_min, lon_max, lat_min, lat_max, zoom)
    tile_data = read_elevation_tiles(x_min, x_max, y_min, y_max, zoom)
    seal_stats = calc_seal_stats(x_min, x_max, y_min, y_max, zoom, tile_data)
    logging.info(f"MAD south seals:\n{seal_stats.south.describe()}")
    logging.info(f"MAD north seals:\n{seal_stats.north.describe()}")
    logging.info(f"MAD y-dir diffs:\n{seal_stats.y.describe()}")
    logging.info(f"MAD west seals:\n{seal_stats.west.describe()}")
    logging.info(f"MAD east seals:\n{seal_stats.east.describe()}")
    logging.info(f"MAD x-dir diffs:\n{seal_stats.x.describe()}")
    logging.info("completed")
