"""
This test verifies hypothesis on the positioning of mapbox raster elevation points.

The hypothesis is that the elevation points are aligned with the bounding box.
This is unlike raster image pixels, where the pixels correspond with the centers of
rectangles which fill the mapbox tile.

The verification is done by comparison of raster elevations on two consecutive zoom levels.
If the elevation points are aligned with the bounding box, it should hold that

e00[0:2:256, 0:2:256] == e[0:128, 0:128]
e01[0:2:256, 0:2:256] == e[0:128, 128:256]
e10[0:2:256, 0:2:256] == e[128:256, 0:128]
e11[0:2:256, 0:2:256] == e[128:256, 128:256]

where e=tile(x,y) is elevation at zoom level Z and
e00=tile(2*x,2*y)
e01=tile(2*x,2*y+1)
e10=tile(2*x+1,2*y)
e11=tile(2*x+1,2*y+1)
are elevations at zoom level Z+1 on the child tiles

The results reject the hypothesis. The four distances are greater than zero,
they are of the same magnitude as the elevation differences within the grid
and they are increasing with zoom out.

We conclude that the Mapbox elevation points correspond to the centers of each 256*256 rectangle
making the tile.

Filip Trojan 2022-01-18
"""

import pytest
import logging
import numpy as np
import pandas as pd
import utils


logfmt = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s'
i2 = slice(0, 256, 2)
i0 = slice(0, 128)
i1 = slice(128, 256)
tol = 1.0  # tolerance for MAD in meters
logging.basicConfig(level=logging.INFO, format=logfmt)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@pytest.mark.parametrize("ix,iy,zoom", [
    (8793, 5613, 14),
    (4397, 2807, 13),
    (2198, 1403, 12),
    (1099, 701, 11),
])
def test_elevation_points_position(ix, iy, zoom):
    tile = utils.MapboxTile(x=ix, y=iy, z=zoom)
    logging.info(f"started with {tile}")
    t00 = utils.MapboxTile(x=2 * ix, y=2 * iy, z=zoom + 1)
    t01 = utils.MapboxTile(x=2 * ix, y=2 * iy + 1, z=zoom + 1)
    t10 = utils.MapboxTile(x=2 * ix + 1, y=2 * iy, z=zoom + 1)
    t11 = utils.MapboxTile(x=2 * ix + 1, y=2 * iy + 1, z=zoom + 1)
    e = tile.elevation()
    e00 = t00.elevation()
    e01 = t01.elevation()
    e10 = t10.elevation()
    e11 = t11.elevation()
    d00 = dist(e00[i2, i2], e[i0, i0])
    d01 = dist(e01[i2, i2], e[i1, i0])
    d10 = dist(e10[i2, i2], e[i0, i1])
    d11 = dist(e11[i2, i2], e[i1, i1])
    logging.info(f"completed MAD00={d00:.1f}m MAD01={d01:.1f}m MAD10={d10:.1f}m MAD11={d11:.1f}m")
