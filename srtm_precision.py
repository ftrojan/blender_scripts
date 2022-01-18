"""
Calculate and print SRTM precision at Beroun point.

Tiles are stored at https://drive.google.com/file/d/138_iPk9mIo3MzEjWE-K4lIzRPYCsgdmt/view?usp=sharing
Each tile is srtm_{a}_{b}.zip
srtm_01_07.zip has srtm_01_07.asc plain text file with the following header:

srtm_01_07.asc:
ncols         6000
nrows         6000
xllcorner     -180
yllcorner     25
cellsize      0.00083333333333333
NODATA_value  -9999

srtm_15_09.asc
ncols         6000
nrows         6000
xllcorner     -110
yllcorner     15
cellsize      0.00083333333333333
NODATA_value  -9999

We infer that
- this tile is 5x5 degrees (5 = 6000*cellsize),
- cellsize is 3 minutes (cellsize = 1/1200, 3 = cellsize*60*60)
- xllcorner = -180 + 5*(a - 1)
- yllcorner = -15 + 5*(b - 1)

The output for center = utils.GeoPoint(longitude=14.0340, latitude=49.9673) was:
dx=59.8m dy=92.7m
"""
import logging
import utils

logfmt = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logfmt)

center = utils.GeoPoint(longitude=14.0340, latitude=49.9673)
cellsize = 1/1200
a = 39
b = 11

g2s = utils.GeoSceneTransformer(center)
dx = utils.geo_distance(center, utils.GeoPoint(longitude=center.longitude + cellsize, latitude=center.latitude))
dy = utils.geo_distance(center, utils.GeoPoint(longitude=center.longitude, latitude=center.latitude + cellsize))
logging.info(f"dx={dx:.1f}m dy={dy:.1f}m")
bb = utils.srtm_tile_bb(a, b)
logging.info(f"srtm_{a:02d}_{b:02d} is {bb}")
