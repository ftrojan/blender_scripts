"""Pipeline utility functions."""
import logging
import os
import json
import math
from typing import NamedTuple, Tuple, Optional
import numpy as np
import geopy.distance
import requests
from PIL import Image


R2D = 180 / math.pi


class GeoPoint(NamedTuple):

    longitude: float
    latitude: float

    def isclose(self, point: 'GeoPoint') -> bool:
        d = geo_distance(self, point)
        return d < 10.0

    def mapbox_tile(self, zoom: int) -> 'MapboxTile':
        mapbox_point = mapbox_geopoint_to_xy(self, zoom)
        return MapboxTile(
            x=int(mapbox_point.x),
            y=int(mapbox_point.y),
            z=zoom,
        )


GeoVector = GeoPoint


class GeoRectangle(NamedTuple):

    sw: GeoPoint
    ne: GeoPoint


class ScenePoint(NamedTuple):

    x: float
    y: float


SceneVector = ScenePoint


class MercatorPoint(NamedTuple):

    x: float
    y: float

    def to_geo(self) -> GeoPoint:
        # https://github.com/mapbox/mercantile/blob/fe3762d14001ca400caf7462f59433b906fc25bd/mercantile/__init__.py#L273
        lng = self.x
        lat = ((math.pi * 0.5) - 2.0 * math.atan(math.exp(self.y))) * R2D
        return GeoPoint(longitude=lng, latitude=lat)


class MapboxTile(NamedTuple):

    x: int
    y: int
    z: int

    def ul(self) -> GeoPoint:
        """Return the upper left GeoPoint of a tile."""
        # https://github.com/mapbox/mercantile/blob/fe3762d14001ca400caf7462f59433b906fc25bd/mercantile/__init__.py#L169
        z2 = math.pow(2, self.z)
        lon_deg = self.x / z2 * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * self.y / z2)))
        lat_deg = math.degrees(lat_rad)
        return GeoPoint(longitude=lon_deg, latitude=lat_deg)

    def br(self) -> GeoPoint:
        """Return the bottom right GeoPoint of a tile."""
        return MapboxTile(self.x + 1, self.y + 1, self.z).ul()

    def tile_key(self) -> str:
        key = f"{self.x}_{self.y}_{self.z}"
        return key

    def elevation_tile_path(self) -> str:
        elevation_tiles_dir = "tile_data"
        path = os.path.join(elevation_tiles_dir, f"elevation_{self.tile_key()}.png")
        return path

    def download_elevation(self):
        # https://docs.mapbox.com/help/troubleshooting/access-elevation-data/
        base_url = "https://api.mapbox.com/v4/mapbox.terrain-rgb"
        url = f"{base_url}/{self.z}/{self.x}/{self.y}.pngraw"
        param = dict(access_token=mapbox_token)
        response = requests.get(url, param)
        if response.ok:
            path = self.elevation_tile_path()
            with open(path, 'wb') as fp:
                fp.write(response.content)
                logging.debug(f"{path} saved")
        else:
            logging.error(response.text)

    def elevation_from_file(self) -> np.ndarray:
        with Image.open(self.elevation_tile_path()) as im:
            logging.debug(f"image size: {im.size}")
            imarray = np.array(im)
            logging.debug(f"image array shape: {imarray.shape}")
            red = imarray[:, :, 0]
            green = imarray[:, :, 1]
            blue = imarray[:, :, 2]
            elevation_array = -10000.0 + 6553.6 * red + 25.6 * green + 0.1 * blue
            return elevation_array

    def elevation(self) -> np.ndarray:
        if not os.path.isfile(self.elevation_tile_path()):
            self.download_elevation()
        e = self.elevation_from_file()
        return e


data_dir = 'pipeline_data'
mapbox_token = "pk.eyJ1IjoiZnRyb2phbiIsImEiOiJjazlxcGN2eWEwM28yM3FwaW9kdThocmNjIn0.7owFB0UrzG_gV8HxIVCX1A"
openrouteservice_token = '5b3ce3597851110001cf62488582da6a244e483ab57317ca5cf326a1'


class GeoSceneTransformer:

    def __init__(self, center: GeoPoint):
        geo_delta = GeoVector(longitude=0.1, latitude=0.1)
        # logging.debug(f"geo_delta: {geo_delta}")
        dx = geo_distance(center, GeoPoint(longitude=center.longitude + geo_delta.longitude, latitude=center.latitude))
        dy = geo_distance(center, GeoPoint(longitude=center.longitude, latitude=center.latitude + geo_delta.latitude))
        # logging.debug(f"scene_delta: {dx} {dy}")
        scene_lambda = SceneVector(
            x=dx / geo_delta.longitude,
            y=dy / geo_delta.latitude,
        )
        self.center = center
        self.matrix_g2s = np.array([
            [scene_lambda.x, 0.0, -scene_lambda.x * center.longitude],
            [0.0, scene_lambda.y, -scene_lambda.y * center.latitude],
        ])
        # logging.debug(f"matrix_g2s: {self.matrix_g2s}")
        self.matrix_s2g = np.array([
            [1.0/scene_lambda.x, 0.0, center.longitude],
            [0.0, 1.0/scene_lambda.y, center.latitude],
        ])
        # logging.debug(f"metrix_s2g: {self.matrix_s2g}")

    @staticmethod
    def homog(points: np.ndarray) -> np.ndarray:
        """
        Convert points to homogeneous coordinates - add 1 as the third element
        :param points: 2*N array of points
        :return: 3*N array of points with 1 added as the third element
        """
        nd, n = points.shape
        assert nd == 2
        points_homog = np.vstack((points, np.ones((1, n))))
        return points_homog

    def geo_to_scene(self, geo: np.ndarray) -> np.ndarray:
        """
        Transform geo coordinates to scene coordinates.

        :param geo: matrix 2*N, where N is number of points to transform.
            Longitude is geo[0,:], latitude is geo[1,:].
        :return: matrix 2*N, x is result[0,:], y is result[1,:]
        """
        geo_homog = self.homog(geo)
        scene = self.matrix_g2s @ geo_homog
        return scene

    def scene_to_geo(self, scene: np.ndarray) -> np.ndarray:
        """
        Transform scene coordinates to geo coordinates.
        :param scene: matrix 2*N, x is scene[0,:], y is scene[1,:]
        :return: matrix 2*N, longitude is result[0,:], latitude is result[1,:].
        """
        scene_homog = self.homog(scene)
        geo = self.matrix_s2g @ scene_homog
        return geo

    @staticmethod
    def geo_line_distance(points: np.ndarray) -> np.ndarray:
        """Calculate distances in meters between geodetic points of a line."""
        nd, n = points.shape
        dist = np.zeros(n - 1)
        for i in range(n - 1):
            p1 = points[:, i]
            p2 = points[:, i + 1]
            gp1 = GeoPoint(longitude=p1[0], latitude=p1[1])
            gp2 = GeoPoint(longitude=p2[0], latitude=p2[1])
            dist[i] = geo_distance(gp1, gp2)
        return dist


def geo_distance(point1: GeoPoint, point2: GeoPoint) -> float:
    """Geodesic distance in meters."""
    p1_geopy = (point1.latitude, point1.longitude)
    p2_geopy = (point2.latitude, point2.longitude)
    dist = geopy.distance.geodesic(p1_geopy, p2_geopy).m
    return dist


def size_png(size: SceneVector, step: SceneVector) -> Tuple[int, int]:
    width = np.round(size.x / step.x)
    height = np.round(size.y / step.y)
    return int(width), int(height)


def mapbox_png(pipeline_id: str, bb: np.ndarray, size_px: Tuple[int, int], style: str):
    bbox_string = f"{bb[0, 0]},{bb[1, 0]},{bb[0, 2]},{bb[1, 2]}"
    width_px, height_px = size_px

    # constants
    base_url = "https://api.mapbox.com/styles/v1/mapbox"
    png_filename = f'mapbox_{style}.png'

    # request
    url = f"{base_url}/{style}/static/[{bbox_string}]/{width_px}x{height_px}"
    param = dict(access_token=mapbox_token)
    logging.debug(url)
    response = requests.get(url, param)

    if response.ok:
        probe_pipeline(pipeline_id)
        output_path = os.path.join(data_dir, pipeline_id, png_filename)
        with open(output_path, 'wb') as output_file:
            output_file.write(response.content)
            logging.info(f"PNG file {width_px}*{height_px} saved to {output_path}.")
    else:
        logging.error(response.text)


def probe_pipeline(pipeline_id: str):
    pipeline_dir = os.path.join(data_dir, pipeline_id)
    if not os.path.isdir(pipeline_dir):
        os.mkdir(pipeline_dir)


def texture(pipeline_id, pipeline, mapbox_job, bbox):
    logging.debug(bbox)
    for jobid, jobspec in mapbox_job.items():
        w, h = size_png(pipeline['size_meters'], jobspec['grid'])
        logging.debug(f"png: {w} * {h}")
        mapbox_png(pipeline_id, bbox, (w, h), style=jobspec['style'])


def elevation_grid(size_meters: SceneVector, grid_step: SceneVector, t: GeoSceneTransformer) -> np.ndarray:
    ax = np.arange(start=-0.5*size_meters.x, stop=+0.5*size_meters.x, step=grid_step.x)
    ay = np.arange(start=-0.5*size_meters.y, stop=+0.5*size_meters.y, step=grid_step.y)
    ax = np.append(ax, +0.5*size_meters.x)
    ay = np.append(ay, +0.5*size_meters.y)
    nx = len(ax)
    ny = len(ay)
    logging.debug(f"ax ({nx}): {ax}, ay ({ny}): {ay}")
    xx, yy = np.meshgrid(ax, ay, indexing='ij')
    n = nx*ny
    logging.debug(f"xx {xx.shape}, yy {yy.shape}, n={n}")
    scene_points = np.zeros((2, n))
    for j in range(ny):
        for i in range(nx):
            k = j*nx + i
            scene_points[0, k] = xx[i, j]
            scene_points[1, k] = yy[i, j]
            # logging.debug(f"({i}, {j}): {scene_points[:, k]}")
    # logging.debug(f"scene points ({n}) = {k}: {scene_points.T}")
    geo_points = t.scene_to_geo(scene_points)
    return geo_points


def elevation(pipeline_id, bb, zoom):
    el = mapbox_elevation(bb, zoom)
    json_filename = 'elevation.json'
    output_path = os.path.join(data_dir, pipeline_id, json_filename)
    with open(output_path, 'w') as output_file:
        json.dump(el, output_file, indent=2)


def get_operouteservis_elevation(geometry: np.ndarray) -> dict:
    """Deprecated. Operouteservice has a limit of 1000 points per query."""
    # constants
    base_url = "https://api.openrouteservice.org/elevation/line"

    # request
    header = {
        'Content-Type': 'application/json',
        'Authorization': openrouteservice_token
    }
    data = {
        "format_in": "polyline",
        "format_out": "polyline",
        "dataset": "srtm",
        "geometry": geometry.T.tolist()
    }
    response = requests.post(base_url, json=data, headers=header)

    if response.ok:
        logging.debug(response.text)
        result = response.json()
    else:
        logging.error(response.text)
        result = {}
    return result


def sec(x):
    return 1 / math.cos(x)


def geo_to_mercator(geo_point: GeoPoint) -> MercatorPoint:
    """https://www.marksmath.org/classes/common/MapProjection.pdf"""
    #  https://github.com/mapbox/mercantile/blob/fe3762d14001ca400caf7462f59433b906fc25bd/mercantile/__init__.py#L268
    if geo_point.latitude <= -90.0:
        y = float("-inf")
    elif geo_point.latitude >= 90.0:
        y = float("inf")
    else:
        phi = math.radians(geo_point.latitude)
        tansec = abs(math.tan(phi) + sec(phi))
        y = math.log(tansec)
    return MercatorPoint(x=geo_point.longitude, y=y)


def mapbox_geopoint_to_xy(geo_point: GeoPoint, zoom: int) -> ScenePoint:
    """Return XY of a GeoPoint in coordinates in which Mapbox references its tiles."""
    # see https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    tile_count = pow(2, zoom)
    mp = geo_to_mercator(geo_point)
    x = (mp.x + 180) / 360
    y = (1 - mp.y / math.pi) / 2
    return ScenePoint(x=tile_count * x, y=tile_count * y)


def bbox_to_mapbox_xy(lon_min, lon_max, lat_min, lat_max, zoom):
    x_min, y_max = mapbox_geopoint_to_xy(GeoPoint(latitude=lat_min, longitude=lon_min), zoom)
    x_max, y_min = mapbox_geopoint_to_xy(GeoPoint(latitude=lat_max, longitude=lon_max), zoom)
    bb_deg = f"({lon_min:3f},{lat_min:3f}), ({lon_max:.3f},{lat_max:.3f})"
    bb_mb = f"({x_min:3f},{y_min:3f}), ({x_max:.3f},{y_max:.3f})"
    logging.debug(f"degrees bbox {bb_deg}, zoom={zoom} -> mapbox bbox {bb_mb}")
    return math.floor(x_min), math.floor(x_max), math.floor(y_min), math.floor(y_max)


def mapbox_elevation(bb: np.ndarray, zoom: int) -> bytes:
    """From 07-training/cesium/2020-05_jimmy/tiles_to_tiff.py"""

    sw, se, ne, nw = tuple(bb.T)
    lon_min, lat_min = tuple(sw)
    lon_max, lat_max = tuple(ne)
    x_min, x_max, y_min, y_max = bbox_to_mapbox_xy(lon_min, lon_max, lat_min, lat_max, zoom)
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    elevation_tiles_dir = "tile_data"
    num_tiles = nx * ny
    logging.info(f"Downloading {nx}*{ny}={num_tiles} tiles to {elevation_tiles_dir}")
    tile_id = 0
    tile_data = {}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_id += 1
            tile = MapboxTile(x, y, zoom)
            logging.debug(f"{tile_id}/{num_tiles}: tile ul={tile.ul()} br={tile.br()}")
            elevation_array = tile.elevation()
            logging.debug(f"el min={np.min(elevation_array):.1f} max={np.max(elevation_array):.1f}")
            tile_data[tile.tile_key()] = dict(
                tile=tile,
                elevation=elevation_array,
            )


def srtm_tile_bb(a: int, b: int) -> GeoRectangle:
    """See srtm_precision."""
    xllcorner = -180 + 5 * (a - 1)
    yllcorner = -15 + 5 * (b - 1)
    xurcorner = xllcorner + 5
    yurcorner = yllcorner + 5
    bb = GeoRectangle(
        sw=GeoPoint(longitude=xllcorner, latitude=yllcorner),
        ne=GeoPoint(longitude=xurcorner, latitude=yurcorner),
    )
    return bb


def pipeline_inputs(pipeline_id: str) -> dict:
    """Read pipeline inputs JSON parameters."""
    input_json_path = os.path.join(data_dir, pipeline_id, "pipeline_mapbox_inputs.json")
    with open(input_json_path, 'r') as fp:
        inputs = json.load(fp)
        logging.debug(inputs)
    return inputs