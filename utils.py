"""Pipeline utility functions."""
import logging
import os
import json
import math
from typing import NamedTuple, Tuple, List
import numpy as np
import geopy.distance
import requests
from PIL import Image

logger = logging.getLogger(__name__)
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

    def __repr__(self):
        h = self.size_m()
        s = f"MapboxTile x={self.x} y={self.y} z={self.z} size={h.x:.0f}*{h.y:.0f}m"
        return s

    def ul(self) -> GeoPoint:
        """Return the upper left GeoPoint of a tile."""
        # https://github.com/mapbox/mercantile/blob/fe3762d14001ca400caf7462f59433b906fc25bd/mercantile/__init__.py#L169
        z2 = math.pow(2, self.z)
        lon_deg = self.x / z2 * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * self.y / z2)))
        lat_deg = math.degrees(lat_rad)
        return GeoPoint(longitude=lon_deg, latitude=lat_deg)

    def ur(self) -> GeoPoint:
        """Return the upper right GeoPoint of the tile."""
        return MapboxTile(self.x + 1, self.y, self.z).ul()

    def bl(self) -> GeoPoint:
        """Return the bottom left GeoPoint of the tile."""
        return MapboxTile(self.x, self.y + 1, self.z).ul()

    def br(self) -> GeoPoint:
        """Return the bottom right GeoPoint of a tile."""
        return MapboxTile(self.x + 1, self.y + 1, self.z).ul()

    def size_m(self) -> SceneVector:
        """Return size of the tile in meters."""
        w = geo_distance(self.ul(), self.ur())
        h = geo_distance(self.ul(), self.bl())
        return SceneVector(x=w, y=h)

    def tile_key(self) -> str:
        key = f"{self.x}_{self.y}_{self.z}"
        return key

    def elevation_tile_path(self) -> str:
        elevation_tiles_dir = "tile_data"
        path = os.path.join(elevation_tiles_dir, f"elevation_{self.tile_key()}.png")
        return path

    def static_tile_path(self, style_id: str, tilesize: int):
        elevation_tiles_dir = "tile_data"
        path = os.path.join(elevation_tiles_dir, f"{style_id}_{self.tile_key()}_{tilesize}.jpeg")
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
                logging.info(f"{path} saved")
        else:
            logging.error(response.text)

    def download_static_tile(self, style_id: str, tilesize: int = 512):
        """
        Download and save raster image 512*512 from mapbox.

        :param style_id: satellite-v9, streets-v11
        :param tilesize: The size in pixels of the returned tile, either 512 or 256.
        """
        # https://docs.mapbox.com/api/maps/static-tiles/
        # /styles/v1/{username}/{style_id}/tiles/{tilesize}/{z}/{x}/{y}{@2x}
        base_url = "https://api.mapbox.com/styles/v1/mapbox"
        url = f"{base_url}/{style_id}/tiles/{tilesize}/{self.z}/{self.x}/{self.y}"
        param = dict(access_token=mapbox_token)
        logger.debug(f"GET {url}")
        response = requests.get(url, param)
        if response.ok:
            path = self.static_tile_path(style_id, tilesize)
            with open(path, 'wb') as fp:
                fp.write(response.content)
                logging.info(f"{path} saved")
        else:
            logging.error(response.text)

    def elevation_from_file(self) -> np.ndarray:
        with Image.open(self.elevation_tile_path()) as im:
            imarray = np.array(im)
            red = imarray[:, :, 0]
            green = imarray[:, :, 1]
            blue = imarray[:, :, 2]
            elevation_array = -10000.0 + 6553.6 * red + 25.6 * green + 0.1 * blue
            return elevation_array

    def static_tile_from_file(self, style_id: str, tilesize: int = 512) -> Image:
        return Image.open(self.static_tile_path(style_id, tilesize))

    def elevation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not os.path.isfile(self.elevation_tile_path()):
            self.download_elevation()
        e = self.elevation_from_file()
        nx, ny = e.shape
        p0 = self.bl()
        p1 = self.ur()
        x = np.linspace(start=p0.longitude, stop=p1.longitude, num=nx)
        y = np.linspace(start=p0.latitude, stop=p1.latitude, num=ny)
        logging.info(f"elevation between {np.min(e)} and {np.max(e)} on {nx}*{ny} grid")
        return x, y, e

    def static_tile(self, style_id: str, tilesize: int = 512) -> Image:
        if not os.path.isfile(self.static_tile_path(style_id, tilesize)):
            self.download_static_tile(style_id, tilesize)
        im = self.static_tile_from_file(style_id, tilesize)
        logging.info(f"static tile {style_id} {im.width}*{im.height}")
        return im


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
        os.makedirs(pipeline_dir, exist_ok=True)


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


def grid_join_tiles(x1: int, x2: int, y1: int, y2: int, z: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join tiles elevation into a single array."""
    tile_nx, tile_ny = 256, 256
    ntx, nty = x2 - x1 + 1, y2 - y1 + 1
    x = np.zeros((ntx * tile_nx))
    y = np.zeros((nty * tile_ny))
    e = np.zeros((ntx * tile_nx, nty * tile_ny))
    for ix, tx in enumerate(range(x1, x2 + 1)):
        for iy, ty in enumerate(range(y1, y2 + 1)):
            tile = MapboxTile(tx, ty, z)
            xx, yy, ee = tile.elevation()
            jx = slice(tile_nx * ix, tile_nx * (ix + 1))
            jy = slice(tile_ny * iy, tile_ny * (iy + 1))
            x[jx] = xx
            y[jy] = yy
            e[jy, jx] = ee
    return x, y, e


def d3grid(x, y, z) -> dict:
    """Create JSON to pass into d3.contours from x vector, y vector and z 2D array."""
    ny = len(y)
    nx = len(x)
    assert z.shape == (ny, nx)
    p = ny * nx
    v = p * [0]
    for iy in range(ny):
        for ix in range(nx):
            v[iy * nx + ix] = z[iy, ix]
    result = dict(
        width=nx,
        height=ny,
        values=v,
        x1=float(x[0]),
        x2=float(x[-1]),
        y1=float(y[0]),
        y2=float(y[-1]),
    )
    logging.info(f"grid {nx}*{ny}={p} created")
    return result


def d3grid_apply(d: dict, fun) -> dict:
    """Apply scalar function fun to values and return dictionary."""
    values = [fun(v) for v in d["values"]]
    d["values"] = values
    return d


def save_d3grid(grid, pipeline_id, filename: str = None):
    if filename is None:
        filename = "elevation_d3.json"
    json_path = os.path.join(data_dir, pipeline_id, filename)
    with open(json_path, 'w') as fp:
        json.dump(grid, fp, indent=2)
    logging.info(f"grid {grid['width']}*{grid['height']} saved to {json_path}")


def texture_join_tiles(inp: dict, tilesize: int = 512) -> dict:
    result = {}
    ntx, nty = inp["x2"] - inp["x1"] + 1, inp["y2"] - inp["y1"] + 1
    for style in inp["styles"]:
        a = {}
        for ix, tx in enumerate(range(inp["x1"], inp["x2"] + 1)):
            for iy, ty in enumerate(range(inp["y1"], inp["y2"] + 1)):
                tile = MapboxTile(tx, ty, inp["zoom"])
                imxy = tile.static_tile(style, tilesize)
                a[ix, iy] = imxy.copy()
        sx = [max([a[ix, iy].width for iy in range(nty)]) for ix in range(ntx)]
        sy = [max([a[ix, iy].height for ix in range(ntx)]) for iy in range(nty)]
        cx = [sum(sx[:ix]) for ix in range(ntx)]
        cy = [sum(sy[:iy]) for iy in range(nty)]
        im = Image.new('RGB', (sum(sx), sum(sy)))
        logging.info(f"created {im}")
        for ix in range(ntx):
            for iy in range(nty):
                imxy = a[ix, iy]
                logging.info(f"pasting {imxy.width}*{imxy.height} at {ix}, {iy} = {cx[ix]}, {cy[iy]}")
                im.paste(imxy, (cx[ix], cy[iy]))
        result[style] = im
    return result


def save_texture(textures: dict, pipeline_id: str):
    for style, im in textures.items():
        texture_path = os.path.join(data_dir, pipeline_id, f"texture_{style}.png")
        im.save(texture_path, format="png")
