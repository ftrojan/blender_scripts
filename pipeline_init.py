"""Initial part of the pipeline."""
import logging
import numpy as np
import utils
from brdy5_function import brdy5


logfmt = '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logfmt)

# variables
pipeline_id = "tok001"
utils.probe_pipeline(pipeline_id)
inputs = utils.pipeline_inputs(pipeline_id)
pipeline = dict(
    center=utils.GeoPoint(longitude=inputs["cen_longitude"], latitude=inputs["cen_latitude"]),
    size_meters=utils.SceneVector(x=inputs["size_meters_x"], y=inputs["size_meters_y"]),
)
logging.debug(pipeline)
mapbox_job = {
    style: dict(grid=utils.SceneVector(x=10, y=10), style=style)
    for style in inputs["styles"]
}
logging.debug(f"mapbox_job: {mapbox_job}")
grid_elevation_meters = utils.SceneVector(x=400, y=400)

g2s = utils.GeoSceneTransformer(center=pipeline['center'])
size = pipeline['size_meters']
bb_scene = np.array([
    [-0.5*size.x, -0.5*size.y],
    [+0.5*size.x, -0.5*size.y],
    [+0.5*size.x, +0.5*size.y],
    [-0.5*size.x, +0.5*size.y],
]).T
logging.debug(f"bb_scene: {bb_scene.T}")
bb_geo = g2s.scene_to_geo(bb_scene)
logging.debug(f"bb_geo: {bb_geo.T}")
dist = g2s.geo_line_distance(bb_geo)
logging.debug(f"bb_geo distances: {dist}")
bb_check = g2s.geo_to_scene(bb_geo)
logging.debug(f"bb_check: {bb_check.T}")
# utils.texture(pipeline_id, pipeline, mapbox_job, bb_geo)
# elgrid_points = utils.elevation_grid(size, grid_elevation_meters, g2s)
# utils.elevation(pipeline_id, bb_geo, zoom=15)
x, y, e = utils.grid_join_tiles(inputs["x1"], inputs["x2"], inputs["y1"], inputs["y2"], inputs["zoom"])
# tile = utils.MapboxTile(x=2198, y=1403, z=12)
# x, y, e = tile.elevation()
grid = utils.d3grid(x, y, e)
utils.save_d3grid(grid, pipeline_id)
grid2 = utils.d3grid_apply(grid, brdy5)
utils.save_d3grid(grid2, pipeline_id, filename="elevation_d3_brdy5.json")
images = utils.texture_join_tiles(inputs)
utils.save_texture(images, pipeline_id)
logging.info("completed")
