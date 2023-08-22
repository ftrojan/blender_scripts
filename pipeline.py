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
x, y, e = utils.grid_join_tiles(inputs)
grid = utils.d3grid(x, y, e)
utils.save_d3grid(grid, pipeline_id)
grid2 = utils.d3grid_apply(grid, brdy5)
utils.save_d3grid(grid2, pipeline_id, filename="elevation_d3_brdy5.json")
images = utils.texture_join_tiles(inputs)
utils.save_texture(images, pipeline_id)
logging.info("completed")
