"""
Mapbox part of the pipeline. Superseded by pipeline_init.

https://docs.mapbox.com/playground/static/
https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/14.034,49.9673,16,0/600x600@2x?access_token=pk.eyJ1IjoiZnRyb2phbiIsImEiOiJjazlxcGN2eWEwM28yM3FwaW9kdThocmNjIn0.7owFB0UrzG_gV8HxIVCX1A
"""
import logging
import os
import requests
import json

logging.basicConfig(level=logging.DEBUG)

# constants
base_url = "https://api.mapbox.com/styles/v1/mapbox"
access_token = "pk.eyJ1IjoiZnRyb2phbiIsImEiOiJjazlxcGN2eWEwM28yM3FwaW9kdThocmNjIn0.7owFB0UrzG_gV8HxIVCX1A"
data_dir = 'pipeline_data'

# variables
pipeline_id = "mustek001"
input_json_path = os.path.join(data_dir, pipeline_id, "pipeline_mapbox_inputs.json")
with open(input_json_path, 'r') as fp:
    inputs = json.load(fp)
    logging.debug(inputs)


for style in inputs['styles']:
    png_filename = f"mapbox_{style}.png"

    # request
    url_bb = ",".join([
        f"{inputs[key]}" for key in [
            "min_longitude",
            "min_latitude",
            "max_longitude",
            "max_latitude"
        ]
    ])
    url_px = f"{inputs['width_px']}x{inputs['height_px']}"
    url = f"{base_url}/{style}/static/[{url_bb}]/{url_px}"
    param = dict(access_token=access_token)
    logging.debug(url)
    response = requests.get(url, param)

    if response.ok:
        output_path = os.path.join(data_dir, pipeline_id, png_filename)
        with open(output_path, 'wb') as output_file:
            output_file.write(response.content)
