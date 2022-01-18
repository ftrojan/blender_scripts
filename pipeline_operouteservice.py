"""
Operouteservice elevation part of the pipeline.

https://openrouteservice.org/dev/#/api-docs/elevation/line/post
"""
import logging
import os
import json
import requests

logging.basicConfig(level=logging.DEBUG)

# variables
pipeline_id = "beroun001"
geometry = [[14.0342, 49.9676], [14.1342, 49.9676]]

# constants
base_url = "https://api.openrouteservice.org/elevation/line"
data_dir = 'pipeline_data'
json_filename = 'elevation.json'

# request
header = {
    'Content-Type': 'application/json',
    'Authorization': '5b3ce3597851110001cf62488582da6a244e483ab57317ca5cf326a1'
}
data = {
    "format_in": "polyline",
    "format_out": "polyline",
    "dataset": "srtm",
    "geometry": geometry
}
response = requests.post(base_url, json=data, headers=header)

if response.ok:
    logging.debug(response.text)
    output_path = os.path.join(data_dir, pipeline_id, json_filename)
    with open(output_path, 'w') as output_file:
        json.dump(response.json(), output_file, indent=2)
else:
    logging.error(response.text)
