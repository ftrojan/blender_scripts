# pip install to python used by blender

/Applications/Blender.app/Contents/Resources/2.92/python/bin/python3.7m -m pip install -e /Users/ftrojan/BlenderScripts/

# Inputs

# Texture - mapbox

https://docs.mapbox.com/playground/static/

https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/14.034,49.9673,16,0/600x600@2x?access_token=pk.eyJ1IjoiZnRyb2phbiIsImEiOiJjazlxcGN2eWEwM28yM3FwaW9kdThocmNjIn0.7owFB0UrzG_gV8HxIVCX1A

# Elevation - openrouteservice

https://openrouteservice.org/dev/#/api-docs/elevation/line/post
example body: {"format_in":"polyline","format_out":"polyline","dataset":"srtm","geometry":[[14.0342,49.9676],[14.1342,49.9676]]}

# Pipeline
- pipeline_init
- pipeline_mapbox
- pipeline_operouteservice
- create_mesh_from_geotiff
- Blender sculpt
- pipeline_map

Seems like the openrouteservice elevation can give maximum 1000 points.

# Mapbox styles

The following Mapbox styles are available to all accounts using a valid access token:
mapbox://styles/mapbox/streets-v11 # good
mapbox://styles/mapbox/outdoors-v11 # it has contour lines which would interefere with our own contour lines
mapbox://styles/mapbox/light-v10 # good
mapbox://styles/mapbox/dark-v10 # not for our use case
mapbox://styles/mapbox/satellite-v9 # good
mapbox://styles/mapbox/satellite-streets-v11 # street names and main street lines