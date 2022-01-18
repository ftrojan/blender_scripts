import logging
import numpy as np
import geotiff

fn = "/Users/ftrojan/BlenderScripts/read_geotiff.log"
# logging.basicConfig(level=logging.DEBUG, filename=fn)
# logging.debug("started")
with open(fn, 'w') as log:
    file_path = "/Users/ftrojan/Oracle Content - Accounts/Oracle Content/07-training/cesium/2020-05_jimmy/output/elevation.tif"
    tf = geotiff.GeoTiff(file_path)
    aa = tf.read()
    log.write(f"all bands array {aa.shape} min={np.min(aa):4.0f} max={np.max(aa):4.0f} mean={np.mean(aa):5.1f}\n")
    # decode data as per https://docs.mapbox.com/help/troubleshooting/access-elevation-data/
    R = aa[:, :, 0]
    log.write(f"R: {R.shape} min={R.min():4.0f} max={R.max():4.0f} mean={R.mean():5.1f}\n")
    G = aa[:, :, 1]
    log.write(f"G: {G.shape} min={G.min():4.0f} max={G.max():4.0f} mean={G.mean():5.1f}\n")
    B = aa[:, :, 2] 
    log.write(f"B: {B.shape} min={B.min():4.0f} max={B.max():4.0f} mean={B.mean():5.1f}\n")
    height = -10000.0 + 6553.6 * R + 25.6 * G + 0.1 * B
    log.write(f"height {height.shape} min={height.min():5.1f} max={height.max():5.1f} mean={height.mean():5.1f}\n")
