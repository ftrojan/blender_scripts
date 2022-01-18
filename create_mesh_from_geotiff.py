# import create_mesh_from_geotiff
# m = importlib.reload(create_mesh_from_geotiff); m.main()
# vertices, edges, faces = m.main()
# import bpy
import numpy as np
import geopy.distance
import geotiff


def tiff_geometry(file_path):
    tf = geotiff.GeoTiff(file_path)
    aa = tf.read()
    bb = tf.tif_bBox
    R = aa[:, :, 0]
    G = aa[:, :, 1]
    B = aa[:, :, 2]
    height = -10000.0 + 6553.6 * R + 25.6 * G + 0.1 * B
    xy0, xy1 = bb
    x0, y0 = xy0
    x1, y1 = xy1
    z0 = np.min(height)
    z1 = np.max(height)
    bb_xyz = ((x0, x1), (y0, y1), (z0, z1))
    return height, bb_xyz


def make_vef_pyramid() -> tuple:
    vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 0.5)]  # (x, y, z)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]  # (v1, v2), vi is zero-based
    faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 4, 3)]  # (v1, v2, v3) counter clockwise, vi is zero-based
    return vertices, edges, faces


def bounding_box_meters(bb_wgs: tuple) -> tuple:
    xx, yy, zz = bb_wgs
    x0, x1 = xx
    y0, y1 = yy
    dx = geopy.distance.geodesic((y0, x0), (y0, x1)).m
    dy = geopy.distance.geodesic((y0, x0), (y1, x0)).m
    print(f"dx={dx:.0f}m dy={dy:.0f}m")
    return (-0.5*dx, +0.5*dx), (-0.5*dy, +0.5*dy), zz


def wgs84_scene(vertices_wgs84, bbox_meters, units_per_meter):
    bbox_units = units_per_meter * bbox_meters
    print(f"bbox_units = {bbox_units}")
    center = np.mean(vertices_wgs84, axis=0)
    lam = [units_per_meter*bbox_meters[0], units_per_meter*bbox_meters[1], units_per_meter]
    print(f"center: {center}")
    vertices_scene = np.array([
        (lam[0]*(x - center[0]), lam[1]*(y - center[1]), lam[2]*z)
        for x, y, z in vertices_wgs84
    ])
    return vertices_scene


def make_vef(height, bb_meters) -> tuple:

    def vertex_idx(i: int, j: int) -> int:
        return i*nx + j

    def edge_x_idx(i: int, j: int) -> int:
        # index of the edge along x axis leading to vertex (i, j)
        return (nx - 1)*i + j - 1

    def edge_y_idx(i: int, j: int) -> int:
        # index of the edge along y axis leading to vertex (i, j)
        return num_edges_x - 1 + (ny - 1)*j + i

    def face_idx(i: int, j: int) -> int:
        return (i-1)*(nx-1) + (j-1)

    scale = 1/100  # 100 meters is one unit
    xxm, yym, zzm = bb_meters
    x0m, x1m = xxm
    y0m, y1m = yym
    z0m, z1m = zzm
    x0 = scale * x0m
    x1 = scale * x1m
    y0 = scale * y0m
    y1 = scale * y1m
    z0 = scale * z0m
    z1 = scale * z1m
    bb_scene = ((x0, x1), (y0, y1), (z0, z1))
    print(f"bbox_scene = {bb_scene}")
    ny, nx = height.shape
    ax = np.linspace(x0, x1, num=nx)
    ay = np.flip(np.linspace(y0, y1, num=ny))
    # preallocate arrays
    num_vertices = nx*ny
    print(f"nx: {nx}, ny: {ny}, vertices: {num_vertices}")
    num_edges_x = (nx - 1)*ny
    num_edges_y = (ny - 1)*nx
    num_edges = num_edges_x + num_edges_y
    print(f"edges: {num_edges_x} x + {num_edges_y} y = {num_edges}")
    num_faces = (nx - 1)*(ny - 1)
    print(f"faces: {nx - 1} * {ny - 1} = {num_faces}")
    vertices = [None] * num_vertices
    edges = [None] * num_edges
    faces = [None] * num_faces
    for i in range(1, ny):
        for j in range(1, nx):
            try:
                # vertices
                vertices[vertex_idx(i, j)] = (ax[j], ay[i], scale * height[i, j])
                if j == 1:
                    vertices[vertex_idx(i, 0)] = (ax[0], ay[i], scale * height[i, 0])
                    vertices[vertex_idx(i-1, 0)] = (ax[0], ay[i-1], scale * height[i-1, 0])
                if i == 1:
                    vertices[vertex_idx(0, j)] = (ax[j], ay[0], scale * height[0, j])
                # edges
                edges[edge_x_idx(i, j)] = (vertex_idx(i, j-1), vertex_idx(i, j))
                edges[edge_y_idx(i, j)] = (vertex_idx(i-1, j), vertex_idx(i, j))
                if i == 1:
                    edges[edge_x_idx(0, j)] = (vertex_idx(0, j-1), vertex_idx(0, j))
                if j == 1:
                    edges[edge_y_idx(i, 0)] = (vertex_idx(i-1, 0), vertex_idx(i, 0))
                # faces
                faces[face_idx(i, j)] = (
                    vertex_idx(i-1, j-1),
                    vertex_idx(i-1, j),
                    vertex_idx(i, j),
                    vertex_idx(i, j-1)
                )
            except IndexError as e:
                print(f"i={i}, j={j}: {e}")
    return vertices, edges, faces


def make_mesh(vertices, edges, faces):
    new_mesh = bpy.data.meshes.new('Beroun_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    return new_mesh


def draw_mesh(new_mesh):
    new_object = bpy.data.objects.new('Beroun', new_mesh)
    master_collection = bpy.context.scene.collection
    master_collection.objects.link(new_object)


def main():
    file_path = "/Users/ftrojan/Oracle Content - Accounts/Oracle Content/07-training/cesium/2020-05_jimmy/output/elevation.tif"
    h, bb = tiff_geometry(file_path)
    print(f"height {h.shape} min={h.min():5.1f} max={h.max():5.1f} mean={h.mean():5.1f}")
    print(f"bbox: {bb}")
    bb_meters = bounding_box_meters(bb)
    print(f"bbox_meters = {bb_meters}")
    h_small = h[0::3, 0::3]  # take every 3rd element in each dimension
    vertices, edges, faces = make_vef(h_small, bb_meters)
    # m = make_mesh(vertices, edges, faces)
    # draw_mesh(m)
    return vertices, edges, faces
