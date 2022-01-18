# https://b3d.interplanety.org/en/how-to-create-mesh-through-the-blender-python-api/
import bpy
 
# make mesh - pyramid with 5 vertices and 4 faces. square 2 units long. top 0.5 units high.
vertices = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 0.5)]  # (x, y, z)
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]  # (v1, v2), vi is zero-based
faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 4, 3)]  # (v1, v2, v3) counter clockwise, vi is zero-based
new_mesh = bpy.data.meshes.new('new_mesh')
new_mesh.from_pydata(vertices, edges, faces)
new_mesh.update()
# make object from mesh
new_object = bpy.data.objects.new('new_object', new_mesh)
# make collection
new_collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(new_collection)
# add object to scene collection
new_collection.objects.link(new_object)