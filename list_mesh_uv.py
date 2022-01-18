# https://docs.blender.org/api/blender_python_api_2_67_release/bmesh.html#customdata-access
import bpy
import bmesh

# Get the active mesh
me = bpy.context.object.data


# Get a BMesh representation
bm = bmesh.new()   # create an empty BMesh
bm.from_mesh(me)   # fill it in from a Mesh

uv_lay = bm.loops.layers.uv.active

for i, face in enumerate(bm.faces):
    print(f"face #{i}/{len(bm.faces)}")
    for j, loop in enumerate(face.loops):
        uv = loop[uv_lay].uv
        vert = loop.vert
        print(f"loop #{j}/{len(face.loops)}: uv={uv} vert={vert.co}")
        # print("Loop UV: %f, %f" % uv[:])
        # print("Loop Vert: (%f,%f,%f)" % vert.co[:])
