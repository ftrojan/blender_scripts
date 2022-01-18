# https://b3d.interplanety.org/en/using-external-ide-pycharm-for-writing-blender-scripts/
import bpy

bpy.ops.mesh.primitive_cube_add(location=[0, 0, 0])
Obj = bpy.context.active_object
mod = Obj.modifiers.new("Bevel", 'BEVEL')
mod.segments = 3
bpy.ops.object.shade_smooth()
mod1 = Obj.modifiers.new("Array", 'ARRAY')
mod1.count = 20
mod2 = Obj.modifiers.new("Array", 'ARRAY')
mod2.relative_offset_displace[0] = 0
mod2.relative_offset_displace[1] = 1
mod2.count = 20
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Array.001")
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Array")
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.separate(type="LOOSE")
bpy.ops.object.editmode_toggle()
bpy.ops.object.randomize_transform(loc=[0, 0, 1])
