import bpy
from random import randint

number = 60
for i in range(number):
    x = randint(-10, 10)
    y = randint(-10, 10)
    z = randint(-10, 10)
    bpy.ops.mesh.primitive_cube_add(location=[x, y, z])
