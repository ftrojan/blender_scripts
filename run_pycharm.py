import bpy
import os

filename = os.path.join("/Users/ftrojan/BlenderScripts", "interplanety_cubes.py")
exec(compile(open(filename).read(), filename, 'exec'))
