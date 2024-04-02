import bpy
import os
import pathlib
from glob import glob

WORKING_DIR = pathlib.Path(__file__).parent.resolve()

def delete_hierarchy(parent_obj):
  for child_obj in parent_obj.children:
      bpy.data.objects.remove(child_obj, do_unlink=True)
  bpy.data.objects.remove(parent_obj, do_unlink=True)

def delete_obj(obj_name='Armature'):
  parent_obj = bpy.data.objects.get(obj_name)
  delete_hierarchy(parent_obj)

def dump_objs(mesh_name):
  os.makedirs(os.path.join(WORKING_DIR, "outputs", mesh_name))

  for i in range(30):
    bpy.context.scene.frame_set(i)
    bpy.ops.export_scene.obj(
      filepath=os.path.join(WORKING_DIR, "outputs", mesh_name, f"{i}.obj"),
      filter_glob="*.obj",
      use_selection=True)

animated_mixamo_file = glob(os.path.join(WORKING_DIR, "animation", "*.fbx"))[0]
bpy.ops.import_scene.fbx(filepath=animated_mixamo_file)
anim_obj = bpy.context.selected_objects[0]
action = anim_obj.animation_data.action

delete_obj()

fbx_files = glob(os.path.join(WORKING_DIR, "fbx_files", "*.fbx"))

for fbx_file in fbx_files:
  bpy.ops.import_scene.fbx(filepath=fbx_file)
  obj = bpy.context.selected_objects[0]
  obj.animation_data_create()
  obj.animation_data.action = action

  filename = os.path.basename(fbx_file)[:-4]
  dump_objs(filename)
  delete_obj()