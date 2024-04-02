

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
import numpy as np
import open3d as o3d

from tqdm import tqdm
import argparse
from natsort import natsorted 

tf.enable_eager_execution()
from waymo_open_dataset.utils import  frame_utils, box_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2, label_pb2 
from waymo_map_utils import get_mask
from lidardm.datasets.utils import encode, decode
from PIL import Image
from lidardm.visualization.cond2rgb import cond_to_rgb_waymo
import matplotlib.pyplot as plt 
import copy

# These functions are modified from Waymo's frame_utils to get lidar scans for only 
# the top Lidar to speed up data_dumping process (by 3x)

# Copyright 2019 The Waymo Open Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def convert_range_image_to_cartesian(frame,
									 range_image,
									 range_image_top_pose,
									 keep_polar_features=False):
	cartesian_range_images = None
	frame_pose = tf.convert_to_tensor(
			value=np.reshape(np.array(frame.pose.transform), [4, 4]))

	# [H, W, 6]
	range_image_top_pose_tensor = tf.reshape(
			tf.convert_to_tensor(value=range_image_top_pose.data),
			range_image_top_pose.shape.dims)
	# [H, W, 3, 3]
	range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
			range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
			range_image_top_pose_tensor[..., 2])
	range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
	range_image_top_pose_tensor = transform_utils.get_transform(
			range_image_top_pose_tensor_rotation,
			range_image_top_pose_tensor_translation)

	for c in frame.context.laser_calibrations:
		if c.name != dataset_pb2.LaserName.TOP:
			continue

		if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
			beam_inclinations = range_image_utils.compute_inclination(
					tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
					height=range_image.shape.dims[0])
		else:
			beam_inclinations = tf.constant(c.beam_inclinations)

		beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
		extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

		range_image_tensor = tf.reshape(
				tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
		pixel_pose_local = None
		frame_pose_local = None
		if c.name == dataset_pb2.LaserName.TOP:
			pixel_pose_local = range_image_top_pose_tensor
			pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
			frame_pose_local = tf.expand_dims(frame_pose, axis=0)
		range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
				tf.expand_dims(range_image_tensor[..., 0], axis=0),
				tf.expand_dims(extrinsic, axis=0),
				tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
				pixel_pose=pixel_pose_local,
				frame_pose=frame_pose_local)

		range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

		if keep_polar_features:
			# If we want to keep the polar coordinate features of range, intensity,
			# and elongation, concatenate them to be the initial dimensions of the
			# returned Cartesian range image.
			range_image_cartesian = tf.concat(
					[range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

		cartesian_range_images = range_image_cartesian

	return cartesian_range_images, frame_pose.numpy()

def convert_range_image_to_point_cloud(frame,
									   range_image,
									   range_image_top_pose,
									   keep_polar_features=False):
	
	points = None

	cartesian_range_images, frame_pose = convert_range_image_to_cartesian(
			frame, range_image, range_image_top_pose, keep_polar_features)

	range_image_tensor = tf.reshape(
			tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
	range_image_mask = range_image_tensor[..., 0] > 0

	range_image_cartesian = cartesian_range_images
	points_tensor = tf.gather_nd(range_image_cartesian,tf.compat.v1.where(range_image_mask))
	points = points_tensor.numpy()

	return points, frame_pose

def parse_range_image_top_lidar(frame):
	range_image = None
	range_image_top_pose = None

	for laser in frame.lasers:
		if laser.name != dataset_pb2.LaserName.TOP:
			continue

		if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
			range_image_str_tensor = tf.io.decode_compressed(
					laser.ri_return1.range_image_compressed, 'ZLIB')
			ri = dataset_pb2.MatrixFloat()
			ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
			range_image = ri

			range_image_top_pose_str_tensor = tf.io.decode_compressed(
					laser.ri_return1.range_image_pose_compressed, 'ZLIB')
			range_image_top_pose = dataset_pb2.MatrixFloat()
			range_image_top_pose.ParseFromString(
					bytearray(range_image_top_pose_str_tensor.numpy()))

			break
	return range_image, range_image_top_pose

def get_lidar_point_cloud(frame):
	range_images, range_image_top_pose = parse_range_image_top_lidar(frame)
	points, pose = convert_range_image_to_point_cloud(frame, range_images, range_image_top_pose)

	return points, pose

def get_dynamic_bbox(frame, speed_threshold):
	bboxes = []
	ids = []
	for label in frame.laser_labels:
		speed = np.linalg.norm(np.array([label.metadata.speed_x, 
								   		 label.metadata.speed_y,
										 label.metadata.speed_z]))
		if speed > speed_threshold:
			box = box_utils.box_to_tensor(label.box).numpy()
			box[-4:-1] += [1, 0.5, 0.5]
			bboxes.append(box)
			current_type = "Unknown"
			if label.type == label_pb2.Label.TYPE_VEHICLE: current_type = "Vehicle"
			elif label.type == label_pb2.Label.TYPE_PEDESTRIAN: current_type = "Pedestrian"
			ids.append(np.array([label.id, current_type, label.box.heading]))

	bboxes = tf.convert_to_tensor(bboxes)
	return bboxes, ids

def filter_dynamic_pcd(points, bboxes_tf):
	if len(bboxes_tf.numpy().shape) != 2:
		return points

	mask = box_utils.is_within_box_3d(points, bboxes_tf).numpy()
	mask = np.any(mask, axis=1)

	points = points[mask == False]
	return points

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Script that preprocess Waymo Open for dynamic filtering and pose dumping')
	parser.add_argument('--tfrecord_paths', type=str, help='tfrecord_paths')
	parser.add_argument('--split', type=str, nargs='+', help='pose file')
	parser.add_argument('--out_dir', type=str, help='output folder')
	parser.add_argument('--viz', action=argparse.BooleanOptionalAction, help='visualize or nah')
	args = parser.parse_args()
 
	for split_type in args.split:
		tfrecords_folder = os.path.join(args.tfrecord_paths, split_type)
		print("Processing", tfrecords_folder)

		tfrecords = natsorted(os.listdir(tfrecords_folder)) 
		for tfrecord in tqdm(tfrecords):
			output_dir = args.out_dir

			output_scan_folder = os.path.join(output_dir, split_type, tfrecord, 'lidar_scans')
			output_lidar_folder = os.path.join(output_dir, split_type, tfrecord, 'lidar_scan_raw')
			output_pose_file = os.path.join(output_dir, split_type, tfrecord, 'poses.txt')
			output_map_folder = os.path.join(output_dir, split_type, tfrecord, 'map')
			output_bbox_coord_folder = os.path.join(output_dir, split_type, tfrecord, 'dynamic_bbox', 'coords')
			output_bbox_ids_folder = os.path.join(output_dir, split_type, tfrecord, 'dynamic_bbox', 'ids_heading')
   
			if os.path.isdir(output_scan_folder):
				print(f'Skipping {tfrecord} as it is existed')
				continue

			os.makedirs(output_scan_folder)
			os.makedirs(output_lidar_folder)
			os.makedirs(output_map_folder)
			os.makedirs(output_bbox_coord_folder)
			os.makedirs(output_bbox_ids_folder)
			
			if args.viz:
				vis = o3d.visualization.Visualizer()
				vis.create_window()
				vis.get_render_option().line_width = 5
				
				mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
							size=5, origin=[0,0,0])
				vis.add_geometry(mesh_frame)

			dataset = tf.data.TFRecordDataset(os.path.join(tfrecords_folder,tfrecord), compression_type='')
			frame_cnt = 0
			poses = []

			first = True
			map_features = None
			for data in dataset:
				frame = dataset_pb2.Frame()
				frame.ParseFromString(bytearray(data.numpy()))

				if(first):
					map_features = frame.map_features
					first = False

				xyz, pose = get_lidar_point_cloud(frame)
				dynamic_bbox, ids = get_dynamic_bbox(frame, speed_threshold=0.1)
				
				map = get_mask(frame, copy.deepcopy(map_features))
				map_encoded = encode(map * 255)
				map_img = Image.fromarray(map_encoded)
				map_img.save(os.path.join(output_map_folder, f'{frame_cnt}.png'))

				# raw lidar scan
				np.save(os.path.join(output_lidar_folder, f'{frame_cnt}.npy'), xyz)

				# filtered lidar scan
				xyz = filter_dynamic_pcd(xyz, dynamic_bbox)
				np.save(os.path.join(output_scan_folder, f'{frame_cnt}.npy'), xyz)

				if len(ids) > 0:
					dynamic_bbox = box_utils.get_upright_3d_box_corners(dynamic_bbox)
					dynamic_bbox = dynamic_bbox.numpy().reshape(-1, 3)

					np.save(os.path.join(output_bbox_coord_folder, f'{frame_cnt}.npy'), dynamic_bbox)
					np.savetxt(os.path.join(output_bbox_ids_folder, f'{frame_cnt}.txt'), np.vstack(ids), delimiter=" ", fmt="%s") 
	
				poses.append(np.append(frame_cnt, pose.flatten()))
	
				if args.viz:
					pcd = o3d.geometry.PointCloud()
					pcd.points = o3d.utility.Vector3dVector(xyz)

					dynamic_bbox = box_utils.get_upright_3d_box_corners(dynamic_bbox)
					dynamic_bbox = dynamic_bbox.numpy().reshape(-1, 3)
					for i in range(len(dynamic_bbox) // 8):
						bbox = o3d.geometry.LineSet()
						bbox.points = o3d.utility.Vector3dVector(dynamic_bbox[8*i:8*i+8, :])
						lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
						[0, 4], [1, 5], [2, 6], [3, 7]])
						bbox.lines = o3d.utility.Vector2iVector(lines)
						vis.add_geometry(bbox)

					vis.add_geometry(pcd)
					vis.poll_events()
					vis.update_renderer()
					vis.clear_geometries()

				frame_cnt += 1
			
			poses = np.vstack(poses)
			np.savetxt(output_pose_file, poses, fmt=' '.join(['%i'] + ['%1.4f']*16))
			if args.viz:
				vis.destroy_window()
