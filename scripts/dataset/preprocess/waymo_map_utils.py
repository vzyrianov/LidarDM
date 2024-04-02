
import dataclasses
import enum
from typing import List

import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objs as go

from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.utils import  frame_utils, box_utils
from waymo_open_dataset.utils.plot_maps import FeatureType, MapPoints, plot_map_points
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow as tf

def get_line_layers(all_points, m_coverage=96, width=640, fill_it=False):
    h, w = width, width
    result = list()
    render = np.zeros((h, w), dtype=np.uint8)

    for map_points in all_points:
      p = np.stack([np.float32(map_points.x), np.float32(map_points.y)])
      
      p = (p * (width / m_coverage)) + (width / 2)
 
      p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
      p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
      p = p[:2].round().astype(np.int32).T

      if(not fill_it):
        result.append(cv2.polylines(render, [p], False, 1, thickness=4))
      else:
        #result.append(cv2.polylines(render, [p], False, 1, thickness=4))
        result.append(cv2.fillPoly(render, [p], 1))

    res = (render > 0.5).astype(np.uint8)

    return res

def plot_bounding(
    frame, m_coverage=96, width=640
):
  h, w = width, width

  result_unknown = list()
  render_unknown = np.zeros((h, w), dtype=np.uint8)
  result_vehicle = list()
  render_vehicle = np.zeros((h, w), dtype=np.uint8)
  result_pedestrian = list()
  render_pedestrian = np.zeros((h, w), dtype=np.uint8)
  result_cyclist = list()
  render_cyclist = np.zeros((h, w), dtype=np.uint8)

  result_unknown_fast = list()
  render_unknown_fast = np.zeros((h, w), dtype=np.uint8)
  result_vehicle_fast = list()
  render_vehicle_fast = np.zeros((h, w), dtype=np.uint8)
  result_pedestrian_fast = list()
  render_pedestrian_fast = np.zeros((h, w), dtype=np.uint8)
  result_cyclist_fast = list()
  render_cyclist_fast = np.zeros((h, w), dtype=np.uint8)

  for label in frame.laser_labels:
    speed_threshold=0.1
    speed = np.linalg.norm(np.array([label.metadata.speed_x, 
                        label.metadata.speed_y,
                     label.metadata.speed_z]))
    is_fast = False

    if speed > speed_threshold:
      #box = box_utils.box_to_tensor(label.box).numpy()
      is_fast=True
      #bboxes.append(box)
    
    
    
    box = box_utils.get_upright_3d_box_corners(tf.expand_dims(box_utils.box_to_tensor(label.box), 0)).numpy()
    p = box[0, 0:4,0:2]
  
    p = (p * (width / m_coverage)) + (width / 2)
    p = p.round().astype(np.int32).T

    if(not is_fast):
      if(label.type == label.TYPE_UNKNOWN):
        result_unknown.append(cv2.fillPoly(render_unknown, [p.T], 1))
      elif(label.type == label.TYPE_VEHICLE):
        result_vehicle.append(cv2.fillPoly(render_vehicle, [p.T], 1))
      elif(label.type == label.TYPE_PEDESTRIAN):
        result_pedestrian.append(cv2.fillPoly(render_pedestrian, [p.T], 1))
      elif(label.type == label.TYPE_CYCLIST):
        result_cyclist.append(cv2.fillPoly(render_cyclist, [p.T], 1))
    else:
      if(label.type == label.TYPE_UNKNOWN):
        result_unknown_fast.append(cv2.fillPoly(render_unknown_fast, [p.T], 1))
      elif(label.type == label.TYPE_VEHICLE):
        result_vehicle_fast.append(cv2.fillPoly(render_vehicle_fast, [p.T], 1))
      elif(label.type == label.TYPE_PEDESTRIAN):
        result_pedestrian_fast.append(cv2.fillPoly(render_pedestrian_fast, [p.T], 1))
      elif(label.type == label.TYPE_CYCLIST):
        result_cyclist_fast.append(cv2.fillPoly(render_cyclist_fast, [p.T], 1))

  res = (np.stack([
    render_unknown, render_vehicle, render_pedestrian, render_cyclist,
    render_unknown_fast, render_vehicle_fast, render_pedestrian_fast, render_cyclist_fast
    ], -1) > 0.5).astype(np.uint8)
  x = res
    
  x = np.rot90(x, 2, (0, 1))
  x = np.flip(x, 1)
  return x
  

def plot_map_features(
    map_features: List[map_pb2.MapFeature],
    transform,
    points_offset
) -> go._figure.Figure:
  """Plots the map data for a Scenario proto from the open dataset.

  Args:
    map_features: A list of map features to be plotted.

  Returns:
    A plotly figure object.
  """
  lane_types = {
      map_pb2.LaneCenter.TYPE_UNDEFINED: FeatureType.UNKNOWN_FEATURE,
      map_pb2.LaneCenter.TYPE_FREEWAY: FeatureType.FREEWAY_LANE,
      map_pb2.LaneCenter.TYPE_SURFACE_STREET: FeatureType.SURFACE_STREET_LANE,
      map_pb2.LaneCenter.TYPE_BIKE_LANE: FeatureType.BIKE_LANE,
  }
  road_line_types = {
      map_pb2.RoadLine.TYPE_UNKNOWN: (
          FeatureType.UNKNOWN_FEATURE
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: (
          FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: (
          FeatureType.SOLID_SINGLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: (
          FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: (
          FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: (   
          FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: (
          FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY
      ),
      map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: (
          FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY
      ),
  }
  road_edge_types = {
      map_pb2.RoadEdge.TYPE_UNKNOWN: FeatureType.UNKNOWN_FEATURE,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: FeatureType.ROAD_EDGE_BOUNDARY,
      map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: FeatureType.ROAD_EDGE_MEDIAN,
  }

  def add_points(
      feature_id: int,
      points: List[map_pb2.MapPoint],
      feature_type: FeatureType,
      map_points: MapPoints,
      is_polygon=False,
      points_offset=None,
      transform=None
  ):
    above_threshold = True

    if feature_type is None:
      return
    for point in points:
      points_np = np.array([point.x - points_offset[0], point.y - points_offset[1], point.z - points_offset[2], 1])

      points_np = transform @ points_np

      point.x = points_np[0]
      point.y = points_np[1]
      point.z = points_np[2]

      map_points.append_point(point, feature_type, feature_id)

      if(point.z < 4):
        above_threshold = False

    if is_polygon:
      map_points.append_point(points[0], feature_type, feature_id)
    
    return not above_threshold

  # Create arrays of the map points to be plotted.
  lane_points = []
  road_line_points = []
  road_edge_points = []
  stop_sign_points = []
  crosswalk_points = []
  speed_bump_points = []
  driveway_points = []



  for feature in map_features:
    if feature.HasField('lane'):
      map_points = MapPoints()
      to_add = add_points(
          feature.id,
          list(feature.lane.polyline),
          lane_types.get(feature.lane.type),
          map_points, points_offset=points_offset, transform=transform
      )
      if(to_add):
        lane_points.append(map_points)
    elif feature.HasField('road_line'):
      map_points = MapPoints()
      feature_type = road_line_types.get(feature.road_line.type)
      to_add = add_points(
          feature.id, list(feature.road_line.polyline), feature_type, map_points, points_offset=points_offset, transform=transform
      )
      if(to_add):
        road_line_points.append(map_points)
    elif feature.HasField('road_edge'):
      map_points = MapPoints()
      feature_type = road_edge_types.get(feature.road_edge.type)
      to_add = add_points(
          feature.id, list(feature.road_edge.polyline), feature_type, map_points, points_offset=points_offset, transform=transform
      )
      if(to_add):
        road_edge_points.append(map_points)
    elif feature.HasField('stop_sign'):
      map_points = MapPoints()
      to_add = add_points(
          feature.id,
          [feature.stop_sign.position],
          FeatureType.STOP_SIGN,
          map_points, points_offset=points_offset, transform=transform
      )
      if(to_add):
        stop_sign_points.append(map_points)
    elif feature.HasField('crosswalk'):
      map_points = MapPoints()
      to_add = add_points(
          feature.id,
          list(feature.crosswalk.polygon),
          FeatureType.CROSSWALK,
          map_points,
          True, points_offset=points_offset, transform=transform
      )
      if(to_add):
        crosswalk_points.append(map_points)
    elif feature.HasField('speed_bump'):
      map_points = MapPoints()
      to_add = add_points(
          feature.id,
          list(feature.speed_bump.polygon),
          FeatureType.SPEED_BUMP,
          map_points,
          True, points_offset=points_offset, transform=transform
      )
      if(to_add):
        speed_bump_points.append(map_points)
    elif feature.HasField('driveway'):
      map_points = MapPoints()
      to_add = add_points(
          feature.id,
          list(feature.driveway.polygon),
          FeatureType.DRIVEWAY,
          map_points,
          True, points_offset=points_offset, transform=transform
      )
      if(to_add):
        driveway_points.append(map_points)
    
  lane_layer = get_line_layers(all_points=lane_points)
  road_line_layer = get_line_layers(all_points=road_line_points)
  road_edge_layer = get_line_layers(all_points=road_edge_points)
  #stop_sign_layer = get_line_layers(all_points=stop_sign_points)
  crosswalk_layer = get_line_layers(all_points=crosswalk_points, fill_it=True)
  #speed_bump_layer = get_line_layers(all_points=speed_bump_points)
  driveway_layer = get_line_layers(all_points=driveway_points)
  #x = np.stack([lane_layer, road_line_layer, road_edge_layer, stop_sign_layer, crosswalk_layer, speed_bump_layer, driveway_layer])
  x = np.stack([lane_layer, road_line_layer, road_edge_layer, crosswalk_layer, driveway_layer])



  x = np.rot90(x, 2, (1, 2))
  x = np.flip(x, 2)
  
  x = np.transpose(x, (1, 2, 0))

  return x



from typing import List

import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

def get_mask(frame: dataset_pb2.Frame, map_features):

  transform = np.linalg.inv(np.reshape(np.array(frame.pose.transform), [4, 4]))
  offset = frame.map_pose_offset
  points_offset = np.array([offset.x, offset.y, offset.z])
  
  figure = plot_map_features(map_features, transform, points_offset)
  
  objects = plot_bounding(frame)

  return np.concatenate([figure, objects], 2)


