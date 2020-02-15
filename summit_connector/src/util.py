import math
import numpy as np
from summit import Summit
import carla
import sys

def in_front(ego_pos, ego_heading, exo_pos):
    ped_dir = (carla.Vector2D(exo_pos[1], ego_pos[1])
                        - carla.Vector2D(ego_pos[0], ego_pos[1])).make_unit_vector()
    car_dir = carla.Vector2D(math.cos(ego_heading), math.sin(ego_heading))
    proj = car_dir.x * ped_dir.x + car_dir.y * ped_dir.y

    if proj > math.cos(3.1415 / 3.0):
        return True
    else:
        return False

def get_signed_angle_diff(vector1, vector2):
    theta = math.atan2(vector1.y, vector1.x) - math.atan2(vector2.y, vector2.x)
    theta = np.rad2deg(theta)
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

def get_position(actor):
    pos3d = actor.get_location()
    return carla.Vector2D(pos3d.x, pos3d.y)

def get_forward_direction(actor):
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)

def get_bounding_box(actor):
    return actor.bounding_box

def get_velocity(actor):
    v = actor.get_velocity()
    return carla.Vector2D(v.x, v.y)

def get_bounding_box_corners(actor, expand=0.0):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(
        actor).make_unit_vector()  # the local x direction (left-handed coordinate system)
    sideward_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction

    half_y_len = bbox.extent.y + expand
    half_x_len = bbox.extent.x + expand

    corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len * forward_vec - half_y_len * sideward_vec]

    return corners
    
def get_vehicle_bounding_box_corners(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))
    half_y_len = bbox.extent.y + 0.3
    half_x_len_forward = bbox.extent.x + 1.0
    half_x_len_backward = bbox.extent.x + 0.1
    corners = [loc - half_x_len_backward * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len_forward * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len_forward * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len_backward * forward_vec - half_y_len * sideward_vec]
    return corners

def get_pedestrian_bounding_box_corners(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))
    # Hardcoded values for pedestrians.
    half_y_len = 0.25
    half_x_len = 0.25
    corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len * forward_vec - half_y_len * sideward_vec]
    return corners
