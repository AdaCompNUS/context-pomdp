import math
import numpy as np
from drunc import Drunc
import carla
import sys


# rotate angle_deg clockwise. note that carla uses left-handed coordinate system, hence it is rotating clockwise
# instead of counter-clockwise as in RVO::Vector2
def rotate(vector, angle_deg):
    angle_rad = 3.1415926 * angle_deg / 180.0
    cs = math.cos(angle_rad)
    sn = math.sin(angle_rad)
    px = vector.x * cs - vector.y * sn
    py = vector.x * sn + vector.y * cs
    return carla.Vector2D(px, py)


def norm(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y)


def normalize(vector):
    n = norm(vector)
    if n == 0:
        return vector
    else:
        return carla.Vector2D(vector.x / n, vector.y / n)


def get_signed_angle_diff(vector1, vector2):
    theta = math.atan2(vector1.y, vector1.x) - math.atan2(vector2.y, vector2.x)
    theta = theta * 180.0 / 3.1415926
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta


def get_position(actor):
    pos3D = actor.get_location()
    return carla.Vector2D(pos3D.x, pos3D.y)


def get_forward_direction(actor):
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)


def in_front(ego_pos, ego_heading, exo_pos):
    ped_dir = normalize(carla.Vector2D(exo_pos[1], ego_pos[1])
                        - carla.Vector2D(ego_pos[0], ego_pos[1]))
    car_dir = carla.Vector2D(math.cos(ego_heading), math.sin(ego_heading))
    proj = car_dir.x * ped_dir.x + car_dir.y * ped_dir.y

    if proj > math.cos(3.1415 / 3.0):
        return True
    else:
        return False


def get_spawn_range(center, size):
    spawn_min = carla.Vector2D(
        center.x - size,
        center.y - size)
    spawn_max = carla.Vector2D(
        center.x + size,
        center.y + size)
    return spawn_min, spawn_max


def dot_product(a, b):
    return a.x * b.x + a.y * b.y


def check_bounds(point, bounds_min, bounds_max):
    return bounds_min.x <= point.x <= bounds_max.x and \
           bounds_min.y <= point.y <= bounds_max.y


def get_spawn_occupancy_map(center_pos, spawn_size_min, spawn_size_max):
    return carla.OccupancyMap(
        carla.Vector2D(center_pos.x - spawn_size_max, center_pos.y - spawn_size_max),
        carla.Vector2D(center_pos.x + spawn_size_max, center_pos.y + spawn_size_max)) \
        .difference(carla.OccupancyMap(
        carla.Vector2D(center_pos.x - spawn_size_min, center_pos.y - spawn_size_min),
        carla.Vector2D(center_pos.x + spawn_size_min, center_pos.y + spawn_size_min)))


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
