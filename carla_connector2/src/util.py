import math

from drunc import Drunc
import carla

# rotate angle_deg clockwise. note that carla uses left-handed coordinate system, hence it is rotating clockwise instead of counter-clockwise as in RVO::Vector2
def rotate(vector, angle_deg):
    angle_rad = 3.1415926 * angle_deg / 180.0
    cs = math.cos(angle_rad)
    sn = math.sin(angle_rad)
    px = vector.x * cs - vector.y * sn
    py = vector.x * sn + vector.y * cs
    return carla.Vector2D(px, py)

def norm(vector):
    return math.sqrt(vector.x*vector.x + vector.y * vector.y)

def normalize(vector):
    n = norm(vector)
    if n == 0:
        return vector
    else:
        return carla.Vector2D(vector.x/n, vector.y/n)

def get_signed_angle_diff(vector1, vector2):
    theta = math.atan2 (vector1.y, vector1.x) - math.atan2 (vector2.y, vector2.x)
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

def get_bounding_box_corners(actor):
        bbox = actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
        forward_vec = normalize(get_forward_direction(actor)) # the local x direction (left-handed coordinate system)
        sideward_vec = rotate(forward_vec, 90.0) # the local y direction

        half_y_len = bbox.extent.y
        half_x_len = bbox.extent.x

        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners
