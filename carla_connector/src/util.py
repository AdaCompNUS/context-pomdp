
import os, sys, glob, numpy, math

carla_root = os.path.expanduser("~/carla/")

api_root = os.path.expanduser("~/carla/PythonAPI")

osm_file_loc = carla_root + 'Data/network.ln'
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob(api_root + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    pass

sys.path.append(api_root)

import carla
from carla import Transform, Location, Rotation, Vector2D
import random
import time

mute_debug = True 
map_type = "osm"  # or "osm"
map_bound = 200
carla_portal  = 2000
carla_ip = '127.0.0.1'

def cv_to_np(c):
    return numpy.array([c.x, c.y, c.z])

def np_to_cv(n):
    return carla.Vector3D(*n)


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
    theta = theta * 180.0/3.1415926
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

def get_actor_flag(actor):
    if isinstance(actor, carla.Vehicle):
        return 'car'
    elif isinstance(actor, carla.Walker):
        return 'ped'
    else:
        return 'unsupported'
