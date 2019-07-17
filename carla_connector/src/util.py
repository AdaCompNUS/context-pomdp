
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

def cv_to_np(c):
    return numpy.array([c.x, c.y, c.z])

def np_to_cv(n):
    return carla.Vector3D(*n)
