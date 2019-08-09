# Example showing usage of LaneNetwork API and mesh spawning.
#
# LaneNetwork is loaded in LibCarla (C++) and exposed through the
# PythonAPI wrapper. Mesh triangles are calculated in this script
# (python) and sent back to LibCarla, where the API function 
# SpawnMesh spawns the mesh in UE.
#
# In future iterations, the mesh calculations will probably be 
# done completely in LibCarla and the concept of mesh triangles
# will completely be hidden from users. i.e. the PythonAPI should
# only contain a world.spawn_map(lane_network) or something like
# that.

import glob
import math
import os
import sys
from util import *

def create_map():
    print('loading map {}...'.format(osm_file_loc))
    lane_network = carla.LaneNetwork.load(osm_file_loc)
    occupancy_map = lane_network.create_occupancy_map()
    print(len(occupancy_map.triangles))
    
    client = carla.Client(carla_ip, carla_portal)
    client.set_timeout(10.0)
    client.get_world().spawn_occupancy_map(occupancy_map,'/Game/Carla/Static/GenericMaterials/Asphalt/M_Asphalt01')
    return client.get_world()

if __name__ == '__main__':
	create_map()