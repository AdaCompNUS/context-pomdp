#!/usr/bin/env python2

import math
import os
import sys
import random
import time

import rospy
import struct

from drunc import Drunc, summit_root
import carla

from std_msgs.msg import Bool

TILE_PATH = os.path.expanduser('~/summit/Data/imagery/{}/{}/{}_{}.jpeg')

def deg2num(zoom, lat_deg, lon_deg):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (zoom, ytile, xtile)

def num2deg(zoom, ytile, xtile):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

EARTH_RADIUS = 6378137.0 # in meters on the equator

def lat2y(a):
    return math.log(math.tan(math.pi / 4 + math.radians(a) / 2)) * EARTH_RADIUS

def lon2x(a):
    return math.radians(a) * EARTH_RADIUS

def project(a): # LatLon -> CARLA coordinates.
    return carla.Vector2D(lat2y(a[0]), lon2x(a[1]))

WALL_MAT = [
        '/Game/Carla/Static/Buildings/aa_wall_mat',
        '/Game/Carla/Static/Buildings/Block08/Block08Wall_1',
        '/Game/Carla/Static/Buildings/Block08/Block08Wall_2',
        '/Game/Carla/Static/Buildings/Apartment02/M_Apartment02_Wall_1',
        '/Game/Carla/Static/Buildings/Apartment02/M_Apartment02Wall_2',
        '/Game/Carla/Static/Buildings/Block01/M_Block01OuterWall',
        '/Game/Carla/Static/Buildings/Block02/M_Block02BaseWall_1',
        '/Game/Carla/Static/Buildings/Block02/M_Block02Wall_2',
        '/Game/Carla/Static/Buildings/Block02/M_Block02WallRoof_1',
        '/Game/Carla/Static/Buildings/Block02/M_Block02WallRoof_2',
        '/Game/Carla/Static/Buildings/Block04/M_Block04Wall_1',
        '/Game/Carla/Static/Buildings/Block04/M_Block04Wall_2',
        '/Game/Carla/Static/Buildings/Block04/M_Block04Wall_3',
        '/Game/Carla/Static/Buildings/Block06/M_Block06StairsWall_1',
        '/Game/Carla/Static/Buildings/Block06/M_Block06Stairswall_2',
        '/Game/Carla/Static/Buildings/House01/M_House01StairsWall_3',
        '/Game/Carla/Static/Buildings/House01/M_House01Wall_1',
        '/Game/Carla/Static/Buildings/House01/M_House01Wall_2',
        '/Game/Carla/Static/Buildings/House12/M_House12Wall_4',
        '/Game/Carla/Static/Buildings/House12/M_House12Wall_6',
        '/Game/Carla/Static/Buildings/TerracedHouse02/M_TerracedHouse02Wall',
        '/Game/Carla/Static/Buildings/TerracedHouse02/M_TerracedHouse02Wall_4',
        '/Game/Carla/Static/Walls/WallTunnel01/M_WallTunnel01',
        '/Game/Carla/Static/Walls/Wall15/T_Wall15']

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()

        self.reload_world()

        self.spawn_imagery = rospy.get_param('~spawn_imagery', True)
        self.spawn_landmarks = rospy.get_param('~spawn_landmarks', True)
        self.meshes_spawned_pub = rospy.Publisher('/meshes_spawned', Bool, queue_size=1, latch=True) 

        print('Spawning meshes...')

        self.mesh_ids = []

        commands = []

        # Roadmark occupancy map.
        if self.roadmark_occupancy_map is not None:
            commands.append(carla.command.SpawnDynamicMesh(
                self.roadmark_occupancy_map.get_mesh_triangles(-0.0),
                '/Game/Carla/Static/GenericMaterials/LaneMarking/M_MarkingLane_W'))

        # Network occupancy map.
        if self.network_occupancy_map is not None:
            commands.append(carla.command.SpawnDynamicMesh(
                self.network_occupancy_map.difference(self.roadmark_occupancy_map).get_mesh_triangles(-0.0),
                '/Game/Carla/Static/GenericMaterials/Masters/LowComplexity/M_Road1'))

        # Sidewalk occupancy.
        if self.sidewalk_occupancy_map is not None:
            commands.append(carla.command.SpawnDynamicMesh(
                self.sidewalk.create_occupancy_map(3.0).get_mesh_triangles(-0.0),
                '/Game/Carla/Static/GenericMaterials/Ground/GroundWheatField_Mat'))
        
        results = self.client.apply_batch_sync(commands)
        self.mesh_ids.extend(result.actor_id for result in results)

        # Imagery.
        if self.spawn_imagery:
            def spawn_tiles(zoom, (min_lat, min_lon), (max_lat, max_lon)):
                bottom_left_id = deg2num(zoom, min_lat, min_lon)
                top_right_id = deg2num(zoom, max_lat, max_lon)
                top_left_id = (zoom, top_right_id[1], bottom_left_id[2])
                bottom_right_id = (zoom, bottom_left_id[1], top_right_id[2])

                if bottom_right_id[1] >= top_left_id[1]:
                    height = bottom_right_id[1] - top_left_id[1] + 1
                else:
                    height = (top_left_id[1] + 1) + (2 ** zoom - bottom_right_id[1])

                if bottom_right_id[2] >= top_left_id[2]:
                    width = bottom_right_id[2] - top_left_id[2] + 1
                else:
                    width = (top_left_id[2] + 1) + (2 ** zoom - bottom_right_id[2])

                for row in range(top_left_id[1], bottom_right_id[1] + 1):
                    for column in range(top_left_id[2], bottom_right_id[2] + 1):

                        path = os.path.join(os.path.expanduser(summit_root+ 'Data/imagery'),
                                            "{}/{}_{}.jpeg".format(zoom, row, column))
                        sys.stdout.flush()
                        if not os.path.exists(path):
                            print("{} for row {} column {} missing in {}Data/imagery".format(path, row, column, summit_root))
                            continue

                        data = []
                        with open(path, "rb") as f:
                            byte = f.read(1)
                            while byte != "":
                                data += [ord(byte)]
                                byte = f.read(1)

                        bounds_min = project(num2deg(zoom, row + 1, column)) + self.network.offset
                        bounds_max = project(num2deg(zoom, row, column + 1)) + self.network.offset

                        self.mesh_ids.append(self.world.spawn_dynamic_tile_mesh(bounds_min, bounds_max, data))

            spawn_tiles(18, self.geo_min, self.geo_max)

        # Landmarks.
        commands = []
        if self.spawn_landmarks and self.landmarks is not None:
            for l in self.landmarks:
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_mesh_triangles(20),
                    random.choice(WALL_MAT)))
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_mesh_triangles(0),
                    random.choice(WALL_MAT)))
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_wall_mesh_triangles(20),
                    random.choice(WALL_MAT)))
            results = self.client.apply_batch_sync(commands)
            self.mesh_ids.extend(result.actor_id for result in results)

        self.meshes_spawned_pub.publish(True)

        print('All meshes spawned.')


    def dispose(self):
        commands = [carla.command.DestroyDynamicMesh(mesh_id) for mesh_id in self.mesh_ids]
        self.client.apply_batch(commands) 

        print('Destroyed meshes.')

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()

    rospy.spin()
    spawn_meshes.dispose()
