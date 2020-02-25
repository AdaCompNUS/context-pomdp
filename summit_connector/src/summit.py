#!/usr/bin/env python2

import os, sys, glob

summit_root = os.path.expanduser("~/summit/")

api_root = os.path.expanduser("~/summit/PythonAPI")
try:
    sys.path.append(glob.glob(api_root + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(os.path.basename(__file__) + ": Cannot locate the CARLA egg file!!!")
    sys.exit()

import carla

from pathlib2 import Path
import random
import rospy
import rospy
import time

DATA_PATH = Path(summit_root)/'Data'   

class Summit(object):

    def __init__(self):
        address = rospy.get_param('address', '127.0.0.1')
        port = rospy.get_param('port', 2000)
        self.map_location = rospy.get_param('map_location', 'meskel_square')
        self.random_seed = rospy.get_param('random_seed', 1)
        self.rng = random.Random(rospy.get_param('random_seed', 0))
    
        sys.stdout.flush()
        with (DATA_PATH/'{}.sim_bounds'.format(self.map_location)).open('r') as f:
            bounds_min = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
            bounds_max = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
            self.bounds_occupancy = carla.OccupancyMap(bounds_min, bounds_max)

        sys.stdout.flush()
        self.sumo_network = carla.SumoNetwork.load(str(DATA_PATH/'{}.net.xml'.format(self.map_location)))
        self.sumo_network_segments = self.sumo_network.create_segment_map()
        self.sumo_network_spawn_segments = self.sumo_network_segments.intersection(carla.OccupancyMap(bounds_min, bounds_max))
        self.sumo_network_spawn_segments.seed_rand(self.rng.getrandbits(32))
        self.sumo_network_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.network.wkt'.format(self.map_location)))

        sys.stdout.flush()
        self.sidewalk = self.sumo_network_occupancy.create_sidewalk(1.5)
        self.sidewalk_segments = self.sidewalk.create_segment_map()
        self.sidewalk_spawn_segments = self.sidewalk_segments.intersection(carla.OccupancyMap(bounds_min, bounds_max))
        self.sidewalk_spawn_segments.seed_rand(self.rng.getrandbits(32))
        self.sidewalk_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.sidewalk.wkt'.format(self.map_location)))

        sys.stdout.flush()
        self.client = carla.Client(address, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        sys.stdout.flush()


    def reload_world(self):
        self.client.reload_world()
        self.world = self.client.get_world()

    def draw_point(self, position, color=carla.Color(255, 0, 0), life_time=-1.0):
        self.world.debug.draw_point(
            carla.Location(position.x, position.y, 0.0),
            0.3, color, life_time)

    def draw_path(self, positions, color=carla.Color(255, 0, 0), life_time=-1.0):
        for i in range(len(positions) - 1):
            self.world.debug.draw_line(
                carla.Location(positions[i].x, positions[i].y, 0.0),
                carla.Location(positions[i + 1].x, positions[i + 1].y, 0.0),
                2,
                color,
                life_time)
