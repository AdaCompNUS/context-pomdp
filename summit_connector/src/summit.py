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
import rospy
import time
import random

import rospy

class Summit(object):

    def __init__(self):
        address = rospy.get_param('address', '127.0.0.1')
        port = rospy.get_param('port', 2000)
        self.map_location = rospy.get_param('map_location', 'meskel_square')
        self.random_seed = rospy.get_param('random_seed', 1)
        self.rng = random.Random(rospy.get_param('random_seed', 0))
        # print("Map location set in SUMMIT: {}".format(self.map_location))
        # print("Random seed set in SUMMIT: {}".format(self.random_seed))
        # sys.stdout.flush()

        self.client = carla.Client(address, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Define bounds.
        if self.map_location == "map":
            self.scenario_center = carla.Vector2D(825, 1500)
            self.scenario_min = carla.Vector2D(450, 1100)
            self.scenario_max = carla.Vector2D(1200, 1900)
            self.geo_min = (1.2894000, 103.7669000)
            self.geo_max = (1.3088000, 103.7853000)
        elif self.map_location == "meskel_square":
            self.scenario_center = carla.Vector2D(450, 400)
            self.scenario_min = carla.Vector2D(350, 300)
            self.scenario_max = carla.Vector2D(550, 500)
            self.geo_min = (9.00802, 38.76009)
            self.geo_max = (9.01391, 38.76603)
        elif self.map_location == "magic":
            self.scenario_center = carla.Vector2D(180, 220)
            self.scenario_min = carla.Vector2D(80, 120)
            self.scenario_max = carla.Vector2D(280, 320)
            self.geo_min = (51.5621800, -1.7729100)
            self.geo_max = (51.5633900, -1.7697300)
        elif self.map_location == "highway":
            self.scenario_center = carla.Vector2D(100, 400)
            self.scenario_min = carla.Vector2D(0, 300)
            self.scenario_max = carla.Vector2D(200, 500)
            self.geo_min = (1.2983800, 103.7777000)
            self.geo_max = (1.3003700, 103.7814900)
        elif self.map_location == "chandni_chowk":
            self.scenario_center = carla.Vector2D(380, 250)
            self.scenario_min = carla.Vector2D(260, 830)
            self.scenario_max = carla.Vector2D(500, 1150)
            self.geo_min = (28.653888, 77.223296)
            self.geo_max = (28.660295, 77.236850)
        elif self.map_location == "shi_men_er_lu":
            self.scenario_center = carla.Vector2D(1010, 1900)
            self.scenario_min = carla.Vector2D(780, 1700)
            self.scenario_max = carla.Vector2D(1250, 2100)
            self.geo_min = (31.229828, 121.438702)
            self.geo_max = (31.242810, 121.464944)
        elif self.map_location == "beijing":
            self.scenario_center = carla.Vector2D(2080, 1860)
            self.scenario_min = carla.Vector2D(490, 1730)
            self.scenario_max = carla.Vector2D(3680, 2000)
            self.geo_min = (39.8992818, 116.4099687)
            self.geo_max = (39.9476116, 116.4438916)

        # print("Loading map data")

        # Load network.
        self.network = carla.SumoNetwork.load(summit_root + 'Data/' + self.map_location + '.net.xml')
        self.network_occupancy_map = carla.OccupancyMap.load(summit_root + 'Data/' + self.map_location + '.wkt')
        self.network_segment_map = self.network.create_segment_map()
        self.network_segment_map.seed_rand(self.rng.getrandbits(32))
        # print("Lane network loaded")
        # sys.stdout.flush()
        # Roadmarks.
        self.roadmark_occupancy_map = self.network.create_roadmark_occupancy_map()
        print("Roadmarks loaded")
        # sys.stdout.flush()
        # Load sidewalk.
        self.sidewalk = self.network_occupancy_map.create_sidewalk(1.5)
        self.sidewalk_occupancy_map = carla.OccupancyMap.load(summit_root + 'Data/' + self.map_location + '.sidewalk.wkt')
        with open(summit_root + 'Data/' + self.map_location + '.sidewalk.mesh', 'r') as file:
            sidewalk_mesh_data = file.read()
        sidewalk_mesh_data = sidewalk_mesh_data.split(',')
        # print("Sidewalk loaded")
        # sys.stdout.flush()
        # Load landmarks.
        self.landmarks = []
        self.landmarks = carla.Landmark.load(summit_root + 'Data/' + self.map_location + '.osm', self.network.offset)
        self.landmarks = [l.difference(self.network_occupancy_map).difference(self.sidewalk_occupancy_map) for l in
                          self.landmarks]
        self.landmarks = [l for l in self.landmarks if not l.is_empty]
        # print("Landmarks loaded")
        # sys.stdout.flush()

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
