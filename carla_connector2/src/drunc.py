import os, sys, glob

carla_root = os.path.expanduser("~/carla/")
api_root = os.path.expanduser("~/carla/PythonAPI")
try:
    sys.path.append(glob.glob(api_root + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Cannot locate the CARLA egg file!!!")
    sys.exit()

import carla
import random

class Drunc(object):
    def __init__(self):
        # Create connection to simulator.
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(1.0)
        self.world = self.client.get_world()

        # Define bounds.
        self.map_bounds_min = carla.Vector2D(450, 1100)
        self.map_bounds_max = carla.Vector2D(1200, 1900)
        
        # Create network related objects.
        with open(carla_root + 'Data/map.net.xml', 'r') as file:
            map_data = file.read()
        self.network = carla.SumoNetwork.load(map_data)
        self.network_occupancy_map = self.network.create_occupancy_map()
        self.sidewalk = carla.Sidewalk(
            self.network_occupancy_map,
            self.map_bounds_min, self.map_bounds_max, ## to check
            3.0, 0.1,
            20.0)
        self.sidewalk_occupancy_map = self.sidewalk.create_occupancy_map()
        self.landmark_map = carla.LandmarkMap.load(
            carla_root + 'Data/map.osm',
            carla.Vector2D(-11551102.28, -143022.13))
        self.landmark_map = self.landmark_map.filter(self.network_occupancy_map).filter(self.sidewalk_occupancy_map)
    
    def in_bounds(self, point):
        return self.map_bounds_min.x <= point.x <= self.map_bounds_max.x and \
               self.map_bounds_min.y <= point.y <= self.map_bounds_max.y

    def rand_bounds_point(self, bounds_min=None, bounds_max=None):
        if bounds_min is None:
            bounds_min = self.map_bounds_min
        if bounds_max is None:
            bounds_max = self.map_bounds_max

        return carla.Vector2D(
                random.uniform(bounds_min.x, bounds_max.x),
                random.uniform(bounds_min.y, bounds_max.y))

    def rand_network_route_point(self, bounds_min=None, bounds_max=None):
        point = self.network.get_nearest_route_point(self.rand_bounds_point(bounds_min, bounds_max))
        while not self.in_bounds(self.network.get_route_point_position(point)):
            point = self.network.get_nearest_route_point(self.rand_bounds_point())
        return point

    def rand_sidewalk_route_point(self, bounds_min=None, bounds_max=None):
        point = self.sidewalk.get_nearest_route_point(self.rand_bounds_point(bounds_min, bounds_max))
        while not self.in_bounds(self.sidewalk.get_route_point_position(point)):
            point = self.sidewalk.get_nearest_route_point(self.rand_bounds_point(bounds_min, bounds_max))
        return point

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
