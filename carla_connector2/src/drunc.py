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

map_location = "map" # NUS 
#map_location = "meskel_square"

class Drunc(object):
    def __init__(self):
        # Create connection to simulator.
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Define bounds.
        if map_location == "map":
            self.map_bounds_min = carla.Vector2D(450, 1100)
            self.map_bounds_max = carla.Vector2D(1200, 1900)
        elif map_location == "meskel_square":
            self.map_bounds_min = carla.Vector2D(-110, -5)
            self.map_bounds_max = carla.Vector2D(1450, 1100)
        
        # Create network related objects.
        self.network = carla.SumoNetwork.load(carla_root + 'Data/' + map_location + '.net.xml')
        self.network_occupancy_map = carla.OccupancyMap.load(carla_root + 'Data/' + map_location + '.wkt')

        with open(carla_root + 'Data/' + map_location + '.mesh', 'r') as file:
            network_mesh_data = file.read()
        network_mesh_data = network_mesh_data.split(',')
        self.network_occupancy_map_mesh_triangles = []
        for i in range(0, len(network_mesh_data), 3):
            self.network_occupancy_map_mesh_triangles.append(carla.Vector3D(
                float(network_mesh_data[i]), 
                float(network_mesh_data[i + 1]), 
                float(network_mesh_data[i + 2])))

        self.sidewalk = self.network_occupancy_map.create_sidewalk(1.5)
        self.sidewalk_occupancy_map = carla.OccupancyMap.load(carla_root + 'Data/' + map_location + '.sidewalk.wkt')

        with open(carla_root + 'Data/' + map_location + '.sidewalk.mesh', 'r') as file:
            sidewalk_mesh_data = file.read()
        sidewalk_mesh_data = sidewalk_mesh_data.split(',')
        self.sidewalk_occupancy_map_mesh_triangles = []
        for i in range(0, len(sidewalk_mesh_data), 3):
            self.sidewalk_occupancy_map_mesh_triangles.append(carla.Vector3D(
                float(sidewalk_mesh_data[i]), 
                float(sidewalk_mesh_data[i + 1]), 
                float(sidewalk_mesh_data[i + 2])))

        self.landmarks = []
        '''
        self.landmarks = carla.Landmark.load(carla_root + 'Data/' + map_location + '.osm', self.network.offset)
        self.landmarks = [l.difference(self.network_occupancy_map).difference(self.sidewalk_occupancy_map) for l in self.landmarks]
        self.landmarks = [l for l in self.landmarks if not l.is_empty]
        '''
    
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
