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
import rospy

# map_location = 'map'
map_location = 'meskel_square'
# map_location = 'magic'
# map_location = 'highway'

class Drunc(object):
    def __init__(self):
        # Create connection to simulator.
        address = rospy.get_param('address', '127.0.0.1')
        port = rospy.get_param('port', 2000)
        self.client = carla.Client(address, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Load network.
        self.network = carla.SumoNetwork.load(carla_root + 'Data/' + map_location + '.net.xml')
        self.network_occupancy_map = carla.OccupancyMap.load(carla_root + 'Data/' + map_location + '.wkt')
        self.network_segment_map = self.network.create_segment_map()

        # Roadmarks.
        self.roadmark_occupancy_map = self.network.create_roadmark_occupancy_map()

        # Load sidewalk.
        self.sidewalk = self.network_occupancy_map.create_sidewalk(1.5)
        self.sidewalk_occupancy_map = carla.OccupancyMap.load(carla_root + 'Data/' + map_location + '.sidewalk.wkt')
        with open(carla_root + 'Data/' + map_location + '.sidewalk.mesh', 'r') as file:
            sidewalk_mesh_data = file.read()
        sidewalk_mesh_data = sidewalk_mesh_data.split(',')

        # Load landmarks.
        self.landmarks = []
        self.landmarks = carla.Landmark.load(carla_root + 'Data/' + map_location + '.osm', self.network.offset)
        self.landmarks = [l.difference(self.network_occupancy_map).difference(self.sidewalk_occupancy_map) for l in self.landmarks]
        self.landmarks = [l for l in self.landmarks if not l.is_empty]
        
        # Define bounds.
        if map_location == "map":
            self.scenario_center = carla.Vector2D(825, 1500)
            self.scenario_min = carla.Vector2D(450, 1100)
            self.scenario_max = carla.Vector2D(1200, 1900)
        elif map_location == "meskel_square":
            self.scenario_center = carla.Vector2D(450, 400)
            self.scenario_min = carla.Vector2D(350, 300)
            self.scenario_max = carla.Vector2D(550, 500)
        elif map_location == "magic":
            self.scenario_center = carla.Vector2D(180, 220)
            self.scenario_min = carla.Vector2D(80, 120)
            self.scenario_max = carla.Vector2D(280, 320)
        elif map_location == "highway":
            self.scenario_center = carla.Vector2D(100, 400)
            self.scenario_min = carla.Vector2D(0, 300)
            self.scenario_max = carla.Vector2D(200, 500)
    
    def in_scenario_bounds(self, point):
        return self.scenario_min.x <= point.x <= self.scenario_max.x and \
               self.scenario_min.y <= point.y <= self.scenario_max.y

    def rand_bounds_point(self, bounds_min=None, bounds_max=None):
        if bounds_min is None:
            bounds_min = self.scenario_min
        if bounds_max is None:
            bounds_max = self.scenario_max

        return carla.Vector2D(
                random.uniform(bounds_min.x, bounds_max.x),
                random.uniform(bounds_min.y, bounds_max.y))

    def rand_sidewalk_route_point(self, bounds_min=None, bounds_max=None):
        point = self.sidewalk.get_nearest_route_point(self.rand_bounds_point(bounds_min, bounds_max))
        while not self.in_scenario_bounds(self.sidewalk.get_route_point_position(point)):
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
