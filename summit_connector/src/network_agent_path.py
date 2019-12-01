import random
import numpy as np
import math

class NetworkAgentPath:
    def __init__(self, summit, min_points, interval):
        self.summit = summit
        self.min_points = min_points
        self.interval = interval
        self.route_points = []

    @staticmethod
    def rand_path(summit, min_points, interval, segment_map, min_safe_points=None, rng=random):
        if min_safe_points is None:
            min_safe_points = min_points

        spawn_point = None
        route_paths = None
        while not spawn_point or len(route_paths) < 1:
            spawn_point = segment_map.rand_point()
            spawn_point = summit.network.get_nearest_route_point(spawn_point)
            route_paths = summit.network.get_next_route_paths(spawn_point, min_safe_points - 1, interval)

        path = NetworkAgentPath(summit, min_points, interval)
        path.route_points = rng.choice(route_paths)[0:min_points]
        return path

    def resize(self, rng=random):
        while len(self.route_points) < self.min_points:
            next_points = self.summit.network.get_next_route_points(self.route_points[-1], self.interval)
            if len(next_points) == 0:
                return False
            self.route_points.append(rng.choice(next_points))
        return True

    def get_min_offset(self, position):
        min_offset = None
        for i in range(len(self.route_points) / 2):
            route_point = self.route_points[i]
            offset = position - self.summit.network.get_route_point_position(route_point)
            offset = offset.length() 
            if min_offset == None or offset < min_offset:
                min_offset = offset
        return min_offset

    def cut(self, position):
        cut_index = 0
        min_offset = None
        min_offset_index = None
        for i in range(len(self.route_points) / 2):
            route_point = self.route_points[i]
            offset = position - self.summit.network.get_route_point_position(route_point)
            offset = offset.length() 
            if min_offset == None or offset < min_offset:
                min_offset = offset
                min_offset_index = i
            if offset <= 1.0:
                cut_index = i + 1

        # Invalid path because too far away.
        if min_offset > 1.0:
            self.route_points = self.route_points[min_offset_index:]
        else:
            self.route_points = self.route_points[cut_index:]

    def get_position(self, index=0):
        return self.summit.network.get_route_point_position(self.route_points[index])

    def get_yaw(self, index=0):
        pos = self.summit.network.get_route_point_position(self.route_points[index])
        next_pos = self.summit.network.get_route_point_position(self.route_points[index + 1])
        return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x))
