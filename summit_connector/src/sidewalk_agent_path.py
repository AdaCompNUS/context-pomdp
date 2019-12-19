import random
import numpy as np
import math
import carla

class SidewalkAgentPath:
    def __init__(self, summit, min_points, interval):
        self.summit = summit
        self.min_points = min_points
        self.interval = interval
        self.route_points = []
        self.route_orientations = []

    @staticmethod
    def rand_path(summit, min_points, interval, bounds_min, bounds_max, rng=None):
        if rng is None:
            rng = random
    
        point = None
        while point is None or not (bounds_min.x <= point_position.x <= bounds_max.x and bounds_min.y <= point_position.y <= bounds_max.y):
            point = carla.Vector2D(rng.uniform(bounds_min.x, bounds_max.x), rng.uniform(bounds_min.y, bounds_max.y))
            point = summit.sidewalk.get_nearest_route_point(point)
            point_position = summit.sidewalk.get_route_point_position(point)

        path = SidewalkAgentPath(summit, min_points, interval)
        path.route_points = [point]
        path.route_orientations = [rng.choice([True, False])]
        path.resize()
        return path

    def resize(self, rng=None):
        if rng is None:
            rng = random

        while len(self.route_points) < self.min_points:
            if rng.random() <= 0.8: #0.01
                adjacent_route_points = self.summit.sidewalk.get_adjacent_route_points(self.route_points[-1], 50.0)
                if adjacent_route_points:
                    self.route_points.append(adjacent_route_points[0])
                    self.route_orientations.append(rng.randint(0, 1) == 1)
                    continue

            if self.route_orientations[-1]:
                self.route_points.append(
                        self.summit.sidewalk.get_next_route_point(self.route_points[-1], self.interval))
                self.route_orientations.append(True)
            else:
                self.route_points.append(
                        self.summit.sidewalk.get_previous_route_point(self.route_points[-1], self.interval))
                self.route_orientations.append(False)

        return True

    def cut(self, position):
        cut_index = 0
        min_offset = 100000.0
        for i in range(len(self.route_points) / 2):
            route_point = self.route_points[i]
            offset = position - self.summit.sidewalk.get_route_point_position(route_point)
            offset = offset.length()
            if offset < min_offset:
                min_offset = offset
            if offset <= 1.0:
                cut_index = i + 1

        self.route_points = self.route_points[cut_index:]
        self.route_orientations = self.route_orientations[cut_index:]

    def get_position(self, index=0):
        return self.summit.sidewalk.get_route_point_position(self.route_points[index])

    def get_yaw(self, index=0):
        pos = self.summit.sidewalk.get_route_point_position(self.route_points[index])
        next_pos = self.summit.sidewalk.get_route_point_position(self.route_points[index + 1])
        return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x))
