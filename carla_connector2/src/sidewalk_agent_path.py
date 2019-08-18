import random
import numpy as np
import math

class SidewalkAgentPath:
    def __init__(self, drunc, min_points, interval):
        self.drunc = drunc
        self.min_points = min_points
        self.interval = interval
        self.route_points = []
        self.route_orientation = None

    @staticmethod
    def rand_path(drunc, min_points, interval):
        path = SidewalkAgentPath(drunc, min_points, interval)
        path.route_points = [drunc.rand_sidewalk_route_point()]
        path.route_orientation = random.choice([True, False])
        path.resize()
        return path

    def resize(self):
        while len(self.route_points) < self.min_points:
            if random.random() <= 0.01:
                adjacent_route_points = self.drunc.sidewalk.get_adjacent_route_points(self.route_points[-1])
                if adjacent_route_points:
                    self.route_points.append(adjacent_route_points[0])
                    self.route_orientation = random.randint(0, 1) == 1
                    continue

            if self.route_orientation:
                self.route_points.append(
                        self.drunc.sidewalk.get_next_route_point(self.route_points[-1], self.interval))
            else:
                self.route_points.append(
                        self.drunc.sidewalk.get_previous_route_point(self.route_points[-1], self.interval))

        return True

    def cut(self, position):
        cut_index = 0
        min_offset = 100000.0
        for i in range(len(self.route_points) / 2):
            route_point = self.route_points[i]
            offset = position - self.drunc.sidewalk.get_route_point_position(route_point)
            offset = offset.length()
            if offset < min_offset:
                min_offset = offset
            if offset < 0.5:
                cut_index = i + 1

        self.route_points = self.route_points[cut_index:]

    def get_position(self, index=0):
        return self.drunc.sidewalk.get_route_point_position(self.route_points[index])

    def get_yaw(self, index=0):
        pos = self.drunc.sidewalk.get_route_point_position(self.route_points[index])
        next_pos = self.drunc.sidewalk.get_route_point_position(self.route_points[index + 1])
        return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x))
