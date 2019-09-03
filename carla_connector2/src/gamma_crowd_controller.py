#!/usr/bin/env python2

import math
import random
import numpy as np
import rospy

from drunc import Drunc
import carla
from network_agent_path import NetworkAgentPath
from sidewalk_agent_path import SidewalkAgentPath
from util import *
import carla_connector2.msg
from peds_unity_system.msg import car_info as CarInfo
import timeit


first_time = True
prev_time = timeit.default_timer()

default_agent_pos = carla.Vector2D(10000, 10000)
default_agent_bbox = []
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,-1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,-1))

class CrowdAgent(object):
    def __init__(self, actor, preferred_speed):
        self.actor = actor
        self.preferred_speed = preferred_speed
        self.actor.set_collision_enabled(False) ## to check. Disable collision will generate vehicles that are overlapping
    
    def get_id(self):
        return self.actor.id

    def get_velocity(self):
        v = self.actor.get_velocity()
        return carla.Vector2D(v.x, v.y)
    
    def get_transform(self):
        return self.actor.get_transform()
    
    def get_bounding_box(self):
        return self.actor.bounding_box
    
    def get_forward_direction(self):
        forward = self.actor.get_transform().get_forward_vector()
        return carla.Vector2D(forward.x, forward.y)

    def get_position(self):
        pos3D = self.actor.get_location()
        return carla.Vector2D(pos3D.x, pos3D.y)
    
    def get_position3D(self):
        return self.actor.get_location()


class CrowdNetworkAgent(CrowdAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkAgent, self).__init__(actor, preferred_speed)
        self.path = path

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction

        half_y_len = bbox.extent.y #+ 0.3
        half_x_len = bbox.extent.x #+ 0.4



        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners
    
    def get_preferred_velocity(self):
        position = self.get_position()

        ## to check
        if not self.path.resize():
            return None
        self.path.cut(position)
        if not self.path.resize():
            return None

        target_position = self.path.get_position(5) ## to check
        velocity = (target_position - position).make_unit_vector()
        return self.preferred_speed * velocity

    def get_path_forward(self):
        position = self.get_position()
        if not self.path.resize():
            return carla.Vector2D(0, 0)
        self.path.cut(position)
        if not self.path.resize():
            return carla.Vector2D(0, 0)

        first_position = self.path.get_position(0)
        second_position = self.path.get_position(1)

        return (second_position - first_position).make_unit_vector()
    
    def get_control(self, velocity):
        steer = get_signed_angle_diff(velocity, self.get_forward_direction())
        min_steering_angle = -45.0
        max_steering_angle = 45.0
        if steer > max_steering_angle:
            steer = max_steering_angle
        elif steer < min_steering_angle:
            steer = min_steering_angle

        k = 1.0
        steer = k * steer / (max_steering_angle - min_steering_angle) * 2.0
        desired_speed = velocity.length()
        #steer_tmp = get_signed_angle_diff(velocity, self.get_forward_direction())
        cur_speed = self.get_velocity().length()
        control = self.actor.get_control()

        # if desired_speed < 0.5:
        #     desired_speed = 0

        k2 = 1.5 #1.5
        k3 = 2.5 #2.5
        if desired_speed - cur_speed > 0:
            control.throttle = k2 * (desired_speed - cur_speed) / desired_speed
            control.brake = 0.0
        elif desired_speed - cur_speed == 0:
            control.throttle = 0.0
            control.brake = 0.0
        else:
            control.throttle = 0
            control.brake = k3 * (cur_speed - desired_speed) / cur_speed

        control.steer = steer
        return control

class CrowdNetworkCarAgent(CrowdNetworkAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkCarAgent, self).__init__(actor, path, preferred_speed)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_type(self):
        return 'car'

class CrowdNetworkBikeAgent(CrowdNetworkAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkBikeAgent, self).__init__(actor, path, preferred_speed)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Bicycle')

    def get_type(self):
        return 'bike'

class CrowdSidewalkAgent(CrowdAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdSidewalkAgent, self).__init__(actor, preferred_speed)
        self.path = path
        
    def get_agent_params(self):
        return carla.AgentParams.get_default('People')
    
    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction. (rotating clockwise by 90 deg)

        # Hardcoded values for people.
        half_y_len = 0.25
        half_x_len = 0.25

        # half_y_len = bbox.extent.y
        # half_x_len = bbox.extent.x

        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners
    
    def get_preferred_velocity(self):
        position = self.get_position()

        if not self.path.resize():
            return None

        self.path.cut(position)

        if not self.path.resize():
            return None

        target_position = self.path.get_position(0)
        velocity = (target_position - position).make_unit_vector()
        return self.preferred_speed * velocity

    def get_path_forward(self):
        return carla.Vector2D(0, 0)
    
    def get_control(self, velocity):
        # velocity = velocity.make_unit_vector() * self.preferred_speed
        return carla.WalkerControl(
                carla.Vector3D(velocity.x, velocity.y, 0),
                1.0, False)

class GammaCrowdController(Drunc):
    def __init__(self):
        super(GammaCrowdController, self).__init__()
        self.network_car_agents = []
        self.network_bike_agents = []
        self.sidewalk_agents = []
        self.gamma = carla.RVOSimulator()
        self.ego_actor = None

        while self.ego_actor is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.ego_actor = actor

        self.walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.vehicles_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        self.cars_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.bikes_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 2]
        
        self.num_network_car_agents = rospy.get_param('~num_network_car_agents')
        self.num_network_bike_agents = rospy.get_param('~num_network_bike_agents')
        self.num_sidewalk_agents = rospy.get_param('~num_sidewalk_agents')
        self.path_min_points = rospy.get_param('~path_min_points')
        self.path_interval = rospy.get_param('~path_interval')
        self.network_agents_pub = rospy.Publisher(
                '/crowd/network_agents', 
                carla_connector2.msg.CrowdNetworkAgentArray, 
                queue_size=1)
        self.sidewalk_agents_pub = rospy.Publisher(
                '/crowd/sidewalk_agents', 
                carla_connector2.msg.CrowdSidewalkAgentArray, 
                queue_size=1)
        self.il_car_info_sub = rospy.Subscriber(
                '/IL_car_info',
                CarInfo,
                self.il_car_info_callback,
                queue_size=1)
        self.initialized = False

        for i in range(self.num_network_car_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Car'), i)
        
        for i in range(self.num_network_bike_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Bicycle'), i + self.num_network_car_agents)
        
        for i in range(self.num_sidewalk_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('People'), i + self.num_network_car_agents + self.num_network_bike_agents)
        
        # For ego vehicle.
        self.gamma.add_agent(carla.AgentParams.get_default('Car'), self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents)

        adding_obstacle = True
        if(adding_obstacle):
            self.add_obstacles()

    def add_obstacles(self):
        polygon_table = self.network_occupancy_map.create_polygon_table(self.map_bounds_min,self.map_bounds_max,100,0.1)
        for r in range(polygon_table.rows):
            for c in range(polygon_table.columns):
                for p in polygon_table.get(r, c):
                    obstacle = []
                    for i in reversed(range(len(p))): 
                        v1 = p[i]
                        obstacle.append(carla.Vector2D(v1.x, v1.y))
                    self.gamma.add_obstacle(obstacle)
        self.gamma.process_obstacles()
   
    def dispose(self):
        commands = []
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_car_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_bike_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.sidewalk_agents)
        self.client.apply_batch(commands)
        print('Destroyed crowd actors.')

    def il_car_info_callback(self, car_info):
        self.ego_car_info = car_info
        i = self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents

        if self.ego_car_info:
            self.gamma.set_agent_position(i, carla.Vector2D(
                self.ego_car_info.car_pos.x,
                self.ego_car_info.car_pos.y))
            self.gamma.set_agent_velocity(i, carla.Vector2D(
                self.ego_car_info.car_vel.x,
                self.ego_car_info.car_vel.y))
            self.gamma.set_agent_heading(i, carla.Vector2D(
                math.cos(self.ego_car_info.car_yaw),
                math.sin(self.ego_car_info.car_yaw)))
            self.gamma.set_agent_bounding_box_corners(i, 
                    [carla.Vector2D(v.x, v.y) for v in self.ego_car_info.car_bbox.points])
            self.gamma.set_agent_pref_velocity(i, carla.Vector2D(
                self.ego_car_info.car_pref_vel.x,
                self.ego_car_info.car_pref_vel.y))
        else:
            self.gamma.set_agent_position(i, default_agent_pos)
            self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
            self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
            self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
    
    def det(self, vector1, vector2):
        #return vector1.x * vector2.y - vector1.y * vector2.x;
        return vector1.y * vector2.x - vector1.x * vector2.y

    def left_of(self, a, b, c): ## if c is at the left side of vector ab, then return Ture, False otherwise
        if self.det(a - c, b - a) > 0:
            return True
        return False
    def in_polygon(self, position, rect):
        if len(rect) < 3:
            return False
        for i in range(0, len(rect)-1):
            if not self.left_of(rect[i], rect[i+1], position):
                return False
        if not self.left_of(rect[len(rect) - 1], rect[0], position):
            return False

        return True

    def dot_product(self, a, b):
        return a.x * b.x + a.y * b.y

    def draw_line(self, pos, vec, color=carla.Color (255,0,0)):
        height = 3
        start = carla.Vector3D(pos.x,pos.y,height)
        end = carla.Vector3D(pos.x+vec.x,pos.y+vec.y,height)
        self.world.debug.draw_line(start, end,  color=color, life_time=0.1)

    def draw_box(self, corners):
        height = 1
        for i in range(len(corners)-1):
            start = carla.Vector3D(corners[i].x,corners[i].y,height)
            end = carla.Vector3D(corners[i+1].x,corners[i+1].y,height)
            self.world.debug.draw_line(start, end,  life_time=0.1)

        start = carla.Vector3D(corners[len(corners)-1].x,corners[len(corners)-1].y,height)
        end = carla.Vector3D(corners[0].x,corners[0].y,height)
        self.world.debug.draw_line(start, end,  life_time=0.1)


    def get_lane_constraints_by_vehicle(self, position, forward_vec):
        sidewalk_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction. (rotating clockwise by 90 deg)

        
        lookahead_x = 20
        lookahead_y = 6

        right_region_corners = []
        right_region_corners.append(position)
        right_region_corners.append(position + lookahead_y*sidewalk_vec)
        right_region_corners.append(position + lookahead_y*sidewalk_vec + lookahead_x*forward_vec)
        right_region_corners.append(position + lookahead_x*forward_vec)

        #self.draw_box(right_region_corners)


        left_region_corners = []
        left_region_corners.append(position)
        left_region_corners.append(position + lookahead_x*forward_vec)
        left_region_corners.append(position - lookahead_y*sidewalk_vec + lookahead_x*forward_vec)
        left_region_corners.append(position - lookahead_y*sidewalk_vec)
        
        #self.draw_box(left_region_corners)

        left_lane_constrained_by_vehicle = False
        right_lane_constrained_by_vehicle = False



        for crowd_agent in self.network_bike_agents: # + self.sidewalk_agents:
            pos_agt = crowd_agent.get_position()
            if (not left_lane_constrained_by_vehicle) and self.in_polygon(pos_agt, left_region_corners): # if it is already constrained, then no need to check other agents
                if self.dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    left_lane_constrained_by_vehicle = True
            if (not right_lane_constrained_by_vehicle) and self.in_polygon(pos_agt, right_region_corners): # if it is already constrained, then no need to check other agents
                if self.dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    right_lane_constrained_by_vehicle = True

        front_car_count = 0
        for crowd_agent in self.network_car_agents: # + self.sidewalk_agents:
            pos_agt = crowd_agent.get_position()
            if self.in_polygon(pos_agt, left_region_corners): # if it is already constrained, then no need to check other agents
                front_car_count += 1
                if (not left_lane_constrained_by_vehicle) and self.dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    left_lane_constrained_by_vehicle = True
            if (not right_lane_constrained_by_vehicle) and self.in_polygon(pos_agt, right_region_corners): # if it is already constrained, then no need to check other agents
                if self.dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    right_lane_constrained_by_vehicle = True


        if front_car_count > 1:
            right_lane_constrained_by_vehicle = True

        return left_lane_constrained_by_vehicle, right_lane_constrained_by_vehicle
        #return False, False

    def get_lane_constraints(self, position, forward_vec):
        # left_lane_constrained = False
        # right_lane_constrained = False
        # nearest_pos_at_sidewalk = self.sidewalk.get_nearest_route_point(position)
        # nearest_pos_at_sidewalk = self.sidewalk.get_route_point_position(nearest_pos_at_sidewalk)
        # dist = (nearest_pos_at_sidewalk - position).length()
        # if dist < 1.5 + 2.0 + 0.6: ## 1.5 = sidewalk_width / 2; 2.0 = lane_width / 2; 0.6 is dist threshold
        #     if self.left_of(position, position + forward_vec, nearest_pos_at_sidewalk):
        #         left_lane_constrained = True
        # return left_lane_constrained, right_lane_constrained
        left_line_end = position + (1.5 + 2.0 + 0.8) * ((forward_vec.rotate(np.deg2rad(-90))).make_unit_vector())
        right_line_end = position + (1.5 + 2.0 + 0.5) * ((forward_vec.rotate(np.deg2rad(90))).make_unit_vector())
        left_lane_constrained_by_sidewalk = self.sidewalk.intersects(position, left_line_end)
        right_lane_constrained_by_sidewalk = self.sidewalk.intersects(position, right_line_end)
        #left_lane_constrained_by_vehicle, right_lane_constrained_by_vehicle = self.get_lane_constraints_by_vehicle(position, forward_vec)

        #return True, True
        #return left_lane_constrained_by_sidewalk or left_lane_constrained_by_vehicle, right_lane_constrained_by_sidewalk or right_lane_constrained_by_vehicle
        return left_lane_constrained_by_sidewalk, right_lane_constrained_by_sidewalk

    def get_spawn_range(self, spawn_size = 150, center_pos = None):
        # if it has specified the center position for spawnning the agents

        if center_pos is None:
            center_pos = self.get_ego_pos()

        if center_pos is not None:
            spawn_min = carla.Vector2D(
                max(self.map_bounds_min.x, center_pos.x - spawn_size), 
                max(self.map_bounds_min.y, center_pos.y - spawn_size))
            spawn_max = carla.Vector2D(
                min(self.map_bounds_max.x, center_pos.x + spawn_size),
                min(self.map_bounds_max.y, center_pos.y + spawn_size))
            return spawn_min, spawn_max
        else:
            return self.map_bounds_min, self.map_bounds_max 

    def get_ego_pos(self):
        if self.ego_actor is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.ego_actor = actor
                    break
        if self.ego_actor is not None:
            ego_position = self.ego_actor.get_location()
            return carla.Vector2D(ego_position.x, ego_position.y)
        else:
            return None

    def compute_intersection_of_two_lines(self, ps, pe, qs, qe):
        ## implement based on http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
        denominator = (qe.x - qs.x) * (ps.y - pe.y) - (ps.x - pe.x) * (qe.y - qs.y)
        if denominator == 0:
            return None

        tp = ((qs.y - qe.y) * (ps.x - qs.x) + (qe.x - qs.x) * (ps.y - qs.y)) / denominator
        tq = ((ps.y - pe.y) * (ps.x - qs.x) + (pe.x - ps.x) * (ps.y - qs.y) ) / denominator
        if tp >= 0 and tp <= 1 and tq >= 0 and tq <= 1:
            return ps + tp * (pe - ps)

        return None

    def compute_intersections(self, rect, lane):
        points_in_rect = [] 
        intersections = []

        for i in range(len(rect)-1):
            start = rect[i]
            end = rect[i+1]
            intersect = self.compute_intersection_of_two_lines(start, end, lane[0], lane[1])
            if intersect is not None:
                if intersect not in intersections:
                    intersections.append(intersect)

        start = rect[len(rect)-1]
        end = rect[0]
        intersect = self.compute_intersection_of_two_lines(start, end, lane[0], lane[1])
        if intersect is not None:
            if intersect not in intersections:
                intersections.append(intersect)

        if self.in_polygon(lane[0], rect):
            points_in_rect.append(lane[0])

        if self.in_polygon(lane[1], rect):
            points_in_rect.append(lane[1])

        return points_in_rect, intersections

    # def compute_intersections(self, rect, lane):
    #     points_in_rect = [] 
    #     intersections = []

    #     for i in range(len(rect)-1):
    #         start = rect[i]
    #         end = rect[i+1]
    #         intersect = self.compute_intersection_of_two_lines(start, end, lane[0], lane[1])
    #         if intersect is not None:
    #             if lane[0] not in points_in_rect:
    #                 if self.left_of(start, end, lane[0]):
    #                     points_in_rect.append(lane[0])
    #             if lane[1] not in points_in_rect:
    #                 if self.left_of(start, end, lane[1]):
    #                     points_in_rect.append(lane[1])
    #             if intersect not in intersections:
    #                 intersections.append(intersect)

    #     start = rect[len(rect)-1]
    #     end = rect[0]
    #     intersect = self.compute_intersection_of_two_lines(start, end, lane[0], lane[1])
    #     if intersect is not None:
    #         if lane[0] not in points_in_rect:
    #             if self.left_of(start, end, lane[0]):
    #                 points_in_rect.append(lane[0])
    #         if lane[1] not in points_in_rect:
    #             if self.left_of(start, end, lane[1]):
    #                 points_in_rect.append(lane[1])
    #         if intersect not in intersections:
    #             intersections.append(intersect)


    #     return points_in_rect, intersections


    def get_feasible_lanes(self, intersecting_lanes, center_pos = None, spawn_size_min = 100, spawn_size_max = 150):

        if intersecting_lanes == []:
            return []

        if center_pos is None:
            center_pos = self.get_ego_pos()
            if center_pos is None:
                return []

        rect_list = []
        corners = []
        corners.append(carla.Vector2D(center_pos.x-spawn_size_min, center_pos.y-spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x-spawn_size_min, center_pos.y-spawn_size_min))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_max, center_pos.y-spawn_size_min))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_max, center_pos.y-spawn_size_max))
        rect_list.append(corners)

        corners = []
        corners.append(carla.Vector2D(center_pos.x-spawn_size_min, center_pos.y-spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x-spawn_size_max, center_pos.y-spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x-spawn_size_max, center_pos.y+spawn_size_min))
        corners.append(carla.Vector2D(center_pos.x-spawn_size_min, center_pos.y+spawn_size_min))
        rect_list.append(corners)


        corners = []
        corners.append(carla.Vector2D(center_pos.x-spawn_size_max, center_pos.y+spawn_size_min))
        corners.append(carla.Vector2D(center_pos.x-spawn_size_max, center_pos.y+spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_min, center_pos.y+spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_min, center_pos.y+spawn_size_min))
        rect_list.append(corners)


        corners = []
        corners.append(carla.Vector2D(center_pos.x+spawn_size_min, center_pos.y+spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_max, center_pos.y+spawn_size_max))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_max, center_pos.y-spawn_size_min))
        corners.append(carla.Vector2D(center_pos.x+spawn_size_min, center_pos.y-spawn_size_min))
        rect_list.append(corners)


        feasible_lane_list = []
        for lane in intersecting_lanes:
            for rect in rect_list:
                points_in_rect, intersections = self.compute_intersections(rect, lane)

                num_intersections = len(intersections)
                if num_intersections == 2:
                    feasible_lane_list.append([intersections[0], intersections[1]])
                elif num_intersections == 1:
                    if len(points_in_rect) == 1: # len(points_in_rect) could be zero if the point is the vertice of rect
                        feasible_lane_list.append([points_in_rect[0], intersections[0]])
                elif num_intersections == 0:
                    if len(points_in_rect) == 2: # len(points_in_rect) could be zero if the lane is outside rect
                        feasible_lane_list.append([points_in_rect[0], points_in_rect[1]])
                # if num_intersections > 2: #lane is overlapping with one edge of rect
                #     continue
               
        return feasible_lane_list

    def get_bounds(self, center_pos = None, spawn_size = 150):

        if center_pos is None:
            center_pos = self.get_ego_pos()
            if center_pos is None:
                return None
        bounds_min = carla.Vector2D(center_pos.x - spawn_size, center_pos.y - spawn_size)
        bounds_max = carla.Vector2D(center_pos.x + spawn_size, center_pos.y + spawn_size)

        return (bounds_min, bounds_max)

    def get_intersecting_lanes(self, center_pos = None, spawn_size = 150):
        bounds = self.get_bounds(center_pos, spawn_size)
        if bounds is None:
            return []
        intersecting_lanes = self.network.query_intersect(*bounds)
        intersecting_lanes = [
                (
                    self.network.edges[rp.edge].lanes[rp.lane].shape[rp.segment],
                    self.network.edges[rp.edge].lanes[rp.lane].shape[rp.segment + 1]
                ) for rp in intersecting_lanes]
        return intersecting_lanes

    def get_lane_prob_list(self, feasible_lane_list):
        dist_total = 0.0
        prob_list = []
        for lane in feasible_lane_list:
            dist = (lane[0]-lane[1]).length()
            prob_list.append(dist)
            dist_total += dist

        for i in range(len(prob_list)):
            prob_list[i] /= dist_total

        return prob_list

    def in_ego_bounds(self, point):
        bounds_min, bounds_max = self.get_spawn_range()

        return bounds_min.x <= point.x <= bounds_max.x and \
               bounds_min.y <= point.y <= bounds_max.y


    def update(self):
        # self.center_pos = carla.Vector2D(450, 400)
        # spawn_size_min = 10
        # spawn_size_max = 200
        # self.intersecting_lanes = self.get_intersecting_lanes(center_pos = self.center_pos, spawn_size = spawn_size_max)
        # self.feasible_lane_list = self.get_feasible_lanes(self.intersecting_lanes, center_pos = self.center_pos, spawn_size_min = spawn_size_min, spawn_size_max = spawn_size_max)

        # for lane in self.feasible_lane_list:
        #     self.draw_line(lane[0], lane[1] - lane[0], carla.Color (255,0,0))

        # self.lane_prob_list = self.get_lane_prob_list(self.feasible_lane_list)
        # self.initialized = True
        if not self.initialized: ## the first time
            if self.ego_actor is None:
                self.center_pos = carla.Vector2D(450, 400)
            else:
                self.center_pos = self.get_ego_pos()
            spawn_size_min = 0
            spawn_size_max = 150 #100

            self.intersecting_lanes = self.get_intersecting_lanes(center_pos = self.center_pos, spawn_size = spawn_size_max)
            self.feasible_lane_list = self.get_feasible_lanes(self.intersecting_lanes, center_pos = self.center_pos, spawn_size_min = spawn_size_min, spawn_size_max = spawn_size_max)
            self.lane_prob_list = self.get_lane_prob_list(self.feasible_lane_list)
            self.initialized = True
        else:
            if self.ego_actor is None:
                self.center_pos = carla.Vector2D(450, 400)
            else:
                self.center_pos = self.get_ego_pos()
            spawn_size_min = 100#10 #150
            spawn_size_max = 150#50 #200
            spawn_min, spawn_max = self.get_spawn_range()
            self.intersecting_lanes = self.get_intersecting_lanes(center_pos = self.center_pos, spawn_size = spawn_size_max )
            self.feasible_lane_list = self.get_feasible_lanes(self.intersecting_lanes, center_pos = self.center_pos, spawn_size_min = spawn_size_min, spawn_size_max = spawn_size_max)
            self.lane_prob_list = self.get_lane_prob_list(self.feasible_lane_list)

        while len(self.network_car_agents) < self.num_network_car_agents:
            path = None
            if len(self.feasible_lane_list) == 0:
                spawn_min, spawn_max = self.get_spawn_range() 
                path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_min, spawn_max)
            else:
                path = NetworkAgentPath.rand_path_fron_feasible_lanes(self, self.path_min_points, self.path_interval, self.feasible_lane_list, prob_list = self.lane_prob_list)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.1
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.cars_blueprints),
                    trans)
            self.world.wait_for_tick(1.0)
            if actor:
                self.network_car_agents.append(CrowdNetworkCarAgent(
                    actor, path, 
                    #5.0 + random.uniform(0.0, 1.5)))
                    5.0 + random.uniform(0.0, 0.5)))
        
        while len(self.network_bike_agents) < self.num_network_bike_agents:
            path = None
            if len(self.feasible_lane_list) == 0:
                spawn_min, spawn_max = self.get_spawn_range() 
                path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_min, spawn_max)
            else:
                path = NetworkAgentPath.rand_path_fron_feasible_lanes(self, self.path_min_points, self.path_interval, self.feasible_lane_list, prob_list = self.lane_prob_list)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.1
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.bikes_blueprints),
                    trans)
            self.world.wait_for_tick(1.0)
            if actor:
                self.network_bike_agents.append(CrowdNetworkBikeAgent(
                    actor, path, 
                    #5.0 + random.uniform(0.0, 1.5)))
                    3.0 + random.uniform(0.0, 0.5)))
      
        while len(self.sidewalk_agents) < self.num_sidewalk_agents:
            spawn_min, spawn_max = self.get_spawn_range()
            path = SidewalkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_min, spawn_max)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.5
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.walker_blueprints),
                    trans)
            self.world.wait_for_tick(1.0)
            if actor:
                self.sidewalk_agents.append(CrowdSidewalkAgent(
                    actor, path, 
                    0.5 + random.uniform(0.0, 1.0)))
    
        commands = []
        
        next_agents = []
        for (i, crowd_agent) in enumerate(self.network_car_agents + self.network_bike_agents + self.sidewalk_agents):
            if not self.in_ego_bounds(crowd_agent.get_position()) or crowd_agent.get_position3D().z < -10:
                next_agents.append(None)
                self.gamma.set_agent_position(i, default_agent_pos)
                self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
                commands.append(carla.command.DestroyActor(crowd_agent.actor.id))
                continue

            self.gamma.set_agent(i, crowd_agent.get_agent_params())
            pref_vel = crowd_agent.get_preferred_velocity()
            if pref_vel:
                self.draw_line(crowd_agent.get_position(), pref_vel, carla.Color (255,0,0))
                self.draw_line(crowd_agent.get_position(), crowd_agent.get_velocity(), carla.Color (0,255,0))
                next_agents.append(crowd_agent)
                self.gamma.set_agent_position(i, crowd_agent.get_position())
                self.gamma.set_agent_velocity(i, crowd_agent.get_velocity())
                self.gamma.set_agent_heading(i, crowd_agent.get_forward_direction())
                self.gamma.set_agent_bounding_box_corners(i, crowd_agent.get_bounding_box_corners())
                self.gamma.set_agent_pref_velocity(i, pref_vel)             
                self.gamma.set_agent_path_forward(i, crowd_agent.get_path_forward())
            #     # left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_forward_direction())
            #     # left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_path_forward())
            #     # start = timeit.default_timer()
            #     left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_path_forward())
            #     # run_time = timeit.default_timer() - start
            #     # print("get_lane_constraints ======================")
            #     # print(run_time)
            #     #self.gamma.set_agent_lane_constraints(i, False, True)
            #     self.gamma.set_agent_lane_constraints(i, right_lane_constrained, left_lane_constrained)  ## to check. It seems that we should set left_lane_constrained to false as currently we do because of the difference of the coordiante systems.
            else:
                next_agents.append(None)
                self.gamma.set_agent_position(i, default_agent_pos)
                self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
                commands.append(carla.command.DestroyActor(crowd_agent.actor.id))

        # start = timeit.default_timer()
        self.gamma.do_step()
        # run_time = timeit.default_timer() - start
        # print("dostep ======================")
        # print(run_time)
        global first_time
        global prev_time
        if first_time:
            prev_time = timeit.default_timer()
            first_time = False
        cur_time = timeit.default_timer()
        simu_time = cur_time - prev_time
        prev_time = cur_time

        #simu_time = 0.05 #self.world.wait_for_tick(1.0).timestamp.delta_seconds
        for (i, crowd_agent) in enumerate(next_agents):
            if crowd_agent:

                vel_to_exe = self.gamma.get_agent_velocity(i)

                #self.draw_line(crowd_agent.get_position(), vel_to_exe, carla.Color (0,0,255))

                cur_vel = crowd_agent.actor.get_velocity()

                cur_vel = carla.Vector2D(cur_vel.x, cur_vel.y)

                angle_diff = get_signed_angle_diff(vel_to_exe, cur_vel)
                if angle_diff > 30 or angle_diff < -30:
                    vel_to_exe = 0.5 * (vel_to_exe + cur_vel)
                    #self.draw_line(crowd_agent.get_position(), vel_to_exe, carla.Color (255,255,255))


                # cur_loc = crowd_agent.actor.get_location()
                # translation = simu_time * vel_to_exe
                # loc = cur_loc + carla.Vector3D(translation.x, translation.y, 0)
                # #crowd_agent.actor.set_location(loc)

                # trans = crowd_agent.actor.get_transform()
                # trans.location = loc
                # if vel_to_exe.length() != 0:
                #     trans.rotation.yaw = np.rad2deg(math.atan2(vel_to_exe.y, vel_to_exe.x))
                # crowd_agent.actor.set_transform(trans)

                control = crowd_agent.get_control(vel_to_exe)
                if type(crowd_agent) is CrowdNetworkCarAgent or type(crowd_agent) is CrowdNetworkBikeAgent:
                    commands.append(carla.command.ApplyVehicleControl(crowd_agent.actor.id, control))
                elif type(crowd_agent) is CrowdSidewalkAgent:
                    commands.append(carla.command.ApplyWalkerControl(crowd_agent.actor.id, control))
        
        self.network_car_agents = [a for a in next_agents if a and type(a) is CrowdNetworkCarAgent]
        self.network_bike_agents = [a for a in next_agents if a and type(a) is CrowdNetworkBikeAgent]
        self.sidewalk_agents = [a for a in next_agents if a and type(a) is CrowdSidewalkAgent]
        
        self.client.apply_batch(commands)
        self.world.wait_for_tick(1.0)

        network_agents_msg = carla_connector2.msg.CrowdNetworkAgentArray()
        network_agents_msg.header.stamp = rospy.Time.now()
        for a in self.network_car_agents + self.network_bike_agents:
            network_agent_msg = carla_connector2.msg.CrowdNetworkAgent()
            network_agent_msg.id = a.get_id()
            network_agent_msg.type = a.get_type()
            network_agent_msg.route_point.edge = a.path.route_points[0].edge
            network_agent_msg.route_point.lane = a.path.route_points[0].lane
            network_agent_msg.route_point.segment = a.path.route_points[0].segment
            network_agent_msg.route_point.offset = a.path.route_points[0].offset
            network_agents_msg.agents.append(network_agent_msg)
        self.network_agents_pub.publish(network_agents_msg)
        
        sidewalk_agents_msg = carla_connector2.msg.CrowdSidewalkAgentArray()
        sidewalk_agents_msg.header.stamp = rospy.Time.now()
        for a in self.sidewalk_agents:
            sidewalk_agent_msg = carla_connector2.msg.CrowdSidewalkAgent()
            sidewalk_agent_msg.id = a.get_id()
            sidewalk_agent_msg.type = 'ped'
            sidewalk_agent_msg.route_point.polygon_id = a.path.route_points[0].polygon_id
            sidewalk_agent_msg.route_point.segment_id = a.path.route_points[0].segment_id
            sidewalk_agent_msg.route_point.offset = a.path.route_points[0].offset
            sidewalk_agent_msg.route_orientation = a.path.route_orientations[0]
            sidewalk_agents_msg.agents.append(sidewalk_agent_msg)
        self.sidewalk_agents_pub.publish(sidewalk_agents_msg)
        
            
if __name__ == '__main__':
    rospy.init_node('gamma_crowd_controller')
    gamma_crowd_controller = GammaCrowdController()

    rate = rospy.Rate(100) ## to check
    #rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        gamma_crowd_controller.update()
        rate.sleep()

    gamma_crowd_controller.dispose()
