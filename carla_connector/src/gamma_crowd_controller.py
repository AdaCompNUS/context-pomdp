#!/usr/bin/env python2

from drunc import Drunc
import carla

import math
import random
import numpy as np
import rospy
import os
import sys

from std_msgs.msg import Bool
from network_agent_path import NetworkAgentPath
from sidewalk_agent_path import SidewalkAgentPath
from util import *
import carla_connector.msg
from peds_unity_system.msg import car_info as CarInfo
import timeit
import time

first_time = True
prev_time = timeit.default_timer()

default_agent_pos = carla.Vector2D(10000, 10000)
default_agent_bbox = []
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,-1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,-1))

def get_position(actor):
    pos3D = actor.get_location()
    return carla.Vector2D(pos3D.x, pos3D.y)
    
def get_forward_direction(actor):
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)

class CrowdAgent(object):
    def __init__(self, actor, preferred_speed):
        self.actor = actor
        self.preferred_speed = preferred_speed
        self.actor.set_collision_enabled(True) ## to check. Disable collision will generate vehicles that are overlapping
        self.stuck_time = None
    
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

    def disable_collision(self):
        self.actor.set_collision_enabled(False)
    
    def get_path_occupancy(self):
        p = [self.get_position()] + [self.path.get_position(i) for i in range(self.path.min_points)]
        return carla.OccupancyMap(p, self.actor.bounding_box.extent.y * 2 + 1.0)


class CrowdNetworkAgent(CrowdAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkAgent, self).__init__(actor, preferred_speed)
        self.path = path

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_bounding_box_corners(self, expand = 0.0):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction

        half_y_len = bbox.extent.y + expand
        half_x_len = bbox.extent.x + expand

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

        k = 1.0 # 1.0
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
    
    def get_bounding_box_corners(self, expand=0.0):
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
        self.initialized = False

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
                carla_connector.msg.CrowdNetworkAgentArray, 
                queue_size=1)
        self.sidewalk_agents_pub = rospy.Publisher(
                '/crowd/sidewalk_agents', 
                carla_connector.msg.CrowdSidewalkAgentArray, 
                queue_size=1)
        self.il_car_info_sub = rospy.Subscriber(
                '/IL_car_info',
                CarInfo,
                self.il_car_info_callback,
                queue_size=1)

        for i in range(self.num_network_car_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Car'), i)
        
        for i in range(self.num_network_bike_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Bicycle'), i + self.num_network_car_agents)
        
        for i in range(self.num_sidewalk_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('People'), i + self.num_network_car_agents + self.num_network_bike_agents)
        
        # For ego vehicle.
        self.gamma.add_agent(carla.AgentParams.get_default('Car'), self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents)

        for triangle in carla.OccupancyMap(self.scenario_min, self.scenario_max).difference(self.network_occupancy_map).get_triangles():
            self.gamma.add_obstacle([triangle.v2, triangle.v1, triangle.v1])
        self.gamma.process_obstacles()
        
        self.start_time = None
        self.stats_total_num_car = 0
        self.stats_total_num_bike = 0
        self.stats_total_num_ped = 0
        self.stats_total_num_stuck_car = 0
        self.stats_total_num_stuck_bike = 0
        self.stats_total_num_stuck_ped = 0
        self.log_file = open(os.path.join(os.path.expanduser('~'), 'gamma_data.txt'), 'w', buffering=0)

        self.do_publish = False 
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_agents)

    def get_bounding_box_corners(self, actor, expand = 0.0):
        bbox = actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
        forward_vec = get_forward_direction(actor).make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction

        half_y_len = bbox.extent.y + expand
        half_x_len = bbox.extent.x + expand

        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners

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
            self.find_ego_actor()
            self.gamma.set_agent_bounding_box_corners(i, self.get_bounding_box_corners(self.ego_actor, 0.2))
            self.gamma.set_agent_pref_velocity(i, carla.Vector2D(
                self.ego_car_info.car_pref_vel.x,
                self.ego_car_info.car_pref_vel.y))
        else:
            self.gamma.set_agent_position(i, default_agent_pos)
            self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
            self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
            self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
    
    def det(self, vector1, vector2):
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

    def get_spawn_range(self, center, size):
        spawn_min = carla.Vector2D(
            center.x - size, 
            center.y - size)
        spawn_max = carla.Vector2D(
            center.x + size,
            center.y + size)
        return (spawn_min, spawn_max)

    def find_ego_actor(self):
        if self.ego_actor is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.ego_actor = actor
                    break

    def check_bounds(self, point, bounds_min, bounds_max):
        return bounds_min.x <= point.x <= bounds_max.x and \
               bounds_min.y <= point.y <= bounds_max.y

    def no_collision(self):
        for (i, crowd_agent) in enumerate(self.network_car_agents + self.network_bike_agents + self.sidewalk_agents):
            crowd_agent.disable_collision()
         
    def get_spawn_occupancy_map(self, center_pos, spawn_size_min, spawn_size_max):
        return carla.OccupancyMap(
                carla.Vector2D(center_pos.x - spawn_size_max, center_pos.y - spawn_size_max),
                carla.Vector2D(center_pos.x + spawn_size_max, center_pos.y + spawn_size_max)) \
            .difference(carla.OccupancyMap(
                carla.Vector2D(center_pos.x - spawn_size_min, center_pos.y - spawn_size_min),
                carla.Vector2D(center_pos.x + spawn_size_min, center_pos.y + spawn_size_min)))

    def update(self):
        update_time = rospy.Time.now()

        # Check for ego actor.
        self.find_ego_actor() 

        # Determine bounds variables.
        if self.ego_actor is None:
            bounds_center = self.scenario_center
            bounds_min = self.scenario_min
            bounds_max = self.scenario_max
        else:
            bounds_center = carla.Vector2D(self.ego_actor.get_location().x, self.ego_actor.get_location().y)
            bounds_min = bounds_center + carla.Vector2D(-150, -150)
            bounds_max = bounds_center + carla.Vector2D(150, 150)
        
        # Determine spawning variables.
        if not self.initialized:
            spawn_size_min = 0
            spawn_size_max = 100
        else:
            spawn_size_min = 100
            spawn_size_max = 150
        spawn_segment_map = self.network_segment_map.intersection(self.get_spawn_occupancy_map(bounds_center, spawn_size_min, spawn_size_max))

        start_t = time.time()
        while len(self.network_car_agents) < self.num_network_car_agents:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.2
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.cars_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.network_car_agents.append(CrowdNetworkCarAgent(
                    actor, path, 
                    5.0 + random.uniform(0.0, 0.5)))
                self.stats_total_num_car += 1
            elapsed_time = time.time() - start_t
            # if elapsed_time > 3:
                # break
            if len(self.network_car_agents) > self.num_network_car_agents/1.2:
                self.do_publish = True;

        start_t = time.time()
        while len(self.network_bike_agents) < self.num_network_bike_agents:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.2
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.bikes_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.network_bike_agents.append(CrowdNetworkBikeAgent(
                    actor, path, 
                    3.0 + random.uniform(0, 0.5)))
                self.stats_total_num_bike += 1
            elapsed_time = time.time() - start_t

            if len(self.network_bike_agents) > self.num_network_bike_agents/2.0:
                self.do_publish = True;
            # if elapsed_time > 3:
                # break

        start_t = time.time()
        while len(self.sidewalk_agents) < self.num_sidewalk_agents:
            spawn_min, spawn_max = self.get_spawn_range(bounds_center, 40)
            path = SidewalkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_min, spawn_max)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.2
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.walker_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.sidewalk_agents.append(CrowdSidewalkAgent(
                    actor, path, 
                    0.5 + random.uniform(0.0, 1.0)))
                self.stats_total_num_ped += 1
            elapsed_time = time.time() - start_t

            if len(self.sidewalk_agents) > self.num_sidewalk_agents/2.0:
                self.do_publish = True;
            # if elapsed_time > 3:
                # break
        
        commands = []
        
        next_agents = []
        for (i, crowd_agent) in enumerate(self.network_car_agents + self.network_bike_agents + self.sidewalk_agents):
            delete = False
            if not delete and not self.check_bounds(crowd_agent.get_position(), bounds_min, bounds_max):
                delete = True
            if not delete and crowd_agent.get_position3D().z < -10:
                delete = True
            if not delete and (type(crowd_agent) is not CrowdSidewalkAgent and \
                    not self.network_occupancy_map.contains(crowd_agent.get_position())):
                delete = True

            if self.initialized:
                if crowd_agent.get_velocity().length() < 0.2:
                    if crowd_agent.stuck_time is not None:
                        if (update_time - crowd_agent.stuck_time).to_sec() >= 5.0:
                            delete = True
                            if type(crowd_agent) is CrowdNetworkCarAgent:
                                self.stats_total_num_stuck_car += 1
                            elif type(crowd_agent) is CrowdNetworkBikeAgent:
                                self.stats_total_num_stuck_bike += 1
                            elif type(crowd_agent) is CrowdSidewalkAgent:
                                self.stats_total_num_stuck_ped += 1
                    else :
                        crowd_agent.stuck_time = update_time
                else:
                    crowd_agent.stuck_time = None

            if delete:
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
                # self.draw_line(crowd_agent.get_position(), pref_vel, carla.Color (255,0,0))
                # self.draw_line(crowd_agent.get_position(), crowd_agent.get_velocity(), carla.Color (0,255,0))
                next_agents.append(crowd_agent)
                self.gamma.set_agent_position(i, crowd_agent.get_position())
                self.gamma.set_agent_velocity(i, crowd_agent.get_velocity())
                self.gamma.set_agent_heading(i, crowd_agent.get_forward_direction())
                self.gamma.set_agent_bounding_box_corners(i, crowd_agent.get_bounding_box_corners(0.3))
                self.gamma.set_agent_pref_velocity(i, pref_vel)             
                self.gamma.set_agent_path_forward(i, crowd_agent.get_path_forward())
                # left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_forward_direction())
                # left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_path_forward())
                # start = timeit.default_timer()
                left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(), crowd_agent.get_path_forward())
                # run_time = timeit.default_timer() - start
                # print("get_lane_constraints ======================")
                # print(run_time)
                #self.gamma.set_agent_lane_constraints(i, False, True)
                self.gamma.set_agent_lane_constraints(i, right_lane_constrained, left_lane_constrained)  ## to check. It seems that we should set left_lane_constrained to false as currently we do because of the difference of the coordiante systems.
            else:
                next_agents.append(None)
                self.gamma.set_agent_position(i, default_agent_pos)
                self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
                commands.append(carla.command.DestroyActor(crowd_agent.actor.id))

        # start = timeit.default_timer()
        try:
            self.gamma.do_step()
        except Exception as e:
            print(e)

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

        if not self.initialized:
            self.start_time = rospy.Time.now()
            self.initialized = True

        #simu_time = 0.05 #self.world.wait_for_tick(5.0).timestamp.delta_seconds
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
        self.world.wait_for_tick(5.0)
        '''
        stats_num_car = 0
        stats_num_bike = 0
        stats_num_ped = 0
        stats_sum_speed_car = 0.0
        stats_sum_speed_bike = 0.0
        stats_sum_speed_ped = 0.0

        for agent in self.network_car_agents:
            stats_num_car += 1
            stats_sum_speed_car += agent.get_velocity().length()
        for agent in self.network_bike_agents:
            stats_num_bike += 1
            stats_sum_speed_bike += agent.get_velocity().length()
        for agent in self.sidewalk_agents:
            stats_num_ped += 1
            stats_sum_speed_ped += agent.get_velocity().length()

        self.log_file.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            (update_time - self.start_time).to_sec(),
            self.stats_total_num_car, 
            self.stats_total_num_bike, 
            self.stats_total_num_ped,
            self.stats_total_num_stuck_car, 
            self.stats_total_num_stuck_bike, 
            self.stats_total_num_stuck_ped,
            stats_num_car,
            stats_num_bike,
            stats_num_ped,
            stats_sum_speed_car,
            stats_sum_speed_bike,
            stats_sum_speed_ped))

        
        print('Time = {}'.format((update_time - self.start_time).to_sec()))
        print('Total spawned = {}, {}, {}'.format(
            self.stats_total_num_car, 
            self.stats_total_num_bike, 
            self.stats_total_num_ped))
        print('Stuck deleted = {}, {}, {}'.format(
            self.stats_total_num_stuck_car, 
            self.stats_total_num_stuck_bike, 
            self.stats_total_num_stuck_ped))
        print('Avg. Instantaneous Speed = {}, {}, {}'.format(stats_avg_speed_car, stats_avg_speed_bike, stats_avg_speed_ped))
        '''

        ''' Temporarily disabled for experiments.
        '''

    def publish_agents(self, tick):
        if self.do_publish is False:
            return

        network_agents_msg = carla_connector.msg.CrowdNetworkAgentArray()
        network_agents_msg.header.stamp = rospy.Time.now()
        for a in self.network_car_agents + self.network_bike_agents:
            network_agent_msg = carla_connector.msg.CrowdNetworkAgent()
            network_agent_msg.id = a.get_id()
            network_agent_msg.type = a.get_type()
            network_agent_msg.route_point.edge = a.path.route_points[0].edge
            network_agent_msg.route_point.lane = a.path.route_points[0].lane
            network_agent_msg.route_point.segment = a.path.route_points[0].segment
            network_agent_msg.route_point.offset = a.path.route_points[0].offset
            network_agents_msg.agents.append(network_agent_msg)
        self.network_agents_pub.publish(network_agents_msg)
        
        sidewalk_agents_msg = carla_connector.msg.CrowdSidewalkAgentArray()
        sidewalk_agents_msg.header.stamp = rospy.Time.now()
        for a in self.sidewalk_agents:
            sidewalk_agent_msg = carla_connector.msg.CrowdSidewalkAgent()
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
    init_time = rospy.Time.now()
    rospy.wait_for_message("/meshes_spawned", Bool)
    rospy.wait_for_message("/IL_car_info", CarInfo)

    gamma_crowd_controller = GammaCrowdController()

    rate = rospy.Rate(100) ## to check
    #rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        start_time = rospy.Time.now()
        gamma_crowd_controller.update()
        #gamma_crowd_controller.no_collision()
        end_time = rospy.Time.now()
        duration = (end_time - start_time).to_sec()
        elapsed = (end_time - init_time).to_sec()
        print('Update = {} ms = {} hz'.format(duration * 1000, 1.0 / duration))
        # print('Crowd update at {}'.format(elapsed))
        # sys.stdout.flush()
        rate.sleep()

    gamma_crowd_controller.dispose()
