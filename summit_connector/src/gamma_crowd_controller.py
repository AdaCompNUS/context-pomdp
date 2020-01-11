#!/usr/bin/env python2
from util import *

import carla

import random
import numpy as np
import rospy
import sys
from std_msgs.msg import Float32, Bool, Int32
from geometry_msgs.msg import Twist
import nav_msgs.msg
from network_agent_path import NetworkAgentPath
from sidewalk_agent_path import SidewalkAgentPath
import msg_builder.msg
from msg_builder.msg import car_info as CarInfo
import timeit
import time
from threading import RLock

default_agent_pos = carla.Vector2D(10000, 10000)
default_agent_bbox = [default_agent_pos + carla.Vector2D(1, -1), default_agent_pos + carla.Vector2D(1, 1),
                      default_agent_pos + carla.Vector2D(-1, 1), default_agent_pos + carla.Vector2D(-1, -1)]

class CrowdAgent(object):
    def __init__(self, summit, actor, preferred_speed, behavior_type = carla.AgentBehaviorType.Gamma):
        self.summit = summit
        self.actor = actor
        self.preferred_speed = preferred_speed
        self.stuck_time = None
        self.path = None
        self.control_velocity = carla.Vector2D(0, 0)
        self.behavior_type = behavior_type

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
        pos3d = self.actor.get_location()
        return carla.Vector2D(pos3d.x, pos3d.y)

    def get_position_3d(self):
        return self.actor.get_location()

    def get_path_occupancy(self):
        p = [self.get_position()] + [self.path.get_position(i) for i in range(self.path.min_points)]
        return carla.OccupancyMap(p, self.actor.bounding_box.extent.y * 2 + 1.0)


class CrowdNetworkAgent(CrowdAgent):
    def __init__(self, summit, actor, path, preferred_speed, behavior_type = carla.AgentBehaviorType.Gamma):
        super(CrowdNetworkAgent, self).__init__(summit, actor, preferred_speed, behavior_type)
        self.path = path

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector()  # the local x direction (left-handed
        # coordinate system)
        side_ward_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction

        half_y_len = bbox.extent.y + 0.3
        half_x_len_forward = bbox.extent.x + 1.0
        half_x_len_backward = bbox.extent.x + 0.1

        corners = [loc - half_x_len_backward * forward_vec + half_y_len * side_ward_vec,
                   loc + half_x_len_forward * forward_vec + half_y_len * side_ward_vec,
                   loc + half_x_len_forward * forward_vec - half_y_len * side_ward_vec,
                   loc - half_x_len_backward * forward_vec - half_y_len * side_ward_vec]

        return corners

    def get_preferred_velocity(self, lane_change_probability=0.0, rng=random):
        position = self.get_position()

        ## to check
        if not self.path.resize():
            return None

        current_offset = self.path.get_min_offset(position)
        nearest_route_point = self.summit.network.get_nearest_route_point(position)
        if nearest_route_point.edge == self.path.route_points[0].edge and nearest_route_point.lane != \
                self.path.route_points[0].lane:
            if rng.uniform(0.0, 1.0) <= lane_change_probability:
                new_path_candidates = self.summit.network.get_next_route_paths(nearest_route_point,
                                                                              self.path.min_points - 1,
                                                                              self.path.interval)
                if len(new_path_candidates) > 0:
                    new_path = NetworkAgentPath(self.summit, self.path.min_points, self.path.interval)
                    new_path.route_points = rng.choice(new_path_candidates)[0:self.path.min_points]
                    self.path = new_path

        self.path.cut(position)
        if not self.path.resize():
            return None

        target_position = self.path.get_position(5)  ## to check
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

    def get_control(self, velocity, k1, k2, k3):
        steer = get_signed_angle_diff(velocity, self.get_forward_direction())
        min_steering_angle = -45.0
        max_steering_angle = 45.0
        if steer > max_steering_angle:
            steer = max_steering_angle
        elif steer < min_steering_angle:
            steer = min_steering_angle

        desired_speed = velocity.length()
        cur_speed = self.get_velocity().length()
        control = self.actor.get_control()

        steer = k1 * steer / (max_steering_angle - min_steering_angle) * 2.0

        if desired_speed > cur_speed:
            control.throttle = k2 * (desired_speed - cur_speed) / desired_speed
            control.brake = 0.0
        elif desired_speed == cur_speed:
            control.throttle = 0.0
            control.brake = 0.00
        elif desired_speed < cur_speed:
            control.throttle = 0.0
            control.brake = k3 * (cur_speed - desired_speed) / cur_speed
        control.steer = steer

        control.manual_gear_shift = True
        control.gear = 1

        return control


class CrowdNetworkCarAgent(CrowdNetworkAgent):
    def __init__(self, summit, actor, path, preferred_speed, behavior_type = carla.AgentBehaviorType.Gamma):
        super(CrowdNetworkCarAgent, self).__init__(summit, actor, path, preferred_speed, behavior_type)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_type(self):
        return 'car'


class CrowdNetworkBikeAgent(CrowdNetworkAgent):
    def __init__(self, summit, actor, path, preferred_speed, behavior_type = carla.AgentBehaviorType.Gamma):
        super(CrowdNetworkBikeAgent, self).__init__(summit, actor, path, preferred_speed, behavior_type)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Bicycle')

    def get_type(self):
        return 'bike'


class CrowdSidewalkAgent(CrowdAgent):
    def __init__(self, summit, actor, path, preferred_speed, behavior_type = carla.AgentBehaviorType.Gamma):
        super(CrowdSidewalkAgent, self).__init__(summit, actor, preferred_speed, behavior_type)
        self.path = path

    def get_agent_params(self):
        return carla.AgentParams.get_default('People')

    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector()  # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction. (rotating clockwise by 90 deg)

        # Hardcoded values for people.
        half_y_len = 0.25
        half_x_len = 0.25

        # half_y_len = bbox.extent.y
        # half_x_len = bbox.extent.x

        corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
                   loc + half_x_len * forward_vec + half_y_len * sideward_vec,
                   loc + half_x_len * forward_vec - half_y_len * sideward_vec,
                   loc - half_x_len * forward_vec - half_y_len * sideward_vec]

        return corners

    def get_preferred_velocity(self, lane_change_probability=0.0, rng=random):
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
        return carla.WalkerControl(
            carla.Vector3D(velocity.x, velocity.y, 0),
            1.0, False)


class GammaCrowdController(Summit):
    def __init__(self):
        super(GammaCrowdController, self).__init__()

        self.ego_car_info = None
        self.network_car_agents = []
        self.network_car_agents_lock = RLock()
        self.network_bike_agents = []
        self.network_bike_agents_lock = RLock()
        self.sidewalk_agents = []
        self.sidewalk_agents_lock = RLock()
        self.gamma = carla.RVOSimulator()
        self.ego_actor = None
        self.initialized = False
        self.start_time = rospy.Time().now()

        self.walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.vehicles_blueprints = self.world.get_blueprint_library().filter('vehicle.*')

        self.cars_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.cars_blueprints = [x for x in self.cars_blueprints if x.id not in ['vehicle.mini.cooperst']]

        self.bikes_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 2]

        self.num_network_car_agents = rospy.get_param('~num_network_car_agents')
        self.num_network_bike_agents = rospy.get_param('~num_network_bike_agents')
        self.num_sidewalk_agents = rospy.get_param('~num_sidewalk_agents')
        self.path_min_points = rospy.get_param('~path_min_points')
        self.path_interval = rospy.get_param('~path_interval')
        self.lane_change_probability = rospy.get_param('~lane_change_probability')
        self.spawn_clearance_ego = rospy.get_param('~spawn_clearance_ego')
        self.spawn_clearance_vehicle = rospy.get_param('~spawn_clearance_vehicle')
        self.spawn_clearance_person = rospy.get_param('~spawn_clearance_person')
        self.crowd_range = rospy.get_param('~crowd_range')
        self.network_agents_pub = rospy.Publisher(
            '/crowd/network_agents',
            msg_builder.msg.CrowdNetworkAgentArray,
            queue_size=1)
        self.sidewalk_agents_pub = rospy.Publisher(
            '/crowd/sidewalk_agents',
            msg_builder.msg.CrowdSidewalkAgentArray,
            queue_size=1)
        self.agents_ready_pub = rospy.Publisher('/agents_ready', Bool, queue_size=1, latch=True)
        self.gamma_cmd_accel_pub = rospy.Publisher('/gamma_cmd_accel', Float32, queue_size=1)
        self.gamma_cmd_speed_pub = rospy.Publisher('/gamma_cmd_speed', Float32, queue_size=1)
        self.gamma_cmd_steer_pub = rospy.Publisher('/gamma_cmd_steer', Float32, queue_size=1)
        self.gamma_lane_change_pub = rospy.Publisher('/gamma_lane_decision', Int32, queue_size=1)

        # Check for ego actor.
        self.find_ego_actor()

        self.il_car_info_sub = rospy.Subscriber(
            '/ego_state',
            CarInfo,
            self.il_car_info_callback,
            queue_size=1)

        for i in range(self.num_network_car_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Car'), i)

        for i in range(self.num_network_bike_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('Bicycle'), i + self.num_network_car_agents)

        for i in range(self.num_sidewalk_agents):
            self.gamma.add_agent(carla.AgentParams.get_default('People'),
                                 i + self.num_network_car_agents + self.num_network_bike_agents)

        # For ego vehicle.
        self.gamma.add_agent(carla.AgentParams.get_default('Car'),
                             self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents)

        for triangle in carla.OccupancyMap(self.scenario_min, self.scenario_max).difference(
                self.network_occupancy_map).get_triangles():
            self.gamma.add_obstacle([triangle.v2, triangle.v1, triangle.v1])
        self.gamma.process_obstacles()

        self.do_publish = False

        self.publish_agents_timer = rospy.Timer(rospy.Duration(0.1), self.publish_agents)
        self.update_spawn_timer = rospy.Timer(rospy.Duration(1.0 / 5), self.update_spawn)
        self.update_destroy_timer = rospy.Timer(rospy.Duration(1.0 / 1), self.update_destroy)
        self.update_gamma_timer = rospy.Timer(rospy.Duration(1.0 / 50), self.update_gamma)
        self.update_agent_controls_timer = rospy.Timer(rospy.Duration(1.0 / 20), self.update_agent_controls)
        self.update_lane_decision = rospy.Timer(rospy.Duration(1.0 / 1), self.make_lane_decision)

    def get_aabb(self, actor):
        bbox = actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
        forward_vec = get_forward_direction(
            actor).make_unit_vector()  # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction
        corners = [loc - bbox.extent.x * forward_vec + bbox.extent.y * sideward_vec,
                   loc + bbox.extent.x * forward_vec + bbox.extent.y * sideward_vec,
                   loc + bbox.extent.x * forward_vec - bbox.extent.y * sideward_vec,
                   loc - bbox.extent.x * forward_vec - bbox.extent.y * sideward_vec]
        return carla.AABB2D(
            carla.Vector2D(
                min(v.x for v in corners),
                min(v.y for v in corners)),
            carla.Vector2D(
                max(v.x for v in corners),
                max(v.y for v in corners)))

    def get_bounding_box_corners(self, actor):
        bbox = actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
        forward_vec = get_forward_direction(
            actor).make_unit_vector()  # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction

        half_y_len = bbox.extent.y + 0.3
        half_x_len = bbox.extent.x + 1.0

        corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
                   loc + half_x_len * forward_vec + half_y_len * sideward_vec,
                   loc + half_x_len * forward_vec - half_y_len * sideward_vec,
                   loc - half_x_len * forward_vec - half_y_len * sideward_vec]

        return corners

    def dispose(self):
        self.publish_agents_timer.shutdown()
        self.update_agent_controls_timer.shutdown()
        self.update_gamma_timer.shutdown()
        self.update_lane_decision.shutdown()
        commands = []
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_car_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_bike_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.sidewalk_agents)
        self.client.apply_batch(commands)
        print('Destroyed crowd actors.')

    def il_car_info_callback(self, car_info):
        self.ego_car_info = car_info

    def det(self, vector1, vector2):
        return vector1.y * vector2.x - vector1.x * vector2.y

    def left_of(self, a, b, c):  # if c is at the left side of vector ab, then return Ture, False otherwise
        if self.det(a - c, b - a) > 0:
            return True
        return False

    def in_polygon(self, position, rect):
        if len(rect) < 3:
            return False
        for i in range(0, len(rect) - 1):
            if not self.left_of(rect[i], rect[i + 1], position):
                return False
        if not self.left_of(rect[len(rect) - 1], rect[0], position):
            return False

        return True

    def draw_line(self, pos, vec, color=carla.Color(255, 0, 0)):
        height = 3
        start = carla.Vector3D(pos.x, pos.y, height)
        end = carla.Vector3D(pos.x + vec.x, pos.y + vec.y, height)
        self.world.debug.draw_line(start, end, color=color, life_time=0.1)

    def draw_box(self, corners):
        height = 1
        for i in range(len(corners) - 1):
            start = carla.Vector3D(corners[i].x, corners[i].y, height)
            end = carla.Vector3D(corners[i + 1].x, corners[i + 1].y, height)
            self.world.debug.draw_line(start, end, life_time=0.1)

        start = carla.Vector3D(corners[len(corners) - 1].x, corners[len(corners) - 1].y, height)
        end = carla.Vector3D(corners[0].x, corners[0].y, height)
        self.world.debug.draw_line(start, end, life_time=0.1)

    def get_lane_constraints_by_vehicle(self, position, forward_vec):
        sidewalk_vec = forward_vec.rotate(np.deg2rad(90))  # the local y direction. (rotating clockwise by 90 deg)

        lookahead_x = 20
        lookahead_y = 6

        right_region_corners = [position, position + lookahead_y * sidewalk_vec,
                                position + lookahead_y * sidewalk_vec + lookahead_x * forward_vec,
                                position + lookahead_x * forward_vec]

        # self.draw_box(right_region_corners)

        left_region_corners = [position, position + lookahead_x * forward_vec,
                               position - lookahead_y * sidewalk_vec + lookahead_x * forward_vec,
                               position - lookahead_y * sidewalk_vec]

        # self.draw_box(left_region_corners)

        left_lane_constrained_by_vehicle = False
        right_lane_constrained_by_vehicle = False

        for crowd_agent in self.network_bike_agents:  # + self.sidewalk_agents:
            pos_agt = crowd_agent.get_position()
            if (not left_lane_constrained_by_vehicle) and self.in_polygon(pos_agt,
                                                                          left_region_corners):  # if it is already
                # constrained, then no need to check other agents
                if dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    left_lane_constrained_by_vehicle = True
            if (not right_lane_constrained_by_vehicle) and self.in_polygon(pos_agt,
                                                                           right_region_corners):  # if it is already
                # constrained, then no need to check other agents
                if dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    right_lane_constrained_by_vehicle = True

        front_car_count = 0
        for crowd_agent in self.network_car_agents:  # + self.sidewalk_agents:
            pos_agt = crowd_agent.get_position()
            if self.in_polygon(pos_agt,
                               left_region_corners):  # if it is already constrained, then no need to check other agents
                front_car_count += 1
                if (not left_lane_constrained_by_vehicle) and dot_product(forward_vec,
                                                                          crowd_agent.get_forward_direction()) < 0:
                    left_lane_constrained_by_vehicle = True
            if (not right_lane_constrained_by_vehicle) and self.in_polygon(pos_agt,
                                                                           right_region_corners):  # if it is already
                # constrained, then no need to check other agents
                if dot_product(forward_vec, crowd_agent.get_forward_direction()) < 0:
                    right_lane_constrained_by_vehicle = True

        if front_car_count > 1:
            right_lane_constrained_by_vehicle = True

        return left_lane_constrained_by_vehicle, right_lane_constrained_by_vehicle
        # return False, False

    def get_lane_constraints(self, position, forward_vec):
        left_line_end = position + (1.5 + 2.0 + 0.8) * ((forward_vec.rotate(np.deg2rad(-90))).make_unit_vector())
        right_line_end = position + (1.5 + 2.0 + 0.5) * ((forward_vec.rotate(np.deg2rad(90))).make_unit_vector())
        left_lane_constrained_by_sidewalk = self.sidewalk.intersects(position, left_line_end)
        right_lane_constrained_by_sidewalk = self.sidewalk.intersects(position, right_line_end)
        return left_lane_constrained_by_sidewalk, right_lane_constrained_by_sidewalk

    def no_collision(self):
        for (i, crowd_agent) in enumerate(self.network_car_agents + self.network_bike_agents + self.sidewalk_agents):
            crowd_agent.disable_collision()

    def find_ego_actor(self):
        if self.ego_actor is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.ego_actor = actor
                    break

    def update_agent_controls(self, event):
        self.network_car_agents_lock.acquire()
        self.network_bike_agents_lock.acquire()
        self.sidewalk_agents_lock.acquire()

        commands = []
        for crowd_agent in self.network_car_agents + self.network_bike_agents + self.sidewalk_agents:
            vel_to_exe = crowd_agent.control_velocity
            cur_vel = crowd_agent.actor.get_velocity()
            cur_vel = carla.Vector2D(cur_vel.x, cur_vel.y)

            angle_diff = get_signed_angle_diff(vel_to_exe, cur_vel)
            if angle_diff > 30 or angle_diff < -30:
                vel_to_exe = 0.5 * (vel_to_exe + cur_vel)

            if type(crowd_agent) is CrowdNetworkCarAgent:
                control = crowd_agent.get_control(vel_to_exe, k1=1.0, k2=2.0, k3=7)
                commands.append(carla.command.ApplyVehicleControl(crowd_agent.actor.id, control))
            elif type(crowd_agent) is CrowdNetworkBikeAgent:
                control = crowd_agent.get_control(vel_to_exe, k1=1.0, k2=2.0, k3=5)
                commands.append(carla.command.ApplyVehicleControl(crowd_agent.actor.id, control))
            elif type(crowd_agent) is CrowdSidewalkAgent:
                control = crowd_agent.get_control(vel_to_exe)
                commands.append(carla.command.ApplyWalkerControl(crowd_agent.actor.id, control))

        self.client.apply_batch_sync(commands)
        
        self.network_car_agents_lock.release()
        self.network_bike_agents_lock.release()
        self.sidewalk_agents_lock.release()

    def dist_to_nearest_agt_in_region(self, position, forward_vec, sidewalk_vec, lookahead_x=30, lookahead_y=4,
                                      ref_point=None, consider_ped=False):

        if ref_point is None:
            ref_point = position

        region_corners = [ref_point - (lookahead_y / 2.0) * sidewalk_vec,
                          ref_point + (lookahead_y / 2.0) * sidewalk_vec,
                          ref_point + (lookahead_y / 2.0) * sidewalk_vec + lookahead_x * forward_vec,
                          ref_point - (lookahead_y / 2.0) * sidewalk_vec + lookahead_x * forward_vec]

        min_dist = lookahead_x
        for crowd_agent in self.network_bike_agents:
            pos_agt = crowd_agent.get_position()
            if self.in_polygon(pos_agt, region_corners):
                min_dist = min(min_dist, (pos_agt - position).length())

        for crowd_agent in self.network_car_agents:
            pos_agt = crowd_agent.get_position()
            if self.in_polygon(pos_agt, region_corners):
                min_dist = min(min_dist, (pos_agt - position).length())

        if consider_ped:
            for crowd_agent in self.sidewalk_agents:
                pos_agt = crowd_agent.get_position()
                if self.in_polygon(pos_agt, region_corners):
                    min_dist = min(min_dist, (pos_agt - position).length())

        return min_dist

    def gamma_lane_change_decision(self):
        if not self.ego_car_info or not self.ego_actor:
            return

        forward_vec = carla.Vector2D(math.cos(self.ego_car_info.car_yaw), math.sin(self.ego_car_info.car_yaw))
        sidewalk_vec = forward_vec.rotate(np.deg2rad(90))  # rotate clockwise by 90 degree

        ego_veh_pos = carla.Vector2D(
            self.ego_car_info.car_pos.x,
            self.ego_car_info.car_pos.y)

        left_ego_veh_pos = ego_veh_pos - 4.0 * sidewalk_vec
        right_ego_veh_pos = ego_veh_pos + 4.0 * sidewalk_vec

        cur_route_point = self.network.get_nearest_route_point(ego_veh_pos)
        left_route_point = self.network.get_nearest_route_point(left_ego_veh_pos)
        right_route_point = self.network.get_nearest_route_point(right_ego_veh_pos)

        left_lane_exist = False
        right_lane_exist = False

        if left_route_point.edge == cur_route_point.edge and left_route_point.lane != cur_route_point.lane:
            left_lane_exist = True
        else:
            left_lane_exist = False
        if right_route_point.edge == cur_route_point.edge and right_route_point.lane != cur_route_point.lane:
            right_lane_exist = True
        else:
            right_lane_exist = False

        min_dist_to_front_veh = self.dist_to_nearest_agt_in_region(ego_veh_pos, forward_vec, sidewalk_vec,
                                                                   lookahead_x=30, lookahead_y=4, ref_point=None)
        min_dist_to_left_front_veh = -1.0
        min_dist_to_right_front_veh = -1.0

        if left_lane_exist:
            # if want to change lane, also need to consider vehicles behind
            min_dist_to_left_front_veh = self.dist_to_nearest_agt_in_region(left_ego_veh_pos, forward_vec, sidewalk_vec,
                                                                            lookahead_x=35, lookahead_y=4,
                                                                            ref_point=left_ego_veh_pos - 12.0 * forward_vec,
                                                                            consider_ped=True)
        if right_lane_exist:
            # if want to change lane, also need to consider vehicles behind
            min_dist_to_right_front_veh = self.dist_to_nearest_agt_in_region(right_ego_veh_pos, forward_vec,
                                                                             sidewalk_vec, lookahead_x=35,
                                                                             lookahead_y=4,
                                                                             ref_point=right_ego_veh_pos - 12.0 * forward_vec,
                                                                             consider_ped=True)

        change_left = -1
        remain = 0
        change_right = 1

        if min_dist_to_front_veh >= 15:
            self.gamma_lane_change_pub.publish(remain)
        else:
            if min_dist_to_left_front_veh > min_dist_to_right_front_veh:
                if min_dist_to_left_front_veh > min_dist_to_front_veh + 3.0:
                    self.gamma_lane_change_pub.publish(change_left)
                else:
                    self.gamma_lane_change_pub.publish(remain)
            else:  # min_dist_to_left_front_veh <= min_dist_to_right_front_veh:
                if min_dist_to_right_front_veh > min_dist_to_front_veh + 3.0:
                    self.gamma_lane_change_pub.publish(change_right)
                else:
                    self.gamma_lane_change_pub.publish(remain)

            # if rng.uniform(0.0, 1.0) <= lane_change_probability:
            #     new_path_candidates = self.summit.network.get_next_route_paths(nearest_route_point, self.path.min_points - 1, self.path.interval)
            #     new_path = NetworkAgentPath(self.summit, self.path.min_points, self.path.interval)
            #     new_path.route_points = rng.choice(new_path_candidates)[0:self.path.min_points]
            #     self.path = new_path

    def make_lane_decision(self, event):
        self.gamma_lane_change_decision()

    def rand_agent_behavior_type(self):
        prob_gamma_agent = 0.8
        prob_simplified_gamma_agent = 0.1
        prob_ttc_agent = 0.1

        prob = self.rng.uniform(0.0, 1.0)

        if prob <= prob_gamma_agent:
            return carla.AgentBehaviorType.Gamma
        elif prob <= prob_gamma_agent + prob_simplified_gamma_agent:
            return carla.AgentBehaviorType.SimplifiedGamma
        else:
            return -1

    def update_spawn(self, event):
        # Determine bounds variables.
        bounds_center = carla.Vector2D(self.ego_actor.get_location().x, self.ego_actor.get_location().y)
        bounds_min = bounds_center + carla.Vector2D(-self.crowd_range, -self.crowd_range)
        bounds_max = bounds_center + carla.Vector2D(self.crowd_range, self.crowd_range)

        # Get segment map within ego range.
        spawn_segment_map = self.network_segment_map.intersection(
            get_spawn_occupancy_map(bounds_center, 0, self.crowd_range))
        spawn_segment_map.seed_rand(self.rng.getrandbits(32))

        # Get AABB.
        exo_aabb_map = carla.AABBMap([self.get_aabb(self.ego_actor)])
        aabb_map = carla.AABBMap(
            [self.get_aabb(agent.actor) for agent in self.network_bike_agents + self.network_car_agents + self.sidewalk_agents] +
            [self.get_aabb(self.ego_actor)])

        # Spawn at most one car.
        self.network_car_agents_lock.acquire()
        do_spawn = len(self.network_car_agents) < self.num_network_car_agents
        self.network_car_agents_lock.release()

        if do_spawn:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map, rng=self.rng)
            agent_behavior_type = self.rand_agent_behavior_type()
            if not aabb_map.intersects(carla.AABB2D(
                    carla.Vector2D(path.get_position(0).x - self.spawn_clearance_vehicle,
                                   path.get_position(0).y - self.spawn_clearance_vehicle),
                    carla.Vector2D(path.get_position(0).x + self.spawn_clearance_vehicle,
                                   path.get_position(0).y +
                                   self.spawn_clearance_vehicle))) and \
                    not exo_aabb_map.intersects(carla.AABB2D(
                        carla.Vector2D(path.get_position(0).x - self.spawn_clearance_ego,
                                       path.get_position(0).y - self.spawn_clearance_ego),
                        carla.Vector2D(path.get_position(0).x + self.spawn_clearance_ego,
                                       path.get_position(0).y + self.spawn_clearance_ego))):
                trans = carla.Transform()
                trans.location.x = path.get_position(0).x
                trans.location.y = path.get_position(0).y
                trans.location.z = 0.2
                trans.rotation.yaw = path.get_yaw(0)
                actor = self.world.try_spawn_actor(self.rng.choice(self.cars_blueprints), trans)
                if actor:
                    actor.set_collision_enabled(False)
                    self.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                    aabb_map.insert(self.get_aabb(actor))
                    self.network_car_agents_lock.acquire()
                    self.network_car_agents.append(CrowdNetworkCarAgent(
                        self, actor, path,
                        5.0 + self.rng.uniform(0.0, 0.5),
                        agent_behavior_type))
                    self.network_car_agents_lock.release()
        if len(self.network_car_agents) > self.num_network_car_agents / 4:
            self.do_publish = True     

        # Spawn at most one bike.
        self.network_bike_agents_lock.acquire()
        do_spawn = len(self.network_bike_agents) < self.num_network_bike_agents
        self.network_bike_agents_lock.release()

        if do_spawn:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map, rng=self.rng)
            agent_behavior_type = self.rand_agent_behavior_type()
            if aabb_map.intersects(carla.AABB2D(
                    carla.Vector2D(path.get_position(0).x - self.spawn_clearance_vehicle,
                                   path.get_position(0).y - self.spawn_clearance_vehicle),
                    carla.Vector2D(path.get_position(0).x + self.spawn_clearance_vehicle,
                                   path.get_position(0).y + self.spawn_clearance_vehicle))) and \
                    not exo_aabb_map.intersects(carla.AABB2D(
                        carla.Vector2D(path.get_position(0).x - self.spawn_clearance_ego,
                                       path.get_position(0).y - self.spawn_clearance_ego),
                        carla.Vector2D(path.get_position(0).x + self.spawn_clearance_ego,
                                       path.get_position(0).y + self.spawn_clearance_ego))):
                trans = carla.Transform()
                trans.location.x = path.get_position(0).x
                trans.location.y = path.get_position(0).y
                trans.location.z = 0.2
                trans.rotation.yaw = path.get_yaw(0)
                actor = self.world.try_spawn_actor(self.rng.choice(self.bikes_blueprints), trans)
                if actor:
                    actor.set_collision_enabled(False)
                    self.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                    aabb_map.insert(self.get_aabb(actor))
                    self.network_bike_agents_lock.acquire()
                    self.network_bike_agents.append(CrowdNetworkBikeAgent(
                        self, actor, path,
                        3.0 + self.rng.uniform(0, 0.5),
                        agent_behavior_type))
                    self.network_bike_agents_lock.release()

        # Spawn at most one pedestrian.
        self.sidewalk_agents_lock.acquire()
        do_spawn = len(self.sidewalk_agents) < self.num_sidewalk_agents
        self.sidewalk_agents_lock.release()

        if do_spawn:
            path = SidewalkAgentPath.rand_path(self, self.path_min_points, self.path_interval, bounds_min, bounds_max, self.rng)
            agent_behavior_type = self.rand_agent_behavior_type()
            if aabb_map.intersects(carla.AABB2D(
                    carla.Vector2D(path.get_position(0).x - self.spawn_clearance_person,
                                   path.get_position(0).y - self.spawn_clearance_person),
                    carla.Vector2D(path.get_position(0).x + self.spawn_clearance_person,
                                   path.get_position(0).y + self.spawn_clearance_person))) and \
                    not exo_aabb_map.intersects(carla.AABB2D(
                        carla.Vector2D(path.get_position(0).x - self.spawn_clearance_ego,
                                       path.get_position(0).y - self.spawn_clearance_ego),
                        carla.Vector2D(path.get_position(0).x + self.spawn_clearance_ego,
                                       path.get_position(0).y + self.spawn_clearance_ego))):
                trans = carla.Transform()
                trans.location.x = path.get_position(0).x
                trans.location.y = path.get_position(0).y
                trans.location.z = 0.2
                trans.rotation.yaw = path.get_yaw(0)
                actor = self.world.try_spawn_actor(self.rng.choice(self.walker_blueprints), trans)
                if actor:
                    actor.set_collision_enabled(False)
                    self.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                    aabb_map.insert(self.get_aabb(actor))
                    self.sidewalk_agents_lock.acquire()
                    self.sidewalk_agents.append(CrowdSidewalkAgent(
                        self, actor, path,
                        0.5 + self.rng.uniform(0.0, 1.0),
                        agent_behavior_type))
                    self.sidewalk_agents_lock.release()

        if not self.initialized and rospy.Time.now() - self.start_time > rospy.Duration.from_sec(5.0):
            self.agents_ready_pub.publish(True)
            self.initialized = True
        if rospy.Time.now() - self.start_time > rospy.Duration.from_sec(10.0):
            self.do_publish = True     

    def update_destroy(self, event):
        update_time = event.current_real

        # Determine bounds variables.
        bounds_center = carla.Vector2D(self.ego_actor.get_location().x, self.ego_actor.get_location().y)
        bounds_min = bounds_center + carla.Vector2D(-self.crowd_range, -self.crowd_range)
        bounds_max = bounds_center + carla.Vector2D(self.crowd_range, self.crowd_range)

        # Delete invalid cars.
        self.network_car_agents_lock.acquire()
        commands = []
        next_agents = []

        for agent in self.network_car_agents:
            delete = False
            if not delete and not check_bounds(agent.get_position(), bounds_min, bounds_max):
                delete = True
            if not delete and agent.get_position_3d().z < -10:
                delete = True
            if not delete and not self.network_occupancy_map.contains(agent.get_position()):
                delete = True
            if agent.get_preferred_velocity(self.lane_change_probability, self.rng) is None:
                delete = True
            if agent.get_velocity().length() < 0.2:
                if agent.stuck_time is not None:
                    if (update_time - agent.stuck_time).to_sec() >= 5.0:
                        delete = True
                else:
                    agent.stuck_time = update_time
            else:
                agent.stuck_time = None
            
            if delete:
                commands.append(carla.command.DestroyActor(agent.actor.id))
            else:
                next_agents.append(agent)

        self.client.apply_batch_sync(commands)
        self.network_car_agents = next_agents
        self.network_car_agents_lock.release()
        
        # Delete invalid bikes.
        self.network_bike_agents_lock.acquire()
        commands = []
        next_agents = []

        for agent in self.network_bike_agents:
            delete = False
            if not delete and not check_bounds(agent.get_position(), bounds_min, bounds_max):
                delete = True
            if not delete and agent.get_position_3d().z < -10:
                delete = True
            if not delete and not self.network_occupancy_map.contains(agent.get_position()):
                delete = True
            if agent.get_preferred_velocity(self.lane_change_probability, self.rng) is None:
                delete = True
            if agent.get_velocity().length() < 0.2:
                if agent.stuck_time is not None:
                    if (update_time - agent.stuck_time).to_sec() >= 5.0:
                        delete = True
                else:
                    agent.stuck_time = update_time
            else:
                agent.stuck_time = None
            
            if delete:
                commands.append(carla.command.DestroyActor(agent.actor.id))
            else:
                next_agents.append(agent)

        self.client.apply_batch_sync(commands)
        self.network_bike_agents = next_agents
        self.network_bike_agents_lock.release()
    
        # Delete invalid pedestrians.
        self.sidewalk_agents_lock.acquire()
        commands = []
        next_agents = []

        for agent in self.sidewalk_agents:
            delete = False
            if not delete and not check_bounds(agent.get_position(), bounds_min, bounds_max):
                delete = True
            if not delete and agent.get_position_3d().z < -10:
                delete = True
            if agent.get_preferred_velocity(self.lane_change_probability, self.rng) is None:
                delete = True
            if agent.get_velocity().length() < 0.2:
                if agent.stuck_time is not None:
                    if (update_time - agent.stuck_time).to_sec() >= 5.0:
                        delete = True
                else:
                    agent.stuck_time = update_time
            else:
                agent.stuck_time = None
            
            if delete:
                commands.append(carla.command.DestroyActor(agent.actor.id))
            else:
                next_agents.append(agent)

        self.client.apply_batch_sync(commands)
        self.sidewalk_agents = next_agents
        self.sidewalk_agents_lock.release()

    def update_gamma(self, event):
        update_time = event.current_real

        self.network_car_agents_lock.acquire()
        self.network_bike_agents_lock.acquire()
        self.sidewalk_agents_lock.acquire()

        agents = self.network_car_agents + \
                [None for _ in range(len(self.network_car_agents), self.num_network_car_agents)] + \
                self.network_bike_agents + \
                [None for _ in range(len(self.network_bike_agents), self.num_network_bike_agents)] + \
                self.sidewalk_agents + \
                [None for _ in range(len(self.sidewalk_agents), self.num_sidewalk_agents)]

        for (i, crowd_agent) in enumerate(agents):
            if crowd_agent is None:
                self.gamma.set_agent_position(i, default_agent_pos)
                self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)
            else:
                self.gamma.set_agent(i, crowd_agent.get_agent_params())
                pref_vel = crowd_agent.get_preferred_velocity(self.lane_change_probability, self.rng)
                if pref_vel:
                    self.gamma.set_agent_position(i, crowd_agent.get_position())
                    self.gamma.set_agent_velocity(i, crowd_agent.get_velocity())
                    self.gamma.set_agent_heading(i, crowd_agent.get_forward_direction())
                    self.gamma.set_agent_bounding_box_corners(i, crowd_agent.get_bounding_box_corners())
                    self.gamma.set_agent_pref_velocity(i, pref_vel)
                    self.gamma.set_agent_path_forward(i, crowd_agent.get_path_forward())
                    if crowd_agent.behavior_type is not -1:
                        self.gamma.set_agent_behavior_type(i, crowd_agent.behavior_type)
                    left_lane_constrained, right_lane_constrained = self.get_lane_constraints(crowd_agent.get_position(),
                                                                                              crowd_agent.get_path_forward())
                    self.gamma.set_agent_lane_constraints(i, right_lane_constrained,
                                                          left_lane_constrained)  # # to check. It seems that we should
                else:
                    agents[i] = None # Ignore for rest of GAMMA loop.
                    self.gamma.set_agent_position(i, default_agent_pos)
                    self.gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                    self.gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                    self.gamma.set_agent_bounding_box_corners(i, default_agent_bbox)

        ego_gamma_index = self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents
        if self.ego_car_info and self.ego_actor:
            self.gamma.set_agent_position(ego_gamma_index, carla.Vector2D(
                self.ego_car_info.car_pos.x,
                self.ego_car_info.car_pos.y))
            self.gamma.set_agent_velocity(ego_gamma_index, carla.Vector2D(
                self.ego_car_info.car_vel.x,
                self.ego_car_info.car_vel.y))
            self.gamma.set_agent_heading(ego_gamma_index, carla.Vector2D(
                math.cos(self.ego_car_info.car_yaw),
                math.sin(self.ego_car_info.car_yaw)))
            self.gamma.set_agent_bounding_box_corners(ego_gamma_index, self.get_bounding_box_corners(self.ego_actor))
            self.gamma.set_agent_pref_velocity(ego_gamma_index, carla.Vector2D(
                self.ego_car_info.car_pref_vel.x,
                self.ego_car_info.car_pref_vel.y))
        else:
            self.gamma.set_agent_position(ego_gamma_index, default_agent_pos)
            self.gamma.set_agent_pref_velocity(ego_gamma_index, carla.Vector2D(0, 0))
            self.gamma.set_agent_velocity(ego_gamma_index, carla.Vector2D(0, 0))
            self.gamma.set_agent_bounding_box_corners(ego_gamma_index, default_agent_bbox)

        try:
            self.gamma.do_step()
        except Exception as e:
            print(e)

        for (i, crowd_agent) in enumerate(agents):
            if crowd_agent:
                if crowd_agent.behavior_type is not -1:
                    crowd_agent.control_velocity = self.gamma.get_agent_velocity(i)
                else:
                    crowd_agent.control_velocity = self.get_ttc_vel(i, crowd_agent, agents)
                    if crowd_agent.control_velocity is None:
                        crowd_agent.control_velocity = self.gamma.get_agent_velocity(i)

        self.network_car_agents_lock.release()
        self.network_bike_agents_lock.release()
        self.sidewalk_agents_lock.release()

        # Publish ego vehicle velocity.
        if self.ego_car_info is not None:
            # pass
            ego_gamma_vel = self.gamma.get_agent_velocity(
                self.num_network_car_agents + self.num_network_bike_agents + self.num_sidewalk_agents)
            ego_cur_vel = carla.Vector2D(self.ego_car_info.car_vel.x, self.ego_car_info.car_vel.y)
            ego_angle_diff = get_signed_angle_diff(ego_gamma_vel, ego_cur_vel)
            if ego_angle_diff > 30 or ego_angle_diff < -30:
                ego_gamma_vel = 0.5 * (ego_gamma_vel + ego_cur_vel)
            ego_control = CrowdNetworkAgent(self, self.ego_actor, None, 5.25).get_control(ego_gamma_vel, k1=1.0, k2=2.0,
                                                                                          k3=2.0)  # TODO: Is there a better way using static methods?
            self.gamma_cmd_accel_pub.publish(ego_control.throttle if ego_control.throttle > 0 else -ego_control.brake)
            self.gamma_cmd_speed_pub.publish(ego_gamma_vel.length())
            self.gamma_cmd_steer_pub.publish(ego_control.steer)

    def get_ttc_vel(self, i, crowd_agent, agents):
        if crowd_agent:

            vel_to_exe = crowd_agent.get_preferred_velocity()
            if not vel_to_exe: # path is not ready.
                return None

            speed_to_exe = crowd_agent.preferred_speed
            for (j, other_crowd_agent) in enumerate(agents):
                if i != j and other_crowd_agent and self.network_occupancy_map.contains(other_crowd_agent.get_position()):
                    s_f = other_crowd_agent.get_velocity().length()
                    d_f = (other_crowd_agent.get_position() - crowd_agent.get_position()).length()
                    d_safe = 5.0
                    a_max = 3.0
                    s = max(0, s_f * s_f + 2 * a_max * (d_f - d_safe))**0.5
                    speed_to_exe = min(speed_to_exe, s)

            cur_vel = crowd_agent.actor.get_velocity()
            cur_vel = carla.Vector2D(cur_vel.x, cur_vel.y)
            angle_diff = get_signed_angle_diff(vel_to_exe, cur_vel)
            if angle_diff > 30 or angle_diff < -30:
                vel_to_exe = 0.5 * (vel_to_exe + cur_vel)

            vel_to_exe = vel_to_exe.make_unit_vector() * speed_to_exe

            return vel_to_exe

        return None

    def publish_agents(self, tick):
        if self.do_publish is False:
            return

        network_agents_msg = msg_builder.msg.CrowdNetworkAgentArray()
        network_agents_msg.header.stamp = rospy.Time.now()
        for a in self.network_car_agents + self.network_bike_agents:
            network_agent_msg = msg_builder.msg.CrowdNetworkAgent()
            network_agent_msg.id = a.get_id()
            network_agent_msg.type = a.get_type()
            network_agent_msg.route_point.edge = a.path.route_points[0].edge
            network_agent_msg.route_point.lane = a.path.route_points[0].lane
            network_agent_msg.route_point.segment = a.path.route_points[0].segment
            network_agent_msg.route_point.offset = a.path.route_points[0].offset
            network_agents_msg.agents.append(network_agent_msg)
        self.network_agents_pub.publish(network_agents_msg)

        sidewalk_agents_msg = msg_builder.msg.CrowdSidewalkAgentArray()
        sidewalk_agents_msg.header.stamp = rospy.Time.now()
        for a in self.sidewalk_agents:
            sidewalk_agent_msg = msg_builder.msg.CrowdSidewalkAgent()
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
    rospy.wait_for_message("/meshes_spawned", Bool)
    rospy.wait_for_message("/ego_state", CarInfo)

    gamma_crowd_controller = GammaCrowdController()
    rospy.on_shutdown(gamma_crowd_controller.dispose)
    rospy.spin()
