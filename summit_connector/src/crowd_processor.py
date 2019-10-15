#!/usr/bin/env python2

from util import *
import carla
import sys

import numpy as np
from collections import defaultdict

import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose, Quaternion, Vector3, Polygon, \
    Point32, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Bool

import msg_builder.msg
from msg_builder.msg import car_info as CarInfo  # panpan


# TODO Speed up spawning logic to prevent hangs in ROS.
class CrowdProcessor(Drunc):
    def __init__(self):
        super(CrowdProcessor, self).__init__()

        self.network_agents = []
        self.sidewalk_agents = []
        self.do_update = False
        self.ego_car_info = None
        self.topological_hash_map = defaultdict(lambda: None)  # TODO Expiry to prevent memory explosion.

        self.network_agents_sub = rospy.Subscriber(
            '/crowd/network_agents',
            msg_builder.msg.CrowdNetworkAgentArray,
            self.network_agents_callback,
            queue_size=1)
        self.sidewalk_agents_sub = rospy.Subscriber(
            '/crowd/sidewalk_agents',
            msg_builder.msg.CrowdSidewalkAgentArray,
            self.sidewalk_agents_callback,
            queue_size=1)
        self.il_car_info_sub = rospy.Subscriber(
            '/ego_state',
            CarInfo,
            self.il_car_info_callback,
            queue_size=1)
        self.agents_pub = rospy.Publisher(
            '/agent_array',
            msg_builder.msg.TrafficAgentArray,
            queue_size=1)
        self.agents_path_pub = rospy.Publisher(
            '/agent_path_array',
            msg_builder.msg.AgentPathArray,
            queue_size=1)
        self.obstacles_pub = rospy.Publisher(
            '/local_obstacles',
            msg_builder.msg.Obstacles,
            queue_size=1)
        self.lanes_pub = rospy.Publisher(
            '/local_lanes',
            msg_builder.msg.Lanes,
            queue_size=1)

    def network_agents_callback(self, agents):
        self.network_agents = agents.agents
        self.do_update = True

    def sidewalk_agents_callback(self, agents):
        self.sidewalk_agents = agents.agents
        self.do_update = True

    def il_car_info_callback(self, car_info):
        self.ego_car_info = car_info

    def draw_path(self, path_msg):
        color_i = 255
        last_loc = None
        for pos_msg in path_msg.poses:
            pos = pos_msg.pose.position
            loc = carla.Location(pos.x, pos.y, 0.1)
            if last_loc is not None:
                self.world.debug.draw_line(last_loc, loc, life_time=0.1,
                                           color=carla.Color(color_i, 0, color_i, 0))
            last_loc = carla.Location(pos.x, pos.y, 0.1)

    def update(self):
        end_time = rospy.Time.now()
        elapsed = (end_time - init_time).to_sec()
        if not self.do_update:
            # print('do_update disabled at {}'.format(elapsed))
            # sys.stdout.flush()
            return

        if not self.ego_car_info:
            print('ego_car not exist yet at {}'.format(elapsed))
            sys.stdout.flush()
            return

        ego_car_position = self.ego_car_info.car_pos
        ego_car_position = carla.Vector2D(
            ego_car_position.x,
            ego_car_position.y)

        agents_msg = msg_builder.msg.TrafficAgentArray()
        agents_path_msg = msg_builder.msg.AgentPathArray()

        current_time = rospy.Time.now()

        # get local occupancy map
        publish_map_rad = 50
        OM_bound = carla.OccupancyMap(
            carla.Vector2D(ego_car_position.x - publish_map_rad, ego_car_position.y - publish_map_rad),
            carla.Vector2D(ego_car_position.x + publish_map_rad, ego_car_position.y + publish_map_rad))

        local_obstacles = OM_bound.difference(self.network_occupancy_map)
        obstacle_msg = msg_builder.msg.Obstacles
        obstacle_msg.contours = []

        for polygon in local_obstacles.get_polygons():
            polygon_tmp = Polygon()
            outer_contour = polygon[0]
            for point in outer_contour:
                polygon_tmp.points.append(Point32(point.x, point.y, 0.0))
            obstacle_msg.contours.append(polygon_tmp)

        self.obstacles_pub.publish(obstacle_msg)

        # get local lane centers
        local_lanes = self.network_segment_map.intersection(OM_bound)
        lane_msg = msg_builder.msg.Lanes
        lane_msg.lane_segments = []

        for lane_seg in local_lanes.get_segments():
            lane_seg_tmp = msg_builder.msg.LaneSeg()
            lane_seg_tmp.start.x = lane_seg.start.x
            lane_seg_tmp.start.y = lane_seg.start.y
            lane_seg_tmp.end.x = lane_seg.end.x
            lane_seg_tmp.end.y = lane_seg.end.y

            lane_msg.lane_segments.append(lane_seg_tmp)

        self.lanes_pub.publish(lane_msg)

        agents = self.network_agents + self.sidewalk_agents
        for agent in agents:
            actor = self.world.get_actor(agent.id)

            if actor is None:
                continue

            actor_pos = carla.Vector2D(actor.get_location().x, actor.get_location().y)

            if (actor_pos - ego_car_position).length() > 50:  # TODO Add as ROS parameter.
                continue

            agent_tmp = msg_builder.msg.TrafficAgent()
            agent_tmp.last_update = current_time
            agent_tmp.id = actor.id
            agent_tmp.type = agent.type
            agent_tmp.pose.position.x = actor.get_location().x
            agent_tmp.pose.position.y = actor.get_location().y
            agent_tmp.pose.position.z = actor.get_location().z
            quat_tf = tf.transformations.quaternion_from_euler(
                0, 0,
                np.deg2rad(actor.get_transform().rotation.yaw))
            agent_tmp.pose.orientation = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])

            agent_tmp.bbox = Polygon()
            corners = get_bounding_box_corners(actor)
            for corner in corners:
                agent_tmp.bbox.points.append(Point32(
                    x=corner.x, y=corner.y, z=0.0))

            agents_msg.agents.append(agent_tmp)

            agent_paths_tmp = msg_builder.msg.AgentPaths()
            agent_paths_tmp.id = actor.id
            agent_paths_tmp.type = agent.type

            if type(agent) is msg_builder.msg.CrowdNetworkAgent:
                initial_route_point = carla.SumoNetworkRoutePoint()
                initial_route_point.edge = agent.route_point.edge
                initial_route_point.lane = agent.route_point.lane
                initial_route_point.segment = agent.route_point.segment
                initial_route_point.offset = agent.route_point.offset

                paths = [[initial_route_point]]
                topological_hash = ''

                # TODO Add 20, 1.0 as ROS paramters.
                for _ in range(20):
                    next_paths = []
                    for path in paths:
                        next_route_points = self.network.get_next_route_points(path[-1], 1.0)
                        next_paths.extend(path + [route_point] for route_point in next_route_points)
                        if len(next_route_points) > 1:
                            topological_hash += '({},{},{},{}={})'.format(
                                path[-1].edge, path[-1].lane, path[-1].segment, path[-1].offset,
                                len(next_route_points))
                    paths = next_paths

                agent_paths_tmp.reset_intention = self.topological_hash_map[agent.id] is None or \
                                                  self.topological_hash_map[agent.id] != topological_hash
                for path in paths:
                    path_msg = Path()
                    path_msg.header.frame_id = 'map'
                    path_msg.header.stamp = current_time
                    for path_point in [actor_pos] + [self.network.get_route_point_position(route_point) for route_point
                                                     in path]:
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.header.stamp = current_time
                        pose_msg.pose.position.x = path_point.x
                        pose_msg.pose.position.y = path_point.y
                        path_msg.poses.append(pose_msg)
                    # self.draw_path(path_msg)
                    agent_paths_tmp.path_candidates.append(path_msg)
                agent_paths_tmp.cross_dirs = []
                self.topological_hash_map[agent.id] = topological_hash

            elif type(agent) is msg_builder.msg.CrowdSidewalkAgent:
                sidewalk_route_point = carla.SidewalkRoutePoint()
                sidewalk_route_point.polygon_id = agent.route_point.polygon_id
                sidewalk_route_point.segment_id = agent.route_point.segment_id
                sidewalk_route_point.offset = agent.route_point.offset
                sidewalk_route_orientation = agent.route_orientation

                path = [sidewalk_route_point]
                for _ in range(20):
                    if sidewalk_route_orientation:
                        path.append(self.sidewalk.get_next_route_point(path[-1], 1.0))  # TODO Add as ROS parameter.
                    else:
                        path.append(self.sidewalk.get_previous_route_point(path[-1], 1.0))  # TODO Add as ROS parameter.
                topological_hash = '{},{}'.format(sidewalk_route_point.polygon_id, agent.route_orientation)

                agent_paths_tmp.reset_intention = self.topological_hash_map[agent.id] is None or \
                                                  self.topological_hash_map[agent.id] != topological_hash
                path_msg = Path()
                path_msg.header.frame_id = 'map'
                path_msg.header.stamp = current_time
                for path_point in [actor_pos] + [self.sidewalk.get_route_point_position(route_point) for route_point in
                                                 path]:
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = 'map'
                    pose_msg.header.stamp = current_time
                    pose_msg.pose.position.x = path_point.x
                    pose_msg.pose.position.y = path_point.y
                    path_msg.poses.append(pose_msg)
                # self.draw_path(path_msg)
                agent_paths_tmp.path_candidates = [path_msg]
                agent_paths_tmp.cross_dirs = [agent.route_orientation]
                self.topological_hash_map[agent.id] = topological_hash

            agents_path_msg.agents.append(agent_paths_tmp)

        agents_msg.header.frame_id = 'map'
        agents_msg.header.stamp = current_time
        self.agents_pub.publish(agents_msg)

        agents_path_msg.header.frame_id = 'map'
        agents_path_msg.header.stamp = current_time
        self.agents_path_pub.publish(agents_path_msg)

        self.do_update = False
        end_time = rospy.Time.now()
        elapsed = (end_time - init_time).to_sec()
        # print('agent_array update = {} ms = {} hz'.format(duration * 1000, 1.0 / duration))
        # print('agent_array update at {}'.format(elapsed))
        # sys.stdout.flush()


if __name__ == '__main__':
    rospy.init_node('crowd_processor')
    init_time = rospy.Time.now()
    rospy.wait_for_message("/meshes_spawned", Bool)
    crowd_processor = CrowdProcessor()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        crowd_processor.update()
        rate.sleep()
