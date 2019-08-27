#!/usr/bin/env python2

import numpy as np
from collections import defaultdict

import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose, Quaternion, Vector3, Polygon, Point32
from nav_msgs.msg import Path

from drunc import Drunc
import carla
import carla_connector2.msg
from peds_unity_system.msg import car_info as CarInfo # panpan
from util import *

'''    
def get_path_candidates(self, num_points, interval):
'''

# TODO Speed up spawning logic to prevent hangs in ROS.
class CrowdProcessor(Drunc):
    def __init__(self):
        super(CrowdProcessor, self).__init__()

        self.network_agents = []
        self.sidewalk_agents = []
        self.do_update = False
        self.ego_car_info = None
        self.topological_hash_map = defaultdict(lambda: None) # TODO Expiry to prevent memory explosion.

        self.network_agents_sub = rospy.Subscriber(
                '/crowd/network_agents', 
                carla_connector2.msg.CrowdNetworkAgentArray,
                self.network_agents_callback,
                queue_size=1)
        self.sidewalk_agents_sub = rospy.Subscriber(
                '/crowd/sidewalk_agents',
                carla_connector2.msg.CrowdSidewalkAgentArray,
                self.sidewalk_agents_callback,
                queue_size=1)
        self.il_car_info_sub = rospy.Subscriber(
                '/IL_car_info',
                CarInfo,
                self.il_car_info_callback,
                queue_size=1)
        self.agents_pub = rospy.Publisher(
                '/agent_array',
                carla_connector2.msg.TrafficAgentArray,
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
                self.world.debug.draw_line(last_loc,loc,life_time = 0.1, 
                    color = carla.Color(color_i,0, color_i,0))
            last_loc = carla.Location(pos.x, pos.y, 0.1)

    def update(self):
        if not self.do_update:
            return

        if not self.ego_car_info:
            return

        current_time = rospy.Time.now()

        ego_car_position = self.ego_car_info.car_pos
        ego_car_position = carla.Vector2D(
                ego_car_position.x,
                ego_car_position.y)

        agents_msg = carla_connector2.msg.TrafficAgentArray()

        current_time = rospy.Time.now()
        agents = self.network_agents + self.sidewalk_agents
        for agent in agents:
            actor = self.world.get_actor(agent.id)

            if actor is None: # or (get_position(actor) - ego_car_position).length() > 50: # TODO Add as ROS parameter.
                continue

            actor_pos = carla.Vector2D(actor.get_location().x, actor.get_location().y)
            
            agent_tmp = carla_connector2.msg.TrafficAgent()
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

            if type(agent) is carla_connector2.msg.CrowdNetworkAgent:
                initial_route_point = carla.SumoNetworkRoutePoint()
                initial_route_point.edge = agent.route_point.edge
                initial_route_point.lane = agent.route_point.lane
                initial_route_point.segment = agent.route_point.segment
                initial_route_point.offset = agent.route_point.offset

                paths = [[initial_route_point]]
                topological_hash = ''
            
                #TODO Add 20, 1.0 as ROS paramters.
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

                agent_tmp.reset_intention = self.topological_hash_map[agent.id] is None or self.topological_hash_map[agent.id] != topological_hash
                for path in paths:
                    path_msg = Path()
                    path_msg.header.frame_id = 'map'
                    path_msg.header.stamp = current_time
                    for path_point in [actor_pos] + [self.network.get_route_point_position(route_point) for route_point in path]:
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.header.stamp = current_time
                        pose_msg.pose.position.x = path_point.x
                        pose_msg.pose.position.y = path_point.y
                        path_msg.poses.append(pose_msg)
                    # self.draw_path(path_msg)
                    agent_tmp.path_candidates.append(path_msg)
                agent_tmp.cross_dirs = []
                self.topological_hash_map[agent.id] = topological_hash

            elif type(agent) is carla_connector2.msg.CrowdSidewalkAgent:
                sidewalk_route_point = carla.SidewalkRoutePoint()
                sidewalk_route_point.polygon_id = agent.route_point.polygon_id
                sidewalk_route_point.segment_id = agent.route_point.segment_id
                sidewalk_route_point.offset = agent.route_point.offset
                sidewalk_route_orientation = agent.route_orientation

                path = [sidewalk_route_point]
                for _ in range(20):
                    if sidewalk_route_orientation:
                        path.append(self.sidewalk.get_next_route_point(path[-1], 1.0)) # TODO Add as ROS parameter.
                    else:
                        path.append(self.sidewalk.get_previous_route_point(path[-1], 1.0)) # TODO Add as ROS parameter.
                topological_hash = '{},{}'.format(sidewalk_route_point.polygon_id, agent.route_orientation)

                agent_tmp.reset_intention = self.topological_hash_map[agent.id] is None or self.topological_hash_map[agent.id] != topological_hash
                path_msg = Path()
                path_msg.header.frame_id = 'map'
                path_msg.header.stamp = current_time
                for path_point in [actor_pos] + [self.sidewalk.get_route_point_position(route_point) for route_point in path]:
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = 'map'
                    pose_msg.header.stamp = current_time
                    pose_msg.pose.position.x = path_point.x
                    pose_msg.pose.position.y = path_point.y
                    path_msg.poses.append(pose_msg)
                self.draw_path(path_msg)
                agent_tmp.path_candidates = [path_msg]
                agent_tmp.cross_dirs = [agent.route_orientation]
                self.topological_hash_map[agent.id] = topological_hash
            
            agents_msg.agents.append(agent_tmp)

        agents_msg.header.frame_id = 'map'
        agents_msg.header.stamp = current_time    
        self.agents_pub.publish(agents_msg)

        self.do_update = False
    
if __name__ == '__main__':
    rospy.init_node('crowd_processor')
    crowd_processor = CrowdProcessor()
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        crowd_processor.update()
        rate.sleep()
