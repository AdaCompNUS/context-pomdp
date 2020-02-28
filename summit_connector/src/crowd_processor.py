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

import Pyro4
import time

start_time = time.time()

Pyro4.config.SERIALIZERS_ACCEPTED.add('serpent')
Pyro4.config.SERIALIZER = 'serpent'
Pyro4.util.SerializerBase.register_class_to_dict(
        carla.SumoNetworkRoutePoint, 	
        lambda o: { 	
            '__class__': 'carla.SumoNetworkRoutePoint',	
            'edge': o.edge,	
            'lane': o.lane,	
            'segment': o.segment,	
            'offset': o.offset	
        })
def dict_to_sumo_network_route_point(c, o):	
    r = carla.SumoNetworkRoutePoint()	
    r.edge = str(o['edge']) # In python2, this is a unicode, so use str() to convert.
    r.lane = o['lane']	
    r.segment = o['segment']	
    r.offset = o['offset']	
    return r	
Pyro4.util.SerializerBase.register_dict_to_class(	
        'carla.SumoNetworkRoutePoint', dict_to_sumo_network_route_point)	
Pyro4.util.SerializerBase.register_class_to_dict(	
        carla.SidewalkRoutePoint, 	
        lambda o: { 	
            '__class__': 'carla.SidewalkRoutePoint',	
            'polygon_id': o.polygon_id,	
            'segment_id': o.segment_id,	
            'offset': o.offset	
        })	
def dict_to_sidewalk_route_point(c, o):	
    r = carla.SidewalkRoutePoint()	
    r.polygon_id = o['polygon_id']	
    r.segment_id = o['segment_id']	
    r.offset = o['offset']
    return r	
Pyro4.util.SerializerBase.register_dict_to_class(	
        'carla.SidewalkRoutePoint', dict_to_sidewalk_route_point)

# TODO Speed up spawning logic to prevent hangs in ROS.
class CrowdProcessor(Summit):
    def __init__(self):
        super(CrowdProcessor, self).__init__()
        pyro_port = rospy.get_param('~pyro_port', '8100')
        self.crowd_service = Pyro4.Proxy('PYRO:crowdservice.warehouse@localhost:{}'.format(pyro_port))
        self.network_agents = []
        self.sidewalk_agents = []
        self.ego_car_info = None
        self.topological_hash_map = defaultdict(lambda: None)  # TODO Expiry to prevent memory explosion.

        self.il_car_info_sub = rospy.Subscriber(
            '/ego_state',
            CarInfo,
            self.il_car_info_callback,
            queue_size=1)
        self.agents_ready_pub = rospy.Publisher(
            '/agents_ready',
            Bool,
            latch=True,
            queue_size=1)
        self.agents_pub = rospy.Publisher(
            '/agent_array',
            msg_builder.msg.TrafficAgentArray,
            queue_size=1)
        self.agents_path_pub = rospy.Publisher(
            '/agent_path_array',
            msg_builder.msg.AgentPathArray,
            queue_size=1)

        self.num_car = rospy.get_param('~num_car', 0)
        self.num_bike = rospy.get_param('~num_bike', 0)
        self.num_ped = rospy.get_param('~num_ped', 0)
        self.total_num_agents = self.num_car + self.num_bike + self.num_ped

        # self.agents_ready_pub.publish(True)

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

        if not self.ego_car_info:
            return

        if len(self.world.get_actors()) > self.total_num_agents / 1.2 or time.time() -start_time > 15.0:
            # print("[crowd_processor.py] {} crowd agents ready".format(
                # len(self.world.get_actors())))
            self.agents_ready_pub.publish(True)
        else:
        	pass
            # print("[crowd_processor.py] {} percent of agents ready".format(
                # len(self.world.get_actors()) / float(self.total_num_agents)))

        ego_car_position = self.ego_car_info.car_pos
        ego_car_position = carla.Vector2D(
            ego_car_position.x,
            ego_car_position.y)

        agents_msg = msg_builder.msg.TrafficAgentArray()
        agents_path_msg = msg_builder.msg.AgentPathArray()

        current_time = rospy.Time.now()

        self.crowd_service.acquire_local_intentions()
        local_intentions = self.crowd_service.local_intentions
        self.crowd_service.release_local_intentions()
        local_intentions_lookup = {}

        for x in local_intentions:
            local_intentions_lookup[x[0]] = x[1:]

        for actor in self.world.get_actors():
            if not actor.id in local_intentions_lookup:
                continue
            if actor is None:
                continue

            local_intention = local_intentions_lookup[actor.id]

            actor_pos = carla.Vector2D(actor.get_location().x, actor.get_location().y)

            if (actor_pos - ego_car_position).length() > 50:  # TODO Add as ROS parameter.
                continue

            agent_tmp = msg_builder.msg.TrafficAgent()
            agent_tmp.last_update = current_time
            agent_tmp.id = actor.id
            agent_tmp.type = {'Car': 'car', 'Bicycle': 'bike', 'People': 'ped'}[local_intention[0]]
            agent_tmp.pose.position.x = actor.get_location().x
            agent_tmp.pose.position.y = actor.get_location().y
            agent_tmp.pose.position.z = actor.get_location().z
            agent_tmp.vel.x = actor.get_velocity().x
            agent_tmp.vel.y = actor.get_velocity().y
            quat_tf = tf.transformations.quaternion_from_euler(
                0, 0,
                np.deg2rad(actor.get_transform().rotation.yaw))
            agent_tmp.pose.orientation = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])

            agent_tmp.bbox = Polygon()
            corners = get_bounding_box_corners(actor, expand=0.3)
            for corner in corners:
                agent_tmp.bbox.points.append(Point32(
                    x=corner.x, y=corner.y, z=0.0))

            agents_msg.agents.append(agent_tmp)

            agent_paths_tmp = msg_builder.msg.AgentPaths()
            agent_paths_tmp.id = actor.id
            agent_paths_tmp.type = {'Car': 'car', 'Bicycle': 'bike', 'People': 'ped'}[local_intention[0]]

            if local_intention[0] in ['Car', 'Bicycle']:
                initial_route_point = carla.SumoNetworkRoutePoint()
                initial_route_point.edge = local_intention[1].edge
                initial_route_point.lane = local_intention[1].lane
                initial_route_point.segment = local_intention[1].segment
                initial_route_point.offset = local_intention[1].offset

                paths = [[initial_route_point]]
                topological_hash = ''

                # TODO Add 20, 1.0 as ROS paramters.
                for _ in range(20):
                    next_paths = []
                    for path in paths:
                        next_route_points = self.sumo_network.get_next_route_points(path[-1], 1.0)
                        next_paths.extend(path + [route_point] for route_point in next_route_points)
                        if len(next_route_points) > 1:
                            topological_hash += '({},{},{},{}={})'.format(
                                path[-1].edge, path[-1].lane, path[-1].segment, path[-1].offset,
                                len(next_route_points))
                    paths = next_paths

                agent_paths_tmp.reset_intention = self.topological_hash_map[actor.id] is None or \
                                                  self.topological_hash_map[actor.id] != topological_hash
                for path in paths:
                    path_msg = Path()
                    path_msg.header.frame_id = 'map'
                    path_msg.header.stamp = current_time
                    for path_point in [actor_pos] + \
                            [self.sumo_network.get_route_point_position(route_point) for route_point in path]:
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.header.stamp = current_time
                        pose_msg.pose.position.x = path_point.x
                        pose_msg.pose.position.y = path_point.y
                        path_msg.poses.append(pose_msg)
                    # self.draw_path(path_msg)
                    agent_paths_tmp.path_candidates.append(path_msg)
                agent_paths_tmp.cross_dirs = []
                self.topological_hash_map[actor.id] = topological_hash

            elif local_intention[0] == 'People':
                sidewalk_route_point = carla.SidewalkRoutePoint()
                sidewalk_route_point.polygon_id = local_intention[1].polygon_id
                sidewalk_route_point.segment_id = local_intention[1].segment_id
                sidewalk_route_point.offset = local_intention[1].offset
                sidewalk_route_orientation = local_intention[2]

                path = [sidewalk_route_point]
                for _ in range(20):
                    if sidewalk_route_orientation:
                        path.append(self.sidewalk.get_next_route_point(path[-1], 1.0))  # TODO Add as ROS parameter.
                    else:
                        path.append(self.sidewalk.get_previous_route_point(path[-1], 1.0))  # TODO Add as ROS parameter.
                topological_hash = '{},{}'.format(sidewalk_route_point.polygon_id, sidewalk_route_orientation)

                agent_paths_tmp.reset_intention = self.topological_hash_map[actor.id] is None or \
                                                  self.topological_hash_map[actor.id] != topological_hash
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
                agent_paths_tmp.cross_dirs = [sidewalk_route_orientation]
                self.topological_hash_map[actor.id] = topological_hash

            agents_path_msg.agents.append(agent_paths_tmp)

        try:
        	agents_msg.header.frame_id = 'map'
        	agents_msg.header.stamp = current_time
        	self.agents_pub.publish(agents_msg)

        	agents_path_msg.header.frame_id = 'map'
        	agents_path_msg.header.stamp = current_time
        	self.agents_path_pub.publish(agents_path_msg)
        except Exception as e:
        	print(e)

        # self.do_update = False
        end_time = rospy.Time.now()
        elapsed = (end_time - init_time).to_sec()
        # print('agent_array update = {} ms = {} hz'.format(duration * 1000, 1.0 / duration))
        # print('agent_array update at {}'.format(elapsed))
        # sys.stdout.flush()


if __name__ == '__main__':
    rospy.init_node('crowd_processor')
    init_time = rospy.Time.now()
    crowd_processor = CrowdProcessor()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        crowd_processor.update()
        rate.sleep()
