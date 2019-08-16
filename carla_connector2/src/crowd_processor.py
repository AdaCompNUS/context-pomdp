#!/usr/bin/env python2

import numpy as np

import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose, Quaternion, Vector3, Polygon, Point32

from drunc import Drunc
import carla
import carla_connector2.msg
from peds_unity_system.msg import car_info as CarInfo # panpan
from util import *

class CrowdProcessor(Drunc):
    def __init__(self):
        super(CrowdProcessor, self).__init__()

        self.network_agents = []
        self.sidewalk_agents = []
        self.do_update = False
        self.ego_car_info = None

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

    def update(self):
        if not self.do_update:
            return

        if not self.ego_car_info:
            return

        ego_car_position = self.ego_car_info.car_pos
        ego_car_position = carla.Vector2D(
                ego_car_position.x,
                ego_car_position.y)

        agents_msg = carla_connector2.msg.TrafficAgentArray()

        current_time = rospy.Time.now()
        agents = self.network_agents + self.sidewalk_agents
        for agent in agents:
            actor = self.world.get_actor(agent.id)

            if actor is None or (get_position(actor) - ego_car_position).length() > 50:
                continue
            
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

            agent_tmp.reset_intention = False # TODO: set this flag when intention changes
            agent_tmp.path_candidates = []
            
            agents_msg.agents.append(agent_tmp)

        self.agents_pub.publish(agents_msg)

        self.do_update = False
    
if __name__ == '__main__':
    rospy.init_node('crowd_processor')
    crowd_processor = CrowdProcessor()
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        crowd_processor.update()
        rate.sleep()
