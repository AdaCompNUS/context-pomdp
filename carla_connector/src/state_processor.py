#!/usr/bin/env python

from util import * 

import rospy
from peds_unity_system.msg import peds_car_info as PedsCarInfo
from peds_unity_system.msg import car_info as CarInfo # panpan
from peds_unity_system.msg import peds_info as PedsInfo
from peds_unity_system.msg import ped_info as PedInfo
from cluster_assoc.msg import pedestrian_array as PedestrainArray
from cluster_assoc.msg import pedestrian as Pedestrian
## TODO: PedestrainArray should be replaced by non-player array

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose, Quaternion
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import tf
odom_broadcaster = tf.TransformBroadcaster()



class StateProcessor(object):
    '''
    Get agents' state info from CARLA and publish them as ROS topics
    '''

    def __init__(self, client, world):
        try:
            self.client = client
            self.world = world

            if not mute_debug:
                print("finding player")

            self.player = None
            while self.player is None:
                self.find_player()

            if not mute_debug:
                print("player found")

            self.initialized = False

            self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
            self.peds_array_pub = rospy.Publisher('pedestrian_array', PedestrainArray, queue_size=10)
            self.car_info_pub = rospy.Publisher('IL_car_info', CarInfo, queue_size=1)
            rospy.Timer(rospy.Duration(0.1), self.publish_data)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()


    def find_player(self):
        # player_measure = measurements.player_measurements
        if self.player is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.player = actor
                    break

    def publish_data(self, tick):

        if not mute_debug:
            print("self.initialized {}".format(self.initialized))
        if self.initialized:
            if not mute_debug:
                print("Publishing state data")

            self.find_player()

            if not mute_debug:
                print("player found")

            self.publish_odom()
            self.publish_il_car()

            # measurements, _ = self.client.read_data()
            self.publish_non_players()


    def publish_odom(self):
        odom = Odometry()
        current_time = rospy.Time.now() 

        frame_id = "odom"
        child_frame_id = "base_link"
        # pos = self.player_measure.transform.location
        # yaw = self.player_measure.rotation.yaw
        pos = self.player.get_location()
        vel = self.player.get_velocity()
        yaw = self.player.get_transform().rotation.yaw
        v_2d = numpy.array([vel.x, vel.y, 0])
        forward = numpy.array([math.cos(yaw), math.sin(yaw), 0])
        speed = numpy.vdot(forward, v_2d)  # numpy.linalg.norm(v)
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        w_yaw = self.player.get_angular_velocity().z

        odom_broadcaster.sendTransform(
            (pos.x, pos.y, pos.z),
            odom_quat,
            current_time,
            child_frame_id,
            frame_id
        )

        odom.header.stamp = current_time
        odom.header.frame_id = frame_id
        odom.pose.pose = Pose(Point(pos.x, pos.y, 0), Quaternion(*odom_quat))
        odom.child_frame_id = child_frame_id
        odom.twist.twist = Twist(Vector3(vel.x, vel.y, vel.z), Vector3(0, 0, w_yaw))
        self.odom_pub.publish(odom)


    def publish_il_car(self):

        car_info_msg = CarInfo()

        pos = self.player.get_location()
        vel = self.player.get_velocity()
        yaw = self.player.get_transform().rotation.yaw
        v_2d = numpy.array([vel.x, vel.y, 0])
        forward = numpy.array([math.cos(yaw), math.sin(yaw), 0])
        speed = numpy.vdot(forward, v_2d)  # numpy.linalg.norm(v)
        steer = self.player.get_control().steer

        car_info_msg.car_pos.x = pos.x
        car_info_msg.car_pos.y = pos.y
        car_info_msg.car_pos.z = 0
        car_info_msg.car_yaw = yaw
        car_info_msg.car_speed = speed
        car_info_msg.car_steer = steer

        self.car_info_pub.publish(car_info_msg)


    def publish_non_players(self):

        actor_list = self.world.get_actors()
        current_time = rospy.Time.now()

        peds_array_msg = PedestrainArray()

        ped_list = actor_list.filter('*pedestrian.*') 
        ped_iter = iter(ped_list)
        actor = next(ped_iter, 'end')
        while actor is not 'end':

            # print("id: {}".format(actor.id))
            # print("loc: {}".format(actor.get_location()))
            # print("vel: {}".format(actor.get_velocity()))
            ped_tmp = Pedestrian()
            ped_tmp.object_label = actor.id
            ped_tmp.global_centroid.x = actor.get_location().x
            ped_tmp.global_centroid.y = actor.get_location().y
            ped_tmp.global_centroid.z = 0
            ped_tmp.last_update = current_time
            peds_array_msg.pd_vector.append(ped_tmp)
            actor = next(ped_iter, 'end')

        peds_array_msg.header.frame_id = "map"
        peds_array_msg.header.stamp = current_time
        self.peds_array_pub.publish(peds_array_msg)

        # vehs_array_msg = VehicleArray()

        # for agent in measurements.non_player_agents:

        #     if agent.HasField('vehicle'):
        #         pass
                # t = agent.vehicle.transform
                # veh_tmp = Vehicle()
                # veh_tmp.object_label = agent.id
                # veh_tmp.global_centroid.x = t.location.x
                # veh_tmp.global_centroid.y = t.location.y
                # veh_tmp.global_centroid.z = 0
                # veh_tmp.last_update = current_time
                # vehs_array_msg.pd_vector.append(veh_tmp)
        #     elif agent.HasField('pedestrian'):
        #         t = agent.pedestrian.transform
        #         ped_tmp = Pedestrian()
        #         ped_tmp.object_label = agent.id
        #         ped_tmp.global_centroid.x = t.location.x
        #         ped_tmp.global_centroid.y = t.location.y
        #         ped_tmp.global_centroid.z = 0
        #         ped_tmp.last_update = current_time
        #         peds_array_msg.pd_vector.append(ped_tmp)

        # peds_array_msg.header.frame_id = "map"
        # peds_array_msg.header.stamp = current_time
        # self.peds_array_pub.publish(peds_array_msg)

        # vehs_array_msg.header.frame_id = "map"
        # vehs_array_msg.header.stamp = current_time
        # self.vehs_array_pub.publish(vehs_array_msg)


if __name__ == '__main__':
    processor = StateProcessor()
