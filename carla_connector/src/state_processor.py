#!/usr/bin/env python

from util import * 

import rospy
from peds_unity_system.msg import peds_car_info as PedsCarInfo
from peds_unity_system.msg import car_info as CarInfo # panpan
from peds_unity_system.msg import peds_info as PedsInfo
from peds_unity_system.msg import ped_info as PedInfo
from carla_connector.msg import agent_array as AgentArray
from carla_connector.msg import traffic_agent as TrafficAgent
## TODO: AgentArray should be replaced by non-player array

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point, Pose, Quaternion
from geometry_msgs.msg import Vector3, Polygon, Point32
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
import tf
import pdb
odom_broadcaster = tf.TransformBroadcaster()



class StateProcessor(object):
    '''
    Get agents' state info from CARLA and publish them as ROS topics
    '''

    def __init__(self, client, world, extractor):
        try:
            self.client = client
            self.world = world
            self.extractor = extractor

            if not mute_debug:
                print("finding player")

            self.player = None
            while self.player is None:
                self.find_player()

            if not mute_debug:
                print("player found")

            self.initialized = False

            self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
            self.agents_array_pub = rospy.Publisher('agent_array', AgentArray, queue_size=10)
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

        # if not mute_debug:
        # print("self.initialized {}".format(self.initialized))
        if self.initialized:
            # if not mute_debug:
            # print("Publishing state data")

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
        
        car_info_msg.car_pos.x = pos.x
        car_info_msg.car_pos.y = pos.y
        car_info_msg.car_pos.z = 0
        car_info_msg.car_yaw = yaw
        car_info_msg.car_speed = speed
                        
        if isinstance(self.player, carla.Vehicle):
            steer = self.player.get_control().steer
            car_info_msg.car_steer = steer
            
        if isinstance(self.player, carla.Walker):
            car_info_msg.car_steer = 0.0

        self.car_info_pub.publish(car_info_msg)

    def update_agent_topic(self, agents_array_msg, actor_list, current_time, actor_query):

        try:
            # print("update topic with {} actors".format(len(actor_list)))

            agent_list = actor_list.filter(actor_query) 
            agent_iter = iter(agent_list)
            actor = next(agent_iter, 'end')
            
            while actor is not 'end':

                if actor.attributes.get('role_name') == 'ego_vehicle':
                    # print("skip player")
                    actor = next(agent_iter, 'end')      
                    continue

                ego_location = self.player.get_location()

                actor_location = actor.get_location()

                # print("actor_location {} {}".format(actor_location.x, actor_location.y))

                dist = norm(ego_location - actor_location)

                if dist > 50:
                    # print("dist {}".format(dist))
                    actor = next(agent_iter, 'end')
                    continue
            
                # print("id: {}".format(actor.id))
                # print("loc: {}".format(actor.get_location()))
                # print("vel: {}".format(actor.get_velocity()))
                agent_tmp = TrafficAgent()
                agent_tmp.last_update = current_time
                agent_tmp.id = actor.id
                actor_flag = get_actor_flag(actor)
                agent_tmp.type = String()
                agent_tmp.type.data = actor_flag
                agent_tmp.pose.position.x = actor.get_location().x
                agent_tmp.pose.position.y = actor.get_location().y
                agent_tmp.pose.position.z = actor.get_location().z
                yaw = self.player.get_transform().rotation.yaw
                quat_tf = tf.transformations.quaternion_from_euler(0, 0, yaw)

                agent_tmp.pose.orientation = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])

                bb = actor.bounding_box

                pos = [bb.location.x, bb.location.y, 0.0]
                ext = [bb.extent.x, bb.extent.y]

                agent_tmp.bbox = Polygon()
                for i,j in ([-1, -1], [1, -1], [1, 1], [-1, 1]):
                    agent_tmp.bbox.points.append(Point32(x=pos[0] + i*ext[0], y=pos[1] + j*ext[1], z=0.0))

                agent_tmp.reset_intention = False # TODO: set this flag when intention changes

                agent_paths = self.extractor.get_cur_paths(actor, [])

                # print("visualize path")
                color_b = random.randint(0, 100)
                color_i = 0
                for route_path in agent_paths:
                    last_loc = None
                    color_i += 50
                    for point in route_path:
                        pos = self.extractor.get_position(point, actor_flag)
                        loc = carla.Location(pos.x, pos.y, 0.1)
                        if last_loc is not None:
                            self.world.debug.draw_line(last_loc,loc,life_time = 0.1, 
                                color = carla.Color(color_i,color_i,color_b,0))
                        last_loc = carla.Location(pos.x, pos.y, 0.1)

                agent_tmp.path_candidates = []

                # TODO: unblock this code block to send intentions to the ros topic
                # actor_flag = get_actor_flag(actor)
                # for path in agent_paths:
                #     nav_path = Path()
                #     nav_path.header.frame_id = "map";
                #     nav_path.header.stamp = current_time;

                #     for route_point in path:
                #         position = self.extractor.get_position(route_point, actor_flag)
                #         nav_path.poses.append(self.extractor.to_pose_stamped_route(
                #             route_point, current_time, actor_flag))

                #     agent_tmp.path_candidates.append(nav_path)

                #     agent_temp.cross_dirs.append(True) # TODO: set this flag correctly   
                
                agents_array_msg.agents.append(agent_tmp)

                actor = next(agent_iter, 'end')
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

        return agents_array_msg

    def publish_non_players(self):

        actor_list = self.world.get_actors()

        # print("[StateProcessor] Publishing {} actors".format(len(actor_list)))
        current_time = rospy.Time.now()

        agents_array_msg = AgentArray()

        actor_query = '*pedestrian.*'
        agents_array_msg = self.update_agent_topic(agents_array_msg, actor_list, current_time, actor_query)

        actor_query = 'vehicle.*'
        agents_array_msg = self.update_agent_topic(agents_array_msg, actor_list, current_time, actor_query)

        agents_array_msg.header.frame_id = "map"
        agents_array_msg.header.stamp = current_time

        # print (agents_array_msg)
        # with open("array.txt", "w") as f: 
        #     f.write(agents_array_msg)
        self.agents_array_pub.publish(agents_array_msg)

if __name__ == '__main__':
    processor = StateProcessor()
