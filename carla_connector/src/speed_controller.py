
from util import * 
from path_smoothing import distance

import rospy
from peds_unity_system.msg import peds_car_info as PedsCarInfo
from peds_unity_system.msg import car_info as CarInfo # panpan
from peds_unity_system.msg import peds_info as PedsInfo
from peds_unity_system.msg import ped_info as PedInfo
from carla_connector.msg import agent_array as AgentArray
from carla_connector.msg import traffic_agent as TrafficAgent
from geometry_msgs.msg import Twist

freq = 10.0
acc = 1.5
delta = acc/freq
max_speed = 3.0


class SpeedController(object):
    def __init__(self, player, client, world):
        try:
            self.client = client
            self.world = world
            self.player = player
            self.proximity = 10000000
            self.initialized = False
            self.player_pos = []
            self.peds_pos = []

            self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=1)
            rospy.Subscriber("agent_array", AgentArray, self.cb_peds, queue_size=1)
            rospy.Subscriber("IL_car_info", CarInfo, self.cb_car, queue_size=1)

            rospy.Timer(rospy.Duration(1.0/freq), self.compute_speed_and_publish)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()


    # @staticmethod
    # def _parse_proximity(self, lidar_measurement):
    #     player_loc = self.player.get_location()

    #     yaw = numpy.deg2rad(self.player.get_transform().rotation.yaw)
    #     heading = numpy.array([math.cos(yaw), math.sin(yaw),0])
    #     self.proximity = 10000000

    #     for location in lidar_measurement:
    #         direc = cv_to_np(location) - cv_to_np(player_loc)
    #         direc = direc / numpy.linalg.norm(direc)

    #         infront = True if (numpy.dot(heading, direc)>0.8) else False

    #         if infront:
    #             dist = player_loc.distance(location)/100.0
    #             if dist < self.proximity:
    #                 self.proximity = dist

    def cal_proximty(self):
        # actor_list = self.world.get_actors()

        # ped_list = actor_list.filter('*pedestrian.*') 
        # ped_iter = iter(ped_list)
        # actor = next(ped_iter, 'end')

        # player_loc = self.player.get_location()
        # self.proximity = 10000000

        # while actor is not 'end':
        #     pos = actor.get_location()
        #     dist = player_loc.distance(pos)
        #     if dist < self.proximity:
        #         self.proximity = dist

        #     actor = next(ped_iter, 'end')

        player_pos = self.player_pos
        self.proximity = 10000000

        for ped_pos in self.peds_pos:
            dist = distance(player_pos, ped_pos)
            if dist < self.proximity:
                self.proximity = dist

    def calculate_player_speed(self):

        self.cal_proximty()

        vel = self.player.get_velocity()
        yaw = numpy.deg2rad(self.player.get_transform().rotation.yaw)
        v_2d = numpy.array([vel.x, vel.y, 0])
        forward = numpy.array([math.cos(yaw), math.sin(yaw), 0])
        speed = numpy.vdot(forward, v_2d)  # numpy.linalg.norm(v)

        return speed


    def compute_speed_and_publish(self, tick):
        if self.initialized:

            cmd = Twist();

            curr_vel = self.calculate_player_speed()

            cmd.angular.z = 0

            if self.proximity > 10:
                cmd.linear.x = curr_vel + delta;
                cmd.linear.y = acc
            elif self.proximity > 8:
                cmd.linear.x = curr_vel
                cmd.linear.y = 0
            elif self.proximity < 6:
                cmd.linear.x = curr_vel - delta;
                cmd.linear.y = -acc

            if curr_vel >2 and cmd.linear.y > 0:
                cmd.linear.x = curr_vel
                cmd.linear.y = 0

            if not mute_debug:
                print("Publishing speed, proximity {}, vel {}, acc {}".format(
                    self.proximity, cmd.linear.x, cmd.linear.y))


            self.pub_cmd_vel.publish(cmd);


    def cb_peds(self, msg):
        ped_list = msg.agents
        self.peds_pos = []
        for ped in ped_list:
            self.peds_pos.append([ped.pos.position.x, ped.pos.position.y])

        if not mute_debug:
            if len(self.peds_pos)>0:
                print('ped_0 info', self.peds_pos[0])

    def cb_car(self, msg):
        self.player_pos = [msg.car_pos.x, msg.car_pos.y]
        if not mute_debug:
            print('car info', self.player_pos)











