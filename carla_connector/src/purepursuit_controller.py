from util import * 


import rospy
import csv
import math
import sys
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from nav_msgs.msg import Path as NavPath
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

PURSUIT_DIST = 3.0 ##1.5 for golfcart
RATIO_ANGULAR = 0.3
WHEEL_DIST = 2.66
#MAX_ANGULAR = 0.20
MAX_ANGULAR = 0.8

MAX_STEERING = 0.66

MAP_FRAME = 'map'


const_speed = 0.47

goal_reached = 0

use_steer_from_path = False # False, if True, steering will be calculated from path
use_steer_from_pomdp = True # True, if True, steering will come from 'cmd_vel' topic

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def norm_angle(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def angle_diff(m1, m0):
    "m1 - m0"
    return norm_angle(m1 - m0)

class Path(object):
    def __init__(self):
        #self.path = load_path(path_fn, reverse)
        self.path = []
        rospy.Subscriber("plan", NavPath, self.cb_path, queue_size=1)

    def cb_path(self, msg):
        path = []
        for i in range(0, len(msg.poses)):
            x = msg.poses[i].pose.position.x
            y = msg.poses[i].pose.position.y
            path.append((x,y))
        self.path = path
        #print path

    def nearest(self, p):
        return min(enumerate(self.path), key=lambda (_, a): dist(a, p))

    def ahead(self, i, d):
        pi = self.path[i]
        while i < len(self.path) and dist(pi, self.path[i]) < d:
            i += 1
        return i

    def pursuit(self, p, d=PURSUIT_DIST):
        if self.path == []:
            return None
        ni, np = self.nearest(p)
        j = self.ahead(ni, d)
        if j>=len(self.path):
            goal_reached = 1
        return self.path[j] if j<len(self.path) else None

    def pursuit_tan(self, p, d=PURSUIT_DIST):
        if self.path == []:
            return None
        if len(self.path) == 1:
            return None
        ni, np = self.nearest(p)
        j = self.ahead(ni, d)
        if j>=len(self.path):
            return None
        if j==len(self.path)-1:
            return math.atan2(self.path[j][1]-self.path[j-1][1], self.path[j][0]-self.path[j-1][0])
        else:
            return math.atan2(self.path[j+1][1]-self.path[j][1], self.path[j+1][0]-self.path[j][0])

class Pursuit(object):
    '''
    Input: the current state of the player vehicle, the path to follow
    Output: control of the player vehicle
    '''
    def __init__(self, player, world, client, bp_lib):
        try:
            self.player = player
            self.world = world
            self.client = client
            self.bp_lib = bp_lib
            self.car_speed = 0.0
            self.car_steer = 0.0
            self.car_acc = 0.0
            self.path = Path()
            self.tm = rospy.Timer(rospy.Duration(0.1), self.cb_pose_timer)  ##0.2 for golfcart; 0.05
            rospy.Subscriber("cmd_vel", Twist, self.cb_speed, queue_size=1)
            self.pub_line = rospy.Publisher("pursuit_line", Marker, queue_size=1)
            self.pub_cmd_steer = rospy.Publisher("IL_steer_cmd", Float32, queue_size=1)
            self.initialized = False
            self.vel = [0.0, 0.0]
            self.markers = []

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for marker in self.markers:
                marker.destroy()
            self.markers = []
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

    def cb_speed(self, msg):
        self.car_speed = msg.linear.x
        self.car_acc = msg.linear.y

        if not mute_debug:
            print("Receiving speed {} acc {}".format(self.car_speed, self.car_acc))

        if use_steer_from_pomdp:
            self.car_steer = msg.angular.z
            if not mute_debug:
                print('Steer from pomdp / drivenet:', self.car_steer)

        elif use_steer_from_path:
            pass #do nothing , code in cb_pose_timer

    def clear_markers(self):
        for marker in self.markers:
            marker.destroy()
        self.markers = []

    def spawn_point_marker(self, point):

        if point is None:
            return

        self.world.debug.draw_point(carla.Location(x=point[0], y=point[1], z=0.3),life_time = 0.1, color=carla.Color(0, 0, 255, 0))


    def cb_pose_timer(self, event):
        if not self.initialized:
            return

        position = (self.player.get_location().x, self.player.get_location().y)
        pursuit_angle = self.path.pursuit_tan(position)
        pursuit_point = self.path.pursuit(position)

        self.spawn_point_marker(pursuit_point)

        if pursuit_angle is None:
            return

        if use_steer_from_path:
            car_yaw = numpy.deg2rad(self.player.get_transform().rotation.yaw)

            # heading = (self.player.get_location().x + 5.0* math.cos(car_yaw), 
                # self.player.get_location().y+ 5.0* math.sin(car_yaw))

            # self.spawn_point_marker(heading, "box02")

            last_steer = self.car_steer

            self.car_steer = self.calc_angular(position, pursuit_angle, car_yaw)

            angular_diff = self.calc_angular_diff(position, pursuit_point, car_yaw)

            self.car_steer = 0.4*self.car_steer + 0.4*angular_diff

            self.vel = self.calc_dir(position, pursuit_point) * 1.2

            # self.car_steer = 0.4* angular_diff

            # self.car_steer = 0.4 * (self.car_steer - last_steer) + self.car_steer

            # self.car_steer = -self.car_steer

            if not mute_debug:
                print('Steer from path:', self.car_steer)

        elif use_steer_from_pomdp:
            pass #do nothing , code in cb_speed

        self.publish_pursuit_line(position, pursuit_point, MAP_FRAME)
        # self.update_carla_control()

        self.clear_markers()


    def calc_angular(self, position, pursuit_angle, car_yaw):
        target = pursuit_angle
        r = angle_diff(target, car_yaw)# * RATIO_ANGULAR
        if r > MAX_ANGULAR:
            r = MAX_ANGULAR
        if r < -MAX_ANGULAR:
            r = -MAX_ANGULAR
        #print "angle diff: ", r

        steering = math.atan2(WHEEL_DIST*r,PURSUIT_DIST)
        if steering < -MAX_STEERING:
            steering = -MAX_STEERING
        if steering > MAX_STEERING:
            steering = MAX_STEERING
        # print "steering: ", steering
        return steering


    def calc_angular_diff(self, position, pursuit_point, car_yaw):
        target = math.atan2(pursuit_point[1] - position[1], pursuit_point[0] - position[0])
        r = angle_diff(target, car_yaw)# * RATIO_ANGULAR
        if r > MAX_ANGULAR:
            r = MAX_ANGULAR
        if r < -MAX_ANGULAR:
            r = -MAX_ANGULAR
        return r

    def calc_dir(self, position, pursuit_point):
        move_dir =  numpy.array([pursuit_point[0] - position[0], pursuit_point[1] - position[1]])
        move_dir = move_dir / numpy.linalg.norm(move_dir)
        return move_dir

    def publish_pursuit_line(self, p, a, frame_id):

        # for visualization
        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = rospy.get_rostime()
        line.ns = "pursuit_line"
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.color.r = 0.8
        line.color.g = 0.2
        line.color.a = 1.0
        line.scale.x = 0.02
        line.points.append(Point(p[0], p[1], 0))
        line.points.append(Point(a[0], a[1], 0))
        self.pub_line.publish(line)

        # ground-truth for imitation learning and let's drive
        steer_to_pub = Float32()

        steer_to_pub.data = self.car_steer
        if not mute_debug:
            print('Publishing IL_steer_cmd:', self.car_steer)
        
        self.pub_cmd_steer.publish(steer_to_pub)

    def reset_spectator(self):
        spectator = self.world.get_spectator()

        pos = self.player.get_transform().location
        yaw = self.player.get_transform().rotation.yaw

        spectator_transform = Transform(Location(pos), Rotation(roll=0,pitch=0,yaw = yaw))

        # self.player.get_transform()
        forward_vector = spectator_transform.get_forward_vector()
        if not mute_debug:
            print(forward_vector)
        view_distance = 8
        spectator_transform.location.x -= forward_vector.x * view_distance
        spectator_transform.location.y -= forward_vector.y * view_distance
        spectator_transform.location.z += 3
        spectator_transform.rotation.pitch -= 20
        spectator.set_transform(spectator_transform)

    def update_carla_control(self):

        if isinstance(self.player, carla.Vehicle):
            control = self.player.get_control()
            control.steer =  self.car_steer # numpy.rad2deg(self.car_steer)

            if self.car_acc > 0:
                control.throttle = self.car_acc
                control.brake = 0.0
            elif self.car_acc == 0:
                control.throttle = 0.0
                control.brake = 0.0
            else:
                control.throttle = 0
                control.brake = -self.car_acc

            # if not mute_debug:
            print("Updating carla control, throttle {}, brake {}, steer {}".format(
                    control.throttle, control.brake, control.steer))

            self.player.apply_control(control)
            
        elif isinstance(self.player, carla.Walker):

            # print("updating velocity ({},{})".format(self.vel[0], self.vel[1]))
            
            control = carla.WalkerControl(
                    carla.Vector3D(self.vel[0], self.vel[1]),
                    1.0, False)
            self.player.apply_control(control)
        # self.reset_spectator()

if __name__=='__main__':

    rospy.init_node('purepursuit')
    pursuit = Pursuit()
    rospy.spin()
