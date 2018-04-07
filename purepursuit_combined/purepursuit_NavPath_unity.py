## socket related modules
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

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

PURSUIT_DIST = 2.5 ##1.5 for golfcart
RATIO_ANGULAR = 0.3
#MAX_ANGULAR = 0.20
MAX_ANGULAR = 0.8

MAP_FRAME = 'map'

car_speed = 0.0
car_steer = 0.0

car_position_x = 0.0
car_position_y = 0.0
car_yaw = 0.0

initialized = False

goal_reached = 0

sio = socketio.Server()
app = Flask(__name__)


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

class Pursuit(object):
    def __init__(self):
        self.path = Path()
        self.tm = rospy.Timer(rospy.Duration(0.05), self.cb_pose_timer)  ##0.2 for golfcart
        rospy.Subscriber("cmd_vel", Twist, self.cb_speed, queue_size=1)
        self.pub_line = rospy.Publisher("pursuit_line", Marker, queue_size=1)

    def cb_speed(self, msg):
        global car_speed
        car_speed = msg.linear.x

    def cb_pose_timer(self, event):
        global car_position_x
        global car_position_y
        if not initialized:
            return
        position = (car_position_x, car_position_y)
        pursuit_point = self.path.pursuit(position)

        if pursuit_point is None:
            return

        global car_steer
        global car_yaw
        car_steer = self.calc_angular(position, pursuit_point, car_yaw)
        self.publish_pursuit_line(position, pursuit_point, MAP_FRAME)

    def calc_angular(self, position, pursuit_point, car_yaw):
        target = math.atan2(pursuit_point[1] - position[1], pursuit_point[0] - position[0])
        r = angle_diff(target, car_yaw)# * RATIO_ANGULAR
        if r > MAX_ANGULAR:
            r = MAX_ANGULAR
        if r < -MAX_ANGULAR:
            r = -MAX_ANGULAR
        return r


    def publish_pursuit_line(self, p, a, frame_id):
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

@sio.on('car_info')
def car_info(sid, data):
    if data:
        global car_position_x
        global car_position_y
        global car_yaw
        global initialized
        initialized = True
        car_position_x = float(data["car_position_x"])
        car_position_y = float(data["car_position_y"])
        car_yaw = float(data["car_yaw"])


@sio.on('get_cmd_vel')
def send_vel_to_unity(sid, data):
    if data:
        if data["mode"] == "Manual":
            sio.emit("cmd_vel",data={},skip_sid=True)
        else:
            global car_speed
            global car_steer
            print car_speed, car_steer
            sio.emit(
                "cmd_vel",
                data={
                    'car_speed': car_speed.__str__(),
                    'car_steer': car_steer.__str__(),
                    'goal_reached': goal_reached.__str__(),
                },
                skip_sid=True)
    else:
        print "no data received"
        sio.emit('empty', data={}, skip_sid=True)

if __name__=='__main__':

    rospy.init_node('purepursuit')
    pursuit = Pursuit()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 6500)), app)
