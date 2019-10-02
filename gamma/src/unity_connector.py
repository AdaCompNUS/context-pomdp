from msg_builder.msg import peds_car_info as PedsCarInfo
from msg_builder.msg import car_info as CarInfo # panpan
from msg_builder.msg import peds_info as PedsInfo
from msg_builder.msg import ped_info as PedInfo
from cluster_assoc.msg import pedestrian_array as PedestrainArray
from cluster_assoc.msg import pedestrian as Pedestrian
from cluster_extraction.msg import cluster as Cluster


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

rospy.init_node('unity_connector')
peds_car_info_pub = rospy.Publisher('peds_car_info', PedsCarInfo, queue_size=10)
car_info_pub = rospy.Publisher('ego_state', CarInfo, queue_size=1) # panpan, for imitation learning data collection


peds_info = PedsInfo()

peds_array_pub = rospy.Publisher('pedestrian_array', PedestrainArray, queue_size=10)


peds_updated = False

sio = socketio.Server()
app = Flask(__name__)


def update_peds(message):
    global peds_info
    global peds_updated
    peds_info = message
    peds_updated = True
    # peds_info = PedsInfo()
    # ped_info_tmp = PedInfo()
    # for ()


def listerner(topic="peds_info"):
    sub = rospy.Subscriber('peds_info', PedsInfo, update_peds)

@sio.on('send_peds_info')
def car_info(sid, data):
    if data:
        global peds_car_info_pub
        global peds_array_pub
        ped_ids = data["ped_ids"]
        ped_ids = ped_ids.split(' ')[0:-1]
        ped_ids = [ int(i) for i in ped_ids]

        
        ped_goal_ids = data["ped_goal_ids"]
        ped_goal_ids = ped_goal_ids.split(' ')[0:-1]
        ped_goal_ids = [ int(i) for i in ped_goal_ids]

        ped_pos_xs = data["ped_pos_xs"]
        ped_pos_xs = ped_pos_xs.split(' ')[0:-1]
        ped_pos_xs = [ float(i) for i in ped_pos_xs]

        ped_pos_ys = data["ped_pos_ys"]
        ped_pos_ys = ped_pos_ys.split(' ')[0:-1]
        ped_pos_ys = [ float(i) for i in ped_pos_ys]

        ped_speeds = data["ped_speeds"]
        ped_speeds = ped_speeds.split(' ')[0:-1]
        ped_speeds = [ float(i) for i in ped_speeds]

        car_pos_x = data["car_position_x"]
        car_pos_x = float(car_pos_x)

        car_pos_y = data["car_position_y"]
        car_pos_y = float(car_pos_y)

        car_yaw = data["car_yaw"]
        car_yaw = float(car_yaw)

        #car_speed = data["car_speed"]
        car_speed = 1.0

        #car_steer = data["car_steer"]
        car_steer = 0.0 #float(car_steer)

        peds_car_info_msg = PedsCarInfo()

        peds_array_msg = PedestrainArray()

        current_time = rospy.Time.now()

        for i in range(0, len(ped_ids)):
            ped_info_tmp = PedInfo()
            ped_info_tmp.ped_id = ped_ids[i]
            ped_info_tmp.ped_goal_id = ped_goal_ids[i]
            ped_info_tmp.ped_speed = ped_speeds[i]
            ped_info_tmp.ped_pos.x = ped_pos_xs[i]
            ped_info_tmp.ped_pos.y = ped_pos_ys[i]
            ped_info_tmp.ped_pos.z = 0
            #print ped_info_tmp
            peds_car_info_msg.peds.append(ped_info_tmp)
            ped_tmp = Pedestrian()
            ped_tmp.object_label = ped_ids[i]
            ped_tmp.global_centroid.x = ped_pos_xs[i]
            ped_tmp.global_centroid.y = ped_pos_ys[i]
            ped_tmp.global_centroid.z = 0
            ped_tmp.last_update = current_time
            peds_array_msg.pd_vector.append(ped_tmp)

        peds_car_info_msg.car.car_pos.x = car_pos_x
        peds_car_info_msg.car.car_pos.y = car_pos_y
        peds_car_info_msg.car.car_pos.z = 0
        peds_car_info_msg.car.car_yaw = car_yaw
        peds_car_info_pub.publish(peds_car_info_msg)

        
        car_info_pub.publish(peds_car_info_msg.car)

        peds_array_msg.header.frame_id = "map"
        peds_array_msg.header.stamp = current_time
        peds_array_pub.publish(peds_array_msg)


@sio.on('get_peds_info')
def send_vel_to_unity(sid, data):
    global peds_updated
    if not peds_updated:
        return
    if data:
        if data["mode"] == "RVO2":
            global peds_info
            ped_ids = ""
            ped_pos_xs = ""
            ped_pos_ys = ""
            ped_goal_ids = ""
            ped_speeds = ""
            for ped in peds_info.peds:
            	ped_ids += ped.ped_id.__str__() + ' '
            	ped_pos_xs += ped.ped_pos.x.__str__() + ' '
            	ped_pos_ys += ped.ped_pos.y.__str__() + ' '
            	ped_goal_ids += ped.ped_goal_id.__str__() + ' '
            	ped_speeds += ped.ped_speed.__str__() + ' '
            sio.emit(
                "next_peds_info",
                data={
                    'ped_ids': ped_ids,
                    'ped_pos_xs': ped_pos_xs,
                    'ped_pos_ys': ped_pos_ys,
                    'ped_goal_ids': ped_goal_ids,
                    'ped_speeds': ped_speeds,
                },
                skip_sid=True)
        else:
            sio.emit("empty",data={},skip_sid=True)
    else:
        print ("no data received")
        sio.emit('empty', data={}, skip_sid=True)
    peds_updated = False

if __name__=='__main__':

    listerner()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 6501)), app)
