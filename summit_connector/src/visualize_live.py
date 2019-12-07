#!/usr/bin/env python2

'''

[Road Lanes]   [Sidewalk Lanes]       [Obstacles]

[Road Agents]  [Sidewalk Agents]  

[Ego Path]

[Ego Agent]

'''

from summit_dql import SummitDQL
import carla
import rospy
import msg_builder.msg
import cv2
import time
import std_msgs.msg

class VisualizeLive(SummitDQL):
    def __init__(self):
        super(VisualizeLive, self).__init__()

    def update(self):
        cv2.imshow('frame1', self.draw_state_frame())
        cv2.imshow('frame2', self.draw_info_frame())

        if cv2.waitKey(1) & 0xFF == ord('q'):
              return False

        return True

if __name__ == '__main__':
    rospy.init_node('visualize_live')
    rospy.wait_for_message("/meshes_spawned", std_msgs.msg.Bool)
    rospy.wait_for_message("/ego_state", msg_builder.msg.car_info)
    visualize_live = VisualizeLive()

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if not visualize_live.update():
            break
        rate.sleep()
        end = time.time()
