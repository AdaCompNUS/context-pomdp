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
import datetime, sys, os
import numpy as np


record_video = False
video_file, video_out = None, None


class VisualizeLive(SummitDQL):
    def __init__(self):
        super(VisualizeLive, self).__init__()
        self.main_timer = rospy.Timer(rospy.Duration(1.0 / fps), self.update)

    def update(self, tick):
        global frame_array, frame_width, frame_height
        frame_array = self.draw_state_frame()
        cv2.imshow('frame', frame_array)

        if cv2.waitKey(1) & 0xFF == ord('q'):
              return False

        return True


if __name__ == '__main__':
    rospy.init_node('visualize_live')
    rospy.wait_for_message("/meshes_spawned", std_msgs.msg.Bool)
    rospy.wait_for_message("/ego_state", msg_builder.msg.car_info)
    fps = 30
    visualize_live = VisualizeLive()

    frame_width = int(2 * visualize_live.range / visualize_live.resolution)
    frame_height = int(2 * visualize_live.range / visualize_live.resolution)
    frame_array = np.zeros((frame_width, frame_height, 3), np.uint8)

    rospy.spin()
