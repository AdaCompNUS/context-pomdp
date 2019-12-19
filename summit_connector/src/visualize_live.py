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
# frame_width, frame_height = None, None
video_file, video_out = None, None


class VisualizeLive(SummitDQL):
    def __init__(self):
        super(VisualizeLive, self).__init__()
        if record_video:
            self.timer = rospy.Timer(rospy.Duration(1.0 / fps), self.record_screen)
        self.main_timer = rospy.Timer(rospy.Duration(1.0 / fps), self.update)

    def update(self, tick):
        global frame_array, frame_width, frame_height
        frame_array[:, 0:frame_height, :] = self.draw_info_frame()
        frame_array[:, frame_height:2*frame_height, :] = self.draw_state_frame()
        cv2.imshow('frame', frame_array)

        if cv2.waitKey(1) & 0xFF == ord('q'):
              return False

        return True

    def record_screen(self, tick):
        if frame_array is None:
            return
        # if frame_array.shape[0] != frame_height or frame_array.shape[1] != frame_width or \
        #         frame_array.shape[2] != 4:
        #     return
        # global record_video
        if record_video:
            global start_time, last_time
            elapsed_time = time.time() - start_time
            frame_time = time.time() - last_time
            if elapsed_time > 5:
                print("elapsed {} frame {}".format(elapsed_time, frame_time))
                # rgb_data = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2RGB)
                video_out.write(frame_array)
                last_time = time.time()


def release_video():
    global record_video
    if record_video:
        global video_out
        print("Exiting visualizer..... releasing video {}".format(video_file))
        record_video = False
        video_out.release()


import atexit

atexit.register(release_video)


if __name__ == '__main__':
    rospy.init_node('visualize_live')
    rospy.wait_for_message("/meshes_spawned", std_msgs.msg.Bool)
    rospy.wait_for_message("/ego_state", msg_builder.msg.car_info)
    fps = 30
    visualize_live = VisualizeLive()

    rate = rospy.Rate(20)

    # global frame_height, frame_width
    frame_width = int(2 * visualize_live.range / visualize_live.resolution)
    frame_height = int(2 * visualize_live.range / visualize_live.resolution)
    frame_array = np.zeros((frame_width, 2*frame_height, 3), np.uint8)

    if record_video:
        datetime_object = datetime.datetime.now()
        date_ = datetime.date
        time_ = datetime.time
        datetime_str = str(datetime_object).replace(' ', '_').replace(':', '_').replace('.', '_')
        video_file = "~/Videos/driving_{}.mp4".format(datetime_str)
        video_file = os.path.expanduser(video_file)
        print("recording video_file: {}".format(video_file))
        sys.stdout.flush()

        video_out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height),
                                    isColor=True)
        start_time = time.time()
        last_time = time.time()

    rospy.spin()

    # while not rospy.is_shutdown():
    #     if not visualize_live.update():
    #         break
    #     rate.sleep()
    #     end = time.time()
