#!/usr/bin/env python2

from summit import Summit
import carla

import random

import numpy as np
import cv2

import rospy
from carla import ColorConverter

from nav_msgs.msg import Odometry
import msg_builder.msg
import datetime, os, sys, time


# TODO Improve logic to allow registration against ego vehicle add/deletion/change.
class Spectator(Summit):
    def __init__(self):
        super(Spectator, self).__init__()

        self.actor = None
        self.camera_sensor_actor = None

        cv2.startWindowThread()
        cv2.namedWindow('spectator')

        for actor in self.world.get_actors():
            if actor.is_alive and actor.attributes.get('role_name') == 'ego_vehicle':
                self.actor = actor
                break

        if self.actor is not None:
            camera_blueprint = random.choice(self.world.get_blueprint_library().filter('sensor.camera.rgb'))
            # TODO Add as ROS parameters.
            camera_blueprint.set_attribute('image_size_x', '1920')
            camera_blueprint.set_attribute('image_size_y', '1080')
            camera_blueprint.set_attribute('enable_postprocess_effects', 'True')
            camera_blueprint.set_attribute('motion_blur_intensity', '0.0')
            camera_blueprint.set_attribute('motion_blur_max_distortion', '0.0')
            camera_blueprint.set_attribute('motion_blur_min_object_screen_size', '0.0')
            self.camera_sensor_actor = self.world.spawn_actor(
                camera_blueprint,
                # carla.Transform(carla.Location(x=-32.0, z=24.0), carla.Rotation(pitch=-30.0)),
                # carla.Transform(carla.Location(x=-16.0, z=12.0), carla.Rotation(pitch=-30.0)),
                # carla.Transform(carla.Location(x=0.0, y=-0.4, z=1.2), carla.Rotation(pitch=0.0)),
                carla.Transform(carla.Location(x=-14.0, y=-0.0, z=20.0),
                    carla.Rotation(pitch=-45.0)),
                # carla.Transform(carla.Location(x=-320.0, z=480.0), carla.Rotation(pitch=-60.0)),
                attach_to=self.actor)
            self.camera_sensor_actor.listen(self.camera_image_callback)

        if record_video:
            self.timer = rospy.Timer(rospy.Duration(1.0 / fps), self.record_screen)

    def record_screen(self, tick):
        if frame_array is None:
            return
        if frame_array.shape[0] != frame_height or frame_array.shape[1] != frame_width or \
                frame_array.shape[2] != 4:
            return

        global record_video
        if record_video:
            global start_time, last_time
            elapsed_time = time.time() - start_time
            frame_time = time.time() - last_time
            if elapsed_time > 9:
                # print("elapsed {} frame {}".format(elapsed_time, frame_time))
                rgb_data = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2RGB)
                video_out.write(rgb_data)
                last_time = time.time()

    def dispose(self):
        if self.camera_sensor_actor is not None:
            if self.camera_sensor_actor.is_alive:
                if self.camera_sensor_actor.is_listening:
                    self.camera_sensor_actor.stop()
                self.camera_sensor_actor.destroy()

        cv2.destroyWindow('spectator')

    def camera_image_callback(self, image):
        global frame_array
        frame_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        frame_array = np.reshape(frame_array, (image.height, image.width, 4))

        cv2.imshow('spectator', frame_array)


def release_video():
    global record_video
    if record_video:
        global video_out
        print("Exiting speculator..... releaseing video {}".format(video_file))
        record_video = False
        video_out.release()


import atexit

atexit.register(release_video)

if __name__ == '__main__':
    rospy.init_node('spectator')
    record_video = False
    if record_video:
        datetime_object = datetime.datetime.now()
        date_ = datetime.date
        time_ = datetime.time
        datetime_str = str(datetime_object).replace(' ', '_').replace(':', '_').replace('.', '_')
        video_file = "~/Videos/driving_{}.mp4".format(datetime_str)
        video_file = os.path.expanduser(video_file)
        print("recording video_file: {}".format(video_file))
        sys.stdout.flush()

        fps = 30
        frame_width = 1920
        frame_height = 1080
        #
        video_out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height),
                                    isColor=True)
        start_time = time.time()
        last_time = time.time()
        frame_array = None

    data = rospy.wait_for_message('/ego_state', msg_builder.msg.car_info)  # /odom, Odometry

    spectator = Spectator()

    rospy.spin()

    spectator.dispose()
