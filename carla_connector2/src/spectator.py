#!/usr/bin/env python2

from drunc import Drunc
import carla

import random

import numpy as np
import cv2

import rospy
from carla import ColorConverter

from nav_msgs.msg import Odometry

# TODO Improve logic to allow registration against ego vehicle add/deletion/change.
class Spectator(Drunc):
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
                carla.Transform(carla.Location(x=0.6, z=1.3), carla.Rotation(pitch=0.0)),

                # carla.Transform(carla.Location(x=-320.0, z=480.0), carla.Rotation(pitch=-60.0)),
                attach_to=self.actor)
            self.camera_sensor_actor.listen(self.camera_image_callback)

    def dispose(self):
        if self.camera_sensor_actor is not None:
            if self.camera_sensor_actor.is_alive:
                if self.camera_sensor_actor.is_listening:
                    self.camera_sensor_actor.stop()
                self.camera_sensor_actor.destroy()

        cv2.destroyWindow('spectator')

    def camera_image_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        cv2.imshow('spectator', array)

if __name__ == '__main__':
    rospy.init_node('spectator')
    
    data = rospy.wait_for_message('/odom', Odometry)

    spectator = Spectator()

    rospy.spin()

    spectator.dispose()
