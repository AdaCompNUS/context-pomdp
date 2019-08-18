#!/usr/bin/env python2

import random

import numpy as np
import cv2

import rospy
from drunc import Drunc
import carla
from carla import ColorConverter

class Spectator(Drunc):
    def __init__(self):
        super(Spectator, self).__init__()

        self.actor = None
        self.camera_sensor_actor = None
    
        cv2.startWindowThread()
        cv2.namedWindow('spectator')

        self.world_tick_callback_id = self.world.on_tick(self.world_tick_callback)

    def dispose(self):
        if self.camera_sensor_actor is not None:
            if self.camera_sensor_actor.is_alive:
                if self.camera_sensor_actor.is_listening:
                    self.camera_sensor_actor.stop()
                self.camera_sensor_actor.destroy()

        cv2.destroyWindow('spectator')

    def world_tick_callback(self, snapshot):
        if self.actor is not None:
            if self.actor.is_alive:
                return
            else:
                self.actor = None
                self.camera_sensor_actor = None
        
        for actor in self.world.get_actors():
            if actor.is_alive and actor.attributes.get('role_name') == 'ego_vehicle':
                self.actor = actor
                break

        if self.actor is not None:
            camera_blueprint = random.choice(self.world.get_blueprint_library().filter('sensor.camera.rgb'))
            # TODO Add as ROS parameters.
            camera_blueprint.set_attribute('image_size_x', '1920')
            camera_blueprint.set_attribute('image_size_y', '1080')
            camera_blueprint.set_attribute('enable_postprocess_effects', 'False')
            self.camera_sensor_actor = self.world.spawn_actor(
                camera_blueprint,
                carla.Transform(carla.Location(x=-32.0, z=24.0), carla.Rotation(pitch=-30.0)),
                attach_to=self.actor)
            self.camera_sensor_actor.listen(self.camera_image_callback)

    def camera_image_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        cv2.imshow('spectator', array)

if __name__ == '__main__':
    rospy.init_node('spectator')
    spectator = Spectator()

    rospy.spin()

    spectator.dispose()
