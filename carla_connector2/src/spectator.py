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
        self.camera_bitmap = None

        self.world_tick_callback_id = self.world.on_tick(self.world_tick_callback)

    def world_tick_callback(self, snapshot):
        if self.actor is not None:
            if self.actor.is_alive:
                return
            else:
                self.actor = None
                self.camera_sensor_actor = None
                self.camera_bitmap = None
        
        for actor in self.world.get_actors():
            if actor.is_alive and actor.attributes.get('role_name') == 'ego_vehicle':
                self.actor = actor
                break

        if self.actor is not None:
            camera_blueprint = random.choice(self.world.get_blueprint_library().filter('sensor.camera.rgb'))
            # TODO Add as ROS parameters.
            camera_blueprint.set_attribute('image_size_x', '800')
            camera_blueprint.set_attribute('image_size_y', '600')
            self.camera_sensor_actor = self.world.spawn_actor(
                camera_blueprint,
                carla.Transform(carla.Location(x=-16.0, z=12.0), carla.Rotation(pitch=-20.0)),
                attach_to=self.actor)
            self.camera_sensor_actor.listen(self.camera_image_callback)

    def camera_image_callback(self, image):
        if self.actor is None or not self.actor.is_alive:
            return

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.camera_bitmap = array
    
    def update(self):
        if self.camera_bitmap is not None:
            cv2.imshow('spectator', self.camera_bitmap)

if __name__ == '__main__':
    rospy.init_node('spectator')
    spectator = Spectator()

    # TODO Put in the correct place.
    cv2.startWindowThread()
    cv2.namedWindow('spectator')

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        spectator.update()
        rate.sleep()

    cv2.destroyWindow('spectator')
