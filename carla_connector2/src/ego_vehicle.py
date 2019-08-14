#!/usr/bin/env python2

import random
import carla
import rospy

from drunc import Drunc
from network_agent_path import NetworkAgentPath

class EgoVehicle(Drunc):
    def __init__(self):
        super(EgoVehicle, self).__init__()

        # Create path.
        self.path = NetworkAgentPath.rand_path(self, 20.0, 1.0)
            
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.*'))
        vehicle_bp.set_attribute('role_name', 'ego_vehicle')
        spawn_position = self.path.get_position()
        spawn_trans = carla.Transform()
        spawn_trans.location.x = spawn_position.x
        spawn_trans.location.y = spawn_position.y
        spawn_trans.location.z = 2.0
        spawn_trans.rotation.yaw = self.path.get_yaw()
        self.actor = self.world.spawn_actor(vehicle_bp, spawn_trans)
        
        self.world.on_tick(self.world_tick_callback)

    def get_position(self):
        location = self.actor.get_location()
        return carla.Vector2D(location.x, location.y)

    def world_tick_callback(self, snapshot):
        if not self.path.resize():
            print('Warning : path too short.')
            return

        self.path.cut(self.get_position())
        
        if not self.path.resize():
            print('Warning : path too short.')
            return

if __name__ == '__main__':
    rospy.init_node('ego_vehicle')
    ego_vehicle = EgoVehicle()
    rospy.spin()
