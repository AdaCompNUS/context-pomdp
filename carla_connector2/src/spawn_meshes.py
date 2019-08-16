#!/usr/bin/env python2

import rospy

from drunc import Drunc
import carla

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()
        
        self.world.spawn_occupancy_map(
            self.occupancy_map, 
            '/Game/Carla/Static/GenericMaterials/Asphalt/M_Asphalt01')
        self.world.spawn_occupancy_map(
            self.sidewalk_occupancy_map,
            '/Game/Carla/Static/GenericMaterials/M_Red')
        self.world.wait_for_tick()

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()
    rospy.spin()
