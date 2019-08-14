#!/usr/bin/env python2

import rospy
from drunc import Drunc

class SpawnMeshes():
    def __init__(self):
        # Create DRUNC adapter.
        self.drunc = Drunc()
        
        # Spawn meshes.
        self.drunc.world.spawn_occupancy_map(
            self.drunc.occupancy_map, 
            '/Game/Carla/Static/GenericMaterials/Asphalt/M_Asphalt01')
        self.drunc.world.spawn_occupancy_map(
            self.drunc.sidewalk_occupancy_map,
            '/Game/Carla/Static/GenericMaterials/M_Red')
        self.drunc.world.wait_for_tick()
        

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()
    rospy.spin()
