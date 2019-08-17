#!/usr/bin/env python2

import rospy

from drunc import Drunc
import carla

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()
        
        triangles = self.occupancy_map.get_mesh_triangles()
        
        self.network_mesh_id = self.world.spawn_dynamic_mesh(
            self.occupancy_map.get_mesh_triangles(), 
            '/Game/Carla/Static/GenericMaterials/Asphalt/M_Asphalt01')
        self.sidewalk_mesh_id = self.world.spawn_dynamic_mesh(
            self.sidewalk_occupancy_map.get_mesh_triangles(),
            '/Game/Carla/Static/GenericMaterials/M_Red')
        self.world.wait_for_tick()

    def dispose(self):
        self.world.destroy_dynamic_mesh(self.network_mesh_id)
        self.world.destroy_dynamic_mesh(self.sidewalk_mesh_id)
        print('Destroyed meshes.')

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()

    rospy.spin()
    spawn_meshes.dispose()
