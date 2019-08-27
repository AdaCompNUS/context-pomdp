#!/usr/bin/env python2

import rospy

from drunc import Drunc
import carla

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()
        
        self.network_mesh_id = self.world.spawn_dynamic_mesh(
            self.network_occupancy_map.get_mesh_triangles(), 
            '/Game/Carla/Static/GenericMaterials/Masters/LowComplexity/M_Road1')
        self.sidewalk_mesh_id = self.world.spawn_dynamic_mesh(
            self.sidewalk_occupancy_map.get_mesh_triangles(),
            '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete')
        self.landmarks_mesh_id = self.world.spawn_dynamic_mesh(
            self.landmark_map.get_mesh_triangles(20),
            '/Game/Carla/Static/GenericMaterials/Masters/M_WallMaster')
        self.world.wait_for_tick(1.0)

    def dispose(self):
        self.world.destroy_dynamic_mesh(self.network_mesh_id)
        self.world.destroy_dynamic_mesh(self.sidewalk_mesh_id)
        self.world.destroy_dynamic_mesh(self.landmarks_mesh_id)
        print('Destroyed meshes.')

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()

    rospy.spin()
    spawn_meshes.dispose()
