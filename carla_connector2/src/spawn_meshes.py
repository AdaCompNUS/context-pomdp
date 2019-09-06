#!/usr/bin/env python2

import rospy

from drunc import Drunc
import carla

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()

        commands = [];
        commands.append(carla.command.SpawnDynamicMesh(
            self.network_occupancy_map.get_mesh_triangles(), 
            '/Game/Carla/Static/GenericMaterials/Masters/LowComplexity/M_Road1'))
        commands.append(carla.command.SpawnDynamicMesh(
            self.sidewalk_occupancy_map.get_mesh_triangles(),
            '/Game/Carla/Static/GenericMaterials/Ground/GroundWheatField_Mat'))

        '''
        for l in self.landmarks:
            commands.append(carla.command.SpawnDynamicMesh(
                l.get_wall_mesh_triangles(20),
                '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete'))
            commands.append(carla.command.SpawnDynamicMesh(
                l.get_outline_mesh_triangles(20),
                '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete'))
        '''

        results = self.client.apply_batch_sync(commands)
        self.mesh_ids = [result.actor_id for result in results] 

    def dispose(self):
        commands = [carla.command.DestroyDynamicMesh(mesh_id) for mesh_id in self.mesh_ids]
        self.client.apply_batch(commands) 

        print('Destroyed meshes.')

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()

    rospy.spin()
    spawn_meshes.dispose()
