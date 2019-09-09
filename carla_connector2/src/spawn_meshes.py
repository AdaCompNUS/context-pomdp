#!/usr/bin/env python2

import rospy

from drunc import Drunc
import carla

from std_msgs.msg import Bool

class SpawnMeshes(Drunc):
    def __init__(self):
        super(SpawnMeshes, self).__init__()

        self.meshes_spawned_pub = rospy.Publisher('/meshes_spawned', Bool, queue_size=1, latch=True) 


        print('Spawning meshes...')
        commands = [];
        if self.network_occupancy_map_mesh_triangles is not None:
            commands.append(carla.command.SpawnDynamicMesh(
                self.network_occupancy_map_mesh_triangles,
                '/Game/Carla/Static/GenericMaterials/Masters/LowComplexity/M_Road1'))
        if self.sidewalk_occupancy_map_mesh_triangles is not None:
            commands.append(carla.command.SpawnDynamicMesh(
                self.sidewalk.create_occupancy_map(3.0).get_mesh_triangles(),
                '/Game/Carla/Static/GenericMaterials/Ground/GroundWheatField_Mat'))
        if self.landmarks is not None:
            for l in self.landmarks:
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_mesh_triangles(20),
                    '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete'))
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_mesh_triangles(0),
                    '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete'))
                commands.append(carla.command.SpawnDynamicMesh(
                    l.get_wall_mesh_triangles(20),
                    '/Game/Carla/Static/GenericMaterials/Ground/Generic_Concrete_Material/M_Generic_Concrete'))

        results = self.client.apply_batch_sync(commands)
        self.mesh_ids = [result.actor_id for result in results] 
        self.meshes_spawned_pub.publish(True)

        print('All meshes spawned.')


    def dispose(self):
        commands = [carla.command.DestroyDynamicMesh(mesh_id) for mesh_id in self.mesh_ids]
        self.client.apply_batch(commands) 

        print('Destroyed meshes.')

if __name__ == '__main__':
    rospy.init_node('spawn_meshes')
    spawn_meshes = SpawnMeshes()

    rospy.spin()
    spawn_meshes.dispose()
