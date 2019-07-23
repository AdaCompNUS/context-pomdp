# Example showing usage of RouteMap API to create pedestrians
# that follow routes in the RouteMap.
#
# RouteMap is calculated in LibCarla (C++) from a LaneNetwork and
# exposed through the PythonAPI wrapper. Here, pedestrians are
# spawned and controlled in a loop to follow the RouteMap.
#
# In future iterations, a CrowdController API will be implemented
# in LibCarla (C++) and exposed through PythonAPI. Generally, the
# stuff in this example will be moved to LibCarla and done directly
# in LibCarla, together with the ORCA/GAMMA routines. For usage,
# the interface would be something like crowd_controller.start(num)
# or something like that.

import glob
import math
import os
import sys
from util import *

# sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#     sys.version_info.major,
#     sys.version_info.minor,
#     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

import numpy as np
import carla
import random
import time
import math

agent_tag_list = ["People", "Car", "Bicycle"]

class CrowdAgent:
    def __init__(self, route_map, actor, pref_speed, agent_tag):
        self.route_map = route_map
        self.actor = actor
        self.pref_speed = pref_speed
        self.path_route_points = []
        self.agent_tag = agent_tag

    def get_velocity(self):
        v = self.actor.get_velocity()
        return carla.Vector2D(v.x, v.y)

    def get_transform(self):
        return self.actor.get_transform()
    
    def get_bounding_box(self):
        return self.actor.bounding_box

    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = normalize(self.get_forward_direction()) # the local x direction (left-handed coordinate system)
        sideward_vec = rotate(forward_vec, 90.0) # the local y direction
        half_y_len = bbox.extent.y
        half_x_len = bbox.extent.x
        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        return corners

    def get_rotation(self):
        return self.actor.get_transform().rotation.yaw
    
    def get_forward_direction(self):
        forward = self.actor.get_transform().get_forward_vector()
        return carla.Vector2D(forward.x, forward.y)

    def get_position(self):
        ## get_transform()
        ### waypoint = map.get_waypoint(vehicle.get_location()) # This waypoint's transform is located on a drivable lane, and it's oriented according to the road direction at that point.
        pos3D = self.actor.get_location()
        return carla.Vector2D(pos3D.x, pos3D.y)

    def get_preferred_velocity(self):
        position = self.get_position()

        if len(self.path_route_points) == 0:
            self.add_closest_route_point_to_path()
        while len(self.path_route_points) < 20 and self.extend_path():
            pass
        if len(self.path_route_points) < 20:
            return None
        
        cut_index = 0
        for i in range(len(self.path_route_points) / 2):
            route_point = self.path_route_points[i]
            offset = position - route_map.get_position(route_point)
            if norm(offset) < 1.0:
                cut_index = i + 1

        self.path_route_points = self.path_route_points[cut_index:]
        target_position = self.route_map.get_position(self.path_route_points[0])
    
        velocity = normalize(target_position - position)

        return self.pref_speed * velocity

    def set_velocity(self, velocity):
        control = carla.WalkerControl(
                carla.Vector3D(velocity.x, velocity.y),
                1.0, False)
        self.actor.apply_control(control)

    def add_closest_route_point_to_path(self):
        self.path_route_points.append(self.route_map.get_nearest_route_point(self.get_position()))
    
    def extend_path(self):
        next_route_points = self.route_map.get_next_route_points(self.path_route_points[-1], 1.0)

        if len(next_route_points) == 0:
            return False

        self.path_route_points.append(random.choice(next_route_points))
        return True

def in_bounds(position):
    return -500 <= position.x <= 500 and -500 <= position.y <= 500

NUM_AGENTS = 100

if __name__ == '__main__':
    lane_network = carla.LaneNetwork.load(osm_file_loc)
    route_map = carla.RouteMap(lane_network)
    gamma = carla.RVOSimulator()

    for i in range(NUM_AGENTS):
        gamma.add_agent(carla.AgentParams.get_default("People"), i)
    
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world();
    walker_blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    vehicles_blueprints = world.get_blueprint_library().filter('vehicle.*')
    bikes_blueprints = [x for x in vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 2]
    cars_blueprints = [x for x in vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    

    crowd_agents = []

    while True:

        while len(crowd_agents) < NUM_AGENTS:

            position = carla.Vector2D(0, 0)
            next_position = carla.Vector2D(0, 0)
            while True:
                position = carla.Vector2D(random.uniform(-500, 500), random.uniform(-500, 500))
                route_point = route_map.get_nearest_route_point(position)
                position = route_map.get_position(route_point)
                if not in_bounds(position):
                    continue
                route_points = route_map.get_next_route_points(route_point, 1.0)
                if len(route_points) == 0:
                    continue
                route_point = random.choice(route_points)
                next_position = route_map.get_position(route_point)


            forward = next_position - position
            yaw_deg = get_signed_angle_diff(forward, carla.Vector2D(1,0))
            rot = carla.Rotation(0, yaw_deg, 0)         
            loc = carla.Location(position.x, position.y, 2.0)
            trans = carla.Transform(loc, rot)

            agent_tag = random.choice(agent_tag_list)
            pref_speed = 1.0
            if agent_tag == "People":
                actor = world.try_spawn_actor(
                    random.choice(walker_blueprints),
                    trans)
                pref_speed = 1.5
            elif agent_tag == "Car":
                actor = world.try_spawn_actor(
                    random.choice(cars_blueprints),
                    trans)
                pref_speed = 4.0
            elif agent_tag == "Bicycle":
                actor = world.try_spawn_actor(
                    random.choice(bikes_blueprints),
                    trans)
                pref_speed = 2.5
            if actor:
                crowd_agents.append(CrowdAgent(route_map, actor, pref_speed, agent_tag))

        world.wait_for_tick()

            # UpdateAllAgentPosition ();
            # UpdateAllAgentBoundingBoxCorners ();
            # UpdateAllAgentHeading ();
            # UpdateAllAgentPrefVelocity ();


            # doStep();
            # UpdateAllAgentVelocity ();

            # RemoveAgentsReachedGoal ();

        next_crowd_agents = []
        for (i, crowd_agent) in enumerate(crowd_agents):
            if not in_bounds(crowd_agent.get_position()):
                next_crowd_agents.append(None)
                crowd_agent.actor.destroy()
                continue
            
            if crowd_agent.agent_tag != gamma.get_agent_type(i):
                gamma.set_agent(i, carla.AgentParams.get_default(crowd_agent.agent_tag))

            pref_vel = crowd_agent.get_preferred_velocity()
            if pref_vel:
                next_crowd_agents.append(crowd_agent)
                gamma.set_agent_position(i, crowd_agent.get_position()) ### update agent position
                gamma.set_agent_velocity(i, crowd_agent.get_velocity()) ### update agent current velocity
                gamma.set_agent_heading(i, crowd_agent.get_forward_direction()) ### update agent heading
                gamma.set_agent_bounding_box_corners(i, crowd_agent.get_bounding_box_corners()) ### update agent bounding box corners
                gamma.set_agent_pref_velocity(i, pref_vel) ### update agent preferred velocity
                # setAgentMaxTrackingAngle
                # setAgentAttentionRadius
                # setAgentResDecRate
            else:
                next_crowd_agents.append(None)
                gamma.set_agent_position(i, crowd_agent.get_position())
                gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                crowd_agent.actor.destroy()
        crowd_agents = next_crowd_agents
        
        gamma.do_step()

        for (i, crowd_agent) in enumerate(crowd_agents):
            if crowd_agent is not None:
                crowd_agent.set_velocity(gamma.get_agent_velocity(i))

        crowd_agents = [w for w in crowd_agents if w is not None]




    # position = carla.Vector2D(random.uniform(-500, 500), random.uniform(-500, 500))
    # print(position)
    # route_point = route_map.get_nearest_route_point(position)
    # position = route_map.get_position(route_point)
    # if in_bounds(position):
    #     rot = carla.Rotation(0,45,0)
    #     loc = carla.Location(position.x, position.y, 2.0)
    #     trans = carla.Transform(loc, rot)
    #     actor = world.try_spawn_actor(
    #         random.choice(cars_blueprints),
    #         trans)
    #     if actor:
    #         car = CrowdAgent(route_map, actor, 10.0)
    #         world.wait_for_tick()
    #         print(car.get_position())
    #         print(car.get_bounding_box().location)
    #         print(car.get_bounding_box().extent)
    #         print(car.get_forward_direction())
    #         corners = car.get_bounding_box_corners()
    #         print(corners[0])
    #         print(corners[1])
    #         print(corners[2])
    #         print(corners[3])
    #         print(get_signed_angle_diff(car.get_forward_direction(), carla.Vector2D(1,0)))
