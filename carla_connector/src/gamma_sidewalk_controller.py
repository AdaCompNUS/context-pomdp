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

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

world = client.get_world()

class CrowdAgent:
    def __init__(self, route_map, actor, pref_speed, agent_tag):
        self.route_map = route_map
        self.actor = actor
        self.pref_speed = pref_speed
        self.path_route_points = []
        self.agent_tag = agent_tag

        self.current_extend_direction = True
        self.res_dec_rate = random.uniform(0.2, 0.8)
        # if agent_tag is "People":
        #     self.pref_speed = random.uniform(0.7, 1.3) * self.pref_speed

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
        if norm(forward_vec) == 0:
            print "no forward vec"
        sideward_vec = rotate(forward_vec, 90.0) # the local y direction

        half_y_len = bbox.extent.y
        half_x_len = bbox.extent.x

        if self.agent_tag == "People":
            half_y_len = 0.23
            half_x_len = 0.23
        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        if half_y_len == 0 or half_x_len == 0:
            print "no bounding_box"
        # print "============"
        # print "forward_vec: ", forward_vec
        # print "half_x_len: ", half_x_len
        # print "half_y_len: ", half_y_len
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
        if self.agent_tag is "People":
            position = self.get_position()

            if len(self.path_route_points) == 0:
                self.add_closest_route_point_to_path()
            
            while len(self.path_route_points) < 20:
                if random.random() <= 0.01:
                    adjacent_route_points = self.route_map.get_adjacent_route_points(self.path_route_points[-1])
                    if adjacent_route_points:
                        self.path_route_points.append(adjacent_route_points[0])
                        self.current_extend_direction = random.randint(0, 1) == 1
                        continue
                if not self.extend_path():
                    break
            if len(self.path_route_points) < 20:
                return None
            
            last_pos = self.route_map.get_route_point_position(self.path_route_points[-1])
            
            cut_index = 0
            for i in range(len(self.path_route_points) / 2):
                route_point = self.path_route_points[i]
                offset = position - self.route_map.get_route_point_position(route_point)
                if norm(offset) < 1.0:
                    cut_index = i + 1

            self.path_route_points = self.path_route_points[cut_index:]
            target_position = self.route_map.get_route_point_position(self.path_route_points[0])
        
            velocity = normalize(target_position - position)

            return self.pref_speed * rotate(velocity, random.uniform(-6.0, 8.0))
        else:
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
                if norm(offset) < 5.0:
                    cut_index = i + 1

            self.path_route_points = self.path_route_points[cut_index:]
            target_position = self.route_map.get_position(self.path_route_points[0])
        
            velocity = normalize(target_position - position)

            return self.pref_speed * velocity

    def set_velocity(self, velocity):
        if self.agent_tag == "People":
            control = carla.WalkerControl(
                    carla.Vector3D(velocity.x, velocity.y),
                    1.0, False)
            self.actor.apply_control(control)
        elif self.agent_tag == "Car" or self.agent_tag == "Bicycle":
            
            steer = get_signed_angle_diff(velocity, self.get_forward_direction())
            min_steering_angle = -45.0
            max_steering_angle = 45.0
            if steer > max_steering_angle:
                steer = max_steering_angle
            elif steer < min_steering_angle:
                steer = min_steering_angle

            k = 1.0
            steer = k * steer / (max_steering_angle - min_steering_angle) * 2.0

            desired_speed = norm(velocity)
            cur_speed = norm(self.get_velocity())

            control = self.actor.get_control()

            k2 = 1.0
            if desired_speed - cur_speed > 0:
                control.throttle = k2 * (desired_speed - cur_speed)/desired_speed
                control.brake = 0.0
            elif desired_speed - cur_speed == 0:
                control.throttle = 0.0
                control.brake = 0.0
            else:
                control.throttle = 0
                control.brake = 0 #k2 * (cur_speed - desired_speed)/cur_speed
            #print "throttle: ", control.throttle
            global world
            if desired_speed == 0:
                print "desired speed: ", desired_speed
                print "current speed: ", cur_speed
                print "throttle: ", control.throttle
                #world.debug.draw_line(self.actor.get_location(), )
                world.debug.draw_line(self.actor.get_location()+carla.Vector3D(0,0,2), self.actor.get_location()+self.actor.get_transform().get_forward_vector()+carla.Vector3D(0,0,2),  life_time=0.2)

            control.steer = steer
            #control.throttle = 0.5
            # control.brake = 0
            self.actor.apply_control(control)


    def add_closest_route_point_to_path(self):
        if self.agent_tag is "People":
            self.path_route_points.append(self.route_map.get_nearest_route_point(self.get_position()))
        else:
            self.path_route_points.append(self.route_map.get_nearest_route_point(self.get_position()))
    
    def extend_path(self):
        if self.agent_tag is "People":
            if self.current_extend_direction:
                self.path_route_points.append(self.route_map.get_next_route_point(self.path_route_points[-1], 1.0))
            else:
                self.path_route_points.append(self.route_map.get_previous_route_point(self.path_route_points[-1], 1.0))
            return True
        else:
            next_route_points = self.route_map.get_next_route_points(self.path_route_points[-1], 1.0)

            if len(next_route_points) == 0:
                return False

            self.path_route_points.append(random.choice(next_route_points))
            return True
            

def in_bounds(position):
    return -200 <= position.x <= 200 and -200 <= position.y <= 200

NUM_AGENTS = 100

default_agent_pos = carla.Vector2D(10000, 10000)
default_agent_bbox = []
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,-1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,-1))

if __name__ == '__main__':
    lane_network = carla.LaneNetwork.load(osm_file_loc)
    route_map = carla.RouteMap(lane_network)

    occupancy_map = lane_network.create_occupancy_map()
    sidewalk = carla.Sidewalk(
        occupancy_map,
        carla.Vector2D(-200, -200), carla.Vector2D(200, 200),
        3.0, 0.1,
        10.0)
    sidewalk_occupancy_map = sidewalk.create_occupancy_map()

    gamma = carla.RVOSimulator()

    for i in range(NUM_AGENTS):
        gamma.add_agent(carla.AgentParams.get_default("People"), i)
    

    world.spawn_occupancy_map(
        occupancy_map, 
        '/Game/Carla/Static/GenericMaterials/Asphalt/M_Asphalt01')
    world.spawn_occupancy_map(
        sidewalk_occupancy_map,
        '/Game/Carla/Static/GenericMaterials/M_Red')
    world.wait_for_tick()
    time.sleep(2)

    walker_blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    vehicles_blueprints = world.get_blueprint_library().filter('vehicle.*')
    bikes_blueprints = [x for x in vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 2]
    cars_blueprints = [x for x in vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 4]

    # for b in cars_blueprints:
    #     print([s.id for s in b])
    

    crowd_agents = []

    while True:

        while len(crowd_agents) < NUM_AGENTS:

            position = carla.Vector2D(0, 0)
            next_position = carla.Vector2D(0, 0)
            while True:
                position = carla.Vector2D(random.uniform(-200, 200), random.uniform(-200, 200))
                route_point = route_map.get_nearest_route_point(position)
                position = route_map.get_position(route_point)
                if not in_bounds(position):
                    continue
                route_points = route_map.get_next_route_points(route_point, 1.0)
                if len(route_points) != 0:
                    route_point = random.choice(route_points)
                    next_position = route_map.get_position(route_point)
                    break


            forward = next_position - position
            yaw_deg = get_signed_angle_diff(forward, carla.Vector2D(1,0))
            rot = carla.Rotation(0, yaw_deg, 0)         
            loc = carla.Location(position.x, position.y, 0.5)
            trans = carla.Transform(loc, rot)

            agent_tag = random.choice(agent_tag_list)
            pref_speed = 1.0
            if agent_tag == "People":
                actor = world.try_spawn_actor(
                    random.choice(walker_blueprints),
                    trans)
                #pref_speed = 1.2
                pref_speed = 0.5 + random.random() * 1.5
                if actor:
                    crowd_agents.append(CrowdAgent(sidewalk, actor, pref_speed, agent_tag))
                    world.wait_for_tick()
            elif agent_tag == "Car":
                actor = world.try_spawn_actor(
                    random.choice(cars_blueprints),
                    trans)
                pref_speed = 6.0
                if actor:
                    crowd_agents.append(CrowdAgent(route_map, actor, pref_speed, agent_tag))
                    world.wait_for_tick()
            elif agent_tag == "Bicycle":
                actor = world.try_spawn_actor(
                    random.choice(bikes_blueprints),
                    trans)
                pref_speed = 6
                if actor:
                    crowd_agents.append(CrowdAgent(route_map, actor, pref_speed, agent_tag))
                    world.wait_for_tick()
            
                #print "spawned agent: ", agent_tag, " at ", actor.get_location()
        # for i in range(0,3):
        #     world.wait_for_tick()
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
            
            if crowd_agent.agent_tag != gamma.get_agent_tag(i):
                gamma.set_agent(i, carla.AgentParams.get_default(crowd_agent.agent_tag))

            pref_vel = crowd_agent.get_preferred_velocity()
            if pref_vel:
                next_crowd_agents.append(crowd_agent)
                gamma.set_agent_position(i, crowd_agent.get_position()) ### update agent position
                gamma.set_agent_velocity(i, crowd_agent.get_velocity()) ### update agent current velocity
                #world.debug.draw_line(crowd_agent.actor.get_location()+carla.Vector3D(0,0,2), crowd_agent.actor.get_location()+crowd_agent.actor.get_velocity()+carla.Vector3D(0,0,2),  life_time=0)
                gamma.set_agent_heading(i, crowd_agent.get_forward_direction()) ### update agent heading
                #world.debug.draw_line(crowd_agent.actor.get_location()+carla.Vector3D(0,0,2), crowd_agent.actor.get_location()+crowd_agent.actor.get_transform().get_forward_vector()+carla.Vector3D(0,0,2), color=carla.Color (0,0,255), life_time=0)
                gamma.set_agent_bounding_box_corners(i, crowd_agent.get_bounding_box_corners()) ### update agent bounding box corners
                gamma.set_agent_pref_velocity(i, pref_vel) ### update agent preferred velocity
                pref_vel_to_draw = carla.Vector3D(pref_vel.x,pref_vel.y,0)
                #world.debug.draw_line(crowd_agent.actor.get_location()+carla.Vector3D(0,0,2), crowd_agent.actor.get_location()+pref_vel_to_draw+carla.Vector3D(0,0,2), color=carla.Color (0,255,0), life_time=0)

                # setAgentMaxTrackingAngle
                # setAgentAttentionRadius
                # setAgentResDecRate

                # if random.randint(0,1) == 0:
                #     gamma.set_agent_res_dec_rate(i, 0.3)
                # else:
                #     gamma.set_agent_res_dec_rate(i, 0.6)
                # gamma.set_agent_res_dec_rate(i, crowd_agent.res_dec_rate)
            else:
                next_crowd_agents.append(None)
                gamma.set_agent_position(i, default_agent_pos)
                gamma.set_agent_pref_velocity(i, carla.Vector2D(0, 0))
                gamma.set_agent_velocity(i, carla.Vector2D(0, 0))
                gamma.set_agent_heading(i, carla.Vector2D(1.0, 0.0)) ### update agent heading
                gamma.set_agent_bounding_box_corners(i, default_agent_bbox) ### update agent bounding box corners                
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
