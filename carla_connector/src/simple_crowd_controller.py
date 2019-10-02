#!/usr/bin/env python2

from drunc import Drunc
import carla

import math
import random
import numpy as np
import rospy

from std_msgs.msg import Bool
from network_agent_path import NetworkAgentPath
from sidewalk_agent_path import SidewalkAgentPath
from util import *
import msg_builder.msg
from msg_builder.msg import car_info as CarInfo
import timeit


first_time = True
prev_time = timeit.default_timer()

default_agent_pos = carla.Vector2D(10000, 10000)
default_agent_bbox = []
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,-1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,1))
default_agent_bbox.append(default_agent_pos + carla.Vector2D(-1,-1))

def get_position(actor):
    pos3D = actor.get_location()
    return carla.Vector2D(pos3D.x, pos3D.y)
    
def get_forward_direction(actor):
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)

class CrowdAgent(object):
    def __init__(self, actor, preferred_speed):
        self.actor = actor
        self.preferred_speed = preferred_speed
        self.actor.set_collision_enabled(True) ## to check. Disable collision will generate vehicles that are overlapping
        self.stuck_time = None
    
    def get_id(self):
        return self.actor.id

    def get_velocity(self):
        v = self.actor.get_velocity()
        return carla.Vector2D(v.x, v.y)
    
    def get_transform(self):
        return self.actor.get_transform()
    
    def get_bounding_box(self):
        return self.actor.bounding_box
    
    def get_forward_direction(self):
        forward = self.actor.get_transform().get_forward_vector()
        return carla.Vector2D(forward.x, forward.y)

    def get_position(self):
        pos3D = self.actor.get_location()
        return carla.Vector2D(pos3D.x, pos3D.y)
    
    def get_position3D(self):
        return self.actor.get_location()

    def disable_collision(self):
        self.actor.set_collision_enabled(False)
  
    def get_path_occupancy(self):
        p = [self.get_position()] + [self.path.get_position(i) for i in range(self.path.min_points)]
        return carla.OccupancyMap(p, self.actor.bounding_box.extent.y * 2 + 1.0)

class CrowdNetworkAgent(CrowdAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkAgent, self).__init__(actor, preferred_speed)
        self.path = path

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_bounding_box_corners(self, expand = 0.0):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction

        half_y_len = bbox.extent.y + expand
        half_x_len = bbox.extent.x + expand

        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners
    
    def get_preferred_velocity(self):
        position = self.get_position()

        ## to check
        if not self.path.resize():
            return None
        self.path.cut(position)
        if not self.path.resize():
            return None

        target_position = self.path.get_position(5) ## to check
        velocity = (target_position - position).make_unit_vector()
        return self.preferred_speed * velocity

    def get_path_forward(self):
        position = self.get_position()
        if not self.path.resize():
            return carla.Vector2D(0, 0)
        self.path.cut(position)
        if not self.path.resize():
            return carla.Vector2D(0, 0)

        first_position = self.path.get_position(0)
        second_position = self.path.get_position(1)

        return (second_position - first_position).make_unit_vector()
    
    def get_control(self, velocity):
        steer = get_signed_angle_diff(velocity, self.get_forward_direction())
        min_steering_angle = -45.0
        max_steering_angle = 45.0
        if steer > max_steering_angle:
            steer = max_steering_angle
        elif steer < min_steering_angle:
            steer = min_steering_angle

        k = 1.0
        steer = k * steer / (max_steering_angle - min_steering_angle) * 2.0
        desired_speed = velocity.length()
        #steer_tmp = get_signed_angle_diff(velocity, self.get_forward_direction())
        cur_speed = self.get_velocity().length()
        control = self.actor.get_control()

        # if desired_speed < 0.5:
        #     desired_speed = 0

        k2 = 1.5 #1.5
        k3 = 2.5 #2.5
        if desired_speed - cur_speed > 0:
            control.throttle = k2 * (desired_speed - cur_speed) / desired_speed
            control.brake = 0.0
        elif desired_speed - cur_speed == 0:
            control.throttle = 0.0
            control.brake = 0.0
        else:
            control.throttle = 0
            control.brake = k3 * (cur_speed - desired_speed) / cur_speed

        control.steer = steer
        return control

class CrowdNetworkCarAgent(CrowdNetworkAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkCarAgent, self).__init__(actor, path, preferred_speed)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Car')

    def get_type(self):
        return 'car'

class CrowdNetworkBikeAgent(CrowdNetworkAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdNetworkBikeAgent, self).__init__(actor, path, preferred_speed)

    def get_agent_params(self):
        return carla.AgentParams.get_default('Bicycle')

    def get_type(self):
        return 'bike'

class CrowdSidewalkAgent(CrowdAgent):
    def __init__(self, actor, path, preferred_speed):
        super(CrowdSidewalkAgent, self).__init__(actor, preferred_speed)
        self.path = path
        
    def get_agent_params(self):
        return carla.AgentParams.get_default('People')
    
    def get_bounding_box_corners(self):
        bbox = self.actor.bounding_box
        loc = carla.Vector2D(bbox.location.x, bbox.location.y) + self.get_position()
        forward_vec = self.get_forward_direction().make_unit_vector() # the local x direction (left-handed coordinate system)
        sideward_vec = forward_vec.rotate(np.deg2rad(90)) # the local y direction. (rotating clockwise by 90 deg)

        # Hardcoded values for people.
        half_y_len = 0.25
        half_x_len = 0.25

        # half_y_len = bbox.extent.y
        # half_x_len = bbox.extent.x

        corners = []
        corners.append(loc - half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec + half_y_len*sideward_vec)
        corners.append(loc + half_x_len*forward_vec - half_y_len*sideward_vec)
        corners.append(loc - half_x_len*forward_vec - half_y_len*sideward_vec)
        
        return corners
    
    def get_preferred_velocity(self):
        position = self.get_position()

        if not self.path.resize():
            return None

        self.path.cut(position)

        if not self.path.resize():
            return None

        target_position = self.path.get_position(0)
        velocity = (target_position - position).make_unit_vector()
        return self.preferred_speed * velocity

    def get_path_forward(self):
        return carla.Vector2D(0, 0)
    
    def get_control(self, velocity):
        # velocity = velocity.make_unit_vector() * self.preferred_speed
        return carla.WalkerControl(
                carla.Vector3D(velocity.x, velocity.y, 0),
                1.0, False)

class SimpleCrowdController(Drunc):
    def __init__(self):
        super(SimpleCrowdController, self).__init__()
        self.network_car_agents = []
        self.network_bike_agents = []
        self.sidewalk_agents = []
        self.initialized = False

        self.walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.vehicles_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        self.cars_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        self.bikes_blueprints = [x for x in self.vehicles_blueprints if int(x.get_attribute('number_of_wheels')) == 2]
        
        self.num_network_car_agents = rospy.get_param('~num_network_car_agents')
        self.num_network_bike_agents = rospy.get_param('~num_network_bike_agents')
        self.num_sidewalk_agents = rospy.get_param('~num_sidewalk_agents')
        self.path_min_points = rospy.get_param('~path_min_points')
        self.path_interval = rospy.get_param('~path_interval')

        self.start_time = None
        self.stats_total_num_car = 0
        self.stats_total_num_bike = 0
        self.stats_total_num_ped = 0
        self.stats_total_num_stuck_car = 0
        self.stats_total_num_stuck_bike = 0
        self.stats_total_num_stuck_ped = 0
        self.log_file = open('/home/leeyiyuan/simple_data.txt', 'w', buffering=0)
   
    def dispose(self):
        commands = []
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_car_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.network_bike_agents)
        commands.extend(carla.command.DestroyActor(a.actor.id) for a in self.sidewalk_agents)
        self.client.apply_batch(commands)
        print('Destroyed crowd actors.')
    
    def get_spawn_range(self, center, size):
        spawn_min = carla.Vector2D(
            center.x - size, 
            center.y - size)
        spawn_max = carla.Vector2D(
            center.x + size,
            center.y + size)
        return (spawn_min, spawn_max)

    def check_bounds(self, point, bounds_min, bounds_max):
        return bounds_min.x <= point.x <= bounds_max.x and \
               bounds_min.y <= point.y <= bounds_max.y

    def get_spawn_occupancy_map(self, center_pos, spawn_size_min, spawn_size_max):
        return carla.OccupancyMap(
                carla.Vector2D(center_pos.x - spawn_size_max, center_pos.y - spawn_size_max),
                carla.Vector2D(center_pos.x + spawn_size_max, center_pos.y + spawn_size_max)) \
            .difference(carla.OccupancyMap(
                carla.Vector2D(center_pos.x - spawn_size_min, center_pos.y - spawn_size_min),
                carla.Vector2D(center_pos.x + spawn_size_min, center_pos.y + spawn_size_min)))

    def update(self):
        update_time = rospy.Time.now()

        # Determine bounds variables.
        bounds_center = self.scenario_center
        bounds_min = self.scenario_min
        bounds_max = self.scenario_max
        
        # Determine spawning variables.
        if not self.initialized:
            spawn_size_min = 0
            spawn_size_max = 100
        else:
            spawn_size_min = 50
            spawn_size_max = 100
        spawn_segment_map = self.network_segment_map.intersection(self.get_spawn_occupancy_map(bounds_center, spawn_size_min, spawn_size_max))

        while len(self.network_car_agents) < self.num_network_car_agents:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.5
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.cars_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.network_car_agents.append(CrowdNetworkCarAgent(
                    actor, path, 
                    5.0 + random.uniform(0, 0.5)))
                self.stats_total_num_car += 1
        
        while len(self.network_bike_agents) < self.num_network_bike_agents:
            path = NetworkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_segment_map)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.5
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.bikes_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.network_bike_agents.append(CrowdNetworkBikeAgent(
                    actor, path, 
                    3.0 + random.uniform(0, 0.5)))
                self.stats_total_num_bike += 1
      
        while len(self.sidewalk_agents) < self.num_sidewalk_agents:
            spawn_min, spawn_max = self.get_spawn_range(bounds_center, 100)
            path = SidewalkAgentPath.rand_path(self, self.path_min_points, self.path_interval, spawn_min, spawn_max)
            trans = carla.Transform()
            trans.location.x = path.get_position(0).x
            trans.location.y = path.get_position(0).y
            trans.location.z = 0.5
            trans.rotation.yaw = path.get_yaw(0)
            actor = self.world.try_spawn_actor(
                    random.choice(self.walker_blueprints),
                    trans)
            self.world.wait_for_tick(5.0)
            if actor:
                self.sidewalk_agents.append(CrowdSidewalkAgent(
                    actor, path, 
                    0.5 + random.uniform(0.0, 1.0)))
                self.stats_total_num_ped += 1
        
        commands = []
        
        next_agents = []
        for (i, crowd_agent) in enumerate(self.network_car_agents + self.network_bike_agents + self.sidewalk_agents):
            delete = False
            if not delete and not self.check_bounds(crowd_agent.get_position(), bounds_min, bounds_max):
                delete = True
            if not delete and crowd_agent.get_position3D().z < -10:
                delete = True
            if not delete and (type(crowd_agent) is not CrowdSidewalkAgent and \
                    not self.network_occupancy_map.contains(crowd_agent.get_position())):
                delete = True

            if self.initialized:
                if crowd_agent.get_velocity().length() < 0.2:
                    if crowd_agent.stuck_time is not None:
                        if (update_time - crowd_agent.stuck_time).to_sec() >= 5.0:
                            delete = True
                            if type(crowd_agent) is CrowdNetworkCarAgent:
                                self.stats_total_num_stuck_car += 1
                            elif type(crowd_agent) is CrowdNetworkBikeAgent:
                                self.stats_total_num_stuck_bike += 1
                            elif type(crowd_agent) is CrowdSidewalkAgent:
                                self.stats_total_num_stuck_ped += 1
                    else :
                        crowd_agent.stuck_time = update_time
                else:
                    crowd_agent.stuck_time = None

            if delete:
                next_agents.append(None)
                commands.append(carla.command.DestroyActor(crowd_agent.actor.id))
                continue

            pref_vel = crowd_agent.get_preferred_velocity()
            if pref_vel:
                next_agents.append(crowd_agent)
            else:
                next_agents.append(None)
                commands.append(carla.command.DestroyActor(crowd_agent.actor.id))

        if not self.initialized:
            self.start_time = rospy.Time.now()
            self.initialized = True
        
        for (i, crowd_agent) in enumerate(next_agents):
            if crowd_agent:
                path_occupancy = crowd_agent.get_path_occupancy()
                speed_to_exe = crowd_agent.preferred_speed
               
                for (j, other_crowd_agent) in enumerate(next_agents):
                    if i != j and other_crowd_agent and path_occupancy.contains(other_crowd_agent.get_position()):
                        s_f = other_crowd_agent.get_velocity().length()
                        d_f = (other_crowd_agent.get_position() - crowd_agent.get_position()).length()
                        d_safe = 5.0
                        a_max = 3.0
                        s = max(0, s_f * s_f + 2 * a_max * (d_f - d_safe))**0.5
                        speed_to_exe = min(speed_to_exe, s)

                vel_to_exe = crowd_agent.get_preferred_velocity()
                cur_vel = crowd_agent.actor.get_velocity()
                cur_vel = carla.Vector2D(cur_vel.x, cur_vel.y)
                angle_diff = get_signed_angle_diff(vel_to_exe, cur_vel)
                if angle_diff > 30 or angle_diff < -30:
                    vel_to_exe = 0.5 * (vel_to_exe + cur_vel)

                vel_to_exe = vel_to_exe.make_unit_vector() * speed_to_exe

                control = crowd_agent.get_control(vel_to_exe)
                if type(crowd_agent) is CrowdNetworkCarAgent:
                    commands.append(carla.command.ApplyVehicleControl(crowd_agent.actor.id, control))
                elif type(crowd_agent) is CrowdNetworkBikeAgent:
                    commands.append(carla.command.ApplyVehicleControl(crowd_agent.actor.id, control))
                elif type(crowd_agent) is CrowdSidewalkAgent:
                    commands.append(carla.command.ApplyWalkerControl(crowd_agent.actor.id, control))
        
        self.network_car_agents = [a for a in next_agents if a and type(a) is CrowdNetworkCarAgent]
        self.network_bike_agents = [a for a in next_agents if a and type(a) is CrowdNetworkBikeAgent]
        self.sidewalk_agents = [a for a in next_agents if a and type(a) is CrowdSidewalkAgent]
        
        self.client.apply_batch(commands)
        self.world.wait_for_tick(5.0)

        stats_num_car = 0
        stats_num_bike = 0
        stats_num_ped = 0
        stats_sum_speed_car = 0.0
        stats_sum_speed_bike = 0.0
        stats_sum_speed_ped = 0.0

        for agent in self.network_car_agents:
            stats_num_car += 1
            stats_sum_speed_car += agent.get_velocity().length()
        for agent in self.network_bike_agents:
            stats_num_bike += 1
            stats_sum_speed_bike += agent.get_velocity().length()
        for agent in self.sidewalk_agents:
            stats_num_ped += 1
            stats_sum_speed_ped += agent.get_velocity().length()

        self.log_file.write('{} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            (update_time - self.start_time).to_sec(),
            self.stats_total_num_car, 
            self.stats_total_num_bike, 
            self.stats_total_num_ped,
            self.stats_total_num_stuck_car, 
            self.stats_total_num_stuck_bike, 
            self.stats_total_num_stuck_ped,
            stats_num_car,
            stats_num_bike,
            stats_num_ped,
            stats_sum_speed_car,
            stats_sum_speed_bike,
            stats_sum_speed_ped))

        '''
        print('Time = {}'.format((update_time - self.start_time).to_sec()))
        print('Total spawned = {}, {}, {}'.format(
            self.stats_total_num_car, 
            self.stats_total_num_bike, 
            self.stats_total_num_ped))
        print('Stuck deleted = {}, {}, {}'.format(
            self.stats_total_num_stuck_car, 
            self.stats_total_num_stuck_bike, 
            self.stats_total_num_stuck_ped))
        print('Avg. Instantaneous Speed = {}, {}, {}'.format(stats_avg_speed_car, stats_avg_speed_bike, stats_avg_speed_ped))
        '''
            
if __name__ == '__main__':
    rospy.init_node('simple_crowd_controller')
    rospy.wait_for_message("/meshes_spawned", Bool)

    simple_crowd_controller = SimpleCrowdController()
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        start_time = rospy.Time.now()
        simple_crowd_controller.update()
        end_time = rospy.Time.now()
        duration = (end_time - start_time).to_sec()
        print('Update = {} ms = {} hz'.format(duration * 1000, 1.0 / duration))
        rate.sleep()

    simple_crowd_controller.dispose()
