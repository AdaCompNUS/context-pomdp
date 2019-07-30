
from util import * 

import rospy
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped
import pdb
import tf

from carla import LaneNetwork
from path_smoothing import interpolate_polyline, smoothing


class PathExtractor(object):
    def __init__(self, player, client, world):
        try:
            self.client = client
            self.world = world
            self.player = player
            self.player_type = 'ped'
            self.player_walking_dir = random.choice([True, False])
            self.path = [] 

            if map_type is "carla":
                self.map = self.world.get_map()
                self.spawn_waypoints = self.map.generate_waypoints(0.5)
            elif map_type is "osm":
                self.lane_net = carla.LaneNetwork.load(osm_file_loc)
                self.route_map= carla.RouteMap(self.lane_net)
                occupancy_map = self.lane_net.create_occupancy_map()
                self.sidewalk = carla.Sidewalk(
                    occupancy_map,
                    carla.Vector2D(-map_bound, -map_bound), carla.Vector2D(map_bound, map_bound),
                    3.0, 0.1,
                    10.0)
                assert(self.lane_net)
                assert(self.route_map)
                assert(self.sidewalk)
            else:
                raise Exception('map type not supported: {}'.format(map_type))


            self.bp_lib = self.world.get_blueprint_library()

            self.initialized = False

            self.markers = []

            if map_type is "carla":
                path = self.rand_path()
                spawn_trans = path[0].transform
                spawn_trans.location.z = 1
                spawn_trans.rotation.pitch = 0
                spawn_trans.rotation.roll = 0
                self.player.set_transform(spawn_trans)
            elif map_type is "osm":
                if self.player_type is 'car':
                    self.spawn_osm_vehicle()
                elif self.player_type is 'ped':
                    self.spawn_osm_pedestrian()

            self.plan_pub = rospy.Publisher('plan', NavPath, queue_size=1)
            rospy.Timer(rospy.Duration(0.1), self.publish_path)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for marker in self.markers:
                marker.destroy()
            self.markers = []
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)

    def get_position(self,route_point, actor_flag=None):
        if actor_flag is None:
            actor_flag = self.player_type
        if actor_flag is 'car':
            return self.route_map.get_position(route_point)
        elif actor_flag is 'ped':
            return self.sidewalk.get_route_point_position(route_point)

    def get_next_position(self, route_point, actor_flag=None, offset = 1.0):
        if actor_flag is None:
            actor_flag = self.player_type
        if actor_flag is 'car':
            next_point = self.route_map.get_next_route_points(route_point, offset)
            return self.get_position(random.choice(next_point), actor_flag)
        elif actor_flag is 'ped':
            next_point = self.sidewalk.get_next_route_point(route_point, offset)
            return self.get_position(next_point, actor_flag)

    def get_yaw_on_route(self, route_point, actor_flag = None):
        if actor_flag is None:
            actor_flag = self.player_type
        pos = self.get_position(route_point, actor_flag)
        next_pos = self.get_next_position(route_point, actor_flag)
        yaw = numpy.rad2deg(math.atan2(next_pos.y-pos.y, next_pos.x-pos.x))
        return yaw

    def get_yaw_on_path(self, path, i):
        pos = path[i,:]
        if i+1 < path.shape[0]:
            next_pos = path[i+1,:]
            yaw = numpy.rad2deg(math.atan2(next_pos[1]-pos[1], next_pos[0]-pos[0]))
        else:
            yaw = 0.0

        return yaw

    def spawn_osm_vehicle(self):
        if map_type is "osm":
            try:

                for i in range(random.randint(0, 1000)):
                    self.route_map.rand_route_point()

                pos = None

                while pos is None or (abs(pos.x)>200 or abs(pos.y) > 200):
                    start_point = self.route_map.rand_route_point()

                    if len(self.route_map.get_next_route_paths(start_point, 20.0, 1.0)) > 1:
                        pos = self.get_position(start_point)

                spawn_trans = Transform()
                
                yaw = self.get_yaw_on_route(start_point, 'car')
                # yaw = start_point.get_orientation()  # TODO: implemented this
                spawn_trans.location.x = pos.x
                spawn_trans.location.y = pos.y
                spawn_trans.rotation.yaw = yaw
                spawn_trans.location.z = 2.0
                spawn_trans.rotation.pitch = 0
                spawn_trans.rotation.roll = 0

                if self.player is None:
                    vehicle_bp = random.choice(self.bp_lib.filter('vehicle.bmw.*'))
                    vehicle_bp.set_attribute('role_name', 'ego_vehicle')

                    self.player = self.world.spawn_actor(vehicle_bp, spawn_trans)
                    time.sleep(1)
                    if True: # not mute_debug:
                        print('New player vehicle in path_extractor at pos {} {} {}'.format(
                            pos.x, pos.y, spawn_trans.location.z))
                else:
                    self.player.set_transform(spawn_trans)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e)
                pdb.set_trace()


    def spawn_osm_pedestrian(self):
        if map_type is "osm":
            try:
                
                pos = None

                while pos is None or (abs(pos.x)>map_bound or abs(pos.y) > map_bound):

                    position = carla.Vector2D(random.uniform(
                        -map_bound/2.0, map_bound/2.0), 
                        random.uniform(-map_bound/2.0, map_bound/2.0))
                    if self.player_type is 'car':
                        start_point = self.route_map.get_nearest_route_point(position)
                        if len(self.sidewalk.get_next_route_paths(start_point, 20.0, 1.0)) > 1:
                            pos = self.get_position(start_point)
                    elif self.player_type is 'ped':
                        start_point = self.sidewalk.get_nearest_route_point(position)
                        pos = self.sidewalk.get_route_point_position(start_point)
                    

                spawn_trans = Transform()
                
                yaw = self.get_yaw_on_route(start_point, 'ped')
                # yaw = start_point.get_orientation()  # TODO: implemented this
                spawn_trans.location.x = pos.x
                spawn_trans.location.y = pos.y
                spawn_trans.rotation.yaw = yaw
                spawn_trans.location.z = 2.0
                spawn_trans.rotation.pitch = 0
                spawn_trans.rotation.roll = 0

                if self.player is None:
                    walker_bp = random.choice(self.bp_lib.filter('walker.pedestrian.*'))
                    walker_bp.set_attribute('role_name', 'ego_vehicle')

                    self.player = self.world.spawn_actor(walker_bp, spawn_trans)
                    time.sleep(1)
                    if True: # not mute_debug:
                        print('New player pedestrian in path_extractor at pos {} {} {}'.format(
                            pos.x, pos.y, spawn_trans.location.z))
                else:
                    self.player.set_transform(spawn_trans)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e)
                pdb.set_trace()

    def extract_path(self, way_point):
        path = None

        is_complete = True

        path = [way_point]
        for _ in range(500):
            next_waypoints = None
            if map_type is "carla":
                next_waypoints = path[-1].next(1.0)
            elif map_type is "osm":
                next_waypoints = self.route_map.get_next_route_points(path[-1], 1.0)

            if not next_waypoints:
                is_complete = False
                break
            path += [random.choice(next_waypoints)]
        if not is_complete:
            return False

        return path

    def rand_path(self):
        if map_type is "carla":
            path = self.extract_path(random.choice(self.spawn_waypoints))
            # if random.choice([True, False]):
            #     path.reverse()
            return path
        else:
            raise Exception('rand_path: map type not supported: {}'.format(map_type))

    def spawn_path_marker(self, path):

        for marker in self.markers:
            marker.destroy()
        self.markers = []

        marker_bp = random.choice(self.bp_lib.filter('*constructioncone*'))

        for i in range(10,20):
            spawn_trans = path[i].transform
            spawn_trans.location.z = 1
            spawn_trans.rotation.pitch = 0
            spawn_trans.rotation.roll = 0

            self.markers.append(self.world.spawn_actor(marker_bp, spawn_trans))

    def extend_path(self, actor=None, path=None):
        if actor is None:
            actor = self.player
        if path is None:
            path = self.path

        actor_flag = ''
        if isinstance(actor, carla.Vehicle):
            actor_flag = 'car'
            next_route_points = self.route_map.get_next_route_points(path[-1], 4.0)

            if len(next_route_points) == 0:
                print('Route point query failure!!! cur path len {}'.format(len(path)))
                return False

            next_point = random.choice(next_route_points)

            if not mute_debug:
                pos = self.get_position(path[-1], actor_flag)
                next_pos = self.get_position(next_point, actor_flag)
                print('cur pos: {} {}'.format(pos.x, pos.y))
                print('next pos {} {}'.format(next_pos.x, next_pos.y))

            path.append(next_point)

        elif isinstance(actor, carla.Walker):
            actor_flag = 'ped'
            if self.player_walking_dir:
                path.append(self.sidewalk.get_next_route_point(path[-1], 1.0))
            else:
                path.append(self.sidewalk.get_previous_route_point(path[-1], 1.0))
        else:
            print("Unsupported player type")
            return False
        
        return True


    def cut_path(self, path, rount_pos, actor_flag):

        try:
            cut_index = 0
            min_offset = 100000.0
            for i in range(len(path) / 2):
                route_point = path[i]

                offset = rount_pos - self.get_position(route_point, actor_flag)
                offset = math.sqrt(offset.x**2 + offset.y**2)

                if offset < min_offset:
                    min_offset = offset
            
                if offset < 0.5:
                    cut_index = i + 1

            if min_offset > 5.0:
                print('Lost track: path min offset {}'.format(min_offset))

            if not mute_debug:
                if cut_index > 0: # not mute_debug:
                    print('cut_index', cut_index)

            if not mute_debug:
                print('cur path len {}'.format(len(path)))
                print('')

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

        return path[cut_index:]
        

    def get_cur_paths(self, actor, path):
        # ego_location = self.player.get_location()
        actor_location = actor.get_location()

        # print('car location {} {}'.format(ego_location.x, ego_location.y))            
        try:
            position = Vector2D(actor_location.x, actor_location.y)
            # print("get_cur_paths for actor", actor)
            actor_flag = ''
            if isinstance(actor, carla.Vehicle):
                actor_flag = 'car'
                # print("actor is vehicle")
                # TODO: change this to extend paths at leaf and trim the start part                
                route_point = self.route_map.get_nearest_route_point(position)
                route_paths = self.route_map.get_next_route_paths(route_point, 20.0, 1.0)
                rount_pos = self.get_position(route_point, actor_flag)

                for i in range(len(route_paths)):
                    route_paths[i] = self.cut_path(route_paths[i], rount_pos, actor_flag)

            elif isinstance(actor, carla.Walker):
                actor_flag = 'ped'
                # print("actor is walker")
                
                route_point = self.sidewalk.get_nearest_route_point(position)
                rount_pos = self.get_position(route_point, actor_flag)

                if len(path) == 0:
                    path.append(route_point)

                while len(path) < 20:
                    # if random.random() <= 0.01:q
                    #     adjacent_route_points = self.sidewalk.get_adjacent_route_points(path[-1])
                    #     if adjacent_route_points:
                    #         path.append(adjacent_route_points[0])
                    #         self.player_walking_dir = random.choice([False, True])
                    #         continue

                    if not self.extend_path(actor, path):
                        break
                    pass

                path = self.cut_path(path, rount_pos, actor_flag)
                route_paths = [path]

            # path = route_paths[0]
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

        return route_paths


    def get_cur_path(self, actor, path):
        if not mute_debug:
            print('parse path')

        if map_type is "carla":
            actor_location = actor.get_location()
            way_point = self.map.get_waypoint(actor_location)  
            path = self.extract_path(way_point)
        elif map_type is "osm":
            
            route_paths = self.get_cur_paths(actor, path)
            path = route_paths[0]
                # color_i = 255
                # last_loc = None
             #    for point in self.path:
             #        pos = self.sidewalk.get_position(point)
             #        loc = carla.Location(pos.x, pos.y, 0.1)
             #        if last_loc is not None:
             #            self.world.debug.draw_line(last_loc,loc,life_time = 0.1, color = carla.Color(color_i,color_i,0,0))
             #        last_loc = carla.Location(pos.x, pos.y, 0.1)
        
        return path

    def get_smooth_path(self, input_path):
        path = []

        # put vehicle current location into path
        # ego_location = self.player.get_location()
        # path.append([ego_location.x,ego_location.y])

        if map_type is "carla":
            for way_point in input_path:
                path.append([way_point.transform.location.x, way_point.transform.location.y])
        elif map_type is "osm":
            for route_point in input_path:
                pos = self.get_position(route_point)
                path.append([pos.x, pos.y])

        path = numpy.array(path)
        interp_path = interpolate_polyline(path, len(input_path)*10)
        smooth_path = smoothing(interp_path)

        return smooth_path


    def to_pose_stamped(self, way_point, current_time):
        pose = PoseStamped()
        pose.header.stamp = current_time
        pose.header.frame_id = "map"
        pose.pose.position.x = way_point.transform.location.x
        pose.pose.position.y = way_point.transform.location.y
        pose.pose.position.z = 0

        yaw = way_point.transform.rotation.yaw

        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose 

    def to_pose_stamped_route(self, route_point, current_time, actor_flag):
        pos = self.get_position(route_point, actor_flag)
        pose = PoseStamped()
        pose.header.stamp = current_time
        pose.header.frame_id = "map"
        pose.pose.position.x = pos.x
        pose.pose.position.y = pos.y
        pose.pose.position.z = 0

        yaw = self.get_yaw_on_route(route_point, actor_flag)

        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose 

    def to_pose_stamped_numpy(self, point, current_time, path, i):
        
        pose = PoseStamped()
        pose.header.stamp = current_time
        pose.header.frame_id = "map"
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = 0

        yaw = self.get_yaw_on_path(path, i)

        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose 

    def publish_path(self, tick):

        if self.initialized:
            if not mute_debug:
                print("Publishing path")

            self.path = self.get_cur_path(self.player, self.path)

            if self.path is not None:

                smooth_path = self.get_smooth_path(self.path)

                # visualize the path

                # print('Visualizing path of len {}'.format(smooth_path.shape[0]))
                # last_loc = None
                # for i in range(smooth_path.shape[0]):
                #     pos = smooth_path[i,:]
                #     loc = carla.Location(pos[0], pos[1], 0.1)
                #     if last_loc is not None:
                #         self.world.debug.draw_line(last_loc,loc,life_time = 0.1, color = carla.Color(0,255,0,0))
                #     last_loc = carla.Location(pos[0], pos[1], 0.1)

                if smooth_path is not None:
                    current_time = rospy.Time.now() 

                    gui_path = NavPath();
                    if len(smooth_path)>0: 
                        gui_path.header.frame_id = "map";
                        gui_path.header.stamp = current_time;

                    for i, waypoint in enumerate(smooth_path): 
                        gui_path.poses.append(
                            self.to_pose_stamped_numpy(
                                waypoint, current_time, smooth_path, i));

                    # print('path start point {} {}'.format(smooth_path[0][0], smooth_path[0][1]))
                    
                    self.plan_pub.publish(gui_path);







