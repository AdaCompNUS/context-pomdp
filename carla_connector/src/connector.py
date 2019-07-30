#!/usr/bin/env python

from util import * 

import rospy
from peds_unity_system.msg import peds_car_info as PedsCarInfo
from peds_unity_system.msg import car_info as CarInfo # panpan
from peds_unity_system.msg import peds_info as PedsInfo
from peds_unity_system.msg import ped_info as PedInfo
from cluster_assoc.msg import pedestrian_array as PedestrainArray
from cluster_assoc.msg import pedestrian as Pedestrian
## TODO: PedestrainArray should be replaced by non-player array

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Odometry
import tf
odom_broadcaster = tf.TransformBroadcaster()


from state_processor import StateProcessor
from purepursuit_controller import Pursuit
from speed_controller import SpeedController
from path_extractor import PathExtractor

from roslaunch.parent import ROSLaunchParent


class Carla_Connector(object):
    def __init__(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
          
            self.world = self.client.get_world()
            # self.world = self.client.load_map('Town03')

            self.bp_lib = self.world.get_blueprint_library()

            if map_type is "carla":
            	self.map = self.world.get_map()
                self.spawn_waypoints = self.map.generate_waypoints(5.0)
            self.actor_dict = dict()
            self.spectator = self.world.get_spectator()
            self.player = None
            self.proximity = 1000000.0

            self.lidar_sensor = None
            self.camera_sensor = None
            self.find_player()
            self.path_extractor = PathExtractor(self.player, self.client, self.world)
            print('-------------- 1 --------------')
            self.find_player()
            self.processor = StateProcessor(self.client, self.world, self.path_extractor)
            self.pursuit = Pursuit(self.player, self.world, self.client, self.bp_lib)
            self.speed_control = SpeedController(self.player, self.client, self.world)


            # self.add_lidar()
            self.add_camera()

            self.pursuit.initialized = True
            self.processor.initialized = True
            self.speed_control.initialized = True
            self.path_extractor.initialized = True
            self.reset_spectator()

            rospy.Timer(rospy.Duration(1.0/30.0), self.cb_update_spectator)  ##0.2 for golfcart; 0.05

            print("Initialization succesful")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print("Launch failed...")
            self.player.destroy()
            pdb.set_trace()
            # sys.exit()

        finally:
            print("Initialization ended")


    def spawn_vehicle(self):
        
        if map_type is "carla":
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle.bmw.*'))

            spawn_point = random.choice(self.world.get_map().get_spawn_points())
            # path = self.path_extractor.rand_path()
            # spawn_trans = path[0].transform
            # spawn_trans.location.z = 5
            # spawn_trans.rotation.pitch = 0
            # spawn_trans.rotation.roll = 0
            # spawn_point = Transform(Location(x=25.68, y=4, z=1))
            vehicle_bp.set_attribute('role_name', 'ego_vehicle')

            self.player = self.world.spawn_actor(vehicle_bp, spawn_point)

            self.reset_spectator()

    def find_player(self):
        # player_measure = measurements.player_measurements
        if self.player is None:
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'ego_vehicle':
                    self.player = actor
                    if True: #not mute_debug:
                        print('Player found in connector')
                    break

        if self.player is not None:
            self.reset_spectator()
        else:
            if True: #not mute_debug:
                print('Player not found in connector')
            if map_type is "carla":
                self.spawn_vehicle()

    def add_lidar(self):
        try:
            if not mute_debug:
                for bp in self.bp_lib.filter('lidar'):
                    print(bp)

            if True: #not mute_debug:
                for bp in self.bp_lib.filter('static'):
                    print(bp)

            lidar_bp = random.choice(self.bp_lib.filter('lidar'))
            lidar_bp.set_attribute('range', '5000') # in centimeter
            lidar_bp.set_attribute('upper_fov', '5')
            lidar_bp.set_attribute('lower_fov', '-5')

            transform = carla.Transform(carla.Location(x=0.8, z=1.7))
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.player)
            # self.lidar_sensor.listen(lambda data: SpeedController._parse_proximity(
            #     self.speed_control,data))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

        finally:

            if not mute_debug:
                print("Lidar added")

    def add_camera(self):
        try:
            camera_bp = random.choice(self.bp_lib.filter('sensor.camera.rgb'))

            transform = carla.Transform(carla.Location(x=0.8, z=1.7))
            # transform = carla.Transform(carla.Location(x=8, z=1.7))

            self.camera_sensor = self.world.spawn_actor(camera_bp, transform, attach_to=self.player)
           
            # self.camera_sensor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            pdb.set_trace()

        finally:

            if not mute_debug:
                print("Lidar added")


    def reset_spectator(self):
        spectator = self.world.get_spectator()

        pos = self.player.get_transform().location
        yaw = self.player.get_transform().rotation.yaw

        spectator_transform = Transform(Location(pos), Rotation(roll=0,pitch=0,yaw = yaw))

        forward_vector = spectator_transform.get_forward_vector()
        if not mute_debug:
            print(forward_vector)
        view_distance = 8
        spectator_transform.location.x -= forward_vector.x * view_distance
        spectator_transform.location.y -= forward_vector.y * view_distance
        spectator_transform.location.z += 3
        spectator_transform.rotation.pitch -= 20
        spectator.set_transform(spectator_transform)


    def cb_update_spectator(self, tick):
        # pass
        self.pursuit.update_carla_control()

        self.reset_spectator()


    def print_actors(self):
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter('*vehicle.*') 
        vehicle_iter = iter(vehicle_list)

        actor = next(vehicle_iter, 'end')
        while actor is not 'end':
            print("--------------------------------------")
            print(actor)

            print("id: {}".format(actor.id))
            print("loc: {}".format(actor.get_location()))

            vel = actor.get_velocity()
            yaw = actor.get_transform().rotation.yaw
            v_2d = numpy.array([vel.x, vel.y, 0])
            forward = numpy.array([math.cos(yaw), math.sin(yaw), 0])
            speed = numpy.vdot(forward, v_2d)  # numpy.linalg.norm(v)

            print("vel: {}".format(actor.get_velocity()))

            print("rot: {}".format(yaw))
            print("steer: {}".format(actor.get_control().steer))
            print("forward_speed: {}".format(speed))

            actor = next(vehicle_iter, 'end')

        ped_list = actor_list.filter('*pedestrian.*') 
        ped_iter = iter(ped_list)
        actor = next(ped_iter, 'end')
        while actor is not 'end':
            print("--------------------------------------")
            print(actor)

            print("id: {}".format(actor.id))
            print("loc: {}".format(actor.get_location()))
            print("vel: {}".format(actor.get_velocity()))

            actor = next(ped_iter, 'end')

    def loop(self):
        try:
            while True:
                self.print_actors()
                time.sleep(1)
            
        except Exception as e:
            print(e)

        finally:
            print("Terminating...")
            self.player.destroy()

            # client.apply_batch([carla.command.DestroyActor(x[0].id) for x in actor_list])

import subprocess
from automatic_control import *

def myhook():
    print "shutdown time!"
    parent.shutdown()
    stop()
    

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

if __name__ == '__main__':

    try:
        # carla_path = sys.path.expanduser("~/carla/")
        # subprocess.Popen("./CarlaUE4.sh",cwd=carla_path, shell=True)
        argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')

        argparser.add_argument(
            '-v', '--verbose',
            action='store_true',
            dest='debug',
            help='print debug information')
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '--res',
            metavar='WIDTHxHEIGHT',
            default='1280x720',
            help='window resolution (default: 1280x720)')
        argparser.add_argument(
            '--filter',
            metavar='PATTERN',
            default='vehicle.*',
            help='actor filter (default: "vehicle.*")')
        argparser.add_argument("-a", "--agent", type=str,
                               choices=["Roaming", "Basic"],
                               help="select which agent to run",
                               default="Basic")


        parent = ROSLaunchParent("mycore", [], is_core=True)     # run_id can be any string
        parent.start()

        rospy.init_node('carla_publishers')
        rospy.on_shutdown(myhook)

        connector = Carla_Connector()

        try:
            args = argparser.parse_args()

            args.width, args.height = [int(x) for x in args.res.split('x')]

            log_level = logging.DEBUG if args.debug else logging.INFO
            logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

            logging.info('listening to server %s:%s', args.host, args.port)

            print(__doc__)

            idle_loop(args)

            parent.shutdown()
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')

        # if not mute_debug:
            # print("Spining...")
        # rospy.spin()
        # connector.loop()

    finally:
        print("Terminating...")
        
        if connector:
            connector.player.destroy()
            if connector.lidar_sensor: 
                connector.lidar_sensor.destroy()
            if connector.camera_sensor: 
                connector.camera_sensor.destroy()
        parent.shutdown()
        


    
     

