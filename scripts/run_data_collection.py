import argparse
import subprocess
from argparse import Namespace
import os
from os import path
import random
import sys
import time
import signal
from os.path import expanduser
from clear_process import clear_process, clear_nodes

sim_mode = "carla"  # "unity"

home = expanduser("~")
nn_folder = 'catkin_ws/src/il_controller/src/'

if not os.path.isdir(os.path.join(home, nn_folder)):
    nn_folder = 'catkin_ws/src/il_controller/src/'

root_path = os.path.join(home, 'driving_data')
if not os.path.isdir(root_path):
    os.makedirs(root_path)

catkin_ws_path = os.path.join(home, 'catkin_ws/')
executable = "cross_bts" # "cross_bts", "cross_bts_tri", "cross_bts_test_3", "cross_bts_test_2"
result_subfolder = ""

drive_net_proc_link = None

config = Namespace()

subfolder, obstacle_file, goal_file, problem_flag, scenario_in_unity, doshift_in_unity, road_width = \
    '', '', '', '', None, None, None
start_x, start_y, goal_x, goal_y, rot_angle, goal_flag = \
    None, None, None, None, None, ''
shell_cmd = ''

map_size = 40.0

from multiprocessing import Process, Queue

global_proc_queue = []

pomdp_proc = None

monitor_worker = None

Timeout_inner_monitor_pid, Timeout_monitor_pid = None, None

NO_NET = 0
IMITATION = 1
LETS_DRIVE = 2
JOINT_POMDP = 3

class SubprocessMonitor(Process):
    def __init__(self, queue, id):
        Process.__init__(self)
        self.queue = queue
        self.id = id
        self.pomdp_proc = None

        for (p_handle, p_name, p_out) in self.queue:
            if "pomdp_proc" in p_name:
                self.pomdp_proc = p_handle

        print("SubprocessMonitor initialized")
    def run(self):
        queue_iter = 0

        time.sleep(1)

        if config.verbosity > 0:
            print("[DEBUG] SubprocessMonitor activated")
        while True:

            if queue_iter >= len(self.queue):
                queue_iter = 0
            p_handle, p_name, p_out = self.queue[queue_iter]
            queue_iter += 1

            # Get the work from the queue and expand the tuple
            process_alive = check_process(p_handle, p_name)

            if process_alive is False:
                # print("Subprocess %s has died!!" % p_name)
                break

            ros_alive = check_ros()
            if not ros_alive:
                print("roscore has died!!")
                break

            time.sleep(1)

        if self.pomdp_proc is not None:
            process_alive = check_process(self.pomdp_proc, "pomdp_proc")

            if process_alive is True:
                print("Killing POMDP planning...")
                os.kill(self.pomdp_proc.pid, signal.SIGKILL)
                if config.verbosity > 0:
                    print("[DEBUG] pomdp_proc killed")
        else:
            if config.verbosity > 0:
                print("[DEBUG] pomdp_proc is None")


def check_process(p_handle, p_name):
    return_code = ''
    try:
        return_code = subprocess.check_output("ps -a " + "| grep " + str(p_handle.pid), shell=True)
        return_code = return_code.decode()
    except Exception:
        # if config.verbosity > 0:
        print('Subprocess', p_name, "has died")
        return False

    feature = 'defunct'
    if feature in return_code:
        print('Subprocess', p_name, "has been defuncted")
        if config.verbosity > 0:
            print("pgrep returned", return_code)
        return False

    if config.verbosity > 1:
        print('Subprocess', p_name, "is alive")
    return True


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    ######################################
    parser.add_argument('--sround', type=int,
                        default=0, help='Start round')
    parser.add_argument('--eround', type=int,
                        default=1, help='End round')
    parser.add_argument('--timeout',
                        type=int,
                        default=1000000,
                        help='Time out for this script')
    parser.add_argument('--verb',
                        type=int,
                        default=0,
                        help='Verbosity')
    parser.add_argument('--test',
                        type=int,
                        default=0,
                        help='Test mode')
    parser.add_argument('--window',
                        type=int,
                        default=0,
                        help='Show unity window')
    parser.add_argument('--record',
                        type=int,
                        default=1,
                        help='record rosbag')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU ID for hyp-despot')
    parser.add_argument('--net',
                        type=str,
                        default="no",
                        help='Use drive net for control: 0 no / 1 imitation / 2 lets_drive')
    parser.add_argument('--model',
                        type=str,
                        default='trained_models/hybrid_unicorn.pth',
                        help='Drive net model name')
    parser.add_argument('--val_model',
                        type=str,
                        default='model_val.pth',
                        help='Drive net value model name')
    parser.add_argument('--t_scale',
                        type=float,
                        default=1.0,
                        help='Factor for scaling down the time in simulation (to search for longer time)')
    parser.add_argument('--make',
                        type=int,
                        default=0,
                        help='Make the simulator package')
    parser.add_argument('--baseline',
                        type=str,
                        default="",
                        help='Which baseline to run')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='carla_port')

    return parser.parse_args()


def update_global_config(cmd_args):
    # Update the global configurations according to command line
    print("Parsing config", flush=True)

    config.start_round = cmd_args.sround
    config.end_round = cmd_args.eround
    config.timeout = cmd_args.timeout
    config.verbosity = cmd_args.verb
    config.show_window = bool(cmd_args.window)
    config.record_bag = bool(cmd_args.record)
    config.gpu_id = int(cmd_args.gpu_id)
    config.port = int(cmd_args.port)
    print(config.port)
    config.ros_port = config.port + 111
    print(config.ros_port)
    config.ros_master_url = "http://localhost:{}".format(config.ros_port)
    config.ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(config.ros_port)
    print(config.ros_pref)
    if "no" in cmd_args.net:
        config.use_drive_net_mode = NO_NET
    elif "imitation" in cmd_args.net:
        config.use_drive_net_mode = IMITATION
    elif "lets_drive" in cmd_args.net:
        config.use_drive_net_mode = LETS_DRIVE
    elif "joint_pomdp" in cmd_args.net:
        config.use_drive_net_mode = JOINT_POMDP
    else:
        print("CAUTION: unsupported drive net mode")
        exit(-1)

    config.model = cmd_args.model
    config.val_model = cmd_args.val_model
    config.time_scale = float(cmd_args.t_scale)
    config.test_mode = bool(cmd_args.test)

    if config.timeout == 1000000:
        config.timeout = 11*120*4

    config.max_launch_wait = 10
    config.make = bool(cmd_args.make)

    if config.make:

        try:
            shell_cmds = ["catkin config --merge-devel", 
                "catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"]
            for shell_cmd in shell_cmds:
                print(shell_cmd, flush=True)
                make_proc = subprocess.call(shell_cmd, cwd=catkin_ws_path, shell = True)

        except Exception as e:
            print(e)
            exit(12)

        print("make done")

    if "joint_pomdp" in cmd_args.baseline:
        config.use_drive_net_mode = JOINT_POMDP
        config.record_bag = 0 
    elif "imitation" in cmd_args.baseline:
        config.use_drive_net_mode = IMITATION
        config.record_bag = 0
    elif "porca" in cmd_args.baseline:
        config.use_drive_net_mode = NO_NET
        config.record_bag = 0
    elif "lets-drive" in cmd_args.baseline:
        config.use_drive_net_mode = LETS_DRIVE
        config.record_bag = 0
    elif "pomdp" in cmd_args.baseline:
        config.use_drive_net_mode = NO_NET
        config.record_bag = 0
    elif cmd_args.baseline is "":
        # not baseline
        config.record_bag = 1

    if cmd_args.baseline is not "":
        print("=> Running {} baseline".format(cmd_args.baseline))
        cmd_args.net = config.use_drive_net_mode
        cmd_args.record = config.record_bag

    print('============== cmd_args ==============')
    print("start_round: " + str(cmd_args.sround))
    print("end_round: " + str(cmd_args.eround))
    print("timeout: " + str(config.timeout))
    print("verbosity: " + str(cmd_args.verb))
    print("window: " + str(cmd_args.window))
    print("record: " + str(cmd_args.record))
    print("gpu id: " + str(cmd_args.gpu_id))
    print("drive net: " + str(cmd_args.net))
    print("nn model: " + str(cmd_args.model))
    print("time scale: " + str(cmd_args.t_scale))
    print("test mode: " + str(cmd_args.test))
    print('============== cmd_args ==============')


def timeout_monitor(pid):
    global shell_cmd

    shell_cmd = "python timeout.py " + str(int(config.timeout/config.time_scale)) + ' ' + str(pid)

    to_proc = subprocess.Popen(shell_cmd.split())

    if config.verbosity > 0:
        print(shell_cmd)

    return to_proc


def inner_timeout_monitor():
    global shell_cmd
    shell_cmd = "python timeout_inner.py " + str(int(200/config.time_scale))
    toi_proc = subprocess.Popen(shell_cmd.split())

    if config.verbosity > 0:
        print(shell_cmd)

    return toi_proc


import rosgraph


def check_ros():
    if rosgraph.is_master_online(master_uri=config.ros_master_url):
        # print('ROS MASTER is Online')
        return True
    else:
        if config.verbosity > 0:
            print('[DEBUG] ROS MASTER is OFFLINE')
        return False


def launch_vm():

    exec(open(home + "/lets_drive_vm/bin/activate").read())

    if config.verbosity > 0:
        print("[DEBUG] virtual machine restarted")


def exit_vm():
    subprocess.call("deactivate", shell=True)

    if config.verbosity > 0:
        print("[DEBUG] virtual machine deactivated")


def launch_ros():
    print("Launching ros", flush=True)
    cmd_args = config.ros_pref+"roscore -p {}".format(config.ros_port)
    if config.verbosity > 0:
        print(cmd_args, flush=True)

    ros_proc = subprocess.Popen(cmd_args, shell=True)

    while check_ros() is False:
        time.sleep(1)
        continue

    if config.verbosity > 0:
        print("[DEBUG] roscore restarted")


def mak_dir(path):
    if sys.version_info[0] > 2:
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def get_debug_file_name(flag, round, run, case):
    return result_subfolder + '_debug/' + flag + '_' + str(round) + '_' + str(
        run) + '_' + str(case) + '.txt'

import glob


def get_bag_file_name(round, run, case):
    dir = result_subfolder

    file_name = problem_flag + '_case_' + goal_flag + '_sample-' + \
                str(round) + '-' + str(run) + '-' + str(case)

    existing_bags = glob.glob(dir + "*.bag")

    # remove existing bags for the same run/problem/case id
    for bag_name in existing_bags:
        if file_name in bag_name:
            print("removing", bag_name)
            os.remove(bag_name)

    existing_active_bags = glob.glob(dir + "*.bag.active")

    # remove existing bags for the same run/problem/case id
    for active_bag_name in existing_active_bags:
        if file_name in active_bag_name:
            print("removing", active_bag_name)
            os.remove(active_bag_name)

    return os.path.join(dir, file_name)


def get_txt_file_name(round, run, case):
    # rand = random.randint(0,10000000)
    return get_bag_file_name(round, run, case) + '.txt'

def choose_problem(run, batch):
    global problem_flag, scenario_in_unity, doshift_in_unity, road_width
    scenario_in_unity = 0
    doshift_in_unity = False
    road_width = None
    if run < 1 * batch:
        problem_flag = "16_8_16"
        scenario_in_unity = 3
        road_width = 8.0
    elif run < 2 * batch:
        problem_flag = "15_10_15"
        scenario_in_unity = 3
        road_width = 10.0
    elif run < 3 * batch:
        problem_flag = "14_12_14"
        scenario_in_unity = 3
        road_width = 12.0
    elif run < 4 * batch:
        problem_flag = "13_14_13"
        scenario_in_unity = 3
        road_width = 14.0
    elif run < 5 * batch:
        problem_flag = "12_16_12"
        scenario_in_unity = 3
        road_width = 16.0
    elif run < 6 * batch:
        problem_flag = "tri_10_10"
        scenario_in_unity = 4
        doshift_in_unity = 0
        road_width = 10.0
    elif run < 7 * batch:
        problem_flag = "tri_10_12"
        scenario_in_unity = 4
        doshift_in_unity = 1
        road_width = 10.0
    elif run < 8 * batch:
        problem_flag = "tri_12_12"
        scenario_in_unity = 4
        doshift_in_unity = 0
        road_width = 12.0
    elif run < 9 * batch:
        problem_flag = "tri_12_14"
        scenario_in_unity = 4
        doshift_in_unity = 1
        road_width = 12.0
    elif run < 10 * batch:
        problem_flag = "tri_14_14"
        scenario_in_unity = 4
        doshift_in_unity = 0
        road_width = 14.0
    elif run < 11 * batch:
        problem_flag = "tri_14_16"
        scenario_in_unity = 4
        doshift_in_unity = 1
        road_width = 14.0
    # test maps
    elif run < 12 * batch:
        problem_flag = "test_map_1"
        scenario_in_unity = 5
        doshift_in_unity = 0  # unknown
        road_width = 10.0  # unknown
    elif run < 13 * batch:
        problem_flag = "test_map_2"
        scenario_in_unity = 6
        doshift_in_unity = 0  # unknown
        road_width = 9.0  # unknown
    elif run < 14 * batch:
        problem_flag = "test_map_3"
        scenario_in_unity = 7
        doshift_in_unity = 0  # unknown
        road_width = 8.0  # unknown


def get_num_cases_in_problem(run, batch):
    num_cases = 0
    if run < 5 * batch:
        num_cases = 2
    elif run < 11 * batch:
        num_cases = 4
    elif run < 12 * batch:
        num_cases = 3
    elif run < 13 * batch:
        num_cases = 2
    elif run < 14 * batch:
        num_cases = 3

    return num_cases


def reset_case_params():
    global goal_flag, goal_value, start_value, goal_value_flag, rot_angle
    global goal_x, goal_y, start_x, start_y

    # goal position
    goal_x = None
    goal_y = None
    goal_value = 19.9 - 2.0

    if config.test_mode:
        goal_value = 11.0

    goal_value_flag = None
    goal_flag = "19d9"

    # start position
    start_x = None
    start_y = None
    start_value = 18.0

    if config.test_mode:
        start_value = 15.0

    # start rotation
    rot_angle = None


def gen_start_shift_sidewise():
    """shift the car side-wise in the road"""
    global road_width
    half_road = road_width / 2.0
    car_half_width = 5.0 / 2.0
    shift_range = half_road - car_half_width - 0.3

    return float('%.1f' % random.uniform(-shift_range, shift_range))


def gen_goal_shift_sidewise():
    """shift the car side-wise in the road"""
    global road_width
    half_road = road_width / 2.0
    car_half_width = 5.0 / 2.0
    shift_range = half_road - car_half_width - 0.5

    print('shift_range', shift_range)
    return float('%.1f' % random.uniform(-shift_range, shift_range))


def gen_rand_shift_along(shift_start=0.0):
    """shift the car along the road"""
    global road_width, start_value
    half_road_length = (map_size - road_width) / 2.0
    car_front_length = 3.6
    shift_range = half_road_length - car_front_length - (map_size / 2.0 - start_value)

    print("gen_rand_shift_along: shift_range: %f" % shift_range)

    return float('%.1f' % random.uniform(shift_start, shift_range))


import numpy


def setup_case_params(run, batch, case, flag='data'):

    assert case >= 1
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag, road_width

    goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise = sample_rand_factors()

    if flag == 'test':
        rand_rot_noise = rand_rot_noise/1.5
        goal_shift_sidewise = goal_shift_sidewise/1.5
        # start_shift_along = 0.0
        start_shift_sidewise = start_shift_sidewise / 1.5

    # quad-cross cases
    if run < 5 * batch:
        quad_cross_params(case, goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise)
        # quad_cross_start_near_goal(case, rand_rot_noise, start_shift_along, start_shift_sidewise)
    # tri-cross cases
    elif run < 11 * batch:
        tri_cross_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise)
        # tri_cross_start_near_goal(case, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise)
    elif run < 12 * batch:
        test_1_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise)
    elif run < 13 * batch:
        test_2_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise)
    elif run < 14 * batch:
        test_3_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise)

    try:
        start_x = float('%.1f' % start_x)
        start_y = float('%.1f' % start_y)
        goal_x = float('%.1f' % goal_x)
        goal_y = float('%.1f' % goal_y)
        rot_angle = float('%.2f' % rot_angle)
    except Exception as e:
        print(e)
        print(start_x, start_x)
        print(start_y, start_y)
        print(goal_x, goal_x)
        print(goal_y, goal_y)
        exit(1)


def tri_cross_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag
    assert(case <= 4)

    if case == 1:
        start_x = -start_value + start_shift_along
        start_y = start_shift_sidewise
        goal_x = goal_value
        goal_y = 0.0 + goal_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_flag = 'west_east'
    elif case == 2:
        start_x = start_value - start_shift_along
        start_y = start_shift_sidewise
        goal_x = -goal_value
        goal_y = 0.0 + goal_shift_sidewise
        rot_angle = math.pi + rand_rot_noise
        goal_flag = 'east_west'
    elif case == 3:
        start_x = start_value - start_shift_along
        start_y = start_shift_sidewise
        rot_angle = math.pi + rand_rot_noise

        goal_y = goal_value
        goal_flag = 'east_north'

        if problem_flag == "tri_10_10":
            goal_x = 0.0 + goal_shift_sidewise
        elif problem_flag == "tri_12_12":
            goal_x = 0.0 + goal_shift_sidewise
        elif problem_flag == "tri_14_14":
            goal_x = 0.0 + goal_shift_sidewise
        elif problem_flag == "tri_10_12":
            goal_x = -7.0 + goal_shift_sidewise
        elif problem_flag == "tri_12_14":
            goal_x = -6.0 + goal_shift_sidewise
        elif problem_flag == "tri_14_16":
            goal_x = -5.0 + goal_shift_sidewise
    elif case == 4:
        half_map_size = map_size / 2.0
        shift_x = 0.0
        if doshift_in_unity:
            shift_x = (half_map_size - (road_width + 2.0) / 2.0) / 2.0  # for shifted tri - road

        start_pivot = [-shift_x, half_map_size]
        start_pivot = numpy.asarray(start_pivot)
        car_dir = [shift_x, -(half_map_size - road_width / 2.0)]
        car_dir = numpy.asarray(car_dir)

        start_y = start_value - start_shift_along
        start_ratio = math.fabs(start_y - math.fabs(start_pivot[1])) / math.fabs(car_dir[1])
        start_x = float(start_pivot[0] + start_ratio * car_dir[0])

        start_x += start_shift_sidewise

        goal_x = goal_value
        goal_y = goal_shift_sidewise
        goal_flag = 'north_east'

        rot_angle = math.atan2(car_dir[1], car_dir[0]) + rand_rot_noise


def test_1_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag
    assert(case <= 3)

    map_shift = 8.5

    if case == 1:
        start_x = -start_value + start_shift_along
        start_y = -3.0 + start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_x = 5.0 - map_shift + goal_shift_sidewise
        goal_y = goal_value
        goal_flag = 'west_north'
    elif case == 2:
        start_x = -start_value + start_shift_along
        start_y = -3.0 + start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_x = 14.49 - map_shift
        goal_y = -13.21
        goal_flag = 'west_south'
    elif case == 3:
        start_x = 14.49 - map_shift
        start_y = -13.21
        goal_x = -start_value # + 2.0
        goal_y = -3.0 + goal_shift_sidewise
        rot_angle = math.pi*0.75 + rand_rot_noise
        goal_flag = 'south_west'


def test_2_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag
    assert(case <= 2)
    if case == 1:
        start_x = -start_value + start_shift_along
        start_y = start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_x = 0.0 + goal_shift_sidewise
        goal_y = goal_value - 1.0
        goal_flag = 'west_north'
    elif case == 2:
        start_x = start_shift_sidewise
        start_y = start_value - start_shift_along
        rot_angle = -math.pi*0.5 + rand_rot_noise
        goal_x = -goal_value
        goal_y = goal_shift_sidewise
        goal_flag = 'north_west'


def test_3_params(case, goal_shift_sidewise, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag
    start_x = -start_value + start_shift_along
    start_y = start_shift_sidewise
    assert (case <= 3)
    if case == 1:
        goal_x = goal_value
        goal_y = 0.0 + goal_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_flag = 'west_east'
    elif case == 2:
        goal_x = 0.0 + goal_shift_sidewise
        goal_y = -goal_value
        rot_angle = 0.0 + rand_rot_noise
        goal_flag = 'west_south'
    elif case == 3:
        goal_x = 0.0 + goal_shift_sidewise
        goal_y = goal_value
        goal_flag = 'west_north'
        rot_angle = 0.0 + rand_rot_noise


def tri_cross_start_near_goal(case, rand_rot_noise, road_width, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag

    shift_start = 3.0
    start_shift_along = gen_rand_shift_along(shift_start=shift_start)
    print("2: start_shift_along: %f" % start_shift_along)
    start_shift_sidewise = start_shift_sidewise * 1.1
    scale = min(0.5, 0.3 * math.sqrt(math.sqrt(start_shift_along)))
    rand_rot_noise = get_rand_rot_noise(start_shift_sidewise, car_half_width=2.5, scale=scale)

    if case == 1:
        start_x = start_value - start_shift_along
        start_y = start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        # goal_flag = 'west_east'
    elif case == 2:
        start_x = -start_value + start_shift_along
        start_y = start_shift_sidewise
        rot_angle = math.pi + rand_rot_noise
        # goal_flag = 'east_west'
    elif case == 3:
        half_map_size = map_size / 2.0
        shift_x = 0.0
        if doshift_in_unity:
            shift_x = (half_map_size - (road_width + 2.0) / 2.0) / 2.0  # for shifted tri - road

        # start_pivot = [-shift_x, half_map_size]
        start_pivot = [0, road_width / 2.0]
        start_pivot = numpy.asarray(start_pivot)
        car_dir = [-shift_x, half_map_size - road_width / 2.0]
        car_dir = numpy.asarray(car_dir)

        start_y = start_value - start_shift_along
        start_ratio = math.fabs(start_y - math.fabs(start_pivot[1])) / math.fabs(car_dir[1])
        start_x = float(start_pivot[0] + start_ratio * car_dir[0])
        start_x += start_shift_sidewise
        rot_angle = math.atan2(car_dir[1], car_dir[0]) + rand_rot_noise
        # goal_flag = 'east_north'
    elif case == 4:
        start_x = start_value - start_shift_along
        start_y = start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        # goal_flag = 'north_east'


def quad_cross_params(case, goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag
    start_x = -start_value + start_shift_along
    start_y = start_shift_sidewise
    assert(case <= 3)
    if case == 1:
        goal_x = goal_value
        goal_y = 0.0 + goal_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        goal_flag = 'west_east'
    elif case == 2:
        goal_x = 0.0 + goal_shift_sidewise
        goal_y = -goal_value
        rot_angle = 0.0 + rand_rot_noise
        goal_flag = 'west_south'
    elif case == 3:
        goal_x = 0.0 + goal_shift_sidewise
        goal_y = goal_value
        goal_flag = 'west_north'
        rot_angle = 0.0 + rand_rot_noise


def quad_cross_start_near_goal(case, rand_rot_noise, start_shift_along, start_shift_sidewise):
    global start_x, start_y, goal_x, goal_y, rot_angle, goal_flag

    shift_start = 3.0
    start_shift_along = gen_rand_shift_along(shift_start=shift_start)
    print("2: start_shift_along: %f" % start_shift_along)
    start_shift_sidewise = start_shift_sidewise * 1.1
    scale = min(0.5, 0.3 * math.sqrt(math.sqrt(start_shift_along)))
    rand_rot_noise = get_rand_rot_noise(start_shift_sidewise, car_half_width=2.5, scale=scale)

    if case == 1:
        start_x = start_value - start_shift_along
        start_y = start_shift_sidewise
        rot_angle = 0.0 + rand_rot_noise
        # goal_flag = 'west_east'
    elif case == 2:
        start_x = start_shift_sidewise
        start_y = -start_value + start_shift_along
        rot_angle = -math.pi/2.0 + rand_rot_noise
        # goal_flag = 'west_south'
    elif case == 3:
        start_x = start_shift_sidewise
        start_y = start_value - start_shift_along
        rot_angle = math.pi/2.0 + rand_rot_noise
        # goal_flag = 'west_north'


def sample_rand_factors():
    # start shift
    start_shift_sidewise = gen_start_shift_sidewise()
    goal_shift_sidewise = gen_goal_shift_sidewise()
    start_shift_along = gen_rand_shift_along()
    rand_rot_noise = get_rand_rot_noise(start_shift_sidewise, car_half_width=0.0)
    print_rand_factors(goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise)
    # rand_rot_noise = 0.0 # debugging 
    start_shift_along = 0.0 
    return goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise


def print_rand_factors(goal_shift_sidewise, rand_rot_noise, start_shift_along, start_shift_sidewise):
    print("----------------- Random factors -----------------")
    print("start_shift_sidewise: %f" % start_shift_sidewise)
    print("start_shift_along: %f" % start_shift_along)
    print("goal_shift_sidewise: %f" % goal_shift_sidewise)
    print("rand_rot_noise: %f" % rand_rot_noise)
    print("----------------- Random factors -----------------")


import math


def get_rand_rot_noise(rand_shift_sidewise, car_half_width, scale=0.2):
    global road_width
    half_road_width = road_width / 2.0
    road_clearance = half_road_width - rand_shift_sidewise - car_half_width
    road_clearance = max(0.0, road_clearance)
    road_clearance = min(3.0, road_clearance)

    return float('%.2f' % random.uniform(-scale * math.sqrt(road_clearance), scale * math.sqrt(road_clearance)))


def print_setting():
    print("- Problem indian_cross_" + problem_flag)
    print("- Executable: " + executable)
    print("- Obstacle file: " + obstacle_file)
    print("- Pedestrian goal file: " + goal_file)
    print("- Car start: " + str(start_x) + "," + str(start_y))
    print("- Car goal: " + str(goal_x) + "," + str(goal_y))
    print(
        "- Scenaro, doshift, and roadedge: " + str(scenario_in_unity) + ", " + str(doshift_in_unity) + ", " + str(
            road_width))


def init_case_dirs():

    global subfolder, obstacle_file, goal_file, result_subfolder

    subfolder = 'map_' + problem_flag + '_case_' + goal_flag

    if cmd_args.baseline is not "":
        result_subfolder = path.join(root_path, 'result', cmd_args.baseline + '_baseline', subfolder)
    else:
        result_subfolder = path.join(root_path, 'result', subfolder)

    mak_dir(result_subfolder)
    mak_dir(result_subfolder + '_debug')
    obstacle_file = catkin_ws_path + "src/Maps/indian_cross_obs_" + problem_flag + ".txt"
    goal_file = catkin_ws_path + "src/Maps/indian_cross_goals_" + problem_flag + ".txt"
    print_setting()


def wait_for(seconds, proc, msg):
    wait_count = 0
    while check_process(proc, msg) is False:
        time.sleep(1)
        wait_count += 1

        if wait_count == seconds:
            break

    return check_process(proc, msg)


def launch_transforms(round, run, case):
    global shell_cmd
    # shell_cmd = 'rosrun tf static_transform_publisher ' + str(start_x) + ' ' + str(start_y) + ' 0.0 ' + str(
    #     rot_angle) + ' 0.0 0.0 /map /odom 10'

    trans_map_out = None
    trans_map_proc = None
    # trans_map_out = open(get_debug_file_name('trans_map_log', round, run, case), 'w')
    # if config.verbosity > 0:
    #     print(shell_cmd)

    # trans_map_proc = subprocess.Popen(shell_cmd.split(), stderr=trans_map_out, stdout=trans_map_out)
    
    shell_cmd = config.ros_pref+'rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 /base_link /laser_frame 10'
    trans_laser_out = open(get_debug_file_name('trans_laser_log', round, run, case), 'w')
    if config.verbosity > 0:
        print(shell_cmd)

    trans_laser_proc = subprocess.Popen(shell_cmd, shell=True, stdout=trans_laser_out, stderr=trans_laser_out)
    print("[INFO] Tranforms setup", end=', ', flush=True)

    # while check_process(trans_map_proc, '[launch] trans_map') is False:
    #     time.sleep(1)
    #     continue

    # wait_for(config.max_launch_wait, trans_map_proc, '[launch] trans_map')

    # while check_process(trans_laser_proc, '[launch] trans_laser') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, trans_laser_proc, '[launch] trans_laser')

    # global_proc_queue.append((trans_map_proc, "trans_map_proc", trans_laser_out))
    global_proc_queue.append((trans_laser_proc, "trans_laser_proc", trans_laser_out))

    return trans_map_proc, trans_laser_proc, trans_map_out, trans_laser_out


def launch_map(round, run, case):
    global shell_cmd

    shell_cmd = config.ros_pref+'rosrun map_server map_server ' + catkin_ws_path + 'src/Maps/indian_cross_' + problem_flag + '.yaml'

    map_server_out = open(get_debug_file_name('map_server_log', round, run, case), 'w')

    if config.verbosity > 0:
        print('')
        print(shell_cmd)
    map_server_proc = subprocess.Popen(shell_cmd, shell=True, stderr=map_server_out, stdout=map_server_out)
    print("Map setup", end=', ', flush=True)

    # while check_process(map_server_proc, '[launch] map_server') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, map_server_proc, '[launch] map_server')

    global_proc_queue.append((map_server_proc, "map_server_proc", map_server_out))

    return map_server_proc, map_server_out


def launch_python_scripts(round, run, case):
    global shell_cmd
    shell_cmd = 'python2.7 ' + root_path + '/Assets/Sensors/Scripts/ROS/OdometryROS.py' + \
                ' --time_scale ' + str.format("%.2f" % config.time_scale)
    odom_out = open(get_debug_file_name('odom_log', round, run, case), 'w')
    if config.verbosity > 0:
        print('')
        print(shell_cmd)
        print(get_debug_file_name('odom_log', round, run, case))

    odom_proc = subprocess.Popen(shell_cmd.split(), stdout=odom_out, stderr=odom_out)
    shell_cmd = 'python2.7 ' + catkin_ws_path + 'src/peds_unity_system/src/unity_connector.py'
    connector_out = open(get_debug_file_name('connector_log', round, run, case), 'w')
    if config.verbosity > 0:
        print(shell_cmd)
        print(get_debug_file_name('connector_log', round, run, case))

    connector_proc = subprocess.Popen(shell_cmd.split(), stdout=connector_out, stderr=connector_out)
    shell_cmd = 'python2.7 ' + catkin_ws_path + 'src/purepursuit_combined/purepursuit_NavPath_unity.py' + \
        ' --time_scale ' + str.format("%.2f" % config.time_scale) + ' --net ' + str(config.use_drive_net_mode)
    pursuit_out = open(get_debug_file_name('pursuit_log', round, run, case), 'w')
    if config.verbosity > 0:
        print(shell_cmd)
        print(get_debug_file_name('pursuit_log', round, run, case))

    pursuit_proc = subprocess.Popen(shell_cmd.split(), stdout=pursuit_out, stderr=pursuit_out)
    print("Python scripts setup", end=', ', flush=True)
    time.sleep(1)

    # while check_process(odom_proc, '[launch] odom') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, odom_proc, '[launch] odom')

    # while check_process(connector_proc, '[launch] connector') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, connector_proc, '[launch] connector')

    # while check_process(pursuit_proc, '[launch] pursuit') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, pursuit_proc, '[launch] pursuit')

    global_proc_queue.append((odom_proc, "odom_proc", odom_out))
    global_proc_queue.append((connector_proc, "connector_proc", connector_out))
    global_proc_queue.append((pursuit_proc, "pursuit_proc", pursuit_out))

    return odom_proc, connector_proc, pursuit_proc, odom_out, connector_out, pursuit_out

def launch_carla_simulator(round, run, case):
    global shell_cmd
    
    launch_summit = True

    if launch_summit:
        shell_cmd = 'DISPLAY= ./CarlaUE4.sh -opengl'
        if config.verbosity > 0:
            print('')
            print(shell_cmd)

        carla_proc = subprocess.Popen(shell_cmd, cwd=os.path.join(home, "summit/LinuxNoEditor"), shell = True)

        wait_for(config.max_launch_wait, carla_proc, '[launch] carla_engine')
        time.sleep(1)   

    shell_cmd = config.ros_pref+'roslaunch connector.launch port:=' + str(config.port)
    if config.verbosity > 0:
        print('')
        print(shell_cmd)
    summit_connector_proc = subprocess.Popen(shell_cmd, shell=True, 
        cwd=os.path.join(home, "catkin_ws/src/summit_connector/launch"))
    wait_for(config.max_launch_wait, summit_connector_proc, '[launch] summit_connector')
   
    crowd_out = open("Crowd_controller_log.txt", 'w')
    global_proc_queue.append((summit_connector_proc, "summit_connector_proc", crowd_out))

    return


def launch_unity_simulator(round, run, case):
    global shell_cmd
    shell_cmd = config.ros_pref+'roslaunch peds_unity_system ped_sim.launch obstacle_file_name:=' + obstacle_file + \
                ' goal_file_name:=' + goal_file + ' time_scale:=' + str.format("%.2f" % config.time_scale)

    ped_sim_out = open(get_debug_file_name('ped_sim_log', round, run, case), 'w')
    if config.verbosity > 0:
        print('')
        print(shell_cmd)
        print(get_debug_file_name('ped_sim_log', round, run, case))

    ped_sim_proc = subprocess.Popen(shell_cmd, shell=True,stderr=ped_sim_out, stdout=ped_sim_out)

    use_unity = True
    unity_proc, unity_out = None, None
    if use_unity:
        window_flag = ''
        if not config.show_window:
            window_flag = ' -batchmode -nographics'
            # window_flag = ' -batchmode -force-opengl'
        shell_cmd = root_path + '/' + executable + window_flag + ' -scenario ' + str(scenario_in_unity) \
                    + ' -startx ' + str(start_x) + ' -starty ' + str(start_y) + ' -doshift ' + str(doshift_in_unity) \
                    + ' -road ' + str(road_width) + ' -rot ' + str(rot_angle) \
                    + ' -time_scale ' + str.format("%.2f" % config.time_scale) \
                    + ' -maxgroupsize ' + str(4)


        unity_out = open(get_debug_file_name('unity_log', round, run, case), 'w')
        if config.verbosity > 0:
            print(shell_cmd)
            print(get_debug_file_name('unity_log', round, run, case))


        unity_proc = subprocess.Popen(shell_cmd.split(), stderr=unity_out, stdout=unity_out)

    print("Simulator setup", end=', ', flush=True)

    wait_for(config.max_launch_wait, ped_sim_proc, '[launch] ped_sim')

    if use_unity:
        wait_for(config.max_launch_wait, unity_proc, '[launch] unity')

    global_proc_queue.append((ped_sim_proc, "ped_sim_proc", ped_sim_out))

    if use_unity:
        global_proc_queue.append((unity_proc, "unity_proc", unity_out))

    return ped_sim_proc, unity_proc, ped_sim_out, unity_out


def launch_path_planner(round, run, case, flag='data'):
    global shell_cmd

    if run < 11:
        inflation_radius = 1.8
    else: # test cases
        inflation_radius = 2.8
    # inflation_radius = float('%.1f' % random.uniform(2.0, 6.0))
    cost_steering_deg = float('%.1f' % random.uniform(1, 4))
    if flag == 'test':
        # inflation_radius = 1.0
        cost_steering_deg = 30
    print('')
    print("----------------- Path planner params -----------------")
    print("inflation_radius: %f" % inflation_radius)
    print("cost_steering_deg: %f" % cost_steering_deg)
    print("----------------- Path planner params -----------------")

    shell_cmd = config.ros_pref+'roslaunch --wait ped_pathplan pathplan.launch inflation_radius:=' + str(inflation_radius) + \
                ' cost_steering_deg:=' + str(cost_steering_deg)

    path_planner_out = open(get_debug_file_name('path_planner_log', round, run, case), 'w')
    if config.verbosity > 0:
        print('')
        print(shell_cmd)
        print(get_debug_file_name('path_planner_log', round, run, case))

    path_planner_proc = subprocess.Popen(shell_cmd, shell=True,stderr=path_planner_out, stdout=path_planner_out)
    # path_planner_proc = subprocess.Popen(shell_cmd.split())

    # while check_process(path_planner_proc, '[launch] path_planner') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, path_planner_proc, '[launch] path_planner')

    global_proc_queue.append((path_planner_proc, "path_planner_proc", path_planner_out))
    
    return path_planner_proc, path_planner_out


def launch_record_bag(round, run, case):
    if config.record_bag:
        global shell_cmd
        if sim_mode is "unity":
            shell_cmd = config.ros_pref+'rosbag record /il_data /map /cmd_vel_to_unity -o ' \
                        + get_bag_file_name(round, run, case) + \
                        ' __name:=bag_record'
        elif sim_mode is "carla":
            shell_cmd = config.ros_pref+'rosbag record /il_data -o ' \
                        + get_bag_file_name(round, run, case) + \
                        ' __name:=bag_record'

        if config.verbosity > 0:
            print('')
            print(shell_cmd)

        record_proc = subprocess.Popen(shell_cmd, shell=True)
        print("Record setup")
        #
        # while check_process(record_proc, '[launch] record') is False:
        #     time.sleep(1)
        #     continue

        wait_for(config.max_launch_wait, record_proc, '[launch] record')

        if config.record_bag:
            global_proc_queue.append((record_proc, "record_proc", None))

        return record_proc
    else:
        return None


def launch_drive_net(round, run, case):

    global drive_net_proc_link
    if config.use_drive_net_mode == IMITATION:

        print("=========================1==========================")

        global shell_cmd
        shell_cmd = 'python3 test.py --batch_size 128 --lr 0.0001 --no_vin 0 --l_h 100 --vinout 28 --w 64 --fit all ' \
                    '--ssm 0.03 --goalx ' + str(goal_x) + ' --goaly ' + str(goal_y) + \
                    ' --modelfile ' + config.model

        # action_hybrid_unicorn_newdata

        drive_net_out = open(get_debug_file_name('drive_net_log', round, run, case), 'w')

        if config.verbosity > 0:
            print('')
            print(shell_cmd)
            print('drive_net_log: ', get_debug_file_name('drive_net_log', round, run, case))

        drive_net_proc = subprocess.Popen(shell_cmd.split(),
                                          cwd=os.path.join(home, nn_folder)
                                          , stderr=drive_net_out, stdout=drive_net_out)
        drive_net_proc_link = drive_net_proc
        time.sleep(1)
        print("Drive_net setup")

        # while check_process(drive_net_proc, '[launch] drive_net') is False:
        #     time.sleep(1)
        #     continue

        wait_for(config.max_launch_wait, drive_net_proc, '[launch] drive_net')

        global_proc_queue.append((drive_net_proc, 'drive_net_proc', drive_net_out))

        return drive_net_proc, drive_net_out
    else:
        return None, None


def launch_rviz(round, run, case):

    global shell_cmd
    shell_cmd = 'rviz'

    if config.verbosity > 0:
        print('')
        print(shell_cmd)

    rviz_out = open(get_debug_file_name('rviz_log', round, run, case), 'w')

    rviz_proc = subprocess.Popen(shell_cmd.split(), stdout=rviz_out, stderr=rviz_out)

    print("Rviz setup")

    # while check_process(rviz_proc, '[launch] rviz') is False:
    #     time.sleep(1)
    #     continue

    wait_for(config.max_launch_wait, rviz_proc, '[launch] rviz')

    global_proc_queue.append((rviz_proc, 'rviz_proc', rviz_out))
    
    return rviz_proc, rviz_out


def launch_pomdp_planner(round, run, case, drive_net_proc):
    global shell_cmd
    global monitor_worker
    pomdp_proc, rviz_out = None, None

    run_query_srvs = False
    if config.use_drive_net_mode == LETS_DRIVE:
        run_query_srvs = True
    
    run_query_val_srvs = False

    query_proc, query_out, query_val_proc, query_val_out = None, None, None, None
    if run_query_srvs:

        shell_cmd = 'python3 query.py --batch_size 128 --lr 0.0001 --no_vin 0 --l_h 100 --vinout 28 --w 64 --fit all ' \
                    '--ssm 0.03 --goalx ' + str(goal_x) + ' --goaly ' + str(goal_y) + \
                    ' --modelfile ' + 'trained_models/hybrid_unicorn.pth'

        query_out = open(get_debug_file_name('query_server_',round, run, case), 'w')

        if config.verbosity > 0:
            print(shell_cmd)
            print(get_debug_file_name('query_server_',round, run, case))

        query_proc = subprocess.Popen(shell_cmd.split(),
                                      stdout=query_out, stderr=query_out,
                                      cwd=catkin_ws_path + 'src/query_nn/src')
    if run_query_val_srvs:

        shell_cmd = "python3 query_val.py"
        query_val_out = open(get_debug_file_name('query_val_server_', round, run, case), 'w')

        if config.verbosity > 0:
            print(shell_cmd)
            print(get_debug_file_name('query_val_server_', round, run, case))

        query_val_proc = subprocess.Popen(shell_cmd.split(),
                                      stdout=query_val_out, stderr=query_val_out,
                                      cwd=catkin_ws_path + 'src/query_nn/src')

    shell_cmd = config.ros_pref+'roslaunch --wait crowd_pomdp_planner is_despot.launch ' \
                + 'goal_x:=' + str(goal_x) + ' goal_y:=' + str(goal_y) + \
                ' obstacle_file_name:=' + obstacle_file + ' goal_file_name:=' + goal_file + \
                ' gpu_id:=' + str(config.gpu_id) + \
                ' net:=' + str(config.use_drive_net_mode) + \
                ' time_scale:=' + str.format("%.2f" % config.time_scale) + \
                ' model_file_name:=' + config.model + \
                ' val_model_name:=' + config.val_model

    # net: 2 for lets_drive, 1 for imitation learning

    pomdp_out = open(get_txt_file_name(round, run, case), 'w')

    print("Search log %s" % pomdp_out.name)

    if config.verbosity > 0:
        print(shell_cmd)

    timeout_flag = False
    start_t = time.time()
    try:

        # pomdp_proc = subprocess.Popen(shell_cmd.split())

        pomdp_proc = subprocess.Popen(shell_cmd, shell=True,stdout=pomdp_out, stderr=pomdp_out)

        print('[INFO] POMDP planning...')

        global_proc_queue.append((pomdp_proc, "pomdp_proc", pomdp_out))

        monitor_subprocess(global_proc_queue)

        if config.use_drive_net_mode >= IMITATION and 'yuanfu' in home:
            rviz_proc, rviz_out = launch_rviz(round, run, case)

        # time.sleep(600)

        if config.test_mode:
            pomdp_proc.wait(timeout=int(120.0/config.time_scale))
        else:
            pomdp_proc.wait(timeout=int(120.0/config.time_scale))
        print("[INFO] waiting successfully ended", flush=True)

        monitor_worker.terminate()

        # time.sleep(180)


    except subprocess.TimeoutExpired:

        print("[INFO] timeout exception caught")
        if check_process(pomdp_proc, "pomdp_proc") is True:
            monitor_worker.terminate()

            os.kill(pomdp_proc.pid, signal.SIGTERM)
            elapsed_time = time.time() - start_t
            print('[INFO] POMDP planner timeout in %f s' % elapsed_time, flush=True)
            timeout_flag = True
    finally:

        if not timeout_flag:
            elapsed_time = time.time() - start_t
            print('[INFO] POMDP planner terminated normally in %f s' % elapsed_time)

        if run_query_srvs:
            print('[INFO] Terminating the nn query server')
            if check_process(query_proc, "terminate"):
                shell_cmd = "rosnode kill /nn_query_node"
                subprocess.call(shell_cmd, shell=True)
                query_out.close()
        if run_query_val_srvs:
            if check_process(query_val_proc, "terminate"):
                shell_cmd = "rosnode kill /val_nn_query_node"
                subprocess.call(shell_cmd, shell=True)
                query_val_out.close()

        # os.kill(query_proc.pid, signal.SIGTERM)

        check_process(record_proc, '[finally] record')
        
        if config.use_drive_net_mode == IMITATION:
            while check_process(drive_net_proc, "[finally] drive_net_proc"):
                time.sleep(1)

        print('End waiting for drive net.')

    # time.sleep(1)

    if config.use_drive_net_mode >= IMITATION:
        if rviz_out:
            if not rviz_out.closed:
                rviz_out.close()

    return pomdp_proc, pomdp_out


def kill_sub_processes(queue):
    if config.verbosity > 0:
        print('[DEBUG] kill sub-processes')

    clear_nodes(port=config.port)
    
    for proc, p_name, p_out in queue:

        if proc is not None:
            try:
                if "record_proc" in p_name and config.record_bag:
                    if check_process(proc, '[kill_sub_processes] record'):
                        kill_record_safe()
                elif "odom_proc" in p_name or "connector_proc" in p_name or "pursuit_proc" in p_name: 
                    os.kill(proc.pid, 9)
                elif "drive_net_proc" in p_name and config.use_drive_net_mode == IMITATION:
                    os.kill(proc.pid, 9)
                    subprocess.call('pkill --ns $$ rviz', shell=True)
                else:
                    os.kill(proc.pid, signal.SIGTERM)
            except Exception as e:
                print(e)
                print("exception when killing {} {}".format(p_name, proc.pid))

    time.sleep(1)

    for proc, p_name, p_out in queue:
        if config.verbosity > 0:
            print('[DEBUG] kill one more time with SIGKILL to ensure killing')

        if check_process(proc, p_name):
            subprocess.call('kill -9 ' + str(proc.pid), shell=True)

    subprocess.call('pkill -9 --ns $$ roslaunch', shell=True)
    subprocess.call('yes | '+ config.ros_pref+'rosclean purge', shell=True)

# print("debug open file count:")
    # subprocess.call('lsof -u `whoami` | wc -l')


def kill_record_safe():

    # shell_cmd = 'pkill -2 --ns $$ record'
    shell_cmd = config.ros_pref+"rosnode kill /bag_record"
    print("[INFO] Ending record: " + shell_cmd)
    subprocess.call(shell_cmd, shell=True)
    iskilled = False
    wait_counter = 0
    while not iskilled:
        try:
            return_code = subprocess.check_output('pgrep record', shell=True)
            if b'record' in return_code:
                if config.verbosity > 0:
                    print('wait for record to be killed')
                time.sleep(1)
                wait_counter += 1
                if config.verbosity > 0:
                    print('.', end='', flush=True)
            else:
                iskilled = True
                if config.verbosity > 0:
                    print('record killed')
            if wait_counter >= 10:
                break
        except Exception as e:
            print(e)
            iskilled = True
            if config.verbosity > 0:
                print('record killed')


def kill_ros():
    if config.verbosity > 0:
        print('[DEBUG] Kill ros')
    subprocess.call('pkill rosmaster', shell=True)
    subprocess.call('pkill roscore', shell=True)
    subprocess.call('pkill rosout', shell=True)


def close_output_files(queue):

    for proc, p_name, p_out in queue:
        if p_out is not None:
            if not p_out.closed:
                p_out.close()


def monitor_subprocess(queue):
    global monitor_worker

    monitor_worker = SubprocessMonitor(queue, 0)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    monitor_worker.daemon = True
    monitor_worker.start()

    if config.verbosity >0:
        print("SubprocessMonitor started")

    for (p_handle, p_name, p_out) in queue:
        check_process(p_handle, '[after monitor launch] ' + p_name)


def kill_inner_timer(Timeout_inner_monitor_pid):
    if config.verbosity > 0:
        print("[DEBUG] Killing inner timer")

    if check_process(Timeout_inner_monitor_pid, 'inner_timer'):
        os.kill(Timeout_inner_monitor_pid.pid, signal.SIGKILL)
        time.sleep(1)
    while check_process(Timeout_inner_monitor_pid, "inner_timer") is True:
        print("INNER TIMER ALIVE")


def kill_outter_timer(Timeout_monitor_pid):
    # kill timeout monitor when terminating:
    if config.verbosity > 0:
        print("[DEBUG] Killing outer timer")

    if check_process(Timeout_monitor_pid, 'outter_timer'):
        os.kill(Timeout_monitor_pid.pid, signal.SIGKILL)
        time.sleep(1)

    while check_process(Timeout_monitor_pid, "outter_timer") is True:
        print("OUTTER TIMER ALIVE")


import atexit


def pass_timers(inner_timer, outter_timer):
    global Timeout_monitor_pid, Timeout_inner_monitor_pid
    Timeout_monitor_pid = outter_timer
    Timeout_inner_monitor_pid = inner_timer


def exit_handler():
    print('My application is ending! Clearing process...')
    kill_inner_timer(Timeout_inner_monitor_pid)
    kill_outter_timer(Timeout_monitor_pid)
    clear_process(clear_outter=True, port = config.port)
    # subprocess.call(root_path + '/clear_process.sh', shell=True)


atexit.register(exit_handler)

if __name__ == '__main__':
    # Parsing training parameters
    cmd_args = parse_cmd_args()
    update_global_config(cmd_args)

    # start the timeout monitor in
    # background and pass the PID:
    pid = os.getpid()
    if config.verbosity > 0:
        print('[DEBUG] pid = ' + str(pid))
    Timeout_monitor_pid = timeout_monitor(pid)

    launch_ros()

    # @atexit.register
    # def goodbye():
    #     print("*******You are now leaving the run_data_collection.py sector*********")
    #     kill_sub_processes(global_proc_queue)

    exp_flag = 'test'
    start_run = 0 # 0: quad, 5: tri, 11: test 3, 12: test 2
    if config.test_mode:
        start_run = 1  # tesing
    batch = 1
    end_run = 1 * batch
    for round in range(config.start_round, config.end_round):
        for run in range(start_run, end_run):
            choose_problem(run, batch)

            num_cases =  1 # get_num_cases_in_problem(run, batch)
            start_case = 3 
            if config.test_mode:
                num_cases = 1  # testing

            # start_case = 3
            # num_cases = 3

            for case in range(start_case, num_cases + start_case):

                # if check_ros() is False:
                    # print("ROS IS BROKEN. RESTARTING")
                    # kill_ros()
                    # launch_ros()

                global_proc_queue.clear() # process monitor queue
                reset_case_params()

                setup_case_params(run, batch, case, exp_flag)

                if config.verbosity > 0:
                    print("[DEBUG] Launch inner timer")
                Timeout_inner_monitor_pid = inner_timeout_monitor()

                init_case_dirs()

                # to make all launches wait for roscore

                # trans_map_proc, trans_laser_proc, trans_map_out, trans_laser_out = launch_transforms(round, run, case)

                # map_server_proc, map_server_out = launch_map(round, run, case)

                if sim_mode is "unity":
                    odom_proc, connector_proc, pursuit_proc, odom_out, connector_out, pursuit_out = \
                        launch_python_scripts(round, run, case)

                    ped_sim_proc, unity_proc, ped_sim_out, unity_out = launch_unity_simulator(round, run, case)


                    path_planner_proc, path_planner_out = launch_path_planner(round, run, case, exp_flag)
                elif sim_mode is "carla":
                    launch_carla_simulator(round, run, case)
                
                record_proc = launch_record_bag(round, run, case)

                # launch_rviz(round, run, case)

                drive_net_proc = None
                if "imitation" in cmd_args.baseline:
                    drive_net_proc, drive_net_out = launch_drive_net(round, run, case)

                if "imitation" in cmd_args.baseline or "pomdp" in cmd_args.baseline or "lets-drive" in cmd_args.baseline:
                    pomdp_proc, pomdp_out = launch_pomdp_planner(round, run, case, drive_net_proc)

                kill_inner_timer(Timeout_inner_monitor_pid)

                close_output_files(global_proc_queue)
                print("[INFO] Finish data: sample_" + str(round) + '_' + str(run) + '_' + str(case))

                kill_sub_processes([]) # global_proc_queue
    
    kill_ros()

    kill_outter_timer(Timeout_monitor_pid)

    clear_process(port=config.port)
