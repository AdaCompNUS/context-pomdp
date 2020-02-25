import argparse
from argparse import Namespace
from os import path
import random
from os.path import expanduser
import glob
from clear_process import *
from timeout import TimeoutMonitor

from summit_simulator import print_flush,  SimulatorAccessories


home = expanduser("~")
root_path = os.path.join(home, 'driving_data')
if not os.path.isdir(root_path):
    os.makedirs(root_path)

ws_root = os.getcwd()
ws_root = os.path.dirname(ws_root)
ws_root = os.path.dirname(ws_root)
print_flush("workspace root: {}".format(ws_root))

config = Namespace()

subfolder = ''
result_subfolder = ""

global_proc_queue = []
pomdp_proc = None

NO_NET = 0
JOINT_POMDP = 1
ROLL_OUT = 2


def parse_cmd_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--t_scale',
                        type=float,
                        default=1.0,
                        help='Factor for scaling down the time in simulation (to search for longer time)')
    parser.add_argument('--make',
                        type=int,
                        default=0,
                        help='Make the simulator package')
    parser.add_argument('--drive_mode',
                        type=str,
                        default="",
                        help='Which drive_mode to run')
    parser.add_argument('--port',
                        type=int,
                        default=2000,
                        help='summit_port')
    parser.add_argument('--maploc',
                        type=str,
                        default="random",
                        help='map location in summit simulator')
    parser.add_argument('--rands',
                        type=int,
                        default=0,
                        help='random seed in summit simulator')
    parser.add_argument('--launch_sim',
                        type=int,
                        default=1,
                        help='choose whether to launch the summit simulator')
    parser.add_argument('--eps_len',
                        type=float,
                        default=70.0,
                        help='Length of episodes in terms of seconds')
    parser.add_argument('--monitor',
                        type=str,
                        default="data_monitor",
                        help='which data monitor to use: data_monitor or summit_dql')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='debug mode')
    parser.add_argument('--num-car',
                        default='20',
                        help='Number of cars to spawn (default: 20)',
                        type=int)
    parser.add_argument('--num-bike',
                        default='20',
                        help='Number of bikes to spawn (default: 20)',
                        type=int)
    parser.add_argument('--num-pedestrian',
                        default='20',
                        help='Number of pedestrians to spawn (default: 20)',
                        type=int)
    return parser.parse_args()


def update_global_config(cmd_args):
    # Update the global configurations according to command line
    print_flush("Parsing config")

    config.start_round = cmd_args.sround
    config.end_round = cmd_args.eround
    config.timeout = cmd_args.timeout
    config.verbosity = cmd_args.verb
    config.show_window = bool(cmd_args.window)
    config.record_bag = bool(cmd_args.record)
    config.gpu_id = int(cmd_args.gpu_id)
    config.port = int(cmd_args.port)
    config.ros_port = config.port + 111
    config.pyro_port = config.port + 6100
    config.ros_master_url = "http://localhost:{}".format(config.ros_port)
    config.ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(config.ros_port)
    config.ros_env = os.environ.copy()
    config.ros_env['ROS_MASTER_URI'] = 'http://localhost:{}'.format(config.ros_port)

    if 'random' in cmd_args.maploc:
        config.summit_maploc = random.choice(['meskel_square', 'magic', 'highway', 'chandni_chowk', 'shi_men_er_lu'])
    else:
        config.summit_maploc = cmd_args.maploc
    config.random_seed = cmd_args.rands
    if cmd_args.rands == -1:
        config.random_seed = random.randint(0, 10000000)

    config.launch_summit = bool(cmd_args.launch_sim)
    config.eps_length = cmd_args.eps_len

    config.monitor = cmd_args.monitor
    config.time_scale = float(cmd_args.t_scale)

    if config.timeout == 1000000:
        config.timeout = 11 * 120 * 4

    config.max_launch_wait = 10
    config.make = bool(cmd_args.make)
    config.debug = bool(cmd_args.debug)

    compile_mode = 'Release'
    if config.debug:
        compile_mode = 'Debug'

    if config.make:
        try:
            shell_cmds = ["catkin config --merge-devel",
                          "catkin build --cmake-args -DCMAKE_BUILD_TYPE=" + compile_mode]
            for shell_cmd in shell_cmds:
                print_flush('[run_data_collection.py] ' + shell_cmd)
                make_proc = subprocess.call(shell_cmd, cwd=ws_root, shell=True)

        except Exception as e:
            print_flush(e)
            exit(12)

        print_flush("[run_data_collection.py] make done")

    if "joint_pomdp" in cmd_args.drive_mode:
        config.drive_mode = JOINT_POMDP
    elif "gamma" in cmd_args.drive_mode:
        config.drive_mode = JOINT_POMDP
    elif "rollout" in cmd_args.drive_mode:
        config.drive_mode = ROLL_OUT
    elif "pomdp" in cmd_args.drive_mode:
        config.drive_mode = NO_NET
    if cmd_args.drive_mode is "":
        # not drive_mode
        config.drive_mode = JOINT_POMDP
        config.record_bag = 1

    if cmd_args.drive_mode is not "":
        print_flush("=> Running {} drive_mode".format(cmd_args.drive_mode))
        cmd_args.record = config.record_bag

    print_flush('============== [run_data_collection.py] cmd_args ==============')
    print_flush("port={}".format(config.port))
    print_flush("ros_port={}".format(config.ros_port))
    print_flush("ros command prefix: {}".format(config.ros_pref))

    print_flush("summit map location: {}".format(config.summit_maploc))
    print_flush("summit random seed: {}".format(config.random_seed))
    print_flush("launch summit: {}".format(config.launch_summit))

    print_flush("start_round: " + str(cmd_args.sround))
    print_flush("end_round: " + str(cmd_args.eround))
    print_flush("timeout: " + str(config.timeout))
    print_flush("verbosity: " + str(cmd_args.verb))
    print_flush("window: " + str(cmd_args.window))
    print_flush("record: " + str(cmd_args.record))
    print_flush("gpu id: " + str(cmd_args.gpu_id))
    print_flush("time scale: " + str(cmd_args.t_scale))
    print_flush('============== [run_data_collection.py] cmd_args ==============')


def mak_dir(path):
    if sys.version_info[0] > 2:
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def get_debug_file_name(flag, round, run):
    return result_subfolder + '_debug/' + flag + '_' + str(round) + '_' + str(
        run) + '.txt'


def get_bag_file_name(round, run):
    dir = result_subfolder

    file_name = 'pomdp_search_log-' + \
                str(round) + '_' + str(run) + '_pid-' + str(os.getpid()) + '_r-' + str(config.random_seed)

    existing_bags = glob.glob(dir + "*.bag")

    # remove existing bags for the same run
    for bag_name in existing_bags:
        if file_name in bag_name:
            print_flush("[run_data_collection.py] removing {}".format(bag_name))
            os.remove(bag_name)

    existing_active_bags = glob.glob(dir + "*.bag.active")

    # remove existing bags for the same run
    for active_bag_name in existing_active_bags:
        if file_name in active_bag_name:
            print_flush("[run_data_collection.py] removing {}".format(active_bag_name))
            os.remove(active_bag_name)

    return os.path.join(dir, file_name)


def get_txt_file_name(round, run):
    return get_bag_file_name(round, run) + '.txt'


def init_case_dirs():
    global subfolder, result_subfolder

    subfolder = config.summit_maploc
    if cmd_args.drive_mode is not "":
        result_subfolder = path.join(root_path, 'result', cmd_args.drive_mode + '_drive_mode', subfolder)
    else:
        result_subfolder = path.join(root_path, 'result', subfolder)

    mak_dir(result_subfolder)
    mak_dir(result_subfolder + '_debug')


def launch_ros():
    print_flush("[run_data_collection.py] Launching ros")
    sys.stdout.flush()
    cmd_args = "roscore -p {}".format(config.ros_port)
    if config.verbosity > 0:
        print_flush(cmd_args)
    ros_proc = subprocess.Popen(cmd_args.split(), env=config.ros_env)

    while check_ros(config.ros_master_url, config.verbosity) is False:
        time.sleep(1)

    if config.verbosity > 0:
        print_flush("[run_data_collection.py] roscore started")
    return ros_proc


def launch_summit_simulator(round, run, cmd_args):
    if config.launch_summit:
        shell_cmd = './CarlaUE4.sh -opengl'
        if config.verbosity > 0:
            print_flush('')
            print_flush('[run_data_collection.py] ' + shell_cmd)

        summit_proc = subprocess.Popen(shell_cmd,
                                       cwd=os.path.join(home, "summit/LinuxNoEditor"),
                                       env=dict(config.ros_env, DISPLAY=''),
                                       shell=True,
                                       preexec_fn=os.setsid)

        wait_for(config.max_launch_wait, summit_proc, 'summit')
        global_proc_queue.append((summit_proc, "summit", None))
        time.sleep(4)

    sim_accesories = SimulatorAccessories(cmd_args, config)
    sim_accesories.start()

    # ros connector for summit
    shell_cmd = 'roslaunch summit_connector connector.launch port:=' + \
                str(config.port) + ' pyro_port:=' + str(config.pyro_port) + \
                ' map_location:=' + str(config.summit_maploc) + \
                ' random_seed:=' + str(config.random_seed)
    if "gamma" in cmd_args.drive_mode:
        print_flush("launching connector with GAMMA controller...")
        shell_cmd = shell_cmd + ' ego_control_mode:=gamma ego_speed_mode:=vel'
    else:
        shell_cmd = shell_cmd + ' ego_control_mode:=other ego_speed_mode:=vel'
    if config.verbosity > 0:
        print_flush('[run_data_collection.py] ' + shell_cmd)
    summit_connector_proc = subprocess.Popen(shell_cmd.split(), env=config.ros_env,
                                             cwd=os.path.join(ws_root, "src/summit_connector/launch"))
    wait_for(config.max_launch_wait, summit_connector_proc, '[launch] summit_connector')
    global_proc_queue.append((summit_connector_proc, "summit_connector_proc", None))

    return sim_accesories


def launch_record_bag(round, run):
    if config.record_bag:
        shell_cmd = 'rosbag record /il_data ' \
                    + '/local_obstacles /local_lanes -o ' \
                    + get_bag_file_name(round, run) + \
                    ' __name:=bag_record'

        if config.verbosity > 0:
            print_flush('')
            print_flush('[run_data_collection.py] ' + shell_cmd)

        record_proc = subprocess.Popen(shell_cmd.split(), env=config.ros_env)
        print_flush("[run_data_collection.py] Record setup")

        wait_for(config.max_launch_wait, record_proc, '[launch] record')
        # if config.record_bag:
        #     global_proc_queue.append((record_proc, "record_proc", None))

        return record_proc
    else:
        return None


def launch_pomdp_planner(round, run):
    global monitor_worker
    pomdp_proc, rviz_out = None, None

    launch_file = 'planner.launch'
    if config.debug:
        launch_file = 'planner_debug.launch'

    shell_cmd = 'roslaunch --wait crowd_pomdp_planner ' + \
                launch_file + \
                ' gpu_id:=' + str(config.gpu_id) + \
                ' mode:=' + str(config.drive_mode) + \
                ' summit_port:=' + str(config.port) + \
                ' time_scale:=' + str.format("%.2f" % config.time_scale) + \
                ' map_location:=' + config.summit_maploc

    pomdp_out = open(get_txt_file_name(round, run), 'w')

    print_flush("=> Search log {}".format(pomdp_out.name))

    if config.verbosity > 0:
        print_flush('[run_data_collection.py] ' + shell_cmd)

    start_t = time.time()
    try:
        pomdp_proc = subprocess.Popen(shell_cmd.split(), env=config.ros_env, stdout=pomdp_out, stderr=pomdp_out)
        print_flush('[run_data_collection.py] POMDP planning...')

        # global_proc_queue.append((pomdp_proc, "main_proc", pomdp_out))
        monitor_subprocess(global_proc_queue)

        pomdp_proc.wait(timeout=int(config.eps_length / config.time_scale))

        print_flush("[run_data_collection.py] episode successfully ended")
    except subprocess.TimeoutExpired:
        print_flush("[run_data_collection.py] episode reaches full length {} s".format(config.eps_length / config.time_scale))
    finally:
        elapsed_time = time.time() - start_t
        print_flush('[run_data_collection.py] POMDP planner exited in {} s'.format(elapsed_time))

    return pomdp_proc, pomdp_out


def monitor_subprocess(queue):
    global monitor_worker

    monitor_worker.feed_queue(queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    monitor_worker.daemon = True
    monitor_worker.start()

    if config.verbosity > 0:
        print_flush("[run_data_collection.py] SubprocessMonitor started")


import atexit

if __name__ == '__main__':
    # Parsing training parameters
    cmd_args = parse_cmd_args()
    update_global_config(cmd_args)

    # start the timeout monitor in
    # background and pass the PID:
    pid = os.getpid()
    if config.verbosity > 0:
        print_flush('[run_data_collection.py] pid = ' + str(pid))

    monitor_worker = SubprocessMonitor(config.ros_port, config.verbosity)
    outter_timer = TimeoutMonitor(pid, int(config.timeout / config.time_scale),
                                  "ego_script_timer", config.verbosity)
    outter_timer.start()
    # ros_proc = launch_ros()


    def exit_handler():

        print_flush('[run_data_collection.py] is ending! Clearing ros nodes...')
        kill_ros_nodes(config.ros_pref)
        print_flush('[run_data_collection.py] is ending! Clearing Processes...')
        try:
            monitor_worker.terminate()
        except Exception as e:
            print_flush(e)
        try:
            sim_accesories.terminate()
        except Exception as e:
            print_flush(e)
        print_flush('[run_data_collection.py] is ending! Clearing timer...')
        try:
            outter_timer.terminate()
        except Exception as e:
            print_flush(e)
        print_flush('[run_data_collection.py] is ending! Clearing subprocesses...')
        clear_queue(global_proc_queue)
        print_flush('exit [run_data_collection.py]')

    atexit.register(exit_handler)

    start_run = 0
    batch = 1
    end_run = 1 * batch
    for round in range(config.start_round, config.end_round):
        for run in range(start_run, end_run):
            global_proc_queue.clear()  # process monitor queue

            init_case_dirs()

            sim_accesories = launch_summit_simulator(round, run, cmd_args)

            record_proc = launch_record_bag(round, run)

            if "pomdp" in cmd_args.drive_mode or "gamma" in cmd_args.drive_mode or "rollout" in cmd_args.drive_mode:
                pomdp_proc, pomdp_out = launch_pomdp_planner(round, run)

            print_flush("[run_data_collection.py] Finish data: sample_{}_{}".format(round, run))
            # kill_ros_nodes(config.ros_pref)
            # monitor_worker.terminate()
            # sim_accesories.terminate()
            # clear_queue(global_proc_queue, other_than='roscore')

    print_flush("[run_data_collection.py] End of run_data_collection script")
    #
    # monitor_worker.terminate()
    # outter_timer.terminate()
