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
import copy
from timeit import default_timer as timer
import glob
import math, numpy
import rosgraph


home = expanduser("~")
root_path = os.path.join(home, 'driving_data')
if not os.path.isdir(root_path):
    os.makedirs(root_path)

ws_root = os.getcwd()
ws_root = os.path.dirname(ws_root)
ws_root = os.path.dirname(ws_root)
print("workspace root: {}".format(ws_root))

config = Namespace()

subfolder = ''
result_subfolder = ""
shell_cmd = ''

from multiprocessing import Process, Queue

global_proc_queue = []

pomdp_proc = None

monitor_worker = None

Timeout_inner_monitor_pid, Timeout_monitor_pid = None, None

NO_NET = 0
JOINT_POMDP = 1 
ROLL_OUT = 2


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
    config.ros_port = config.port + 111
    config.ros_master_url = "http://localhost:{}".format(config.ros_port)
    config.ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(config.ros_port)
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
        config.timeout = 11*120*4

    config.max_launch_wait = 10
    config.make = bool(cmd_args.make)
    config.debug= bool(cmd_args.debug)
    
    compile_mode = 'Release'
    if config.debug:
        compile_mode = 'Debug'
    
    if config.make:
        try:
            shell_cmds = ["catkin config --merge-devel",
                "catkin build --cmake-args -DCMAKE_BUILD_TYPE=" + compile_mode]
            for shell_cmd in shell_cmds:
                print(shell_cmd, flush=True)
                make_proc = subprocess.call(shell_cmd, cwd=ws_root, shell = True)

        except Exception as e:
            print(e)
            exit(12)

        print("make done")

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
        print("=> Running {} drive_mode".format(cmd_args.drive_mode))
        cmd_args.record = config.record_bag

    print('============== cmd_args ==============')
    print("port={}".format(config.port))
    print("ros_port={}".format(config.ros_port))
    print("ros command prefix: {}".format(config.ros_pref))

    print("summit map location: {}".format(config.summit_maploc))
    print("summit random seed: {}".format(config.random_seed))
    print("launch summit: {}".format(config.launch_summit))

    print("start_round: " + str(cmd_args.sround))
    print("end_round: " + str(cmd_args.eround))
    print("timeout: " + str(config.timeout))
    print("verbosity: " + str(cmd_args.verb))
    print("window: " + str(cmd_args.window))
    print("record: " + str(cmd_args.record))
    print("gpu id: " + str(cmd_args.gpu_id))
    print("time scale: " + str(cmd_args.t_scale))
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
    shell_cmd = "python timeout_inner.py " + str(int((config.eps_length + 10)/config.time_scale))
    toi_proc = subprocess.Popen(shell_cmd.split())

    if config.verbosity > 0:
        print(shell_cmd)

    return toi_proc


def check_ros():
    if rosgraph.is_master_online(master_uri=config.ros_master_url):
        return True
    else:
        if config.verbosity > 0:
            print('[DEBUG] ROS MASTER is OFFLINE')
        return False


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


def get_debug_file_name(flag, round, run):
    return result_subfolder + '_debug/' + flag + '_' + str(round) + '_' + str(
        run) + '.txt'


def get_bag_file_name(round, run):
    dir = result_subfolder

    file_name = 'pomdp_search_log-' + \
                str(round) + '_' + str(run) + '_pid-'+ str(os.getpid()) + '_r-'+ str(config.random_seed)

    existing_bags = glob.glob(dir + "*.bag")

    # remove existing bags for the same run
    for bag_name in existing_bags:
        if file_name in bag_name:
            print("removing", bag_name)
            os.remove(bag_name)

    existing_active_bags = glob.glob(dir + "*.bag.active")

    # remove existing bags for the same run
    for active_bag_name in existing_active_bags:
        if file_name in active_bag_name:
            print("removing", active_bag_name)
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


def wait_for(seconds, proc, msg):
    wait_count = 0
    while check_process(proc, msg) is False:
        time.sleep(1)
        wait_count += 1

        if wait_count == seconds:
            break

    return check_process(proc, msg)


def launch_summit_simulator(round, run):
    global shell_cmd 
    
    if config.launch_summit:
        shell_cmd = 'DISPLAY= ./CarlaUE4.sh -opengl'
        # shell_cmd = './CarlaUE4.sh -opengl'
        if config.verbosity > 0:
            print('')
            print(shell_cmd)

        summit_proc = subprocess.Popen(shell_cmd, cwd=os.path.join(home, "summit/LinuxNoEditor"), shell = True)

        wait_for(config.max_launch_wait, summit_proc, '[launch] summit_engine')
        time.sleep(3)   

    shell_cmd = config.ros_pref+'roslaunch connector.launch port:=' + \
    	str(config.port) + ' map_location:=' + str(config.summit_maploc) + \
        ' random_seed:=' + str(config.random_seed)

    if "gamma" in cmd_args.drive_mode:
        print("launching connector with GAMMA controller...")
        shell_cmd = shell_cmd + ' ego_control_mode:=gamma ego_speed_mode:=vel'
    else:
        shell_cmd = shell_cmd + ' ego_control_mode:=other ego_speed_mode:=vel'

    if config.verbosity > 0:
        print('')
        print(shell_cmd)
    summit_connector_proc = subprocess.Popen(shell_cmd, shell=True, 
        cwd=os.path.join(ws_root, "src/summit_connector/launch"))
    wait_for(config.max_launch_wait, summit_connector_proc, '[launch] summit_connector')
   
    crowd_out = open("Crowd_controller_log.txt", 'w')
    global_proc_queue.append((summit_connector_proc, "summit_connector_proc", crowd_out))

    return


def launch_record_bag(round, run):
    if config.record_bag:
        global shell_cmd
        shell_cmd = config.ros_pref+'rosbag record /il_data ' \
                    + '/local_obstacles /local_lanes -o ' \
                    + get_bag_file_name(round, run) + \
                    ' __name:=bag_record'

        if config.verbosity > 0:
            print('')
            print(shell_cmd)

        record_proc = subprocess.Popen(shell_cmd, shell=True)
        print("Record setup")

        wait_for(config.max_launch_wait, record_proc, '[launch] record')
        if config.record_bag:
            global_proc_queue.append((record_proc, "record_proc", None))

        return record_proc
    else:
        return None


def launch_pomdp_planner(round, run):
    global shell_cmd
    global monitor_worker
    pomdp_proc, rviz_out = None, None

    launch_file = 'planner.launch'
    if config.debug:
        launch_file = 'planner_debug.launch'

    shell_cmd = config.ros_pref + 'roslaunch --wait crowd_pomdp_planner ' + \
                launch_file + \
                ' gpu_id:=' + str(config.gpu_id) + \
                ' mode:=' + str(config.drive_mode) + \
                ' summit_port:=' + str(config.port) + \
                ' time_scale:=' + str.format("%.2f" % config.time_scale) + \
                ' map_location:=' + config.summit_maploc

    pomdp_out = open(get_txt_file_name(round, run), 'w')

    print("Search log %s" % pomdp_out.name)

    if config.verbosity > 0:
        print(shell_cmd)

    timeout_flag = False
    start_t = time.time()
    try:
        pomdp_proc = subprocess.Popen(shell_cmd, shell=True,stdout=pomdp_out, stderr=pomdp_out)
        print('[INFO] POMDP planning...')

        global_proc_queue.append((pomdp_proc, "pomdp_proc", pomdp_out))
        monitor_subprocess(global_proc_queue)
        pomdp_proc.wait(timeout=int(config.eps_length/config.time_scale))

        print("[INFO] waiting successfully ended", flush=True)
        monitor_worker.terminate()

    except subprocess.TimeoutExpired:
        print("[INFO] episode reaches full length {} s".format(config.eps_length/config.time_scale))
        if check_process(pomdp_proc, "pomdp_proc") is True:
            monitor_worker.terminate()
            os.kill(pomdp_proc.pid, signal.SIGTERM)
            elapsed_time = time.time() - start_t
            print('[INFO] POMDP planner exited in %f s' % elapsed_time, flush=True)
            timeout_flag = True
    finally:
        if not timeout_flag:
            elapsed_time = time.time() - start_t
            print('[INFO] POMDP planner exited in %f s' % elapsed_time)
        check_process(record_proc, '[finally] record')

    return pomdp_proc, pomdp_out


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

    start_run = 0 
    batch = 1
    end_run = 1 * batch
    for round in range(config.start_round, config.end_round):
        for run in range(start_run, end_run):
            global_proc_queue.clear() # process monitor queue

            if config.verbosity > 0:
                print("[DEBUG] Launch inner timer")
            Timeout_inner_monitor_pid = inner_timeout_monitor()

            init_case_dirs()

            launch_summit_simulator(round, run)
            
            record_proc = launch_record_bag(round, run)

            if "pomdp" in cmd_args.drive_mode or "gamma" in cmd_args.drive_mode or "rollout" in cmd_args.drive_mode:
                pomdp_proc, pomdp_out = launch_pomdp_planner(round, run)

            kill_inner_timer(Timeout_inner_monitor_pid)

            close_output_files(global_proc_queue)
            print("[INFO] Finish data: sample_{}_{}".format(round, run)) 
    
    kill_ros()

    kill_outter_timer(Timeout_monitor_pid)

    clear_process(port=config.port)

