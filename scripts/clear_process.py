import subprocess
import time
import psutil
import signal
import sys, os
import rosgraph
from multiprocessing import Process
summit_launched = False

def print_flush(msg):
    print(msg)
    sys.stdout.flush()


def check_process(p_handle, p_name, verbosity=1):
    global summit_launched

    if p_name == 'summit':
        try:
            os.killpg(p_handle.pid, 0)
        except Exception as e:
            if summit_launched:
                print_flush('Subprocess {} has died'.format(p_name))
            print_flush(e)
            return False
    else:
        # if not p_handle.is_alive():
        #     print_flush('Subprocess {} has died'.format(p_name))
        #     return False
        # if not psutil.pid_exists(p_handle.pid):
        #     print_flush('Subprocess {} has died'.format(p_name))
        #     return False
        try:
            os.kill(p_handle.pid, 0)
        except Exception as e:
            print_flush('Subprocess {} has died'.format(p_name))
            print_flush(e)
            return False

    if p_name == 'summit':
        summit_launched = True

    if verbosity > 1:
        # print_flush('Subprocess {} pid={} pgid={} is alive'.format(p_name, p_handle.pid, os.getpgid(p_handle.pid)))
        print_flush('Subprocess {} pid={} is alive'.format(p_name, p_handle.pid))

    return True


def clear_queue(queue, other_than='nothing'):

    print_flush("clearing queue {}".format(queue))
    for proc, p_name, p_out in reversed(queue):
        if other_than in p_name:
            continue

        print_flush("killing {}".format(p_name))
        proc.kill()
        proc.communicate()

        # if check_process(proc, p_name, verbosity=2) is False:
        #     continue

        # subprocess.call("kill -9 {}".format(proc.pid))
        # os.kill(proc.pid, signal.SIGKILL)
        if p_name == 'summit' or p_name == 'roscore':
            if check_process(proc, p_name, verbosity=2):
                os.killpg(proc.pid, signal.SIGKILL)
        # os.killpg(proc.pid, signal.SIGINT)
        # os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        # patience = 0
        while check_process(proc, p_name, verbosity=2) is True:
            proc.kill()
            proc.communicate()

            # print_flush("Warning: {} pid={} pgid={} cpid={} still alive after killing".format(
            #     p_name, proc.pid, os.getpgid(proc.pid), os.getgid()))
            # if patience == 0:
            #     # proc.kill()
            #     # os.kill(proc.pid, signal.SIGKILL)
            #     # os.killpg(proc.pid, signal.SIGKILL)
            #     os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            #     break
            # patience += 1
            time.sleep(1)

        # release output streams
        if p_out is not None:
            if not p_out.closed:
                p_out.close()

        # print_flush("{} killed".format(p_name))


def wait_for(seconds, proc, msg):
    wait_count = 0
    while check_process(proc, msg) is False:
        time.sleep(1)
        wait_count += 1

        if wait_count == seconds:
            break

    return check_process(proc, msg)


def check_ros(url, verbosity):
    if rosgraph.is_master_online(master_uri=url):
        return True
    else:
        if verbosity > 0:
            print_flush('[DEBUG] ROS MASTER is OFFLINE')
        return False


def kill_ros_nodes(ros_pref):
    subprocess.call(ros_pref + ' rosnode kill -a', shell=True)
    time.sleep(2)
    clear_ros_log(ros_pref)


def clear_ros_log(ros_pref):
    shell_cmds = [ros_pref + 'yes | rosclean purge']
    print_flush("[INFO] Cleaning ros: {}".format(shell_cmds))
    for shell_cmd in shell_cmds:
        subprocess.call(shell_cmd, shell=True)


class SubprocessMonitor(Process):
    def __init__(self, ros_port, verbosity=1):
        Process.__init__(self)
        self.queue = []
        self.queue_iter = 0

        self.ros_port = ros_port
        self.ros_master_url = "http://localhost:{}".format(self.ros_port)
        self.ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(self.ros_port)

        self.main_proc = None
        self.verbosity = verbosity
        print_flush("SubprocessMonitor initialized")

    def feed_queue(self, queue):
        self.queue = self.queue + queue
        for (p_handle, p_name, p_out) in self.queue:
            if "main_proc" in p_name:
                self.main_proc = p_handle

    def next(self):
        if self.queue_iter >= len(self.queue):
            self.queue_iter = 0
        p_handle, p_name, p_out = self.queue[self.queue_iter]
        self.queue_iter += 1
        return p_handle, p_name, p_out

    def run(self):
        time.sleep(1)

        if self.verbosity > 0:
            print_flush("[DEBUG] SubprocessMonitor activated")
        while True:
            p_handle, p_name, p_out = self.next()

            if not check_process(p_handle, p_name):
                break

            if not check_ros(self.ros_master_url, self.verbosity):
                print_flush("roscore has died!!")
                break

            time.sleep(1)

        # after termination of loop
        if self.main_proc is not None:
            if check_process(self.main_proc, "main_proc"):
                print_flush("Killing main waiting process...")
                os.killpg(self.main_proc.pid, signal.SIGKILL)
        else:
            if self.verbosity > 0:
                print_flush("[DEBUG] main_proc is None")

    def terminate(self):
        print_flush("Terminating subprocess monitor")
        # self.clear()
        super(SubprocessMonitor, self).terminate()


if __name__ == '__main__':
    port = 2000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    def clear_all(ros_port):
        ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(ros_port)
        kill_ros_nodes(ros_pref)
        clear_ros_log(ros_pref)
        print_flush("[INFO] clear_process.py")

        subprocess.call('pkill -9 CarlaUE4-Linux-', shell=True)
        time.sleep(1)
        subprocess.call('pkill -9 record', shell=True)
        subprocess.call('pkill -9 python', shell=True)

        subprocess.call('yes | rosclean purge', shell=True)
        subprocess.call('pkill -9 roslaunch', shell=True)
        subprocess.call('pkill -9 rosmaster', shell=True)
        subprocess.call('pkill -9 roscore', shell=True)
        subprocess.call('pkill -9 rosout', shell=True)
        subprocess.call('pkill rviz', shell=True)

    clear_all(str(port + 111))
