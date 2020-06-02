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
                print_flush('[clear_process.py] Subprocess {} has died'.format(p_name))
            return False
    else:
        try:
            os.kill(p_handle.pid, 0)
        except Exception as e:
            print_flush('[clear_process.py] Subprocess {} has died'.format(p_name))
            return False

    if p_name == 'summit':
        summit_launched = True

    if verbosity > 1:
        print_flush('[clear_process.py] Subprocess {} pid={} is alive'.format(p_name, p_handle.pid))

    return True


def clear_queue(queue, other_than='nothing'):

    print_flush("[clear_process.py] clearing queue {}".format(queue))
    for proc, p_name, p_out in reversed(queue):
        if other_than in p_name:
            continue

        if check_process(proc, p_name) is False:
            continue

        print_flush("[clear_process.py] killing {}".format(p_name))
        try:
            patience = 0
            while check_process(proc, p_name):
                if patience < 2:
                    if p_name == 'summit':
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        parent = psutil.Process(proc.pid)
                        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                            child.kill()
                        parent.kill()
                        proc.communicate()
                else:
                    proc.kill()
                    proc.communicate()

                time.sleep(1)
                patience += 1
        except Exception as e:
            print_flush(e)

        # release output streams
        if p_out is not None:
            if not p_out.closed:
                p_out.close()


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
            print_flush('[clear_process.py] ROS MASTER is OFFLINE')
        return False


def kill_ros_nodes(ros_pref):
    try:
        clear_ros_log(ros_pref)

        cmd_arg = ros_pref + 'rosnode list | grep -v rosout | '
        cmd_arg += ros_pref + 'xargs rosnode kill'
        print_flush('[clear_process.py] ' + cmd_arg)
        subprocess.call(cmd_arg, shell=True)
    except Exception as e:
        print_flush(e)


def clear_ros_log(ros_pref):
    try:
        shell_cmds = [ros_pref + 'yes | rosclean purge']
        print_flush("[clear_process.py]  Cleaning ros: {}".format(shell_cmds))
        for shell_cmd in shell_cmds:
            subprocess.call(shell_cmd, shell=True)
    except Exception as e:
        print_flush(e)


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
        print_flush("[clear_process.py] SubprocessMonitor initialized")

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
            print_flush("[clear_process.py] SubprocessMonitor activated")
        while True:
            p_handle, p_name, p_out = self.next()

            if not check_process(p_handle, p_name):
                break

            if not check_ros(self.ros_master_url, self.verbosity):
                print_flush("[clear_process.py] roscore has died!!")
                break

            time.sleep(1)

        # after termination of loop
        if self.main_proc is not None:
            if check_process(self.main_proc, "main_proc"):
                print_flush("[clear_process.py] Killing main waiting process...")
                os.killpg(self.main_proc.pid, signal.SIGKILL)
        else:
            if self.verbosity > 1:
                print_flush("[clear_process.py] main_proc is None")

    # def terminate(self):
    #     print_flush("[clear_process.py] Terminating subprocess monitor")
    #     # self.clear()
    #     super(SubprocessMonitor, self).terminate()


if __name__ == '__main__':
    port = 2000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    def clear_all(ros_port):
        ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(ros_port)
        kill_ros_nodes(ros_pref)
        print_flush("[clear_process.py]  clear_process.py")

        subprocess.call('pkill -9 CarlaUE4-Linux-', shell=True)
        subprocess.call('pkill -9 ped_pomdp', shell=True)
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
