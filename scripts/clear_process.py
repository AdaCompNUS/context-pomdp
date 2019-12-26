import subprocess
import time
import psutil
import sys 

def clear_nodes(port=2000):
    ros_port = port + 111
    print("ros_port = {}".format(ros_port))
    ros_master_url = "http://localhost:{}".format(ros_port)
    ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(ros_port)

    shell_cmds = []
    shell_cmds.append("pkill -2 roslaunch")
    shell_cmds.append("pkill -2 record")

    print("[INFO] Ending nodes: ", shell_cmds)
    for shell_cmd in shell_cmds:
        subprocess.call(shell_cmd, shell=True)


def clear_process(clear_outter=False, port=2000):
    print("[INFO] clear_process")

    clear_nodes(port)    
    subprocess.call('pkill -9 CarlaUE4-Linux-', shell=True)
    time.sleep(1)
    subprocess.call('pkill -9 record', shell=True)

    subprocess.call('pkill -9 ped_pomdp', shell=True)
    subprocess.call('pkill -9 local_frame', shell=True)
    subprocess.call('pkill -9 vel_publisher', shell=True)
    subprocess.call('pkill -9 roslaunch', shell=True)
    subprocess.call('pkill -9 python2.7', shell=True)
    subprocess.call('pkill -9 python2', shell=True)
    subprocess.call('yes | rosclean purge', shell=True)
    subprocess.call('pkill -9 rosmaster', shell=True)
    subprocess.call('pkill -9 roscore', shell=True)
    subprocess.call('pkill -9 rosout', shell=True)
    subprocess.call('pkill rviz', shell=True)

    for p in psutil.process_iter(attrs=['pid', 'name']):
        cmd_args = ' '.join(p.cmdline())
        if 'timeout_inner.py' in cmd_args:
            pid = p.info['pid']
            print('found ' + cmd_args)
            subprocess.call('kill -9 ' + str(pid), shell=True)

        if clear_outter:
            if 'timeout.py' in cmd_args:
                pid = p.info['pid']
                print('found ' + cmd_args)
                subprocess.call('kill -9 ' + str(pid), shell=True)

        if 'sleep' in p.info['name']:
            pid = p.info['pid']
            print('found ' + p.info['name'])
            subprocess.call('kill -9 ' + str(pid), shell=True)


if __name__ == '__main__':
    port = 2000                                                          
    if len(sys.argv) > 1:                                                        
        port = int(sys.argv[1]) 
    clear_process(port=port)
