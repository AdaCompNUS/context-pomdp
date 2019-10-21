import atexit
import time
import os
import subprocess
import signal

carla_proc = None
docker_proc = None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--port',
                                            type=int,
                                            default="2000",
                                            help='carla port')

    parser.add_argument('--sport',
                                            type=int,
                                            default="2001",
                                            help='carla port')

    config = parser.parse_args()

    shell_cmd = "export SDL_VIDEODRIVER=offscreen"
    subprocess.call(shell_cmd, shell = True)

    shell_cmd = "bash " + os.path.expanduser("~/summit/LinuxNoEditor/CarlaUE4.sh") + " -carla-rpc-port={} -carla-streaming-port={}".format(config.port, config.sport)

    carla_proc = subprocess.Popen(shell_cmd.split())

    print("Ececuting: "+shell_cmd)
    time.sleep(1)

    shell_cmd = 'python launch_docker.py'

    print("Ececuting: "+shell_cmd)
    docker_proc = subprocess.call(shell_cmd, shell = True)

    @atexit.register
    def goodbye():
        print "You are now leaving the Python sector."
        subprocess.call('pkill -P ' + str(carla_proc.pid), shell=True)
        os.kill(carla_proc.pid, signal.SIGKILL)




