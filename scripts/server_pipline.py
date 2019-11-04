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

    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='GPU to use')

    parser.add_argument('--port',
                                            type=int,
                                            default="",
                                            help='carla port')

    parser.add_argument('--sport',
                                            type=int,
                                            default="",
                                            help='carla port')

    config = parser.parse_args()

    if config.port is "":
        config.port = str(2000 + condig.gpu*1000)
        config.sport = config.port +1

    shell_cmd = "export SDL_VIDEODRIVER=offscreen"
    subprocess.call(shell_cmd, shell = True)

    shell_cmd = "export CUDA_VISIBLE_DEVICES=" + str(config.gpu)
    subprocess.call(shell_cmd, shell = True)

    shell_cmd = "bash " + os.path.expanduser("~/summit/LinuxNoEditor/CarlaUE4.sh") + " -carla-rpc-port={} -carla-streaming-port={}".format(config.port, config.sport)

    carla_proc = subprocess.Popen(shell_cmd.split())

    print("Ececuting: "+shell_cmd)
    time.sleep(1)

    shell_cmd = 'python launch_docker.py --port {} --gpu {}'.format(config.port, config.gpu)

    print("Ececuting: "+shell_cmd)
    docker_proc = subprocess.call(shell_cmd, shell = True)

    @atexit.register
    def goodbye():
        print "You are now leaving the Python sector."
        subprocess.call('pkill -P ' + str(carla_proc.pid), shell=True)
        os.kill(carla_proc.pid, signal.SIGKILL)




