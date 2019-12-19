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

    parser.add_argument('--trials',
                                            type=int,
                                            default=10,
                                            help='number of trials')

    parser.add_argument('--port',
                                            type=int,
                                            default=0,
                                            help='carla port')

    parser.add_argument('--sport',
                                            type=int,
                                            default=0,
                                            help='carla port')

    config = parser.parse_args()

    if config.port == 0:
        config.port = 2000 + int(config.gpu)*1000
        config.sport = int(config.port) + 1

    shell_cmd = "export SDL_VIDEODRIVER=offscreen"
    subprocess.call(shell_cmd, shell = True)

    shell_prefix = "CUDA_VISIBLE_DEVICES=" + str(config.gpu) + " "
    # subprocess.call(shell_cmd, shell = True)

    carla_proc = None



    for trial in range(config.trials):
        shell_cmd = shell_prefix + "bash " + os.path.expanduser("~/summit/LinuxNoEditor/CarlaUE4.sh") + " -carla-rpc-port={} -carla-streaming-port={}".format(config.port, config.sport)

        carla_proc = subprocess.Popen(shell_cmd, shell = True, preexec_fn=os.setsid)

        @atexit.register
        def goodbye():
            print "Exiting server pipeline."
            if carla_proc:
                os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)

        print("Ececuting: "+shell_cmd)
        time.sleep(1)

        shell_cmd = 'python launch_docker.py --port {} --gpu {}'.format(config.port, config.gpu)

        print("Ececuting: "+shell_cmd)
        docker_proc = subprocess.call(shell_cmd, shell = True)

        print "Docker exited, closing simulator."
        # subprocess.call('pkill -P ' + str(carla_proc.pid), shell=True)
        os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
        #time.sleep(3)


