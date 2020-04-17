import atexit
import time
import os
import subprocess
import signal

summit_proc = None
docker_proc = None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--summit_dir',
                        type=str,
                        default='~/summit/',
                        help='root of the summit directory')

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
                        help='summit port')

    parser.add_argument('--sport',
                        type=int,
                        default=0,
                        help='summit port')

    parser.add_argument('--mode',
                        type=str,
                        default='joint_pomdp',
                        help='driving mode')

    parser.add_argument('--record',
                        type=int,
                        default=0,
                        help='record bags')

    config = parser.parse_args()

    if config.port == 0:
        config.port = 2000 + int(config.gpu)*1000
        config.sport = int(config.port) + 1

    shell_cmd = "export SDL_VIDEODRIVER=offscreen"
    subprocess.call(shell_cmd, shell = True)
    shell_prefix = "CUDA_VISIBLE_DEVICES=" + str(config.gpu) + " "

    # shell_cmd = "export CUDA_VISIBLE_DEVICES=" + str(config.gpu)
    # subprocess.call(shell_cmd, shell = True)
    # shell_prefix = "DISPLAY=" + " "

    summit_proc = None

    for trial in range(config.trials):
        shell_cmd = shell_prefix + "bash " + \
            os.path.expanduser(config.summit_dir + "CarlaUE4.sh") + \
            " -carla-rpc-port={} -carla-streaming-port={}".format(config.port, config.sport)
            # " -opengl -carla-rpc-port={} -carla-streaming-port={}".format(config.port, config.sport)

        summit_proc = subprocess.Popen(shell_cmd, shell = True, preexec_fn=os.setsid)

        @atexit.register
        def goodbye():
            print("Exiting server pipeline.")
            if summit_proc:
                os.killpg(os.getpgid(summit_proc.pid), signal.SIGKILL)

        print("Ececuting: "+shell_cmd)
        time.sleep(1)

        shell_cmd = 'python launch_docker.py --port {} --gpu {} --recordbag {} --mode {}'.format(
                config.port, config.gpu, config.record, config.mode)

        print("Executing: "+shell_cmd)
        docker_proc = subprocess.call(shell_cmd, shell = True)

        print("Docker exited, closing simulator.")
        os.killpg(os.getpgid(summit_proc.pid), signal.SIGKILL)


