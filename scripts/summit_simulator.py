from multiprocessing import Process
import collections
import sys
from os.path import expanduser

summit_scripts = expanduser("~/summit/PythonAPI/examples")
sys.path.append(summit_scripts)
import gamma_crowd, spawn_imagery, spawn_meshes


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


class SimulatorAccessories(Process):
    def __init__(self, cmd_args, config):
        Process.__init__(self)
        self.verbosity = config.verbosity

        Args = collections.namedtuple('args', 'host port pyroport dataset num_car num_bike num_pedestrian seed collision'
                                              ' clearance_car clearance_bike clearance_pedestrian'
                                              ' speed_car speed_bike speed_pedestrian'
                                              ' lane_change_probability cross_probability stuck_speed stuck_duration')

        # Spawn meshes.
        self.args = Args(
            host='127.0.0.1',
            port=cmd_args.port,
            pyroport = config.pyro_port,
            dataset=config.summit_maploc,
            num_car=cmd_args.num_car,
            num_bike=cmd_args.num_bike,
            num_pedestrian=cmd_args.num_pedestrian,
            speed_car=4.0,
            speed_bike=2.0,
            speed_pedestrian=1.0,
            seed=-1,
            collision=False,
            clearance_car=7.0,
            clearance_bike=7.0,
            clearance_pedestrian=1.0,
            lane_change_probability=0.0,
            cross_probability=0.1,
            stuck_speed=0.2,
            stuck_duration=5.0)

    def run(self):
        if self.verbosity > 0:
            print_flush("[summit_simulator.py] spawning meshes")
        spawn_meshes.main(self.args)

        # Spawn imagery.
        if self.verbosity > 0:
            print_flush("[summit_simulator.py] spawning imagery")
        spawn_imagery.main(self.args)

        # Spawn crowd.
        if self.verbosity > 0:
            print_flush("[summit_simulator.py] Spawning crowd")
        gamma_crowd.main(self.args)
