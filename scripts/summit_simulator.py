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
        self.cmd_args = cmd_args
        self.config = config

        Args = collections.namedtuple('args', 'host port dataset num_car num_bike num_pedestrian seed collision'
                                              ' clearance_car clearance_bike clearance_pedestrian'
                                              ' lane_change_probability cross_probability stuck_speed stuck_duration')

        # Spawn meshes.
        self.args = Args(
            host='127.0.0.1',
            port=self.cmd_args.port,
            pyroport = self.cmd_args.port + 6100,
            dataset=self.config.summit_maploc,
            num_car=self.cmd_args.num_car,
            num_bike=self.cmd_args.num_bike,
            num_pedestrian=self.cmd_args.num_pedestrian,
            seed=-1,
            collision=True,
            clearance_car=7.0,
            clearance_bike=7.0,
            clearance_pedestrian=1.0,
            lane_change_probability=0.0,
            cross_probability=0.1,
            stuck_speed=0.2,
            stuck_duration=5.0)

    def run(self):
        if self.config.verbosity > 0:
            print_flush("[summit_simulator.py] spawning meshes")
        spawn_meshes.main(self.args)

        # Spawn imagery.
        if self.config.verbosity > 0:
            print_flush("[summit_simulator.py] spawning imagery")
        spawn_imagery.main(self.args)

        # Spawn crowd.
        if self.config.verbosity > 0:
            print_flush("[summit_simulator.py] Spawning crowd")
        gamma_crowd.main(self.args)
