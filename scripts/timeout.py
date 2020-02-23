import time
import subprocess
from multiprocessing import Process


class TimeoutMonitor(Process):
    def __init__(self, pid, timeout, name, verbosity=1):
        Process.__init__(self)
        self.monitor_pid = pid
        self.verbosity = verbosity
        self.timeout = timeout
        self.proc_name = name

    def run(self):
        time.sleep(self.timeout)
        subprocess.call('kill ' + str(self.monitor_pid), shell=True)
        print("timeout: terminate {} script".format(self.proc_name))
        self.terminate()
