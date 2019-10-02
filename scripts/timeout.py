from clear_process import clear_process
import sys
import time
import subprocess

if __name__ == '__main__':
	arg_len = len(sys.argv)
	assert(arg_len > 2)
	sleep_time = int(sys.argv[1])
	kill_id = int(sys.argv[2])
	
	time.sleep(sleep_time)

	print("timeout: terminate ego script")

	subprocess.call('kill ' + str(kill_id), shell=True)

	
