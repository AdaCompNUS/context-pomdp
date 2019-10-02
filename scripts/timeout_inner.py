from clear_process import clear_process
import sys
import time


if __name__ == '__main__':
	arg_len = len(sys.argv)
	assert(arg_len > 1)
	sleep_time = int(sys.argv[1])

	time.sleep(sleep_time)

	print("timeout_inner: ./clear_process.sh")

	clear_process()