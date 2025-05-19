import time

log_path = "/a.txt"
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
current_time_log = 'start time: {}'.format(current_time)

log_path = log_path.replace(".txt", f"_{current_time}.txt")

print(log_path)