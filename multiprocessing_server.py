

import multiprocessing
from time import sleep


def run_in_multiprocess(number):
    print("hello world " + str(number))
    sleep(1)
    print("Done sleeping")


def success(done):
    def func(x):
        done[0]=True
    return func


done = [False]
available_processors = multiprocessing.cpu_count()
print("Number of processors available: " + str(available_processors))
pool = multiprocessing.Pool(processes=available_processors)


pool.map_async(run_in_multiprocess, [x for x in range(0,available_processors)],
               callback=success(done),
               error_callback=lambda err: print(err))

while not done[0]:
    print("Waiting")
    sleep(1)