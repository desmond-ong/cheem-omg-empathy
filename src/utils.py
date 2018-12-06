import time
import math
import os


def time_since(start):
    s = time.time() - start
    m = math.floor(s / 60)
    s = s - m * 60
    return "{}m {:.0f}s".format(m, s)


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)