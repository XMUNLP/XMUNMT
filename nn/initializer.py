# initializer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np


def uniform_initializer(shape, low=-0.08, high=0.08):
    return np.random.uniform(low, high, shape)


def zeros_initializer(shape):
    return np.zeros(shape)
