# initializer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


def uniform_initializer(shape, low=-0.05, high=0.05):
    return numpy.random.uniform(low, high, shape)


def zeros_initializer(shape):
    return numpy.zeros(shape)
