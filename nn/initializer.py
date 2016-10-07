# initializer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


def uniform_initializer(low=-0.08, high=0.08):
    def initializer(shape, dtype):
        return numpy.random.uniform(low, high, shape).astype(dtype)
    return initializer


def zeros_initializer():
    def initializer(shape, dtype):
        return numpy.zeros(shape).astype(dtype)
    return initializer
