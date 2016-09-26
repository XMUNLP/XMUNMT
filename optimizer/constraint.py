# constraint.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano


def global_norm(grad):
    norm = theano.tensor.sqrt(sum(theano.tensor.sum(g ** 2) for g in grad))
    return norm


def clip_by_value(grad, lower, upper):
    return [theano.tensor.clip(x, lower, upper) for x in grad]


def clip_by_global_norm(grad, threshold, epsilon = 1e-7):
    norm = global_norm(grad)
    dtype = numpy.dtype(theano.config.floatX).type
    target_norm = theano.tensor.clip(norm, 0, dtype(threshold))
    multiplier = target_norm / (dtype(epsilon) + norm)
    grad = [step * multiplier for step in grad]
    return grad
