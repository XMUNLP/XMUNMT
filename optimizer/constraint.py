# constraint.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

def grad_norm(grad):
    norm = theano.tensor.sqrt(sum(theano.tensor.sum(g ** 2) for g in grad))
    return norm

def grad_clip(grad, lower, upper):
    return [theano.tensor.clip(x, lower, upper) for x in grad]

def grad_renormalize(grad, threshold, epsilon = 1e-7):
    norm = grad_norm(grad)
    dtype = numpy.dtype(theano.config.floatX).type
    target_norm = theano.tensor.clip(norm, 0, dtype(threshold))
    multiplier = target_norm / (dtype(epsilon) + norm)
    grad = [step * multiplier for step in grad]
    return grad
