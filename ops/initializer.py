# initializer.py
# used in variable, variable_scope or get_variable
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano


# usage:
# init = random_uniform_initializer()
# shape = [1, 2, 3]
# var = variable("var", shape, initializer=init)


def zeros_initializer(dtype=theano.config.floatX):
    def _initializer(shape, dtype=dtype):
        return numpy.zeros(shape).astype(dtype)
    return _initializer


def ones_initializer(dtype=theano.config.floatX):
    def _initializer(shape, dtype=dtype):
        return numpy.ones(shape).astype(dtype)

    return _initializer


def constant_initializer(value=0.0, dtype=theano.config.floatX):
    def _initializer(shape, dtype=dtype):
        return value * numpy.ones(shape).astype(dtype)
    return _initializer


def random_uniform_initializer(minval=0.0, maxval=1.0,
                               dtype=theano.config.floatX):
    def _initializer(shape, dtype=dtype):
        return numpy.random.uniform(minval, maxval, shape).astype(dtype)
    return _initializer


def uniform_unit_scaling_initializer(factor=1.0, dtype=theano.config.floatX):
    def _initializer(shape, dtype=dtype):
        scale_shape = shape
        input_size = 1.0

        for dim in scale_shape[:-1]:
            input_size *= float(dim)

        input_size = max(input_size, 1.0)
        max_val = numpy.sqrt(3 / input_size) * factor
        return numpy.random.uniform(-max_val, max_val, shape).astype(dtype)

    return _initializer

