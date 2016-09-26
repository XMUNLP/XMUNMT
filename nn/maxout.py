# maxout.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from linear import linear
from utils import get_or_default
from config import config, option
from initializer import zeros_initializer, uniform_initializer


class maxout_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "linear"
    * maxpart: int, maxpart number, default 2
    * concat: bool, True to concate weights, False to use seperate weights
    * bias: config.option, set bias.use=True to use bias, set bias.initializer
            to set initializer
    * weight: config.option, output_major=True to change weigth matrix
              to [output_size, input_size], weight.initializer to set
              initializer
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "maxout")
        self.concat = get_or_default(kwargs, "concat", False)
        self.bias = option(use=True, initializer=zeros_initializer)
        self.weight = option(output_major=False,
                             initializer=uniform_initializer)


# maxout unit
# input_size: dimension of x
# output_size: dimension of y
class maxout:

    def __init__(self, input_size, output_size, maxpart=2,
                 config=maxout_config()):
        scope = config.scope
        k = maxpart

        transform = linear(input_size, output_size * k, config)

        def forward(inputs):
            z = transform(inputs)
            shape = list(z.shape)
            shape[-1] /= k
            shape += [k]

            z = z.reshape(shape)
            y = theano.tensor.max(z, len(shape) - 1)

            return y

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = transform.parameter

    def __call__(self, inputs):
        return self.forward(inputs)
