# feedforward.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from linear import linear
from utils import get_or_default
from config import config, option
from initializer import uniform_initializer, zeros_initializer


class feedforward_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "linear"
    * activation: function, activation function, default sigmoid
    * concat: bool, True to concate weights, False to use seperate weights
    * multibias: bool, True to use bias per input, only works when
    *            concat = False
    * bias: config.option, set bias.use=True to use bias, set bias.initializer
            to set initializer
    * weight: config.option, output_major=True to change weigth matrix
              to [output_size, input_size], weight.initializer to set
              initializer
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "feedforward")
        self.activation = get_or_default(kwargs, "activation",
                                         theano.tensor.nnet.sigmoid)
        self.concat = get_or_default(kwargs, "concat", False)
        self.multibias = get_or_default(kwargs, "multibias", False)
        self.bias = option(use=True, initializer=zeros_initializer)
        self.weight = option(output_major=False,
                             initializer=uniform_initializer)


class feedforward:

    def __init__(self, input_size, output_size, config=feedforward_config()):
        scope = config.scope
        activation = config.activation

        transform = linear(input_size, output_size, config)

        def forward(x):
            y = transform(x)
            return activation(y)

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = transform.parameter

    def __call__(self, x):
        return self.forward(x)
