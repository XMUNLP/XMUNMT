# maxout.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import config

from linear import linear
from utils import update_option, add_parameters

# maxout unit
# input_size: dimension of x
# output_size: dimension of y
# available options:
# 1. name: str, default 'maxout'
# 2. bias: boolean, True to use bias, False to not use bias
# 3. weight: boolean, True stands for Wx, False stands for xW
# 4. target: target device, default 'auto'
class maxout:

    def __init__(self, input_size, output_size, **option):
        opt = config.maxout_option()
        update_option(opt, option)

        k = opt['maxpart']
        name = opt['name']
        transform = linear(input_size, output_size * k, **opt)

        params = []
        add_parameters(params, name, *transform.parameter)

        def forward(inputs):
            z = transform(inputs)
            shape = list(z.shape)
            shape[-1] /= k
            shape += [k]

            z = z.reshape(shape)
            y = theano.tensor.max(z, len(shape) - 1)

            return y

        self.name = name
        self.option = option
        self.forward = forward
        self.parameter = params

    def __call__(self, inputs):
        return self.forward(inputs)
