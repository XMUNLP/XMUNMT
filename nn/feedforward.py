# feedforward.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import config

from linear import linear
from utils import update_option

# feedforward neural network
# y = f(Wx + b) or y = f(xW + b)
# input_size: dimension of x
# output_size: dimension of y
# available options:
# 1. name: str, default 'feedforward'
# 2. bias: boolean, True to use bias, False to not use bias
# 3. weight: boolean, True stands for Wx, False stands for xW
# 4. function: activation function, default: theano.tensor.nnet.sigmoid
# 5. target: target device, default 'auto'
class feedforward:

    def __init__(self, input_size, output_size, **option):
        opt = config.feedforward_option()
        update_option(opt, option)

        transform = linear(input_size, output_size, **opt)
        function = opt['function']

        def forward(x):
            y = transform(x)
            return function(y)

        self.name = opt['name']
        self.option = opt
        self.forward = forward
        self.parameter = transform.parameter

    def __call__(self, x):
        return self.forward(x)
