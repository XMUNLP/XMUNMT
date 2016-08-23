# linear.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import config

from variable import variable
from utils import update_option, add_parameters, uniform_tensor

# linear map
# y = Wx + b or y = xW + b
# input_size: dimension of x
# output_size: dimension of y
# available options:
# 1. name: str, default 'linear'
# 2. bias: boolean, True to use bias, False not to use bias
# 3. weight: boolean, True stands for Wx, False stands for xW
# 4. variant: str, 'standard' or 'tied-weight'
# 4. target: target device, default 'auto'
class linear:

    def __init__(self, input_size, output_size, **option):
        opt = config.linear_option()
        update_option(opt, option)

        weight = []
        target = opt['target']
        variant = opt['variant']

        if not isinstance(opt['weight'], (list, tuple)):
            opt['weight'] = [opt['weight'], uniform_tensor]

        if not isinstance(opt['bias'], (list, tuple)):
            opt['bias'] = [opt['bias'], uniform_tensor]

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        transpose, initializer = opt['weight']

        # linear
        if variant == 'standard':
            for i, isize in enumerate(input_size):
                name = 'weight:' + str(i)

                if transpose:
                    shape = (output_size, isize)
                    w = variable(name, shape, initializer, target)
                else:
                    shape = (isize, output_size)
                    w = variable(name, shape, initializer, target)

                weight.append(w)
        else:
            isize = sum(input_size)

            if transpose:
                shape = (output_size, isize)
                w = variable('weight:0', shape, initializer, target)
            else:
                shape = (isize, output_size)
                w = variable('weight:0', shape, initializer, target)

            weight.append(w)

        name = opt['name']
        params = []

        add_parameters(params, name, *weight)

        bias, initializer = opt['bias']

        if bias:
            bias = variable('bias:0', (output_size,), initializer, target)
            add_parameters(params, name, bias)

        def forward(x):
            if not isinstance(x, (list, tuple)):
                x = [x]

            if len(x) != len(input_size):
                raise RuntimeError('unmatched inputs and weights')

            outs = []

            if variant == 'standard':
                for v, w in zip(x, weight):
                    if transpose:
                        outs.append(theano.dot(v, w.transpose()))
                    else:
                        outs.append(theano.dot(v, w))
            else:
                if len(x) == 1:
                    x = x[0]
                else:
                    x = theano.tensor.concatenate(x, -1)
                if transpose:
                    outs.append(theano.dot(x, weight[0].transpose()))
                else:
                    outs.append(theano.dot(x, weight[0]))

            if bias:
                outs.append(bias)

            y = theano.tensor.add(*outs)

            return y

        self.name = name
        self.option = opt
        self.parameter = params
        self.forward = forward

    def __call__(self, x):
        return self.forward(x)
