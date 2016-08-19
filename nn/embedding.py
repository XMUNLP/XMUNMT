# embedding.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import config

from variable import variable
from utils import update_option, add_parameters, uniform_tensor

# embedding
# representing embedding layer
# num: number of entries
# dim: vector dimension
# available options:
# 1. name: str, default 'embedding'
# 2. bias: boolean, True to use bias, False to not use bias, default False
# 3. init: initialize embedding
# 4. target: target device, default 'auto'
class embedding:

    def __init__(self, num, dim, **option):
        opt = config.embedding_option()
        update_option(opt, option)

        name = opt['name']
        init = opt['init']
        target = opt['target']
        params = []

        if not init:
            init = uniform_tensor

        emb = variable('embedding:0', (num, dim), init, target)
        add_parameters(params, name, emb)

        if not isinstance(opt['bias'], (list, tuple)):
            opt['bias'] = [opt['bias'], uniform_tensor]

        bias, init = opt['bias']

        if bias:
            bias = variable('bias:0', (dim,), init, target)
            add_parameters(params, name, bias)

        def forward(indices):
            shape = list(indices.shape) + [-1]
            values = emb[indices.flatten()]
            values = values.reshape(shape)

            if not bias:
                return values
            else:
                return values + bias

        self.name = name
        self.option = opt
        self.forward = forward
        self.parameter = params

    def __call__(self, indices):
        return self.forward(indices)
