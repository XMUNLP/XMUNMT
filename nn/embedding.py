# embedding.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from utils import get_or_default
from config import config, option
from variable import variable, variable_scope
from initializer import uniform_initializer, zeros_initializer


class embedding_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "embedding"
    * initializer: function, use to initialize embedding
    * bias: config.option, set bias.use=True to use bias, set bias.initializer
            to select initializer
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "embedding")
        self.initializer = get_or_default(kwargs, "initializer",
                                          uniform_initializer)
        self.bias = option(use=True, initializer=zeros_initializer)


def embedding_lookup(params, ids):
    shape = list(ids.shape) + [-1]
    values = params[ids.flatten()]
    values = values.reshape(shape)

    return values


# embedding
# representing embedding layer
# num: number of entries
# dim: vector dimension
class embedding:

    def __init__(self, num, dim, config):
        dtype = config.dtype
        scope = config.scope
        initializer = config.initializer
        use_bias, b_initializer = tuple(config.bias)

        params = []

        with variable_scope(scope):
            emb = variable("embedding", (num, dim), initializer, dtype)
            params.append(emb)
            if use_bias:
                bias = variable("bias", (dim,), initializer, dtype)
                params.append(bias)

        def forward(indices):
            values = embedding_lookup(emb, indices)

            if not bias:
                return values
            else:
                return values + bias

        self.scope = scope
        self.config = config
        self.forward = forward
        self.parameter = params
        self.embedding = emb

    def __call__(self, indices):
        return self.forward(indices)
