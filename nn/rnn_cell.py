# rnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from nn import feedforward
from ops import variable_scope


class rnn_cell(object):

    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("abstract method")

    @property
    def input_size(self):
        raise NotImplementedError("abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("abstract method")

    def zero_state(self, batch_size, dtype):
        raise NotImplementedError("abstract method")


class gru_cell(rnn_cell):

    def __init__(self, size):
        if not isinstance(size, (list, tuple)):
            raise ValueError("size argument must be [input_size, output_size]")

        input_size, output_size = size

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        self._size = (tuple(input_size), output_size)

    def __call__(self, inputs, state, scope=None):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = self.input_size
        output_size = self.output_size

        if len(inputs) != len(input_size):
            raise RuntimeError("unmatched elements: inputs and input_size")

        size = [list(input_size) + [output_size], output_size]

        with variable_scope(scope or "gru_cell"):
            new_inputs = list(inputs[:]) + [state]
            r = feedforward(new_inputs, size, False, scope="reset_gate")
            u = feedforward(new_inputs, size, False, scope="update_gate")
            new_inputs = list(inputs[:]) + [r * state]
            c = feedforward(new_inputs, size, True,
                            activation=theano.tensor.tanh, scope="candidate")

            new_state = (1.0 - u) * state + u * c

        return new_state, new_state

    @property
    def state_size(self):
        return self._size[1]

    @property
    def input_size(self):
        return self._size[0]

    @property
    def output_size(self):
        return self._size[1]

    def zero_state(self, batch_size, dtype=None):
        state_size = self.state_size
        raise theano.tensor.zeros([batch_size, state_size], dtype=dtype)
