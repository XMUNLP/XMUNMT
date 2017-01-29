# rnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from dropout import dropout
from nn import linear, feedforward
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
        output_size = self.output_size
        return theano.tensor.zeros([batch_size, output_size], dtype=dtype)


class lstm_cell(rnn_cell):

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

        size = [list(input_size) + [output_size], 4 * output_size]

        with variable_scope(scope or "lstm"):
            c, h = state
            new_inputs = list(inputs[:]) + [h]
            concat = linear(new_inputs, size, True, concat=True, scope="gates")

            i, j, f, o = theano.tensor.split(concat, [output_size] * 4, 4, -1)

            j = theano.tensor.tanh(j)
            # input, forget, output gate
            i = theano.tensor.nnet.sigmoid(i)
            f = theano.tensor.nnet.sigmoid(f)
            o = theano.tensor.nnet.sigmoid(o)

            new_c = c * f + i * j
            # no output activation
            new_h = new_c * o
            new_state = (new_c, new_h)

        return new_h, new_state

    @property
    def state_size(self):
        return (self._size[1], self._size[1])

    @property
    def input_size(self):
        return self._size[0]

    @property
    def output_size(self):
        return self._size[1]

    def zero_state(self, batch_size, dtype=None):
        output_size = self.output_size
        n = len(self.state_size)
        size = [batch_size, output_size]
        state = [theano.tensor.zeros(size, dtype=dtype) for i in range(n)]
        return tuple(state)


class dropout_wrapper(rnn_cell):

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        if not isinstance(cell, rnn_cell):
            raise TypeError("cell must be an instance of rnn_cell")

        if input_keep_prob > 1.0 or input_keep_prob <= 0.0:
            raise ValueError("input_keep_prob must in range (0.0, 1.0]")

        if output_keep_prob > 1.0 or output_keep_prob <= 0.0:
            raise ValueError("output_keep_prob must in range (0.0, 1.0]")

        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    def __call__(self, inputs, state, scope=None):
        if self._input_keep_prob < 1.0:
            inputs = dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, scope)
        if self._output_keep_prob < 1.0:
            output = dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype=None):
        return self._cell.zero_state(batch_size, dtype)


class multi_rnn_cell(rnn_cell):

    def __init__(self, cells):
        if not cells:
            raise ValueError("must specify at least one cell")

        if not isinstance(cells, (list, tuple)):
            cells = [cells]

        self._cells = cells

    def __call__(self, inputs, state, c_inputs=None, scope=None):
        with variable_scope(scope or "multi_rnn_cell"):
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with variable_scope("cell_%d" % i):
                    cur_state = state[i]

                    if c_inputs:
                        if not isinstance(inputs, (list, tuple)):
                            cur_inp = [inputs]
                        if not isinstance(c_inputs, (list, tuple)):
                            c_inputs = [c_inputs]
                        cur_inp = list(cur_inp) + list(c_inputs)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def input_size(self):
        return self._cells[0].input_size

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype=None):
        cells = self._cells
        return tuple([cell.zero_state(batch_size, dtype) for cell in cells])
