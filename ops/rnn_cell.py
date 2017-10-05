# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from nn import linear
from util import nest, state_size_with_prefix


class RNNCell(object):

    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("abstract method")

    def zero_state(self, batch_size, dtype):
        state_size = self.state_size

        if nest.is_sequence(state_size):
            state_size_flat = nest.flatten(state_size)
            flat_shape = [state_size_with_prefix(s, prefix=[batch_size])
                          for s in state_size_flat]
            zeros_flat = [tf.zeros(tf.stack(s), dtype=dtype)
                          for s in flat_shape]

            for s, z in zip(state_size_flat, zeros_flat):
                z.set_shape(state_size_with_prefix(s, prefix=[None]))

            zeros = nest.pack_sequence_as(structure=state_size,
                                          flat_sequence=zeros_flat)
        else:
            zeros_size = state_size_with_prefix(state_size, prefix=[batch_size])
            zeros = tf.zeros(tf.stack(zeros_size), dtype=dtype)
            zeros.set_shape(state_size_with_prefix(state_size, prefix=[None]))

        return zeros


class BasicRNNCell(RNNCell):

    def __init__(self, num_units, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell"):
            output = linear([inputs, state], self._num_units, True,
                            scope=scope)
            output = self._activation(output)
        return output, output


class GRUCell(RNNCell):

    def __init__(self, num_units, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell"):
            with tf.variable_scope("gates"):
                r_u = linear([inputs, state], 2 * self._num_units, True,
                             scope=scope)
                r, u = tf.split(r_u, 2, 1)
                r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
            with tf.variable_scope("candidate"):
                c = self._activation(linear([inputs, r * state],
                                            self._num_units, True,
                                            scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class BasicLSTMCell(RNNCell):

    def __init__(self, num_units, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell"):
            c, h = state
            concat = linear([inputs, h], 4 * self._num_units, True,
                            scope="gates")

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, 1)

            j = self._activation(j)
            new_c = c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * j
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = (new_c, new_h)

        return new_h, new_state


class DropoutWrapper(RNNCell):

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        if (isinstance(input_keep_prob, float) and
            not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between"
                             "0 and 1: %d" % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
            not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter output_keep_prob must be between"
                             "0 and 1: %d" % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if self._input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self._input_keep_prob, self._seed)

        output, new_state = self._cell(inputs, state, scope)

        if self._output_keep_prob < 1:
            output = tf.nn.dropout(output, self._output_keep_prob, self._seed)

        return output, new_state


class multi_rnn_cell(RNNCell):

    def __init__(self, cells, residual=None):
        if not cells:
            raise ValueError("must specify at least one cell")

        if residual and residual > len(cells):
            raise ValueError("residual connection unused")

        if not isinstance(cells, (list, tuple)):
            cells = [cells]

        self._cells = cells
        self._residual = residual

    def __call__(self, inputs, state, c_inputs=None, scope=None):
        if isinstance(inputs, (list, tuple)):
            raise ValueError("multi_rnn_cell only support one inputs, "
                             "consider concatenate multiple inputs into one")

        with tf.variable_scope(scope or "multi_rnn_cell"):
            pre_inp = None
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    cur_state = state[i]

                    if self._residual and i >= self._residual:
                        cur_inp = cur_inp + pre_inp

                    pre_inp = cur_inp

                    if c_inputs is not None:
                        if not isinstance(c_inputs, (list, tuple)):
                            c_inputs = [c_inputs]
                        cur_inp = [cur_inp] + list(c_inputs)
                        cur_inp = tf.concat(cur_inp, 1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size


class MultiRNNCell(RNNCell):

    def __init__(self, cells):
        if not cells:
            raise ValueError("Must specify at least one cell.")
        self._cells = cells

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "multi_rnn_cell"):
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    if not nest.is_sequence(state):
                        raise ValueError("Expected a tuple")
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states
