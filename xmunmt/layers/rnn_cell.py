# coding=utf-8
# Copyright 2017 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from xmunmt.layers.nn import linear


class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell

    Args:
        num_units: int, The number of units in the RNN cell.
        reuse: (optional) Python boolean describing whether to reuse
            variables in an existing scope.  If not `True`, and the existing
            scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                              scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                              scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")

            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class LSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, output_activation=None):
        super(LSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._output_activation = output_activation

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "lstm_cell"):
            c, h = state
            concat = linear([inputs, h], 4 * self._num_units, True,
                            scope="gates")

            i, j, f, o = tf.split(concat, 4, 1)

            j = self._activation(j)
            new_c = c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) * j

            if not self._output_activation:
                new_h = new_c * tf.nn.sigmoid(o)
            else:
                new_h = self._output_activation(new_c) * tf.nn.sigmoid(o)

            new_state = (new_c, new_h)

        return new_h, new_state

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units
