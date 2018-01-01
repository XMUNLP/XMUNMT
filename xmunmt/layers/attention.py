# coding=utf-8
# Copyright 2018 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from xmunmt.layers.nn import linear


def attention_bias(inputs, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        mask = inputs
        bias = (1.0 - mask) * inf
        return bias


def attention(query, memories, bias, hidden_size, cache=None, reuse=None,
              dtype=None, scope=None):
    """ Standard attention layer

    Args:
        query: A tensor with shape [batch, key_size]
        memories: A tensor with shape [batch, memory_size, key_size]
        bias: A tensor with shape [batch, memory_size]
        hidden_size: An integer
        cache: A dictionary of precomputed value
        reuse: A boolean value, whether to reuse the scope
        dtype: An optional instance of tf.DType
        scope: An optional string, the scope of this layer

    Return:
        A tensor with shape [batch, value_size] and a Tensor with
        shape [batch, memory_size]
    """

    with tf.variable_scope(scope or "attention", reuse=reuse,
                           values=[query, memories, bias], dtype=dtype):
        mem_shape = tf.shape(memories)
        key_size = memories.get_shape().as_list()[-1]

        if cache is None:
            k = tf.reshape(memories, [-1, key_size])
            k = linear(k, hidden_size, False, False, scope="k_transform")

            if query is None:
                return {"key": k}
        else:
            k = cache["key"]

        q = linear(query, hidden_size, False, False, scope="q_transform")
        k = tf.reshape(k, [mem_shape[0], mem_shape[1], hidden_size])

        hidden = tf.tanh(q[:, None, :] + k)
        hidden = tf.reshape(hidden, [-1, hidden_size])

        logits = linear(hidden, 1, False, False, scope="logits")
        logits = tf.reshape(logits, [-1, mem_shape[1]])

        if bias is not None:
            logits = logits + bias

        alpha = tf.nn.softmax(logits)

        outputs = {
            "value": tf.reduce_sum(alpha[:, :, None] * memories, axis=1),
            "weight": alpha
        }

    return outputs
