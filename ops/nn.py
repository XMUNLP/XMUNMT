# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf


def linear(inputs, output_size, bias, concat=False, multibias=False,
           dtype=None, scope=None):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    input_size = [item.get_shape()[1].value for item in inputs]

    if len(inputs) != len(input_size):
        raise RuntimeError("unmatched elements found: inputs and input_size")

    results = []

    with tf.variable_scope(scope or "linear"):
        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(1, inputs)

            shape = [input_size, output_size]
            matrix = tf.get_variable("matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        if bias:
            shape = [output_size]
            if not multibias:
                bias = tf.get_variable("bias", shape, dtype=dtype)
                results.append(bias)
            else:
                for i in range(len(input_size)):
                    name = "bias_%d" % i
                    bias = tf.get_variable(name, shape, dtype=dtype)
                    results.append(bias)

    if len(results) == 1:
        return results[0]

    return reduce(tf.add, results)
