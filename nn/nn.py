# nn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from ops import variable_scope, get_variable


def embedding_lookup(params, ids):
    shape = list(ids.shape) + [-1]
    values = params[ids.flatten()]
    values = values.reshape(shape)

    return values


# size: [input_size, output_size]
def linear(inputs, size, bias, concat=False, dtype=None, scope=None):
    if not isinstance(size, (list, tuple)):
        raise ValueError("size argument must be (input_size, output_size)")

    input_size, output_size = size

    if not isinstance(input_size, (list, tuple)):
        input_size = [input_size]

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if len(inputs) != len(input_size):
        raise RuntimeError("unmatched elements found: inputs and input_size")

    results = []

    with variable_scope(scope):
        if concat:
            input_size = sum(input_size)
            inputs = theano.tensor.concatenate(inputs, -1)

            shape = [input_size, output_size]
            matrix = get_variable("matrix", shape, dtype=dtype)
            results.append(theano.dot(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = get_variable(name, shape, dtype=dtype)
                results.append(theano.dot(inputs[i], matrix))

        if bias:
            shape = [output_size]
            bias = get_variable("bias", shape, dtype=dtype)
            results.append(bias)

    if len(results) == 1:
        return results[0]

    return reduce(theano.tensor.add, results)


def feedforward(inputs, size, bias, activation=theano.tensor.nnet.sigmoid,
                concat=False, dtype=None, scope=None):
    scope = scope or "feedforward"
    return activation(linear(inputs, size, bias, concat, dtype, scope))


def maxout(inputs, size, maxpart, bias, concat=False, dtype=None, scope=None):
    scope = scope = "maxout"
    size[-1] = size[-1] * maxpart

    output = linear(inputs, size, bias, concat, dtype, scope)
    shape = list(output.shape)
    shape[-1] /= maxpart
    shape += [maxpart]
    output = output.reshape(shape)
    output = theano.tensor.max(output, len(shape) - 1)

    return output
