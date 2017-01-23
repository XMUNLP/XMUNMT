# function.py
# a wrapper for Theano's function
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import utils
import theano


def function(inputs, outputs, use_extension=True, **kwargs):
    if not use_extension:
        return theano.function(inputs, outputs, **kwargs)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
        post_proc = True
    else:
        post_proc = False

    nest_outputs = outputs
    flat_inputs = utils.flatten(inputs)
    flat_outputs = utils.flatten(outputs)

    fn = theano.function(flat_inputs, flat_outputs, **kwargs)

    def wrapper(*inputs):
        inputs = utils.flatten(inputs)
        outputs = fn(*inputs)

        if post_proc:
            return outputs[0]

        return utils.pack_sequence_as(nest_outputs, outputs)

    return wrapper
