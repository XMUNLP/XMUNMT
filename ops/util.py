# utils.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape


__all__ = [
    "nest",
    "on_device",
    "infer_state_dtype",
    "state_size_with_prefix"
]


def on_device(fn, device):
    if device:
        with tf.device(device):
            return fn()
    else:
        return fn()


def infer_state_dtype(explicit_dtype, state):
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError("state has tensors of different inferred_dtypes."
                             "Unable to infer a single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype


def state_size_with_prefix(state_size, prefix=None):
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be"
                            "a list.")
        result_state_size = prefix + result_state_size
    return result_state_size
