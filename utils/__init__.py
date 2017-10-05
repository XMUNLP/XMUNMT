# utils/__init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from tensorflow.python.util import nest


def add_if_not_exsit(option, key, value):
    if key not in option:
        option[key] = value


def get_or_default(opt, key, default):
    if key in opt:
        return opt[key]
    return default


def function(inputs, outputs, updates=None):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    fetches = nest.flatten(outputs)

    if updates:
        fetches.append(updates)

    def func(*values, **option):
        feed_dict = {}
        flat_inputs = nest.flatten(inputs)
        flat_values = nest.flatten(values)

        for inp, val in zip(flat_inputs, flat_values):
            feed_dict[inp] = val

        if "session" not in option:
            session = None
        else:
            session = option["session"]

        sess = session or tf.get_default_session()
        results = sess.run(fetches, feed_dict=feed_dict)

        if updates:
            results = results[:-1]

        results = nest.pack_sequence_as(outputs, results)

        if len(results) == 1:
            return results[0]

        return results

    return func
