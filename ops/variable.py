# variable.py
# managed variables
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from collection import add_to_collection, get_collection

__all__ = [
    "variable",
    "global_variables",
    "trainable_variables"
]


_GLOBAL_VARIABLES_KEY = "variables"
_TRAINABLE_VARIABLES_KEY = "trainable_variables"


# a wrapper for theano.shared
def variable(initial_value=None, trainable=True, name=None,
             dtype=theano.config.floatX):
    global _TRAINABLE_VARIABLES
    global _ALL_VARIABLES

    if initial_value is None:
        raise ValueError("initial_value must not be None")

    if callable(initial_value):
        val = initial_value()
    else:
        val = initial_value

    var = theano.shared(val, name=name, borrow=True)

    if trainable:
        add_to_collection(_TRAINABLE_VARIABLES_KEY, var)

    add_to_collection(_GLOBAL_VARIABLES_KEY, var)

    return var


def global_variables():
    return get_collection(_GLOBAL_VARIABLES_KEY)


def trainable_variables():
    return get_collection(_TRAINABLE_VARIABLES_KEY)
