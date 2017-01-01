# regularizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import numbers

from collection import add_to_collection, get_collection


_REGULARIZATION_LOSSES_KEYS = "regularization_losses"


__all__ = ["l1_regularizer",
           "l2_regularizer",
           "sum_regularizer",
           "apply_regularization",
           "add_regularization_loss",
           "get_regularization_loss"]


def add_regularization_loss(loss):
    add_to_collection(_REGULARIZATION_LOSSES_KEYS, loss)


def get_regularization_loss():
    loss_list = get_collection(_REGULARIZATION_LOSSES_KEYS)

    if not loss_list:
        return None

    return reduce(theano.tensor.add, loss_list)


def sum_regularizer(regularizer_list):
    regularizer_list = [reg for reg in regularizer_list if reg is not None]

    if not regularizer_list:
        return None

    def sum_reg(weights):
        regularizer_tensors = [reg(weights) for reg in regularizer_list]
        return reduce(theano.tensor.add, regularizer_tensors)

    return sum_reg


def apply_regularization(regularizer, weights_list):
    if not weights_list:
        raise ValueError("no weights to regularize")

    penalties = [regularizer(w) for w in weights_list]
    penalties = [p if p is not None else 0.0 for p in penalties]

    for p in penalties:
      if p.ndim != 0:
          raise ValueError("regularizer must return a scalar tensor")

    summed_penalty = reduce(theano.tensor.sum, penalties)
    add_regularization_loss(summed_penalty)
    return summed_penalty


def l1_regularizer(scale):
    if isinstance(scale, numbers.Integral):
        raise ValueError("scale cannot be an integer: %s" % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError("scale must >= 0")
        if scale == 0.:
            return lambda _: None

    def l1(weights, name=None):
        return scale * theano.tensor.sum(theano.tensor.abs_(weights))

    return l1


def l2_regularizer(scale):
    if isinstance(scale, numbers.Integral):
        raise ValueError("scale cannot be an integer: %s" % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError("scale must >= 0")
        if scale == 0.:
            return lambda _: None

    def l2(weights):
        return scale * 0.5 * theano.tensor.sum(weights ** 2)

    return l2
