# regularizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

from collection import add_to_collection, get_collection


_REGULARIZATION_LOSSES_KEYS = "regularization_losses"


def add_regularization_loss(loss):
    add_to_collection(_REGULARIZATION_LOSSES_KEYS, loss)


def get_regularization_loss():
    return get_collection(_REGULARIZATION_LOSSES_KEYS)


def l1_regularizer(scale, scope=None):
    raise NotImplemented("currently not available")


def l2_regularizer(scale, scope=None):
    raise NotImplemented("currently not available")
