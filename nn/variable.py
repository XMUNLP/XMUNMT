# variable.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from utils import uniform_tensor

def variable(name, shape, initializer = uniform_tensor,
             target = 'auto', dtype = theano.config.floatX):
    var = initializer(shape).astype(dtype)
    # borrow does not work when using GPU
    if target != 'auto':
        var = theano.shared(var, name = name, borrow = True, target = target)
    else:
        var = theano.shared(var, name = name, borrow = True)

    return var
