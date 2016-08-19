# updates.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
from collections import OrderedDict

def apply_momentum(updates, params, momentum):
    sharedvars = []
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow = True)
        var = numpy.zeros(value.shape, dtype = value.dtype)
        velocity = theano.shared(var, broadcastable = param.broadcastable)
        sharedvars.append(velocity)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return sharedvars, updates

def apply_nesterov_momentum(updates, params, momentum):
    sharedvars = []
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        var = numpy.zeros(value.shape, dtype=value.dtype)
        velocity = theano.shared(var, broadcastable = param.broadcastable)
        sharedvars.append(velocity)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return sharedvars, updates

def sgd_updates(params, grads, lr):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        updates[p] = p - lr * g

    return [], updates

def adagrad_updates(params, grads, lr, epsilon):
    sharedvars = []
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        accu = theano.shared(var, broadcastable = param.broadcastable)
        accu_new = accu + (grad ** 2)
        delta = lr * grad / theano.tensor.sqrt(accu_new + epsilon)
        sharedvars.append(accu)
        updates[accu] = accu_new
        updates[param] = param - delta

    return sharedvars, updates

def rmsprop_updates(params, grads, lr, rho, epsilon, variant = 'hinton'):
    sharedvars = []
    updates = OrderedDict()

    if variant == 'hinton':
        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            var = numpy.zeros_like(value)
            accu = theano.shared(var, broadcastable = param.broadcastable)
            accu_new = rho * accu + (1 - rho) * grad ** 2
            delta = lr * grad / (theano.tensor.sqrt(accu_new) + epsilon)
            sharedvars.append(accu)
            updates[accu] = accu_new
            updates[param] = param - delta
    elif variant == 'graves':
        for param, grad in zip(params, grads):
            value = numpy.zeros_like(param.get_value(borrow = True))
            accu = theano.shared(value, broadcastable = param.broadcastable)
            gaccu = theano.shared(value, broadcastable = param.broadcastable)

            accu_new = rho * accu + (1 - rho) * (grad ** 2)
            gaccu_new = rho * gaccu + (1 - rho) * grad

            sharedvars.append(accu)
            sharedvars.append(gaccu)
            updates[accu] = accu_new
            updates[gaccu] = gaccu_new

            denorm = theano.tensor.sqrt(accu_new - gaccu_new ** 2 + epsilon)
            delta = lr * grad / denorm
            updates[param] = param - delta
    else:
        raise RuntimeError('error: unknown variant')

    return sharedvars, updates

def rmsprop_momentum_updates(params, grads, lr, rho, epsilon, momentum):
    sharedvars = []
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = numpy.zeros_like(param.get_value(borrow = True))
        accu = theano.shared(value, broadcastable = param.broadcastable)
        grad_accu = theano.shared(value, broadcastable = param.broadcastable)
        velocity = theano.shared(value, broadcastable = param.broadcastable)

        accu_new = rho * accu + (1 - rho) * (grad ** 2)
        grad_accu_new = rho * grad_accu + (1 - rho) * grad

        sharedvars.append(accu)
        sharedvars.append(grad_accu)
        sharedvars.append(velocity)
        updates[accu] = accu_new
        updates[grad_accu] = grad_accu_new

        denorm = theano.tensor.sqrt(accu_new - grad_accu_new ** 2 + epsilon)
        velocity_new = momentum * velocity - lr * grad / denorm
        updates[velocity] = velocity_new
        updates[param] = param + velocity_new

    return sharedvars, updates

def adadelta_updates(params, grads, lr, rho, epsilon):
    sharedvars = []
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        accu = theano.shared(var, broadcastable = param.broadcastable)
        delta_accu = theano.shared(var, broadcastable = param.broadcastable)

        sharedvars.append(accu)
        sharedvars.append(delta_accu)

        accu_new = rho * accu + (1 - rho) * (grad ** 2)
        updates[accu] = accu_new

        update = (grad * theano.tensor.sqrt(delta_accu + epsilon) /
                  theano.tensor.sqrt(accu_new + epsilon))
        updates[param] = param - lr * update

        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return sharedvars, updates

def adam_updates(params, grads, lr, beta1, beta2, epsilon):
    sharedvars = []
    updates = OrderedDict()

    t_prev = theano.shared(numpy.asarray(0.0, dtype = theano.config.floatX))
    t = t_prev + 1
    a_t = lr * theano.tensor.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    sharedvars.append(t_prev)

    for param, g_t in zip(params, grads):
        value = param.get_value(borrow = True)
        var = numpy.zeros_like(value)
        m_prev = theano.shared(var, broadcastable = param.broadcastable)
        v_prev = theano.shared(var, broadcastable = param.broadcastable)

        sharedvars.append(m_prev)
        sharedvars.append(v_prev)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * (g_t ** 2)
        step = a_t * m_t / (theano.tensor.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t

    return sharedvars, updates
