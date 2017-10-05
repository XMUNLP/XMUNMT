# optimizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import tensorflow as tf

from utils import function


def sgd_update(grad, var, lr):
    delta = lr * grad
    return tf.assign_sub(var, delta)


def adam_update(grad, var, a_t, m, v, lr, beta1, beta2, eps):
    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * tf.square(grad)
    delta = a_t * m_t / (tf.sqrt(v_t) + eps)

    update_mt = tf.assign(m, m_t)
    update_vt = tf.assign(v, v_t)
    update_delta = tf.assign_sub(var, delta)

    return tf.group(update_mt, update_vt, update_delta)


def rmsprop_update(grad, var, ms, mg, lr, rho, eps):
    new_ms = rho * ms + (1.0 - rho) * tf.square(grad)
    new_mg = rho * mg + (1.0 - rho) * grad

    delta = lr * grad / tf.sqrt(new_ms - tf.square(new_mg) + eps)

    update_ms = tf.assign(ms, new_ms)
    update_mg = tf.assign(mg, new_mg)
    update_var = tf.assign_sub(var, delta)

    return tf.group(update_ms, update_mg, update_var)


def gradient_updates(grads_and_vars):
    updates = []

    for grad, var in grads_and_vars:
        if isinstance(grad, tf.Tensor):
            updates.append(tf.assign(var, grad))
        else:
            new_var = tf.assign(var, tf.zeros_like(var))
            updates.append(tf.scatter_add(new_var, grad.indices, grad.values))

    return tf.group(*updates)


def sgd_updates(grads_and_vars, lr):
    updates = []
    for grad, var in grads_and_vars:
        updates.append(sgd_update(grad, var, lr))

    return tf.group(*updates)


def adam_updates(grads_and_vars, slot_vars, lr, beta1, beta2, eps):
    updates = []
    t = slot_vars[0]
    slot_vars = slot_vars[1:]

    new_t = t + 1
    a = lr * tf.sqrt(1 - tf.pow(beta2, new_t)) / (1 - tf.pow(beta1, new_t))

    updates.append(tf.assign(t, new_t))

    for gv, sv in zip(grads_and_vars, slot_vars):
        grad, var = gv
        m, v = sv
        updates.append(adam_update(grad, var, a, m, v, lr, beta1, beta2, eps))

    return tf.group(*updates)


def rmsprop_updates(grads_and_vars, slot_vars, lr, rho, eps):
    updates = []

    for gv, sv in zip(grads_and_vars, slot_vars):
        grad, var = gv
        ms, mg = sv
        updates.append(rmsprop_update(grad, var, ms, mg, lr, rho, eps))

    return tf.group(*updates)


def create_zeros_slot(primary, name, dtype=None):
    if dtype is None:
        dtype = primary.dtype
    shape = primary.get_shape().as_list()
    init_val = tf.zeros_initializer()(shape, dtype=dtype)
    var = tf.Variable(init_val, name=name, trainable=False)
    return var


class optimizer:

    def __init__(self, model, **option):
        loss = model.cost
        inputs = model.inputs
        outputs = model.outputs

        if "norm" not in option:
            option["norm"] = False

        if "constraint" not in option:
            option["constraint"] = None

        params = tf.trainable_variables()

        grads = tf.gradients(loss, params, colocate_gradients_with_ops=True,
                             gate_gradients=True)

        if option["norm"]:
            normval = tf.global_norm(grads)
            outputs = outputs[:]
            outputs.append(normval)

        if option["constraint"]:
            method, value = option["constraint"]
            if method == "value":
                min_v = value[0]
                max_v = value[1]
                grads = [tf.clip_by_value(g, min_v, max_v) for g in grads]
            if method == "norm":
                grads, normval = tf.clip_by_global_norm(grads, value)

        gvars = []
        gvars_and_vars = []
        grads_and_gvars = []

        for grad, var in zip(grads, params):
            if grad is None:
                continue
            slotvar = create_zeros_slot(var, "gradient")
            gvars.append(slotvar)
            gvars_and_vars.append((slotvar, var))
            grads_and_gvars.append([grad, slotvar])

        grad_updates = gradient_updates(grads_and_gvars)
        placeholders = []

        if "algorithm" not in option:
            option["algorithm"] = "sgd"

        if option["algorithm"] == "sgd":
            varlist = []
            lr = tf.placeholder(tf.float32, [])
            defaults = [('alpha', 1.0)]
            placeholders.append(lr)
            var_updates = sgd_updates(gvars_and_vars, lr)
        elif option["algorithm"] == "rmsprop":
            lr = tf.placeholder(tf.float32, [])
            rho = tf.placeholder(tf.float32, [])
            eps = tf.placeholder(tf.float32, [])
            varlist = []
            svars = []

            for gvar in gvars:
                ms = create_zeros_slot(gvar, "mean_square")
                mg = create_zeros_slot(gvar, "mean_gradient")
                svars.append([ms, mg])
                varlist.extend([ms, mg])

            placeholders.append(lr)
            placeholders.append(rho)
            placeholders.append(eps)
            defaults = [('alpha', 1e-2), ('rho', 0.99), ('epsilon', 1e-8)]
            var_updates = rmsprop_updates(gvars_and_vars, svars, lr, rho, eps)
        elif option["algorithm"] == "adam":
            lr = tf.placeholder(tf.float32, [])
            beta1 = tf.placeholder(tf.float32, [])
            beta2 = tf.placeholder(tf.float32, [])
            eps = tf.placeholder(tf.float32, [])

            t = tf.Variable(0.0, name="adam_t", dtype=tf.float32,
                            trainable=False)
            varlist = [t]
            svars = [t]

            for gvar in gvars:
                m = create_zeros_slot(gvar, "m")
                v = create_zeros_slot(gvar, "v")
                svars.append([m, v])
                varlist.extend([m, v])

            placeholders.append(lr)
            placeholders.append(beta1)
            placeholders.append(beta2)
            placeholders.append(eps)
            defaults = [("alpha", 1e-3), ("beta1", 0.9), ("beta2", 0.999),
                        ("epsilon", 1e-8)]
            var_updates = adam_updates(gvars_and_vars, svars, lr, beta1, beta2,
                                       eps)
        else:
            raise ValueError("unknown algorithm %s" % option["algorithm"])

        optimize = function(inputs, outputs, updates=grad_updates)
        update = function(placeholders, [], updates=var_updates)

        def wrapper(**option):
            values = []
            for item in defaults:
                name = item[0]
                val = item[1]
                if name not in option:
                    option[name] = val
                values.append(option[name])
            return update(*values)

        self.optimize = optimize
        self.update = wrapper
        self.option = option
        self.variables = varlist
