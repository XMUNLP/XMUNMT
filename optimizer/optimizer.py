# optimizer.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

import updates
import constraint

from ops import trainable_variables
from collections import OrderedDict


class optimizer:

    def __init__(self, model, **option):
        cost = model.cost
        inputs = model.inputs
        outputs = model.outputs
        scan_updates = model.updates

        if "variables" not in option or not option["variables"]:
            params = trainable_variables()

        grads = theano.grad(cost, params)
        gradsref = grads

        vec = [theano.shared(numpy.zeros_like(p.get_value())) for p in params]

        if "algorithm" not in option:
            option["algorithm"] = "sgd"

        if "variant" not in option:
            option["variant"] = None

        if "constraint" not in option:
            option["constraint"] = None

        if "momentum" not in option:
            option["momentum"] = False

        if "norm" not in option:
            option["norm"] = True

        if "nesterov" not in option:
            option["nesterov"] = False

        if "initialize" not in option:
            option["initialize"] = False

        if "nanguard" not in option:
            option["nanguard"] = False

        algorithm = option["algorithm"]
        variant = option["variant"]
        variant = [variant] if variant != None else []

        if option["norm"]:
            normval = constraint.global_norm(grads)
            outputs = outputs[:]
            outputs.insert(1, normval)

        if option["constraint"]:
            method, value = option["constraint"]
            if method == "value":
                grads = constraint.clip_by_value(grads, value[0], value[1])
            if method == "norm":
                grads = constraint.clip_by_global_norm(grads, value)

        if option["nanguard"]:
            gnorm = constraint.global_norm(gradsref)
            isnan = theano.tensor.isnan(gnorm)
            isinf = theano.tensor.isinf(gnorm)
            notfinite = theano.tensor.or_(isnan, isinf)
            newgrads = []
            for p, g in zip(params, grads):
                newgrads.append(theano.tensor.switch(notfinite, 0.1 * p, g))
            grads = newgrads

        if option["nesterov"]:
            option["momentum"] = False

        gup = []

        # append update rules
        if isinstance(scan_updates, OrderedDict):
            for key, value in scan_updates.iteritems():
                gup.append((key, value))
        else:
            gup.extend(scan_updates)

        for v, g in zip(vec, grads):
            gup.append((v, g))

        if algorithm == "sgd":
            alpha = theano.tensor.scalar()
            hparams = [alpha]
            defaults = [("alpha", 1.0)]
            svar, pup = updates.sgd_updates(params, vec, *hparams)
        elif algorithm == "adagrad":
            alpha = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, epsilon]
            defaults = [("alpha", 1.0), ("epsilon", 1e-6)]
            svar, pup = updates.adagrad_updates(params, vec, *hparams)
        elif algorithm == "rmsprop":
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon]
            defaults = [("alpha", 1e-2), ("rho", 0.99), ("epsilon", 1e-8)]
            rmsparam = hparams + variant
            svar, pup = updates.rmsprop_updates(params, vec, *rmsparam)
        elif algorithm == "rmsprop_momentum":
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            momentum = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon, momentum]
            defaults = [("alpha", 1e-4), ("rho", 0.95), ("epsilon", 1e-4)]
            defaults.append(("moment", 0.9))
            svar, pup = updates.rmsprop_momentum_updates(params, vec, *hparams)
        elif algorithm == "adadelta":
            alpha = theano.tensor.scalar()
            rho = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, rho, epsilon]
            defaults = [("alpha", 1.0), ("rho", 0.95), ("epsilon", 1e-6)]
            svar, pup = updates.adadelta_updates(params, vec, *hparams)
        elif algorithm == "adam":
            alpha = theano.tensor.scalar()
            beta1 = theano.tensor.scalar()
            beta2 = theano.tensor.scalar()
            epsilon = theano.tensor.scalar()
            hparams = [alpha, beta1, beta2, epsilon]
            defaults = [("alpha", 0.001), ("beta1", 0.9), ("beta2", 0.999)]
            defaults.append(("epsilon", 1e-8))
            svar, pup = updates.adam_updates(params, vec, *hparams)
        else:
            raise "Error: " + algorithm + " is not supported"

        # restore variables used by optimizer
        if option["initialize"]:
            values = option["initialize"]
            for v1, v2 in zip(svar, values):
                v1.set_value(v2)

        if option["momentum"]:
            momentum = theano.tensor.scalar()
            hparams.append(momentum)
            defaults.append(("momentum", 0.9))
            pup = updates.apply_momentum(pup, params, momentum)

        if option["nesterov"]:
            momentum = theano.tensor.scalar()
            hparams.append(momentum)
            defaults.append(("momentum", 0.9))
            pup = updates.apply_momentum(pup, params, momentum)

        optimize = theano.function(inputs, outputs, updates = gup)
        update = theano.function(hparams, [], updates = pup)

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
        self.algorithm = algorithm
        self.parameter = svar
