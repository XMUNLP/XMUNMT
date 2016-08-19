# gru.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import config

from feedforward import feedforward
from utils import update_option, add_parameters
from utils import extract_option, add_if_not_exsit

# gated recurrent unit
# available options:
# 1. name
# 2. variant: 'tied-weight' or 'standard'
# 3. update-gate/weight: only works when variant == 'standard'
# 4. update-gate/bias: only works when variant == 'standard'
# 5. reset-gate/weight: only works when variant == 'standard'
# 6. reset-gate/bias: only works when variant == 'standard'
# 7. gates/weight: only works when variant == 'tied-weight'
# 8. gates/bias: only works when variant == 'tiede-weigth'
# 9. transform/weight
# 10. transform/bias
# 11. target: target device, default: 'auto'
class gru:

    def __init__(self, input_size, output_size, **option):
        # inherit option
        if 'target' in option:
            add_if_not_exsit(option, 'reset-gate/target', option['target'])
            add_if_not_exsit(option, 'update-gate/target', option['target'])
            add_if_not_exsit(option, 'transform/target', option['target'])
            add_if_not_exsit(option, 'gates/target', option['target'])

        if 'variant' in option:
            add_if_not_exsit(option, 'reset-gate/variant', option['variant'])
            add_if_not_exsit(option, 'update-gate/variant', option['variant'])
            add_if_not_exsit(option, 'transform/variant', option['variant'])
            add_if_not_exsit(option, 'gates/variant', option['variant'])

        opt = config.gru_option()
        update_option(opt, option)

        variant = opt['variant']

        ropt = extract_option(opt, 'reset-gate')
        uopt = extract_option(opt, 'update-gate')
        topt = extract_option(opt, 'transform')
        gopt = extract_option(opt, 'gates')
        ropt['name'] = 'reset-gate'
        uopt['name'] = 'update-gate'
        topt['name'] = 'transform'
        gopt['name'] = 'gates'
        topt['function'] = theano.tensor.tanh

        modules = []

        if not isinstance(input_size, (list, tuple)):
            input_size = [input_size]

        if variant == 'standard':
            isize = input_size + [output_size]
            osize = output_size
            rgate = feedforward(isize, osize, **ropt)
            ugate = feedforward(isize, osize, **uopt)
            trans = feedforward(isize, osize, **topt)
            modules.append(rgate)
            modules.append(ugate)
            modules.append(trans)
        else:
            isize = input_size + [output_size]
            osize = output_size
            gates = feedforward(isize, 2 * osize, **gopt)
            trans = feedforward(isize, osize, **topt)
            modules.append(gates)
            modules.append(trans)

        name = opt['name']
        params = []

        for m in modules:
            add_parameters(params, name, *m.parameter)

        def forward(x, h):
            if not isinstance(x, (list, tuple)):
                x = [x]

            if variant == 'standard':
                reset_gate = modules[0]
                update_gate = modules[1]
                transform = modules[2]
                r = reset_gate(x + [h])
                u = update_gate(x + [h])
                t = transform(x + [r * h])
            else:
                gates = modules[0]
                transform = modules[1]
                r_u = gates(x + [h])
                size1 = r_u.shape[-1] / 2
                size2 = r_u.shape[-1] - size1
                r, u = theano.tensor.split(r_u, (size1, size2), 2, -1)
                t = transform(x + [r * h])

            y = (1.0 - u) * h + u * t

            return y

        self.name = name
        self.option = opt
        self.forward = forward
        self.parameter = params

    def __call__(self, x, h):
        return self.forward(x, h)
