# config.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from utils import merge_option

def linear_option():
    opt = {}
    opt['name'] = 'linear'
    opt['weight'] = True
    opt['bias'] = True
    opt['variant'] = 'standard'
    opt['target'] = 'auto'

    return opt

def embedding_option():
    opt = {}
    opt['name'] = 'embedding'
    opt['bias'] = False
    opt['init'] = None
    opt['target'] = 'auto'

    return opt

def feedforward_option():
    opt = {}
    opt['name'] = 'feedforward'
    opt['weight'] = True
    opt['bias'] = True
    opt['variant'] = 'standard'
    opt['function'] = theano.tensor.nnet.sigmoid
    opt['target'] = 'auto'

    return opt

def maxout_option():
    opt = {}
    opt['name'] = 'maxout'
    opt['maxpart'] = 2
    opt['weight'] = True
    opt['bias'] = True
    opt['variant'] = 'standard'
    opt['target'] = 'auto'

    return opt

def gru_option():
    opt = {}
    opt['name'] = 'gru'
    opt['variant'] = 'standard'
    opt['target'] = 'auto'
    lopt = linear_option()
    merge_option(opt, 'reset-gate', lopt)
    merge_option(opt, 'update-gate', lopt)
    merge_option(opt, 'transform', lopt)
    merge_option(opt, 'gates', lopt)

    return opt
