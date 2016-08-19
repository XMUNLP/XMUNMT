# __init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import re
import numpy

def add_if_not_exsit(option, key, value):
    if key not in option:
        option[key] = value

def merge_option(opt1, prefix, opt2):
    for key, value in opt2.iteritems():
        newkey = prefix + '/' + key
        opt1[newkey] = value

def extract_option(opt, prefix):
    newopt = {}

    for key in opt:
        keys = key.split('/')
        if keys[0] == prefix:
            if len(keys) > 1:
                newopt['/'.join(keys[1:])] = opt[key]

    return newopt

def add_parameters(plist, prefix, *params):
    for param in params:
        key = prefix + '/' + param.name
        param.name = key
        plist.append(param)

def update_option(opt1, opt2):
    for key, value in opt2.iteritems():
        opt1[key] = value

def change_option(opt, pattern, value):
    for key in opt:
        if not re.match(pattern, key):
            continue
        opt[key] = value

def uniform_tensor(shape, low = -0.05, high = 0.05):
    return numpy.random.uniform(low, high, shape)
