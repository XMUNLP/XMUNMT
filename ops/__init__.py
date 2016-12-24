# __init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

from variable import variable, global_variables, trainable_variables
from variable_scope import variable_scope, get_variable_scope, get_variable
from initializer import zeros_initializer, ones_initializer
from initializer import constant_initializer, random_uniform_initializer
from initializer import uniform_unit_scaling_initializer


__all__ = [
    "variable",
    "global_variables",
    "trainable_variables",
    "variable_scope",
    "get_variable_scope",
    "get_variable",
    "zeros_initializer",
    "ones_initializer",
    "constant_initializer",
    "random_uniform_initializer",
    "uniform_unit_scaling_initializer"
]
