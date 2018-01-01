# coding=utf-8
# Copyright 2018 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xmunmt.models.rnnsearch


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return xmunmt.models.rnnsearch.RNNsearch
    else:
        raise LookupError("Unknown model %s" % name)
