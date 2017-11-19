# coding=utf-8
# Copyright 2017 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

import xmunmt.models.rnnsearch


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return xmunmt.models.rnnsearch.RNNsearch
    else:
        raise LookupError("Unknown model %s" % name)
