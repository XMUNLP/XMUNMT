# convert_model.py
# convert old model to new model
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import six
import sys
import numpy
import cPickle


def loadvocab(name):
    fd = open(name, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def loadmodel(name):
    fd = open(name, "r")
    opt = cPickle.load(fd)

    if "validate" in opt:
        opt["validation"] = opt["validate"]
        del(opt["validate"])

    if "ref" in opt:
        opt["references"] = opt["ref"]
        del(opt["ref"])

    if not "bosid" in opt:
        opt["bosid"] = 0

    if not "eosid" in opt:
        opt["eosid"] = 0

    name_or_param = cPickle.load(fd)

    if not isinstance(name_or_param[0], six.string_types):
        param = name_or_param
        return opt, param
    else:
        name_list = name_or_param

    try:
        params = cPickle.load(fd)
    except:
        fd.close()
        fd = open(name, "r")
        dummy = cPickle.load(fd)
        dummy = cPickle.load(fd)
        name_list = dummy
        params = dict(numpy.load(fd))
        params = [params[s] for s in name_list]

    return opt, params


def get_rnnsearch_keys():
    keys = []

    keys.append("rnnsearch/source_embedding/embedding")
    keys.append("rnnsearch/target_embedding/embedding")
    keys.append("rnnsearch/encoder/forward/gru_cell/gates/matrix_0")
    keys.append("rnnsearch/encoder/forward/gru_cell/gates/matrix_1")
    keys.append("rnnsearch/encoder/forward/gru_cell/gates/bias")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/bias")
    keys.append("rnnsearch/encoder/backward/gru_cell/gates/matrix_0")
    keys.append("rnnsearch/encoder/backward/gru_cell/gates/matrix_1")
    keys.append("rnnsearch/encoder/backward/gru_cell/gates/bias")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/bias")
    keys.append("rnnsearch/decoder/initial/matrix_0")
    keys.append("rnnsearch/decoder/initial/bias")
    keys.append("rnnsearch/decoder/attention/attention_w/matrix_0")
    keys.append("rnnsearch/decoder/attention/attention_w/bias")
    keys.append("rnnsearch/decoder/below/gru_cell/gates/matrix_0")
    keys.append("rnnsearch/decoder/below/gru_cell/gates/matrix_1")
    keys.append("rnnsearch/decoder/below/gru_cell/gates/bias")
    keys.append("rnnsearch/decoder/below/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/decoder/below/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/decoder/below/gru_cell/candidate/bias")
    keys.append("rnnsearch/decoder/attention/query_w/matrix_0")
    keys.append("rnnsearch/decoder/attention/attention_v/matrix_0")
    keys.append("rnnsearch/decoder/attention/attention_v/bias")
    keys.append("rnnsearch/decoder/above/gru_cell/gates/matrix_0")
    keys.append("rnnsearch/decoder/above/gru_cell/gates/matrix_1")
    keys.append("rnnsearch/decoder/above/gru_cell/gates/bias")
    keys.append("rnnsearch/decoder/above/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/decoder/above/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/decoder/above/gru_cell/candidate/bias")
    keys.append("rnnsearch/decoder/deepout/matrix_0")
    keys.append("rnnsearch/decoder/deepout/matrix_1")
    keys.append("rnnsearch/decoder/deepout/matrix_2")
    keys.append("rnnsearch/decoder/deepout/bias_0")
    keys.append("rnnsearch/decoder/deepout/bias_1")
    keys.append("rnnsearch/decoder/deepout/bias_2")
    keys.append("rnnsearch/decoder/logits/matrix_0")
    keys.append("rnnsearch/decoder/logits/bias")

    return keys


if __name__ == "__main__":
    mapping = {18:24, 19:25, 20:26, 21:18, 22:19, 23:20, 24:21, 25:22, 26:23}

    opt, params = loadmodel(sys.argv[1])

    newparams = {}

    for i in range(len(params)):
        if i not in mapping:
            newparams[i] = params[i]
        else:
            newparams[mapping[i]] = params[i]

    params = [newparams[i] for i in range(len(params))]

    names = get_rnnsearch_keys()
    params = dict([(name, val) for name, val in zip(names, params)])

    fd = open(sys.argv[2], "w")
    cPickle.dump(opt, fd)
    cPickle.dump(names, fd)
    numpy.savez(fd, **params)
