# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


__all__ = ["get_length", "convert_data"]


def get_length(line):
    return len(line.strip().split())


def tokenize(data):
    return data.split()


def to_word_id(data, voc, unk="UNK"):
    newdata = []
    unkid = voc[unk]

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    return newdata


def convert_to_array(data, dtype):
    batch = len(data)
    data_len = map(len, data)
    max_len = max(data_len)

    seq = numpy.zeros((max_len, batch), "int32")
    mask = numpy.zeros((max_len, batch), dtype)

    for idx, item in enumerate(data):
        seq[:data_len[idx], idx] = item
        mask[:data_len[idx], idx] = 1.0

    return seq, mask


def convert_data(data, voc, unk="UNK", eos="<eos>", dtype="float32"):
    data = [tokenize(item) + [eos] for item in data]
    data = to_word_id(data, voc, unk)
    seq, mask = convert_to_array(data, dtype)

    return seq, mask
