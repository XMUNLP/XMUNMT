# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np


__all__ = ["data_length", "convert_data"]


def data_length(line):
    return len(line.split())


def convert_data(data, voc, unk="UNK", eos="<eos>", time_major=True):
    # tokenize
    data = [line.split() + [eos] for line in data]

    unkid = voc[unk]

    newdata = []

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    data = newdata

    lens = [len(tokens) for tokens in data]

    n = len(lens)
    maxlen = np.max(lens)

    batch_data = np.zeros((n, maxlen), "int32")
    data_length = np.array(lens)

    for idx, item in enumerate(data):
        batch_data[idx, :lens[idx]] = item

    if time_major:
        batch_data = batch_data.transpose()

    return batch_data, data_length
