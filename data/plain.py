# plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy

def tokenize(data):
    return data.split()

def numberize(data, voc, unk = 'UNK'):
    newdata = []
    unkid = voc[unk]

    for d in data:
        idlist = [voc[w] if w in voc else unkid for w in d]
        newdata.append(idlist)

    return newdata

def normalize(bat):
    blen = [len(item) for item in bat]

    n = len(bat)
    maxlen = numpy.max(blen)

    b = numpy.zeros((maxlen, n), 'int32')
    m = numpy.zeros((maxlen, n), 'float32')

    for idx, item in enumerate(bat):
        b[:blen[idx], idx] = item
        m[:blen[idx], idx] = 1.0

    return b, m

def processdata(data, voc, unk = 'UNK', eos = '<eos>'):
    data = [tokenize(item) + [eos] for item in data]
    data = numberize(data, voc, unk)
    data, mask = normalize(data)

    return data, mask
