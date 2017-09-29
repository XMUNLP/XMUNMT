# vocab.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import cPickle
import collections


def load_vocab(filename):
    fd = open(filename, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, counts = list(zip(*count_pairs))

    return words, counts
