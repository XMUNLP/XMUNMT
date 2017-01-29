# align.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


__all__ = ["convert_align"]


def convert_align(src, tgt, align, normalize=True):
    src_words = [item.strip().split() for item in src]
    tgt_words = [item.strip().split() for item in tgt]
    batch = len(align)

    max_src_len = max(map(lambda x: len(x), src_words)) + 1
    max_tgt_len = max(map(lambda x: len(x), tgt_words)) + 1

    align_array = numpy.zeros([max_tgt_len, max_src_len, batch], "float32")

    for i, (src, tgt, agn) in enumerate(zip(src_words, tgt_words, align)):
        item = agn.strip().split()
        item = map(lambda x: x.split("-"), item)
        item = map(lambda x: (int(x[0]), int(x[1])), item)

        slen = len(src)
        tlen = len(tgt)

        for align_item in item:
            src, tgt = align_item
            align_array[tgt, src, i] = 1.0

        align_array[tlen, slen, i] = 1.0

        if not normalize:
            continue

        for j in range(tlen):
            align_sum = numpy.sum(align_array[j, :, i])

            if align_sum > 0.0:
                align_array[j, :, i] /= align_sum

    return align_array
