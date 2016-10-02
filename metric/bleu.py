# bleu.py
# code modified from nltk.align.bleu
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import re
import math


# tokenization from mteval
def tokenization(text):
    # language-independent part
    text = text.replace("<skipped>", "")
    text = text.replace("-\n", " ")
    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt", ">")

    # lowercase
    text = text.lower()
    # tokenize punctuation
    text = re.sub("([\{-~\[-` -&\(-\+:-@/])", " \g<1> ", text)
    # tokenize period and comma unless preceded by a digit
    text = re.sub("([^0-9])([.,])", "\g<1> \g<2> ", text)
    # tokenize period and comma unless follwed by a digit
    text = re.sub("([.,])([^0-9])", " \g<1> \g<2>", text)
    # tokenize dash when preceded by a digit
    text = re.sub("([0-9])(-)", "\g<1> \g<2> ", text)
    # one space only between words
    text = " ".join(text.split())
    # no leading and trailing space
    text = text.strip()

    return text


def count_ngrams(seq, n):
    counts = {}
    length = len(seq)

    for i in range(length):
        if i + n <= length:
            ngram = " ".join(seq[i : i + n])
            if ngram not in counts:
                counts[ngram] = 0
            counts[ngram] += 1

    return counts


def closest_length(candidate, references):
    clen = len(candidate)
    closest_diff = 9999
    closest_len = 9999

    for reference in references:
        rlen = len(reference)
        diff = abs(rlen - clen)

        if diff < closest_diff:
            closest_diff = diff
            closest_len = rlen
        elif diff == closest_diff:
            closest_len = rlen if rlen < closest_len else closest_len

    return closest_len


def shortest_length(references):
    return min([len(ref) for ref in references])


def modified_precision(candidate, references, n):
    counts = count_ngrams(candidate, n)

    if len(counts) == 0:
        return 0, 0

    max_counts = {}
    for reference in references:
        ref_counts = count_ngrams(reference, n)
        for ngram in counts:
            mcount = 0 if ngram not in max_counts else max_counts[ngram]
            rcount = 0 if ngram not in ref_counts else ref_counts[ngram]
            max_counts[ngram] = max(mcount, rcount)

    clipped_counts = {}

    for ngram, count in counts.items():
        clipped_counts[ngram] = min(count, max_counts[ngram])

    return float(sum(clipped_counts.values())), float(sum(counts.values()))


def brevity_penalty(trans, refs, mode="closest"):
    bp_c = 0.0
    bp_r = 0.0

    for candidate, references in zip(trans, refs):
        bp_c += len(candidate)

        if mode == "shortest":
            bp_r += shortest_length(references)
        else:
            bp_r += closest_length(candidate, references)

    bp = 1.0

    if bp_c <= bp_r:
        bp = math.exp(1.0 - bp_r / bp_c)

    return bp


# trans: a list of tokenized sentence
# refs: a list of list of tokenized reference sentences
def bleu(trans, refs, bp="closest", smooth=False, n=4, weight=None):
    p_norm = [0 for i in range(n)]
    p_denorm = [0 for i in range(n)]

    for candidate, references in zip(trans, refs):
        for i in range(n):
            ccount, tcount = modified_precision(candidate, references, i + 1)
            p_norm[i] += ccount
            p_denorm[i] += tcount

    bleu_n = [0 for i in range(n)]

    for i in range(n):
        # add one smoothing
        if smooth and i > 0:
            p_norm[i] += 1
            p_denorm[i] += 1

        if p_norm[i] == 0 or p_denorm[i] == 0:
            bleu_n[i] = -9999
        else:
            bleu_n[i] = math.log(float(p_norm[i]) / float(p_denorm[i]))

    bp = brevity_penalty(trans, refs, bp)

    bleu = bp * math.exp(sum(bleu_n) / float(n))

    return bleu
