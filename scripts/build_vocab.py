# build_vocab.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse
import collections


def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts


def special_tokens(string):
    if not string:
        return []
    else:
        return string.strip().split(":")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, ids = list(zip(*pairs))

    with open(name, "w") as f:
        for word in words:
            f.write(word + "\n")


def parse_args():
    msg = "build vocabulary"
    parser = argparse.ArgumentParser(description=msg)

    msg = "input corpus"
    parser.add_argument("corpus", help=msg)
    msg = "output vocabulary name"
    parser.add_argument("output", default="vocab.txt", help=msg)
    msg = "limit"
    parser.add_argument("--limit", default=0, type=int, help=msg)
    msg = "add special token, separated by colon"
    parser.add_argument("--special", type=str, help=msg)

    return parser.parse_args()


def main(args):
    vocab = {}
    limit = args.limit
    count = 0

    words, counts = count_words(args.corpus)
    special = special_tokens(args.special)

    for token in special:
        vocab[token] = len(vocab)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        if word in vocab:
            print("warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq

    save_vocab(args.output, vocab)

    print("total words: %d" % sum(counts))
    print("unique words: %d" % len(words))
    print("vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    args = parse_args()
    main(args)
