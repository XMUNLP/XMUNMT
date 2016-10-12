# build_dictionary.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse
import collections


def parseargs():
    msg = "build dictionary using alignment files"
    parser = argparse.ArgumentParser(description=msg)

    msg = "source file"
    parser.add_argument("source", help=msg)
    msg = "target file"
    parser.add_argument("target", help=msg)
    msg = "alignment file"
    parser.add_argument("alignment", help=msg)
    msg = "output file name"
    parser.add_argument("output", help=msg)
    msg = "min alignment count"
    parser.add_argument("--count", default=100, type=int, help=msg)

    return parser.parse_args()


def main(args):
    source = open(args.source)
    target = open(args.target)
    align = open(args.alignment)

    counter = collections.Counter()

    for src, tgt, agn in zip(source, target, align):
        source_words = src.strip().split()
        target_words = tgt.strip().split()
        alignment = agn.strip().split()
        alignment = map(lambda item: item.split("-"), alignment)
        alignment = map(lambda item: (int(item[0]), int(item[1])), alignment)

        for (sid, tid) in alignment:
            source_word = source_words[sid]
            target_word = target_words[tid]

            counter.update([(source_word, target_word)])

    alignment_pair = {}
    normalize_count = {}

    # prune
    for item in counter.iteritems():
        (source_word, target_word), count = item

        if count < args.count:
            continue

        alignment_pair[(source_word, target_word)] = count

        if source_word in normalize_count:
            normalize_count[source_word] = normalize_count[source_word] + count
        else:
            normalize_count[source_word] = count

    fd = open(args.output, "w")

    count_pairs = sorted(alignment_pair.items())

    for (source_word, target_word), count in count_pairs:
        prob = float(count) / float(normalize_count[source_word])
        fd.write(source_word + " " + target_word + " " + str(prob))
        fd.write("\n")

    fd.close()
    source.close()
    target.close()
    align.close()


if __name__ == "__main__":
    args = parseargs()
    main(args)
