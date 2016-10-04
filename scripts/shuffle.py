# shuffle.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import argparse


def parseargs():
    desc = "shuffle corpus"
    parser = argparse.ArgumentParser(description = desc)

    msg = "input corpora"
    parser.add_argument("--corpus", nargs="+", required=True, help=msg)
    msg = "output file suffix"
    parser.add_argument("--suffix", type=str, default="shuf", help=msg)
    msg = "random seed"
    parser.add_argument("--seed", type=int, help=msg)

    return parser.parse_args()


def main(args):
    name = args.corpus
    suffix = "." + args.suffix
    stream = [open(item, "r") for item in name]
    data = [fd.readlines() for fd in stream]
    minlen = min([len(lines) for lines in data])

    if args.seed:
        numpy.random.seed(args.seed)

    indices = numpy.arange(minlen)
    numpy.random.shuffle(indices)

    newstream = [open(item + suffix, "w") for item in name]

    for idx in indices:
        lines = [item[idx] for item in data]

        for line, fd in zip(lines, newstream):
            fd.write(line)

    for fdr, fdw in zip(stream, newstream):
        fdr.close()
        fdw.close()


if __name__ == "__main__":
    arg = parseargs()
    main(arg)
