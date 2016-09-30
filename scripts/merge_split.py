#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys


def merge_corpus(outname, *clist):
    corpus_list = clist
    ofile = open(outname, "w")
    files = [open(name, "r") for name in corpus_list]

    for linetup in zip(*files):
        linelist = [item.strip() for item in linetup]
        lenlist = [len(line.split()) for line in linelist]
        ofile.write(str(max(lenlist)) + " ")
        ofile.write(" ".join([str(item) for item in lenlist]))
        ofile.write(" ||| ")
        ofile.write(" ||| ".join(linelist))
        ofile.write("\n")

    for fd in files:
        fd.close()

    ofile.close()


def split_corpus(inname, *clise):
    corpus_list = clise
    ifile = open(inname, "r")
    files = [open(name, "w") for name in corpus_list]

    for line in ifile:
        slist = line.strip().split(" ||| ")

        if len(slist) - 1 != len(files):
            msg = "need file: " + str(len(slist) - 1) + " "
            msg += "given file: " + str(len(files))
            raise RuntimeError(msg)

        llist = slist[1:]

        for nline, nfile in zip(llist, files):
            nline = nline.strip()
            nfile.write(nline + "\n")

    for fd in files:
        fd.close()

    ifile.close()


if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)

    if argc < 4:
        print("merge_split -m fo, fi1, fi2, ... fin")
        print("merge_split -s fi, fo1, fo2, ... fon")
        sys.exit(-1)

    mode = argv[1]

    if mode != "-s" and mode != "-m":
        print("unknow mode")
        sys.exit(-1)

    if mode == "-m":
        fo = argv[2]
        fin = argv[3:]
        merge_corpus(fo, *fin)
    else:
        fi = argv[2]
        fout = argv[3:]
        split_corpus(fi, *fout)
