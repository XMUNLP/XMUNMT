# coding=utf-8
# Copyright 2017 Natural Language Processing Lab of Xiamen University
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


def encode():
    for line in sys.stdin:
        line = line.decode("utf-8")
        token = []
        token_list = []

        for char in line:
            if char == " ":
                if token:
                    token_list.append(token)
                token_list.append(["_"])
                token = []
            elif char == "_":
                if token:
                    token_list.append(token)
                token_list.append(["@_@"])
                token = []
            elif char == "-":
                if token:
                    token_list.append(token)
                token_list.append(["@-@"])
                token = []
            elif ord(char) < 256:
                token.append(char.encode("utf-8"))
            else:
                if token:
                    token_list.append(token)
                token_list.append([char.encode("utf-8")])
                token = []

        if token is not None:
            token_list.append(token)

        tokens = ["".join(item) for item in token_list]
        encoded = " ".join(tokens)
        sys.stdout.write(encoded)


def decode():
    for line in sys.stdin:
        tokens = line.strip().split()
        token_list = []

        for token in tokens:
            if token == "@_@":
                token_list.append("_")
            elif token == "_":
                token_list.append(" ")
            elif token == "@-@":
                token_list.append("-")
            else:
                token_list.append(token)

        decoded = "".join(token_list)
        sys.stdout.write(decoded + "\n")


def usage():
    print("usage: char_utils.py encode < input > output\n"
          "       char_utils.py decode < input > output")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    if sys.argv[1] == "encode":
        encode()
    elif sys.argv[1] == "decode":
        decode()
    else:
        usage()
