# commonwords.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import cPickle

def loadpkl(name):
    fd = open(name, 'r')
    vocab = cPickle.load(fd)
    fd.close()
    return vocab

if __name__ == '__main__':
    voc1 = loadpkl(sys.argv[1])
    voc2 = loadpkl(sys.argv[2])
    count = 0

    for words in voc1:
        if words in voc2:
            count += 1

    print count
