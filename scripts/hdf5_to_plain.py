# hdf5_to_plain.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import sys
import numpy
import tables
import cPickle

def to_string(bline, voc):
    sline = [voc[k] for k in bline]
    return " ".join(sline)

if __name__ == "__main__":
    hfd = tables.open_file(sys.argv[1], "r")
    vfd = open(sys.argv[2], "r")
    ofd = open(sys.argv[3], "w")

    voc = cPickle.load(vfd)

    pnode = hfd.get_node("/phrases")
    inode = hfd.get_node("/indices")

    data_len = inode.shape[0]
    idxs = numpy.arange(data_len)
    
    for i in idxs:
        pos = inode[i]["pos"]
        length = inode[i]["length"]
        bline = pnode[pos:(pos + length)]
        ofd.write(to_string(bline, voc))
        ofd.write("\n")

    hfd.close()
    vfd.close()
    ofd.close()
