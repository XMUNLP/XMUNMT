# reader.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


# lowest-level stream
class textreader:

    def __init__(self, name, shuffle=False, readall=False):
        if not isinstance(name, (list, tuple)):
            name = [name]

        stream = [open(item, "r") for item in name]

        if shuffle or readall:
            texts = [fd.readlines() for fd in stream]
        else:
            texts = None

        if shuffle:
            readall = True

            if not isinstance(shuffle, bool):
                randstate = numpy.random.RandomState(shuffle)
                shuffle = randstate.shuffle
            else:
                shuffle = numpy.random.shuffle

            linecnt = min([len(text) for text in texts])
            indices = numpy.arange(linecnt)
            shuffle(indices)
        else:
            indices = None
            shuffle = False

        self.eos = False
        self.count = 0
        self.names = name
        self.texts = texts
        self.stream = stream
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def readline(self):
        # read directly from memory
        if self.texts:
            linecnt = min([len(text) for text in self.texts])
            # end of file
            if self.count == linecnt:
                return None

            if self.shuffle:
                texts = [text[self.indices[self.count]] for text in self.texts]
            else:
                texts = [text[self.count] for text in self.texts]
        else:
            # read from file
            texts = [fd.readline() for fd in self.stream]
            flag = any([line == "" for line in texts])

            if flag:
                return None

        self.count += 1
        texts = [text.strip() for text in texts]

        return texts

    def next(self):
        data = self.readline()

        if data == None:
            self.reset()
            raise StopIteration

        return data

    def reset(self):
        self.count = 0
        self.eos = False

        for fd in self.stream:
            fd.seek(0)

        if self.shuffle:
            linecnt = min([len(text) for text in self.texts])
            indices = numpy.arange(linecnt)
            self.shuffle(indices)
            self.indices = indices

    def close(self):
        for fd in self.stream:
            fd.close()

    def get_indices(self):
        return self.indices

    def set_indices(self, indices):
        self.indices = indices
