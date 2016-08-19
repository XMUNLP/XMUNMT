# batchstream.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

class batchstream:

    def __init__(self, name, batch_size = 1):
        if type(name) != list:
            name = [name]

        stream = [open(item, 'r') for item in name]

        self.filename = name
        self.stream = stream
        self.eos = False
        self.count = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_data(self):

        if self.eos == True:
            return None

        count = 0
        data = [[] for i in xrange(len(self.stream))]

        try:
            while True:
                line = [item.readline() for item in self.stream]

                for i in xrange(len(line)):
                    s = line[i]
                    if s == '':
                        raise IOError
                    data[i].append(s.strip())

                count = count + 1
                self.count = self.count + 1

                if count >= self.batch_size:
                    break
        except IOError:
            self.eos = True

        if count == 0:
            return None

        return data

    def next(self):
        data = self.get_data()

        if data == None:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return data

    def reset(self):
        self.count = 0
        self.eos = False

        for f in self.stream:
            f.seek(0)

    def close(self):
        for fd in self.stream:
            fd.close()
