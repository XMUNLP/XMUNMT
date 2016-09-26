# config.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn


class option(object):

    def __init__(self, **opt):
        self.__dict__.update(opt)

    def __iter__(self):
        return self.__dict__.itervalues()


class config(object):

    def __getitem__(self, key):
        key_list = key.split("/")
        if len(key_list) == 1:
            return getattr(self, key)
        else:
            obj = getattr(self, key_list[0])
            return getattr(obj, "/".join(key_list[1:]))

    def __setitem__(self, key, value):
        key_list = key.split("/")
        if len(key_list) == 1:
            setattr(self, key, value)
        else:
            obj = getattr(self, key_list[0])
            setattr(obj, "/".join(key_list[1:]), value)
