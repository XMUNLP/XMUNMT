# config.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import re


def count_wildcard(keys):
    return sum([key == "*" for key in keys])


def to_regex(key):
    return key.replace("*", ".*")


class option(object):

    def __init__(self, **opt):
        self.__dict__.update(opt)

    def __iter__(self):
        return self.__dict__.itervalues()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return self.__dict__

    def keys(self):
        return list(self.__dict__)


class config(object):

    def __getitem__(self, key):
        key_list = key.split("/")

        if count_wildcard(key_list):
            raise ValueError("key cannot contain wildcard character '*'")

        if len(key_list) == 1:
            return getattr(self, key)
        else:
            obj = getattr(self, key_list[0])
            return obj.__getitem__("/".join(key_list[1:]))

    def __setitem__(self, key, value):
        key_list = key.split("/")
        count = count_wildcard(key_list)

        if count > 1:
            raise ValueError("key cannot contain more than one '*' character")

        if key_list[0] != "*":
            if len(key_list) == 1:
                setattr(self, key, value)
            else:
                obj = getattr(self, key_list[0])
                obj.__setitem__("/".join(key_list[1:]), value)
        else:
            keylist = self.keys()
            pattern = to_regex(key)

            for item in keylist:
                if re.match(pattern, item):
                    self.__setitem__(item, value)

    def to_dict(self):
        dic = {}

        for key in self.__dict__:
            attr = getattr(self, key)
            if isinstance(attr, (config, option)):
                subdict = attr.to_dict()
                for subkey in subdict:
                    mainkey = key + "/" + subkey
                    dic[mainkey] = subdict[subkey]
            else:
                dic[key] = attr

        return dic

    def keys(self):
        keylist = []

        for key in self.__dict__:
            attr = getattr(self, key)
            if isinstance(attr, (config, option)):
                subkeylist = attr.keys()
                for subkey in subkeylist:
                    mainkey = key + "/" + subkey
                    keylist.append(mainkey)
            else:
                keylist.append(key)

        return keylist

    def update(self, **kwargs):
        for key, value in kwargs.iteritems():
            self.__setitem__(key, value)
