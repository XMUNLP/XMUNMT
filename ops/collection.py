# collection.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import re
import six


__all__ = ["add_to_collection",
           "add_to_collections",
           "get_collection_ref",
           "get_collection"]


_COLLECTIONS = {}


def add_to_collection(name, value):
    global _COLLECTIONS

    if name not in _COLLECTIONS:
        _COLLECTIONS[name] = [value]
    else:
        _COLLECTIONS[name].append(value)


def add_to_collections(self, names, value):
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
        add_to_collection(name, value)


def get_collection_ref(name):
    coll_list = _COLLECTIONS.get(name, None)

    if coll_list is None:
        coll_list = []
        _COLLECTIONS[name] = coll_list

    return coll_list


def get_collection(name, scope=None):
    coll_list = _COLLECTIONS.get(name, None)

    if coll_list is None:
        return []

    if scope is None:
        return list(coll_list)
    else:
        c = []
        regex = re.compile(scope)
        for item in coll_list:
            if hasattr(item, "name") and regex.match(item.name):
                c.append(item)
        return c
