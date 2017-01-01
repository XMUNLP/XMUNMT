# scan.py
# a wrapper for Theano's scan
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano

from collections import OrderedDict
from collection import add_to_collection, get_collection


__all__ = ["scan", "get_updates", "merge_updates"]


_SCAN_UPDATES_KEYS = "scan_updates"


def tuple_to_dict(tuples):
    if isinstance(tuples, OrderedDict):
        return tuples
    else:
        updates = OrderedDict()

        for (key, value) in tuples:
            updates[key] = value

        return updates


def merge_updates(updates, new_updates):
    new_updates = tuple_to_dict(new_updates)

    for key, value in new_updates.iteritems():
        updates[key] = value

    return updates


def get_updates():
    updates_list = get_collection(_SCAN_UPDATES_KEYS)

    return reduce(merge_updates, [OrderedDict()] + list(updates_list))


def scan(fn, sequences=None, outputs_info=None, non_sequences=None, **kwargs):
    outputs, updates = theano.scan(fn, sequences, outputs_info, non_sequences,
                                   **kwargs)

    add_to_collection(_SCAN_UPDATES_KEYS, updates)

    return outputs
