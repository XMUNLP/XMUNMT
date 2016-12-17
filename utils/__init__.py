# utils/__init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import six
import collections


def add_if_not_exsit(option, key, value):
    if key not in option:
        option[key] = value


def get_or_default(opt, key, default):
    if key in opt:
        return opt[key]
    return default


def is_sequence(seq):
    if isinstance(seq, six.string_types):
        return False

    if isinstance(seq, collections.Sequence):
        return True

    return False


def sequence_like(instance, args):
    if (isinstance(instance, tuple) and
        hasattr(instance, "_fields") and
        isinstance(instance._fields, collections.Sequence) and
        all(isinstance(f, six.string_types) for f in instance._fields)):
        return type(instance)(*args)
    else:
        return type(instance)(args)


def recursive_yield(nest):
    for n in nest:
        if is_sequence(n):
            for ni in recursive_yield(n):
                yield ni
        else:
            yield n


def recursive_assert(nest1, nest2):
    if is_sequence(nest1) != is_sequence(nest2):
        raise ValueError("structure: %s vs %s" % (nest1, nest2))

    if is_sequence(nest1):
        type1 = type(nest1)
        type2 = type(nest2)
        if type1 != type2:
            raise TypeError("strcture type: %s vs %s" % (type1, type2))

        for n1, n2 in zip(nest1, nest2):
            recursive_assert(n1, n2)


def flatten(nest):
    return list(recursive_yield(nest)) if is_sequence(nest) else [nest]


def assert_same_structure(nest1, nest2):
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("structure: %s vs %s" % (nest1, nest2))

    recursive_assert(nest1, nest2)


def flatten_dictionary(dictionary):
    if not isinstance(dictionary, dict):
        raise TypeError("input must be a dictionary")

    flat_dictionary = {}

    for i, v in six.iteritems(dictionary):
        if not is_sequence(i):
            if i in flat_dictionary:
                raise ValueError("key %s is not unique" % i)
            flat_dictionary[i] = v
        else:
            flat_i = flatten(i)
            flat_v = flatten(v)
            if len(flat_i) != len(flat_v):
                raise ValueError("could not flatten dictionary")
            for new_i, new_v in zip(flat_i, flat_v):
                if new_i in flat_dictionary:
                    raise ValueError("%s is not unique" % (new_i))
                flat_dictionary[new_i] = new_v
    return flat_dictionary


def packed_nest_with_indices(structure, flat, index):
    packed = []
    for s in structure:
        if is_sequence(s):
            new_index, child = packed_nest_with_indices(s, flat, index)
            packed.append(sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")

    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError("flat_sequence is not a scalar")

        return flat_sequence[0]

    flat_structure = flatten(structure)

    if len(flat_structure) != len(flat_sequence):
        raise ValueError("structure: %s vs %s" % (structure, flat_sequence))

    _, packed = packed_nest_with_indices(structure, flat_sequence, 0)
    return sequence_like(structure, packed)
