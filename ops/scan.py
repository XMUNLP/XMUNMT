# scan.py
# a wrapper for Theano's scan
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import utils
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


def get_updates(key="training"):
    updates_list = get_collection(_SCAN_UPDATES_KEYS + "/" + key)

    return reduce(merge_updates, [OrderedDict()] + list(updates_list))


def scan(fn, sequences=None, outputs_info=None, non_sequences=None,
         return_updates=False, use_extension=False, **kwargs):
    if sequences is None:
        sequences = []

    if outputs_info is None:
        outputs_info = []

    if non_sequences is None:
        non_sequences = []

    # support nested structure for sequences, outputs_info and non_sequences
    if use_extension:
        if isinstance(outputs_info, dict):
            raise ValueError("only support nested structure, not dict")

        nest_sequences = sequences
        nest_outputs_info = outputs_info
        nest_non_sequences = non_sequences

        # inputs to Theano's scan
        sequences = utils.flatten(nest_sequences)
        outputs_info = utils.flatten(nest_outputs_info)
        non_sequences = utils.flatten(nest_non_sequences)

        # input structure for fn
        nest_rec_info = []

        for item in nest_outputs_info:
            if item is not None:
                nest_rec_info.append(item)

        rec_info = utils.flatten(nest_rec_info)

        n_seq = len(sequences)
        n_rec = len(rec_info)

        for item in rec_info:
            if item is not None:
                continue
            raise ValueError("None can only appear in the outer level of "
                             "outputs_info")

        inner_fn = fn

        def wrapper_fn(*args):
            seqs = args[:n_seq]
            recs = args[n_seq : n_seq + n_rec]
            nonseq = args[n_seq + n_rec : ]
            nest_seqs = utils.pack_sequence_as(nest_sequences, seqs)
            nest_recs = utils.pack_sequence_as(nest_rec_info, recs)
            nest_nonseq = utils.pack_sequence_as(nest_non_sequences, nonseq)
            newargs = list(nest_seqs) + list(nest_recs) + list(nest_nonseq)

            nest_outs = inner_fn(*newargs)

            return utils.flatten(nest_outs)

        fn = wrapper_fn

    outputs, updates = theano.scan(fn, sequences, outputs_info, non_sequences,
                                   **kwargs)

    if use_extension:
        outputs = utils.pack_sequence_as(nest_outputs_info, outputs)

    if "key" not in kwargs or not kwargs["key"]:
        key = "training"
    else:
        key = kwargs["key"]

    if return_updates:
        return outputs, updates

    add_to_collection(_SCAN_UPDATES_KEYS + "/" + key, updates)

    return outputs
