# __init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

from beam import beam
from utils import flatten, pack_sequence_as


__all__ = ["beam", "select_nbest"]


# nested: a nested structure of shape batch * dim
# indices: indices to select
def select_nbest(nested, indices):
    if not isinstance(nested, (list, tuple)):
        return nested[indices]

    flat_list = flatten(nested)
    selected_list = [item[indices] for item in flat_list]

    return pack_sequence_as(nested, selected_list)
