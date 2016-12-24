# dtype.py
# helper functions for dtype
# author: Playinf
# email: playinf@stu.xmu.edu.cn


_ALL_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128"
]


_INT_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64"
]

_FLOAT_DTYPES = [
    "float16",
    "float32",
    "float64"
]


def is_dtype(dtype):
    if dtype not in _ALL_DTYPES:
        return False
    return True


def is_integer_dtype(dtype):
    if not is_dtype(dtype):
        raise TypeError("data type '%s' not understood" % dtype)
    if dtype in _INT_DTYPES:
        return True
    return False


def is_floating_dtype(dtype):
    if not is_dtype(dtype):
        raise TypeError("data type '%s' not understood" % dtype)
    if dtype in _FLOAT_DTYPES:
        return True
    return False
