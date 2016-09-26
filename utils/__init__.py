# utils/__init__.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn


def add_if_not_exsit(option, key, value):
    if key not in option:
        option[key] = value


def get_or_default(opt, key, default):
    if key in opt:
        return opt[key]
    return default
