# dropout.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops


def dropout(x, keep_prob, noise_shape=None, seed=None):
    if keep_prob > 1.0 or keep_prob <= 0:
        raise ValueError("keep_prob must be in range (0, 1]")

    if noise_shape:
        shape = noise_shape
    else:
        shape = x.shape

    mask = ops.random.binomial(shape, keep_prob, dtype=x.dtype, seed=seed)

    return x * (1.0 / keep_prob) * mask
