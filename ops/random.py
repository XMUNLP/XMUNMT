# random.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import theano
import theano.sandbox.rng_mrg


_RANDOM_STREAM = theano.sandbox.rng_mrg.MRG_RandomStreams()


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    _RANDOM_STREAM.seed(seed)
    return _RANDOM_STREAM.normal(shape, mean, stddev, dtype=dtype)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    _RANDOM_STREAM.seed(seed)
    return _RANDOM_STREAM.uniform(shape, minval, maxval, dtype=dtype)


def binomial(shape, prob, num_samples=1, dtype=None, seed=None):
    _RANDOM_STREAM.seed(seed)
    return _RANDOM_STREAM.binomial(shape, num_samples, prob, dtype=dtype)


def multinomial(dist, num_samples=1, seed=None):
    if dist.ndim != 2:
        raise ValueError("dist is assumed to have shape [batch, dim]")

    _RANDOM_STREAM.seed(seed)
    return _RANDOM_STREAM.multinomial(n=num_samples, pvals=dist)
