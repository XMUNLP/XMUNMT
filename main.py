#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import ops
import sys
import math
import time
import cPickle
import argparse

import numpy as np
import tensorflow as tf

from metric import bleu
from model import NMT, beamsearch
from optimizer import Optimizer
from data import TextReader, TextIterator
from data.plain import convert_data, data_length
from data.vocab import load_vocab, invert_vocab


def serialize(name, option):
    fd = open(name, "w")
    session = tf.get_default_session()
    params = tf.trainable_variables()
    names = [p.name for p in params]
    vals = dict([(p.name, session.run(p)) for p in params])

    if option["indices"] != None:
        indices = option["indices"]
        vals["indices"] = indices
        option["indices"] = None
    else:
        indices = None

    cPickle.dump(option, fd)
    cPickle.dump(names, fd)
    # compress
    np.savez(fd, **vals)

    # restore
    if indices is not None:
        option["indices"] = indices

    fd.close()


# load model from file
def load_model(name):
    fd = open(name, "r")
    option = cPickle.load(fd)
    names = cPickle.load(fd)
    vals = dict(np.load(fd))

    params = [(n, vals[n]) for n in names]

    if "indices" in vals:
        option["indices"] = vals["indices"]

    fd.close()

    # small fix for compatibility
    if "embedding" not in option:
        option["embedding"] = option["embdim"][0]
        option["attention"] = option["hidden"][2]
        option["hidden"] = option["hidden"][0]

    return option, params


def match_variables(variables, values, ignore_prefix=True):
    var_dict = {}
    val_dict = {}
    matched = []
    not_matched = []

    for var in variables:
        if ignore_prefix:
            name = "/".join(var.name.split("/")[1:])
        else:
            name = var.name

        var_dict[name] = var

    for (name, val) in values:
        if ignore_prefix:
            name = "/".join(name.split("/")[1:])
        val_dict[name] = val

    # matching
    for name in var_dict:
        var = var_dict[name]

        if name in val_dict:
            val = val_dict[name]
            matched.append([var, val])
        else:
            not_matched.append(var)

    return matched, not_matched


def restore_variables(matched, not_matched):
    session = tf.get_default_session()

    for var, val in matched:
        session.run(var.assign(val))

    for var in not_matched:
        sys.stderr.write("%s NOT restored\n" % var.name)


def set_variables(variables, values):
    values = [item[1] for item in values]
    session = tf.get_default_session()

    for p, v in zip(variables, values):
        with tf.device("/cpu:0"):
            session.run(p.assign(v))


def count_parameters(variables):
    n = 0
    session = tf.get_default_session()

    for item in variables:
        v = session.run(item)
        n += v.size

    return n


def load_references(names, case=True):
    references = []
    reader = TextReader(names)
    stream = TextIterator(reader, size=[1, 1])

    for data in stream:
        newdata= []
        for batch in data:
            line = batch[0]
            words = line.strip().split()
            if not case:
                lower = [word.lower() for word in words]
                newdata.append(lower)
            else:
                newdata.append(words)

        references.append(newdata)

    stream.close()

    return references


def translate(model, corpus, **opt):
    fd = open(corpus, "r")
    svocab = model.option["vocabulary"][0][0]
    unk_symbol = model.option["unk"]
    eos_symbol = model.option["eos"]

    trans = []

    for line in fd:
        line = line.strip()
        data, length = convert_data([line], svocab, unk_symbol, eos_symbol)
        hls = beamsearch(model, data, **opt)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans


def save_model(model, name, reader, option, **variables):
    option["indices"] = reader.get_indices()
    option["count"] = [variables["step"], reader.count]
    serialize(name, option)


def validate_model(model, valset, refset, search_option,
                   name, reader, option, **variables):
    trans = translate(model, valset, **search_option)
    bleu_score = bleu(trans, refset)

    step = variables["step"]
    epoch = variables["epoch"]
    global_step = variables["global_step"]

    msg = "global_step: %d, epoch: %d, step: %d, bleu: %2.4f"
    print(msg % (global_step, epoch, step, bleu_score))

    best_score = option["bleu"]
    if bleu_score > best_score:
        option["bleu"] = bleu_score
        save_model(model, name, reader, option, **variables)


def parseargs_train(args):
    msg = "training nmt"
    usage = "main.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "source and target corpus"
    parser.add_argument("--corpus", nargs=2, help=msg)
    msg = "source and target vocabulary"
    parser.add_argument("--vocab", nargs=2, help=msg)
    msg = "model name to save or saved model to initalize, required"
    parser.add_argument("--model", required=True, help=msg)

    msg = "source and target embedding size, default 512"
    parser.add_argument("--embedding", type=int, help=msg)
    msg = "source, target and alignment hidden size, default 1024"
    parser.add_argument("--hidden", type=int, help=msg)
    msg = "attention hidden dimension, default 2048"
    parser.add_argument("--attention", type=int, help=msg)

    msg = "maximum training epoch, default 5"
    parser.add_argument("--maxepoch", type=int, help=msg)
    msg = "learning rate, default 5e-4"
    parser.add_argument("--alpha", type=float, help=msg)
    msg = "batch size, default 128"
    parser.add_argument("--batch", type=int, help=msg)
    msg = "optimizer, default adam"
    parser.add_argument("--optimizer", type=str, help=msg)
    msg = "gradient clipping, default 1.0"
    parser.add_argument("--norm", type=float, help=msg)
    msg = "early stopping iteration, default 0"
    parser.add_argument("--stop", type=int, help=msg)
    msg = "decay factor, default 0.5"
    parser.add_argument("--decay", type=float, help=msg)
    msg = "initialization scale, default 0.08"
    parser.add_argument("--scale", type=float, help=msg)
    msg = "L1 regularizer scale"
    parser.add_argument("--l1-scale", type=float, help=msg)
    msg = "L2 regularizer scale"
    parser.add_argument("--l2-scale", type=float, help=msg)
    msg = "dropout keep probability"
    parser.add_argument("--keep-prob", type=float, help=msg)
    msg = "random seed, default 1234"
    parser.add_argument("--seed", type=int, help=msg)

    msg = "validation dataset"
    parser.add_argument("--validation", type=str, help=msg)
    msg = "reference data"
    parser.add_argument("--references", type=str, nargs="+", help=msg)

    # data processing
    msg = "sort batches"
    parser.add_argument("--sort", type=int, help=msg)
    msg = "shuffle every epcoh"
    parser.add_argument("--shuffle", type=int, help=msg)
    msg = "source and target sentence limit, default 50 (both), 0 to disable"
    parser.add_argument("--limit", type=int, nargs='+', help=msg)

    # save frequency
    msg = "save frequency, default 1000"
    parser.add_argument("--freq", type=int, help=msg)
    msg = "sample frequency, default 50"
    parser.add_argument("--sfreq", type=int, help=msg)
    msg = "validate frequency, default 1000"
    parser.add_argument("--vfreq", type=int, help=msg)

    # control beamsearch
    msg = "beam size, default 10"
    parser.add_argument("--beamsize", type=int, help=msg)
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", type=int, help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    msg = "reset"
    parser.add_argument("--reset", type=int, default=0, help=msg)

    return parser.parse_args(args)


def parseargs_decode(args):
    msg = "translate using exsiting nmt model"
    usage = "main.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    msg = "use arithmetic mean instead of geometric mean"
    parser.add_argument("--arithmetic", action="store_true", help=msg)
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    # running device
    msg = "gpu id, -1 to use cpu"
    parser.add_argument("--gpuid", type=int, default=0, help=msg)

    return parser.parse_args(args)


# default options
def default_option():
    option = {
        "corpus": None,
        "vocab": None,
        "embedding": 512,
        "hidden": 1024,
        "attention": 2048,
        "alpha": 5e-4,
        "batch": 128,
        "optimizer": "adam",
        "norm": 5.0,
        "stop": 0,
        "decay": 0.5,
        "scale": 0.08,
        "l1_scale": None,
        "l2_scale": None,
        "keep_prob": None,
        "cost": 0.0,
        "count": [0, 0],
        "epoch": 1,
        "maxepoch": 5,
        "sort": 20,
        "shuffle": False,
        "limit": [50, 50],
        "freq": 1000,
        "vfreq": 1000,
        "sfreq": 50,
        "seed": 1234,
        "validation": None,
        "references": None,
        "bleu": 0.0,
        "indices": None,
        "beamsize": 10,
        "normalize": False,
        "maxlen": None,
        "minlen": None,
        "unk": "UNK",
        "eos": "</s>"
    }

    return option


def args_to_dict(arguments):
    return arguments.__dict__


def override_if_not_none(opt1, opt2, key):
    if key in opt2:
        value = opt2[key]
    else:
        value = None

    if value is not None:
        opt1[key] = value
    else:
        if key not in opt1:
            opt1[key] = None


# override default options
def override(option, newopt):

    # training corpus
    if newopt["corpus"] is None and option["corpus"] is None:
        raise RuntimeError("error: no training corpus specified")

    # vocabulary
    if newopt["vocab"] is None and option["vocab"] is None:
        raise RuntimeError("error: no training vocabulary specified")

    override_if_not_none(option, newopt, "corpus")

    # vocabulary and model paramters cannot be overrided
    if option["vocab"] is None:
        option["vocab"] = newopt["vocab"]

        svocab = load_vocab(option["vocab"][0])
        tvocab = load_vocab(option["vocab"][1])
        isvocab = invert_vocab(svocab)
        itvocab = invert_vocab(tvocab)

        option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, newopt, "embedding")
        override_if_not_none(option, newopt, "hidden")
        override_if_not_none(option, newopt, "attention")

    # training options
    override_if_not_none(option, newopt, "maxepoch")
    override_if_not_none(option, newopt, "alpha")
    override_if_not_none(option, newopt, "batch")
    override_if_not_none(option, newopt, "optimizer")
    override_if_not_none(option, newopt, "norm")
    override_if_not_none(option, newopt, "stop")
    override_if_not_none(option, newopt, "decay")
    override_if_not_none(option, newopt, "scale")
    override_if_not_none(option, newopt, "l1_scale")
    override_if_not_none(option, newopt, "l2_scale")
    override_if_not_none(option, newopt, "keep_prob")

    # runtime information
    override_if_not_none(option, newopt, "validation")
    override_if_not_none(option, newopt, "references")
    override_if_not_none(option, newopt, "freq")
    override_if_not_none(option, newopt, "vfreq")
    override_if_not_none(option, newopt, "sfreq")
    override_if_not_none(option, newopt, "seed")
    override_if_not_none(option, newopt, "sort")
    override_if_not_none(option, newopt, "shuffle")
    override_if_not_none(option, newopt, "limit")
    override_if_not_none(option, newopt, "bleu")
    override_if_not_none(option, newopt, "indices")

    override_if_not_none(option, newopt, "cost")
    override_if_not_none(option, newopt, "step")
    override_if_not_none(option, newopt, "epoch")
    override_if_not_none(option, newopt, "global_cost")
    override_if_not_none(option, newopt, "global_step")
    override_if_not_none(option, newopt, "local_cost")
    override_if_not_none(option, newopt, "local_step")

    # beamsearch
    override_if_not_none(option, newopt, "beamsize")
    override_if_not_none(option, newopt, "normalize")
    override_if_not_none(option, newopt, "maxlen")
    override_if_not_none(option, newopt, "minlen")


def print_option(option):
    isvocab = option["vocabulary"][0][1]
    itvocab = option["vocabulary"][1][1]

    print ""
    print "options"

    print "corpus:", option["corpus"]
    print "vocab:", option["vocab"]
    print "vocabsize:", [len(isvocab), len(itvocab)]

    print "embedding:", option["embedding"]
    print "hidden:", option["hidden"]
    print "attention:", option["attention"]

    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]
    print "keep-prob:", option["keep_prob"]

    print "validation:", option["validation"]
    print "references:", option["references"]
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "sfreq:", option["sfreq"]
    print "seed:", option["seed"]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]
    print "scale:", option["scale"]
    print "L1-scale:", option["l1_scale"]
    print "L2-scale:", option["l2_scale"]

    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]

    # special symbols
    print "unk:", option["unk"]
    print "eos:", option["eos"]


def skip_stream(stream, count):
    for i in range(count):
        stream.next()


def get_filename(name):
    s = name.split(".")
    return s[0]


def train(args):
    option = default_option()

    # predefined model names
    pathname, basename = os.path.split(args.model)
    modelname = get_filename(basename)
    autoname = os.path.join(pathname, modelname + ".autosave.pkl")
    bestname = os.path.join(pathname, modelname + ".best.pkl")

    # load models
    if os.path.exists(args.model):
        opt, params = load_model(args.model)
        override(option, opt)
        init = False
    else:
        init = True
        params = None

    override(option, args_to_dict(args))
    print_option(option)

    # load references
    if option["references"]:
        references = load_references(option["references"])
    else:
        references = None

    # input corpus
    batch = option["batch"]
    sortk = option["sort"] or 1
    shuffle = option["seed"] if option["shuffle"] else None
    reader = TextReader(option["corpus"], shuffle)
    processor = [data_length, data_length]
    stream = TextIterator(reader, [batch, batch * sortk], processor,
                          option["limit"], option["sort"])

    if shuffle and option["indices"] is not None:
        reader.set_indices(option["indices"])

    if args.reset:
        option["count"] = [0, 0]
        option["epoch"] = 0
        option["cost"] = 0.0

    skip_stream(reader, option["count"][1])

    # beamsearch option
    search_opt = {
        "beamsize": option["beamsize"],
        "normalize": option["normalize"],
        "maxlen": option["maxlen"],
        "minlen": option["minlen"]
    }

    # misc
    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs
    unk = option["unk"]
    eos = option["eos"]

    scale = option["scale"]

    # set seed
    np.random.seed(option["seed"])
    tf.set_random_seed(option["seed"])

    initializer = tf.random_uniform_initializer(-scale, scale)
    model = NMT(option["embedding"], option["hidden"], option["attention"],
                len(isvocab), len(itvocab), initializer=initializer)

    model.option = option

    # create optimizer
    optim = Optimizer(model, algorithm=option["optimizer"], norm=True,
                        constraint=("norm", option["norm"]))

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    with tf.Session(config=config):
        tf.global_variables_initializer().run()

        print "parameters:", count_parameters(tf.trainable_variables())

        if not init:
            set_variables(tf.trainable_variables(), params)

        def lr_decay_fn(*args, **kwargs):
            global_step = kwargs["global_step"]
            step = kwargs["step"]
            epoch = kwargs["epoch"]
            option["alpha"] = option["alpha"] * option["decay"]
            msg = "G/E/S: %d/%d/%d  alpha: %f"
            print(msg % (global_step, epoch, step, option["alpha"]))

        def train_step_fn(data, **variables):
            alpha = option["alpha"]
            global_step = variables["global_step"]
            step = variables["step"]
            epoch = variables["epoch"]

            xdata, xlen = convert_data(data[0], svocab, unk, eos)
            ydata, ylen = convert_data(data[1], tvocab, unk, eos)

            t1 = time.time()
            cost, norm = optim.optimize(xdata, xlen, ydata, ylen)
            optim.update(alpha=alpha)
            t2 = time.time()

            cost = cost * len(ylen) / sum(ylen)

            msg = "G/E/S: %d/%d/%d cost: %f norm: %f time: %f"
            print(msg % (global_step, epoch, step, cost, norm, t2 - t1))

            return cost / math.log(2)

        def sample_fn(*args, **kwargs):
            data = args[0]
            batch = len(data[0])
            ind = np.random.randint(0, batch)
            sdata = data[0][ind]
            tdata = data[1][ind]
            xdata, xlen = convert_data(data[0], svocab, unk, eos)
            xdata = xdata[:, ind:ind + 1]
            xlen = xlen[ind:ind+1]
            hls = beamsearch(model, xdata, xlen,  **search_opt)
            best, score = hls[0]
            print(sdata)
            print(tdata)
            print(" ".join(best[:-1]))

        def cost_summary(*args, **kwargs):
            cost = kwargs["local_cost"]
            global_cost = kwargs["global_cost"]
            step = kwargs["local_step"]
            global_step = kwargs["global_step"]

            ac, gac = cost / step, global_cost / global_step

            print("averaged cost: %f/%f" % (ac, gac))

        def stop_fn(*args, **kwargs):
            if option["maxepoch"] < kwargs["epoch"]:
                raise StopIteration

        def save_fn(*args, **kwargs):
            save_model(model, autoname, reader, option, **kwargs)

        def validate_fn(*args, **kwargs):
            if option["validation"] and references:
                validate_model(model, option["validation"], references,
                               search_opt, bestname, reader, option, **kwargs)

        # global/epoch
        lr_decay_hook = ops.train_loop.hook(option["stop"], 1, lr_decay_fn)
        # local
        save_hook = ops.train_loop.hook(0, option["freq"], save_fn)
        e_save_hook = ops.train_loop.hook(0, 2, save_fn)
        # local
        sample_hook = ops.train_loop.hook(0, option["sfreq"], sample_fn)
        # global/local/epoch
        validate_hook = ops.train_loop.hook(0, option["vfreq"], validate_fn)
        e_validate_hook = ops.train_loop.hook(0, 1, validate_fn)
        # epoch
        cost_summary_hook = ops.train_loop.hook(0, 1, cost_summary)
        # global/epoch
        stop_hook = ops.train_loop.hook(0, 1, stop_fn)

        global_level_hooks = []
        local_level_hooks = [save_hook, sample_hook, validate_hook]
        epoch_level_hooks = [lr_decay_hook, cost_summary_hook, e_save_hook,
                             e_validate_hook, stop_hook]

        ops.train_loop.train_loop(stream, train_step_fn, option,
                                  global_level_hooks, local_level_hooks,
                                  epoch_level_hooks)

    stream.close()


def decode(args):
    option, values = load_model(args.model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if args.gpuid >= 0:
        config.gpu_options.visible_device_list = "%d" % args.gpuid

    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    unk_sym = option["unk"]
    eos_sym = option["eos"]

    count = 0

    doption = {
        "maxlen": args.maxlen,
        "minlen": args.minlen,
        "beamsize": args.beamsize,
        "normalize": args.normalize
    }

    # create graph
    model = NMT(option["embedding"], option["hidden"], option["attention"],
                len(isvocab), len(itvocab))

    model.option = option

    with tf.Session(config=config):
        tf.global_variables_initializer().run()
        set_variables(tf.trainable_variables(), values)

        while True:
            line = sys.stdin.readline()

            if line == "":
                break

            data = [line]
            seq, seq_len = convert_data(data, svocab, unk_sym, eos_sym)
            t1 = time.time()
            tlist = beamsearch(model, seq, seq_len, **doption)
            t2 = time.time()

            if len(tlist) == 0:
                sys.stdout.write("\n")
                score = -10000.0
            else:
                best, score = tlist[0]
                sys.stdout.write(" ".join(best[:-1]))
                sys.stdout.write("\n")

            count = count + 1
            sys.stderr.write(str(count) + " ")
            sys.stderr.write(str(score) + " " + str(t2 - t1) + "\n")


def helpinfo():
    print "usage:"
    print "\tmain.py <command> [<args>]"
    print "using main.py train --help to see training options"
    print "using main.py translate --help to see translation options"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == "train":
            print "training command:"
            print " ".join(sys.argv)
            args = parseargs_train(sys.argv[2:])
            train(args)
        elif command == "translate":
            sys.stderr.write(" ".join(sys.argv))
            sys.stderr.write("\n")
            args = parseargs_decode(sys.argv[2:])
            decode(args)
        else:
            helpinfo()
