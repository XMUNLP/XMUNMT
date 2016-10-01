#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import math
import time
import cPickle
import argparse

import numpy as np

from metric import bleu
from optimizer import optimizer
from data import textreader, textiterator, processdata, getlen
from model.rnnsearch import rnnsearch, rnnsearch_config, beamsearch


# standard rnnsearch configuration (groundhog version)
def get_config():
    config = rnnsearch_config()

    config["*/concat"] = False
    config["*/output_major"] = False

    # embedding
    config["source_embedding/bias/use"] = True
    config["target_embedding/bias/use"] = True

    # encoder
    config["encoder/forward_rnn/reset_gate/bias/use"] = False
    config["encoder/forward_rnn/update_gate/bias/use"] = False
    config["encoder/forward_rnn/candidate/bias/use"] = True
    config["encoder/backward_rnn/reset_gate/bias/use"] = False
    config["encoder/backward_rnn/update_gate/bias/use"] = False
    config["encoder/backward_rnn/candidate/bias/use"] = True

    # decoder
    config["decoder/init_transform/bias/use"] = True
    config["decoder/annotation_transform/bias/use"] = False
    config["decoder/state_transform/bias/use"] = False
    config["decoder/context_transform/bias/use"] = False
    config["decoder/rnn/reset_gate/bias/use"] = False
    config["decoder/rnn/update_gate/bias/use"] = False
    config["decoder/rnn/candidate/bias/use"] = True
    config["decoder/maxout/bias/use"] = True
    config["decoder/deepout/bias/use"] = False
    config["decoder/classify/bias/use"] = True

    return config


def loadvocab(file):
    fd = open(file, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def uniform(params, lower, upper, dtype="float32"):

    for p in params:
        s = p.get_value().shape
        v = np.random.uniform(lower, upper, s).astype(dtype)
        p.set_value(v)


def parameters(params):
    n = 0

    for item in params:
        v = item.get_value()
        n += v.size

    return n


def serialize(name, model):
    fd = open(name, "w")
    option = model.option
    params = model.parameter

    pval = [p.get_value() for p in params]

    cPickle.dump(option, fd)
    cPickle.dump(pval, fd)

    fd.close()


# load model from file
def loadmodel(name):
    fd = open(name, "r")
    option = cPickle.load(fd)
    params = cPickle.load(fd)

    return option, params


def set_variables(params, values):
    for p, v in zip(params, values):
        p.set_value(v)


def loadreferences(names, case=True):
    references = []
    reader = textreader(names)
    stream = textiterator(reader, size=[1, 1])

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


def validate(scorpus, tcorpus, model, batch):

    if not scorpus or not tcorpus:
        return None

    reader = textreader([scorpus, tcorpus])
    stream = textiterator(reader, [batch, batch])
    svocabs, tvocabs = model.vocabulary
    totcost = 0.0
    count = 0

    for data in stream:
        xdata, xmask = processdata(data[0], svocabs[0], model.option["eos"])
        ydata, ymask = processdata(data[1], tvocabs[0], model.option["eos"])
        cost = model.compute(xdata, xmask, ydata, ymask)
        cost = cost[0]
        cost = cost * ymask.shape[1] / ymask.sum()
        totcost += cost / math.log(2)
        count = count + 1

    stream.close()

    bpc = totcost / count

    return bpc


def translate(model, corpus, **opt):
    fd = open(corpus, "r")
    svocab = model.option["vocabulary"][0][0]
    trans = []

    for line in fd:
        line = line.strip()
        data, mask = processdata([line], svocab, eos=model.option["eos"])
        hls = beamsearch(model, data, **opt)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans


def parseargs_train(args):
    msg = "training rnnsearch"
    usage = "rnnsearch.py train [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description = msg, usage = usage)

    # training corpus
    msg = "source and target corpus"
    parser.add_argument("--corpus", nargs=2, help=msg)
    # training vocabulary
    msg = "source and target vocabulary"
    parser.add_argument("--vocab", nargs=2, help=msg)
    # output model
    msg = "model name to save or saved model to initalize, required"
    parser.add_argument("--model", required=True, help=msg)

    # embedding size
    msg = "source and target embedding size, default 620"
    parser.add_argument("--embdim", nargs=2, type=int, help=msg)
    # hidden size
    msg = "source, target and alignment hidden size, default 1000"
    parser.add_argument("--hidden", nargs=3, type=int, help=msg)
    # maxout dim
    msg = "maxout hidden dimension, default 500"
    parser.add_argument("--maxhid", type=int, help=msg)
    # maxout number
    msg = "maxout number, default 2"
    parser.add_argument("--maxpart", default=2, type=int, help=msg)
    # deepout dim
    msg = "deepout hidden dimension, default 620"
    parser.add_argument("--deephid", type=int, help=msg)

    # epoch
    msg = "maximum training epoch, default 5"
    parser.add_argument("--maxepoch", type=int, help=msg)
    # learning rate
    msg = "learning rate, default 5e-4"
    parser.add_argument("--alpha", type=float, help=msg)
    # momentum
    msg = "momentum, default 0.0"
    parser.add_argument("--momentum", type=float, help=msg)
    # batch
    msg = "batch size, default 128"
    parser.add_argument("--batch", type=int, help=msg)
    # training algorhtm
    msg = "optimizer, default rmsprop"
    parser.add_argument("--optimizer", type=str, help=msg)
    # gradient renormalization
    msg = "gradient renormalization, default 1.0"
    parser.add_argument("--norm", type=float, help=msg)
    # early stopping
    msg = "early stopping iteration, default 0"
    parser.add_argument("--stop", type=int, help=msg)
    # decay factor
    msg = "decay factor, default 0.5"
    parser.add_argument("--decay", type=float, help=msg)
    # random seed
    msg = "random seed, default 1234"
    parser.add_argument("--seed", type=int, help=msg)

    # compute bit per cost
    msg = "compute bit per cost on validate dataset"
    parser.add_argument("--bpc", action="store_true", help=msg)
    # validate data
    msg = "validate dataset"
    parser.add_argument("--validate", type=str, help=msg)
    # reference
    msg = "reference data"
    parser.add_argument("--ref", type=str, nargs="+", help=msg)

    # data processing
    msg = "sort batches"
    parser.add_argument("--sort", type=int, help=msg)
    msg = "shuffle every epcoh"
    parser.add_argument("--shuffle", type=int, help=msg)
    msg = "source and target sentence limit, default 50 (both))"
    parser.add_argument("--limit", type=int, nargs='+', help=msg)


    # save frequency
    msg = "save frequency, default 1000"
    parser.add_argument("--freq", type=int, help=msg)
    # sample frequency
    msg = "sample frequency, default 50"
    parser.add_argument("--sfreq", type=int, help=msg)
    # validate frequency
    msg = "validate frequency, default 1000"
    parser.add_argument("--vfreq", type=int, help=msg)

    # control beamsearch
    msg = "beam size, default 10"
    parser.add_argument("--beamsize", type=int, help=msg)
    # normalize
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", type=int, help=msg)
    # max length
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    # min length
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    return parser.parse_args(args)


def parseargs_decode(args):
    msg = "translate using exsiting nmt model"
    usage = "rnnsearch.py translate [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    # input model
    msg = "trained model"
    parser.add_argument("--model", required=True, help=msg)
    # beam size
    msg = "beam size"
    parser.add_argument("--beamsize", default=10, type=int, help=msg)
    # normalize
    msg = "normalize probability by the length of cadidate sentences"
    parser.add_argument("--normalize", action="store_true", help=msg)
    # max length
    msg = "max translation length"
    parser.add_argument("--maxlen", type=int, help=msg)
    # min length
    msg = "min translation length"
    parser.add_argument("--minlen", type=int, help=msg)

    return parser.parse_args(args)


# default options
def getoption():
    option = {}

    # training corpus and vocabulary
    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = [620, 620]
    option["hidden"] = [1000, 1000, 1000]
    option["maxpart"] = 2
    option["maxhid"] = 500
    option["deephid"] = 620

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["momentum"] = 0.0
    option["optimizer"] = "rmsprop"
    option["variant"] = "graves"
    option["norm"] = 1.0
    option["stop"] = 0
    option["decay"] = 0.5

    # runtime information
    option["cost"] = 0
    option["count"] = 0
    option["epoch"] = 0
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["sfreq"] = 50
    option["seed"] = 1234
    option["validate"] = None
    option["ref"] = None

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    return option


def override_if_not_none(option, args, key):
    value = args.__dict__[key]
    option[key] = value if value != None else option[key]


# override default options
def override(option, args):

    # training corpus
    if args.corpus == None and option["corpus"] == None:
        raise ValueError("error: no training corpus specified")

    # vocabulary
    if args.vocab == None and option["vocab"] == None:
        raise ValueError("error: no training vocabulary specified")

    if args.limit and len(args.limit) > 2:
        raise ValueError("error: invalid number of --limit argument (<=2)")

    if args.limit and len(args.limit) == 1:
        args.limit = args.limit * 2

    override_if_not_none(option, args, "corpus")

    # vocabulary and model paramters cannot be overrided
    if option["vocab"] == None:
        option["vocab"] = args.vocab
        svocab = loadvocab(args.vocab[0])
        tvocab = loadvocab(args.vocab[1])
        isvocab = invertvoc(svocab)
        itvocab = invertvoc(tvocab)

        # compatible with groundhog
        option["source_eos_id"] = len(isvocab)
        option["target_eos_id"] = len(itvocab)

        option["eos"] = "<eos>"
        svocab["<eos>"] = option["source_eos_id"]
        tvocab["<eos>"] = option["target_eos_id"]
        isvocab[option["source_eos_id"]] = "<eos>"
        itvocab[option["target_eos_id"]] = "<eos>"

        option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, args, "embdim")
        override_if_not_none(option, args, "hidden")
        override_if_not_none(option, args, "maxhid")
        override_if_not_none(option, args, "maxpart")
        override_if_not_none(option, args, "deephid")

    # training options
    override_if_not_none(option, args, "maxepoch")
    override_if_not_none(option, args, "alpha")
    override_if_not_none(option, args, "momentum")
    override_if_not_none(option, args, "batch")
    override_if_not_none(option, args, "optimizer")
    override_if_not_none(option, args, "norm")
    override_if_not_none(option, args, "stop")
    override_if_not_none(option, args, "decay")

    # runtime information
    override_if_not_none(option, args, "validate")
    override_if_not_none(option, args, "ref")
    override_if_not_none(option, args, "freq")
    override_if_not_none(option, args, "vfreq")
    override_if_not_none(option, args, "sfreq")
    override_if_not_none(option, args, "seed")
    override_if_not_none(option, args, "sort")
    override_if_not_none(option, args, "shuffle")
    override_if_not_none(option, args, "limit")

    # beamsearch
    override_if_not_none(option, args, "beamsize")
    override_if_not_none(option, args, "normalize")
    override_if_not_none(option, args, "maxlen")
    override_if_not_none(option, args, "minlen")


def print_option(option):
    isvocab = option["vocabulary"][0][1]
    itvocab = option["vocabulary"][1][1]

    print ""
    print "options"

    print "corpus:", option["corpus"]
    print "vocab:", option["vocab"]
    print "vocabsize:", [len(isvocab), len(itvocab)]

    print "embdim:", option["embdim"]
    print "hidden:", option["hidden"]
    print "maxhid:", option["maxhid"]
    print "maxpart:", option["maxpart"]
    print "deephid:", option["deephid"]

    print "maxepoch:", option["maxepoch"]
    print "alpha:", option["alpha"]
    print "momentum:", option["momentum"]
    print "batch:", option["batch"]
    print "optimizer:", option["optimizer"]
    print "norm:", option["norm"]
    print "stop:", option["stop"]
    print "decay:", option["decay"]

    print "validate:", option["validate"]
    print "ref:", option["ref"]
    print "freq:", option["freq"]
    print "vfreq:", option["vfreq"]
    print "sfreq:", option["sfreq"]
    print "seed:", option["seed"]
    print "sort:", option["sort"]
    print "shuffle:", option["shuffle"]
    print "limit:", option["limit"]

    print "beamsize:", option["beamsize"]
    print "normalize:", option["normalize"]
    print "maxlen:", option["maxlen"]
    print "minlen:", option["minlen"]


def skipstream(stream, count):
    for i in range(count):
        stream.next()


def getfilename(name):
    s = name.split(".")
    return s[0]


def train(args):
    option = getoption()

    if os.path.exists(args.model):
        option, params = loadmodel(args.model)
        init = False
    else:
        init = True

    override(option, args)
    print_option(option)

    # set seed
    np.random.seed(option["seed"])

    if option["ref"]:
        references = loadreferences(option["ref"])
    else:
        references = None

    svocabs, tvocabs = option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    autoname = os.path.join(pathname, modelname + ".autosave.pkl")
    bestname = os.path.join(pathname, modelname + ".best.pkl")
    batch = option["batch"]
    sortk = option["sort"] or 1
    shuffle = option["seed"] if option["shuffle"] else None
    reader = textreader(option["corpus"], shuffle)
    processor = [getlen, getlen] if option["sort"] else None
    stream = textiterator(reader, [batch, batch * sortk], processor,
                          option["limit"], option["sort"])

    if shuffle and "indices" in option and option["indices"] is not None:
        reader.set_indices(option["indices"])

    skipstream(stream, option["count"])
    epoch = option["epoch"]
    maxepoch = option["maxepoch"]
    option["model"] = "rnnsearch"

    model = rnnsearch(get_config(), **option)

    if init:
        uniform(model.parameter, -0.08, 0.08)
    else:
        set_variables(model.parameter, params)

    print "parameters:", parameters(model.parameter)

    # tuning option
    toption = {}
    toption["algorithm"] = option["optimizer"]
    toption["variant"] = option["variant"]
    toption["constraint"] = ("norm", option["norm"])
    toption["norm"] = True
    toption["initialize"] = option["shared"] if "shared" in option else False
    trainer = optimizer(model, **toption)
    alpha = option["alpha"]

    # beamsearch option
    doption = {}
    doption["beamsize"] = option["beamsize"]
    doption["normalize"] = option["normalize"]
    doption["maxlen"] = option["maxlen"]
    doption["minlen"] = option["minlen"]

    best_score = 0.0

    for i in range(epoch, maxepoch):
        totcost = 0.0
        for data in stream:
            xdata, xmask = processdata(data[0], svocab, eos=option["eos"])
            ydata, ymask = processdata(data[1], tvocab, eos=option["eos"])

            t1 = time.time()
            cost, norm = trainer.optimize(xdata, xmask, ydata, ymask)
            trainer.update(alpha = alpha)
            t2 = time.time()

            option["count"] += 1
            count = option["count"]

            cost = cost * ymask.shape[1] / ymask.sum()
            totcost += cost / math.log(2)
            print i + 1, count, cost, norm, t2 - t1

            option["cost"] = totcost

            # save model
            if count % option["freq"] == 0:
                svars = [p.get_value() for p in trainer.parameter]
                model.option = option
                model.option["shared"] = svars
                model.option["indices"] = reader.get_indices()
                serialize(autoname, model)

            if count % option["vfreq"] == 0:
                if option["validate"] and references:
                    trans = translate(model, option["validate"], **doption)
                    bleu_score = bleu(trans, references)
                    print "bleu: %2.4f" % bleu_score
                    if bleu_score > best_score:
                        best_score = bleu_score
                        model.option = option
                        model.option["shared"] = None
                        model.option["indices"] = None
                        serialize(bestname, model)

            if count % option["sfreq"] == 0:
                ind = np.random.randint(0, batch)
                sdata = data[0][ind]
                tdata = data[1][ind]
                xdata = xdata[:, ind : ind + 1]
                hls = beamsearch(model, xdata)
                if len(hls) > 0:
                    best, score = hls[0]
                    print sdata
                    print tdata
                    print " ".join(best[:-1])
                else:
                    print sdata
                    print tdata
                    print "warning: no translation"

        print "--------------------------------------------------"

        if option["vfreq"] and references:
            trans = translate(model, option["validate"], **doption)
            bleu_score = bleu(trans, references)
            print "iter: %d, bleu: %2.4f" % (i + 1, bleu_score)
            if bleu_score > best_score:
                best_score = bleu_score
                model.option = option
                model.option["shared"] = None
                model.option["indices"] = None
                serialize(bestname, model)

        print "averaged cost: ", totcost / option["count"]
        print "--------------------------------------------------"

        # early stopping
        if i + 1 >= option["stop"]:
            alpha = alpha * option["decay"]

        stream.reset()
        option["epoch"] = i + 1
        option["count"] = 0
        option["alpha"] = alpha
        model.option = option

        # update autosave
        svars = [p.get_value() for p in trainer.parameter]
        model.option = option
        model.option["shared"] = svars
        model.option["indices"] = reader.get_indices()
        serialize(autoname, model)

    print "best(bleu): %2.4f" % best_score

    stream.close()


def decode(args):
    option, params = loadmodel(args.model)
    model = rnnsearch(get_config(), **option)

    set_variables(model.parameter, params)

    svocabs, tvocabs = model.option["vocabulary"]
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    option = {}
    option["maxlen"] = args.maxlen
    option["minlen"] = args.minlen
    option["beamsize"] = args.beamsize
    option["normalize"] = args.normalize

    while True:
        line = sys.stdin.readline()

        if line == "":
            break

        data = [line]
        seq, mask = processdata(data, svocab, eos=option["eos"])
        t1 = time.time()
        tlist = beamsearch(model, seq, **option)
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
    print "\trnnsearch.py <command> [<args>]"
    print "using rnnsearch.py train --help to see training options"
    print "using rnnsearch.py translate --help to see translation options"


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
