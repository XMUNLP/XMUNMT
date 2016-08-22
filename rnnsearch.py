#!/usr/bin/python
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import math
import time
import numpy
import cPickle
import argparse

from optimizer import optimizer
from data import batchstream, processdata
from metric.bleu import bleu
from model.rnnsearch import rnnsearch, beamsearch

def loadvocab(file):
    fd = open(file, 'r')
    vocab = cPickle.load(fd)
    fd.close()
    return vocab

def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v

def uniform(params, lower, upper, precision = 'float32'):

    for p in params:
        s = p.get_value().shape
        v = numpy.random.uniform(lower, upper, s).astype(precision)
        p.set_value(v)

def parameters(params):
    n = 0

    for item in params:
        v = item.get_value()
        n += v.size

    return n

def serialize(name, model):
    fd = open(name, 'w')
    option = model.option
    params = model.parameter
    cPickle.dump(option, fd)

    pval = {}

    for param in params:
        pval[param.name] = param.get_value()

    cPickle.dump(pval, fd)
    fd.close()

# load model from file
def loadmodel(name):
    fd = open(name, 'r')
    option = cPickle.load(fd)
    params = cPickle.load(fd)
    model = rnnsearch(**option)
    mparams = {}

    for param in model.parameter:
        mparams[param.name] = param

    for pname in params:
        keys = pname.split('/')
        newkey = ['rnnsearch'] + keys[1:]
        newkey = '/'.join(newkey)

        if newkey in mparams:
            mparams[newkey].set_value(params[pname])

    fd.close()

    return model

def loadreferences(names, case = True):
    references = []
    stream = batchstream(names)

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

    stream = batchstream([scorpus, tcorpus], batch)
    svocabs, tvocabs = model.vocabulary
    totcost = 0.0
    count = 0

    for data in stream:
        xdata, xmask = processdata(data[0], svocabs[0])
        ydata, ymask = processdata(data[1], tvocabs[0])
        cost = model.compute(xdata, xmask, ydata, ymask)
        cost = cost[0]
        cost = cost * ymask.shape[1] / ymask.sum()
        totcost += cost / math.log(2)
        count = count + 1

    stream.close()

    bpc = totcost / count

    return bpc

def translate(model, corpus, **opt):
    fd = open(corpus, 'r')
    svocab = model.option['vocabulary'][0][0]
    trans = []

    for line in fd:
        line = line.strip()
        data, mask = processdata([line], svocab)
        hls = beamsearch(model, data, **opt)
        if len(hls) > 0:
            best, score = hls[0]
            trans.append(best[:-1])
        else:
            trans.append([])

    fd.close()

    return trans

def parseargs_train(args):
    desc = 'training rnnsearch'
    usage = 'rnnsearch.py train [<args>] [-h | --help]'
    parser = argparse.ArgumentParser(description = desc, usage = usage)

    # training corpus
    desc = 'source and target corpus'
    parser.add_argument('--corpus', nargs = 2, help = desc)
    # training vocabulary
    desc = 'source and target vocabulary'
    parser.add_argument('--vocab', nargs = 2, help = desc)
    # output model
    desc = 'model name to save or saved model to initalize, required'
    parser.add_argument('--model', required = True, help = desc)

    # embedding size
    desc = 'source and target embedding size, default 620'
    parser.add_argument('--embdim', nargs = 2, type = int, help = desc)
    # hidden size
    desc = 'source, target and alignment hidden size, default 1000'
    parser.add_argument('--hidden', nargs = 3, type = int, help = desc)
    # maxout dim
    desc = 'maxout hidden dimension, default 500'
    parser.add_argument('--maxhid', type = int, help = desc)
    # maxout number
    desc = 'maxout number, default 2'
    parser.add_argument('--maxpart', default = 2, type = int, help = desc)
    # deepout dim
    desc = 'deepout hidden dimension, default 620'
    parser.add_argument('--deephid', type = int, help = desc)

    # epoch
    desc = 'maximum training epoch, default 5'
    parser.add_argument('--maxepoch', type = int, help = desc)
    # learning rate
    desc = 'learning rate, default 5e-4'
    parser.add_argument('--alpha', type = float, help = desc)
    # momentum
    desc = 'momentum, default 0.0'
    parser.add_argument('--momentum', type = float, help = desc)
    # batch
    desc = 'batch size, default 128'
    parser.add_argument('--batch', type = int, help = desc)
    # training algorhtm
    desc = 'optimizer, default rmsprop'
    parser.add_argument('--optimizer', type = str, help = desc)
    # gradient renormalization
    desc = 'gradient renormalization, default 1.0'
    parser.add_argument('--norm', type = float, help = desc)
    # early stopping
    desc = 'early stopping iteration, default 0'
    parser.add_argument('--stop', type = int, help = desc)
    # decay factor
    desc = 'decay factor, default 0.5'
    parser.add_argument('--decay', type = float, help = desc)
    # random seed
    desc = 'random seed, default 1234'
    parser.add_argument('--seed', type = int, help = desc)

    # compute bit per cost
    desc = 'compute bit per cost on validate dataset'
    parser.add_argument('--bpc', action = 'store_true', help = desc)
    # validate data
    desc = 'validate dataset'
    parser.add_argument('--validate', type = str, help = desc)
    # reference
    desc = 'reference data'
    parser.add_argument('--ref', type = str, nargs = '+', help = desc)

    # save frequency
    desc = 'save frequency, default 1000'
    parser.add_argument('--freq', type = int, help = desc)
    # sample frequency
    desc = 'sample frequency, default 50'
    parser.add_argument('--sfreq', type = int, help = desc)
    # validate frequency
    desc = 'validate frequency, default 1000'
    parser.add_argument('--vfreq', type = int, help = desc)

    # control beamsearch
    desc = 'beam size'
    parser.add_argument('--beam-size', type = int, help = desc)
    # normalize
    desc = 'normalize probability by the length of cadidate sentences'
    parser.add_argument('--normalize', type = bool, help = desc)
    # max length
    desc = 'max translation length'
    parser.add_argument('--maxlen', type = int, help = desc)
    # min length
    desc = 'min translation length'
    parser.add_argument('--minlen', type = int, help = desc)

    return parser.parse_args(args)

def parseargs_decode(args):
    desc = 'translate using exsiting nmt model'
    usage = 'rnnsearch.py translate [<args>] [-h | --help]'
    parser = argparse.ArgumentParser(description = desc, usage = usage)

    # input model
    desc = 'trained model'
    parser.add_argument('--model', required = True, help = desc)
    # beam size
    desc = 'beam size'
    parser.add_argument('--beam-size', default = 10, type = int, help = desc)
    # normalize
    desc = 'normalize probability by the length of cadidate sentences'
    parser.add_argument('--normalize', action = 'store_true', help = desc)
    # max length
    desc = 'max translation length'
    parser.add_argument('--maxlen', type = int, help = desc)
    # min length
    desc = 'min translation length'
    parser.add_argument('--minlen', type = int, help = desc)

    return parser.parse_args(args)

# default options
def getoption():
    option = {}

    # training corpus and vocabulary
    option['corpus'] = None
    option['vocab'] = None

    # model parameters
    option['embdim'] = [620, 620]
    option['hidden'] = [1000, 1000, 1000]
    option['maxpart'] = 2
    option['maxhid'] = 500
    option['deephid'] = 620

    # tuning options
    option['alpha'] = 5e-4
    option['batch'] = 128
    option['momentum'] = 0.0
    option['optimizer'] = 'rmsprop'
    option['variant'] = 'graves'
    option['norm'] = 1.0
    option['stop'] = 0
    option['decay'] = 0.5

    # runtime information
    option['cost'] = 0
    option['count'] = 0
    option['epoch'] = 0
    option['maxepoch'] = 5
    option['freq'] = 1000
    option['vfreq'] = 1000
    option['sfreq'] = 50
    option['seed'] = 1234
    option['validate'] = None
    option['ref'] = None

    # beam search
    option['beamsize'] = 10
    option['normalize'] = False
    option['maxlen'] = None
    option['minlen'] = None

    return option

def override_if_not_none(option, args, key):
    value = args.__dict__[key]
    option[key] = value if value != None else option[key]

# override default options
def override(option, args):

    # training corpus
    if args.corpus == None and option['corpus'] == None:
        raise RuntimeError('error: no training corpus specified')

    # vocabulary
    if args.vocab == None and option['vocab'] == None:
        raise RuntimeError('error: no training vocabulary specified')

    override_if_not_none(option, args, 'corpus')

    # vocabulary and model paramters cannot be overrided
    if option['vocab'] == None:
        option['vocab'] = args.vocab
        svocab = loadvocab(args.vocab[0])
        tvocab = loadvocab(args.vocab[1])
        isvocab = invertvoc(svocab)
        itvocab = invertvoc(tvocab)

        # compatible with groundhog
        option['source_eos_id'] = len(isvocab)
        option['target_eos_id'] = len(itvocab)

        option['eos'] = '<eos>'
        svocab['<eos>'] = option['source_eos_id']
        tvocab['<eos>'] = option['target_eos_id']
        isvocab[option['source_eos_id']] = '<eos>'
        itvocab[option['target_eos_id']] = '<eos>'

        option['vocabulary'] = [[svocab, isvocab], [tvocab, itvocab]]

        # model parameters
        override_if_not_none(option, args, 'embdim')
        override_if_not_none(option, args, 'hidden')
        override_if_not_none(option, args, 'maxhid')
        override_if_not_none(option, args, 'maxpart')
        override_if_not_none(option, args, 'deephid')

    # training options
    override_if_not_none(option, args, 'maxepoch')
    override_if_not_none(option, args, 'alpha')
    override_if_not_none(option, args, 'momentum')
    override_if_not_none(option, args, 'batch')
    override_if_not_none(option, args, 'optimizer')
    override_if_not_none(option, args, 'norm')
    override_if_not_none(option, args, 'stop')
    override_if_not_none(option, args, 'decay')

    # runtime information
    override_if_not_none(option, args, 'validate')
    override_if_not_none(option, args, 'ref')
    override_if_not_none(option, args, 'freq')
    override_if_not_none(option, args, 'vfreq')
    override_if_not_none(option, args, 'sfreq')
    override_if_not_none(option, args, 'seed')

    # beamsearch
    override_if_not_none(option, args, 'beam_size')
    override_if_not_none(option, args, 'normalize')
    override_if_not_none(option, args, 'maxlen')
    override_if_not_none(option, args, 'minlen')

def print_option(option):
    isvocab = option['vocabulary'][0][1]
    itvocab = option['vocabulary'][1][1]

    print ''
    print 'options'

    print 'corpus:', option['corpus']
    print 'vocab:', option['vocab']
    # exclude <eos> symbol
    print 'vocabsize:', [len(isvocab) - 1, len(itvocab) - 1]

    print 'embdim:', option['embdim']
    print 'hidden:', option['hidden']
    print 'maxhid:', option['maxhid']
    print 'maxpart:', option['maxpart']
    print 'deephid:', option['deephid']

    print 'maxepoch:', option['maxepoch']
    print 'alpha:', option['alpha']
    print 'momentum:', option['momentum']
    print 'batch:', option['batch']
    print 'optimizer:', option['optimizer']
    print 'norm:', option['norm']
    print 'stop:', option['stop']
    print 'decay:', option['decay']

    print 'validate:', option['validate']
    print 'ref:', option['ref']
    print 'freq:', option['freq']
    print 'vfreq:', option['vfreq']
    print 'sfreq:', option['sfreq']
    print 'seed:', option['seed']

    print 'beamsize:', option['beamsize']
    print 'normalize:', option['normalize']
    print 'maxlen:', option['maxlen']
    print 'minlen:', option['minlen']

def skipstream(stream, count):
    for i in range(count):
        stream.next()

def getfilename(name):
    s = name.split('.')
    return s[0]

def train(args):
    option = getoption()
    init = True

    if os.path.exists(args.model):
        model = loadmodel(args.model)
        option = model.option
        init = False
    else:
        init = True

    override(option, args)
    print_option(option)

    # set seed
    numpy.random.seed(option['seed'])

    if option['ref']:
        references = loadreferences(option['ref'])
    else:
        references = None

    svocabs, tvocabs = option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    pathname, basename = os.path.split(args.model)
    modelname = getfilename(basename)
    batch = option['batch']
    stream = batchstream(option['corpus'], batch)

    skipstream(stream, option['count'])
    epoch = option['epoch']
    maxepoch = option['maxepoch']
    option['model'] = 'rnnsearch'

    if init:
        model = rnnsearch(**option)
        uniform(model.parameter, -0.08, 0.08)

    # tuning option
    toption = {}
    toption['algorithm'] = option['optimizer']
    toption['variant'] = option['variant']
    toption['constraint'] = ('norm', option['norm'])
    toption['norm'] = True
    toption['initialize'] = option['shared'] if 'shared' in option else False
    trainer = optimizer(model, **toption)
    alpha = option['alpha']

    # beamsearch option
    doption = {}
    doption['beamsize'] = option['beamsize']
    doption['normalize'] = option['normalize']
    doption['maxlen'] = option['maxlen']
    doption['minlen'] = option['minlen']

    print 'parameters:', parameters(model.parameter)

    best_score = 0.0

    for i in range(epoch, maxepoch):
        totcost = 0.0
        for data in stream:
            xdata, xmask = processdata(data[0], svocab)
            ydata, ymask = processdata(data[1], tvocab)

            t1 = time.time()
            cost, norm = trainer.optimize(xdata, xmask, ydata, ymask)
            trainer.update(alpha = alpha)
            t2 = time.time()

            option['count'] += 1
            count = option['count']

            cost = cost * ymask.shape[1] / ymask.sum()
            totcost += cost / math.log(2)
            print i + 1, count, cost, norm, t2 - t1

            option['cost'] = totcost

            # save model
            if count % option['freq'] == 0:
                svars = [p.get_value() for p in trainer.parameter]
                model.option = option
                model.option['shared'] = svars
                filename = os.path.join(pathname, modelname + '.autosave.pkl')
                serialize(filename, model)

            if count % option['vfreq'] == 0:
                if option['validate'] and references:
                    trans = translate(model, option['validate'], **doption)
                    bleu_score = bleu(trans, references)
                    print 'bleu: %2.4f' % bleu_score
                    if bleu_score > best_score:
                        best_score = bleu_score
                        model.option = option
                        model.option['shared'] = False
                        bestname = modelname + '.best.pkl'
                        filename = os.path.join(pathname, bestname)
                        serialize(filename, model)

            if count % option['sfreq'] == 0:
                ind = numpy.random.randint(0, batch)
                sdata = data[0][ind]
                tdata = data[1][ind]
                xdata = xdata[:, ind:ind + 1]
                hls = beamsearch(model, xdata)
                if len(hls) > 0:
                    best, score = hls[0]
                    print sdata
                    print tdata
                    print ' '.join(best[:-1])
                else:
                    print sdata
                    print tdata
                    print 'warning: no translation'

        print '--------------------------------------------------'

        if option['vfreq'] and references:
            trans = translate(model, option['validate'], **doption)
            bleu_score = bleu(trans, references)
            print 'iter: %d, bleu: %2.4f' % (i + 1, bleu_score)
            if bleu_score > best_score:
                best_score = bleu_score
                model.option = option
                model.option['shared'] = False
                bestname = modelname + '.best.pkl'
                filename = os.path.join(pathname, bestname)
                serialize(filename, model)

        print 'averaged cost: ', totcost / option['count']
        print '--------------------------------------------------'

        # early stopping
        if i >= option['stop']:
            alpha = alpha * option['decay']

        stream.reset()
        option['epoch'] = i + 1
        option['count'] = 0
        option['alpha'] = alpha
        model.option = option

        # update autosave
        filename = os.path.join(pathname, modelname + '.autosave.pkl')
        svars = [p.get_value() for p in trainer.parameter]
        model.option = option
        model.option['shared'] = svars
        serialize(filename, model)

    print 'best(bleu): %2.4f' % best_score

    stream.close()

def decode(args):
    model = loadmodel(args.model)

    svocabs, tvocabs = model.option['vocabulary']
    svocab, isvocab = svocabs
    tvocab, itvocab = tvocabs

    count = 0

    option = {}
    option['maxlen'] = args.maxlen
    option['minlen'] = args.minlen
    option['beamsize'] = args.beam_size
    option['normalize'] = args.normalize

    while True:
        line = sys.stdin.readline()

        if line == '':
            break

        data = [line]
        seq, mask = processdata(data, svocab)
        t1 = time.time()
        tlist = beamsearch(model, seq, **option)
        t2 = time.time()

        if len(tlist) == 0:
            sys.stdout.write('\n')
            score = -10000.0
        else:
            best, score = tlist[0]
            sys.stdout.write(' '.join(best[:-1]))
            sys.stdout.write('\n')

        count = count + 1
        sys.stderr.write(str(count) + ' ')
        sys.stderr.write(str(score) + ' ' + str(t2 - t1) + '\n')

def helpinfo():
    print 'usage:'
    print '\trnnsearch.py <command> [<args>]'
    print 'using rnnsearch.py train --help to see training options'
    print 'using rnnsearch.py translate --help to see translation options'

if __name__ == '__main__':
    if len(sys.argv) == 1:
        helpinfo()
    else:
        command = sys.argv[1]
        if command == 'train':
            print 'training command:'
            print ' '.join(sys.argv)
            args = parseargs_train(sys.argv[2:])
            train(args)
        elif command == 'translate':
            sys.stderr.write(' '.join(sys.argv))
            sys.stderr.write('\n')
            args = parseargs_decode(sys.argv[2:])
            decode(args)
        else:
            helpinfo()
