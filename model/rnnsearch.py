# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

from utils import extract_option, update_option
from utils import add_if_not_exsit, add_parameters
from nn import linear, embedding, feedforward, gru, maxout

# standard rnnsearch configuration
def rnnsearch_config():
    opt = {}

    # embedding
    opt['source-embedding/bias'] = True
    opt['target-embedding/bias'] = True

    # encoder
    opt['encoder/forward-rnn/variant'] = 'standard'
    opt['encoder/backward-rnn/variant'] = 'standard'
    opt['encoder/forward-rnn/reset-gate/weight'] = False
    opt['encoder/forward-rnn/reset-gate/bias'] = False
    opt['encoder/forward-rnn/update-gate/weight'] = False
    opt['encoder/forward-rnn/update-gate/bias'] = False
    opt['encoder/forward-rnn/transform/weight'] = False
    opt['encoder/forward-rnn/transform/bias'] = True
    opt['encoder/backward-rnn/reset-gate/weight'] = False
    opt['encoder/backward-rnn/reset-gate/bias'] = False
    opt['encoder/backward-rnn/update-gate/weight'] = False
    opt['encoder/backward-rnn/update-gate/bias'] = False
    opt['encoder/backward-rnn/transform/weight'] = False
    opt['encoder/backward-rnn/transform/bias'] = True

    # decoder
    opt['decoder/init-transform/variant'] = 'standard'
    opt['decoder/annotation-transform/variant'] = 'standard'
    opt['decoder/state-transform/variant'] = 'standard'
    opt['decoder/context-transform/variant'] = 'standard'
    opt['decoder/rnn/variant'] = 'standard'
    opt['decoder/maxout/variant'] = 'standard'
    opt['decoder/deepout/variant'] = 'standard'
    opt['decoder/classify/variant'] = 'standard'

    opt['decoder/init-transform/weight'] = False
    opt['decoder/init-transform/bias'] = True
    opt['decoder/annotation-transform/weight'] = False
    opt['decoder/annotation-transform/bias'] = False
    opt['decoder/state-transform/weight'] = False
    opt['decoder/state-transform/bias'] = False
    opt['decoder/context-transform/weight'] = False
    opt['decoder/context-transform/bias'] = False
    opt['decoder/rnn/reset-gate/weight'] = False
    opt['decoder/rnn/reset-gate/bias'] = False
    opt['decoder/rnn/update-gate/weight'] = False
    opt['decoder/rnn/update-gate/bias'] = False
    opt['decoder/rnn/transform/weight'] = False
    opt['decoder/rnn/transform/bias'] = True
    opt['decoder/maxout/weight'] = False
    opt['decoder/maxout/bias'] = True
    opt['decoder/deepout/weight'] = False
    opt['decoder/deepout/bias'] = False
    opt['decoder/classify/weight'] = False
    opt['decoder/classify/bias'] = True

    return opt

class encoder:

    def __init__(self, input_size, hidden_size, **option):
        opt = option

        fopt = extract_option(opt, 'forward-rnn')
        bopt = extract_option(opt, 'backward-rnn')
        fopt['name'] = 'forward-rnn'
        bopt['name'] = 'backward-rnn'

        forward_encoder = gru(input_size, hidden_size, **fopt)
        backward_encoder = gru(input_size, hidden_size, **bopt)

        params = []
        add_parameters(params, 'encoder', *forward_encoder.parameter)
        add_parameters(params, 'encoder', *backward_encoder.parameter)

        def forward(x, mask, initstate):
            def forward_step(x, m, h):
                nh = forward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            def backward_step(x, m, h):
                nh = backward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            seq = [x, mask]
            hf, u = theano.scan(forward_step, seq, [initstate])

            seq = [x[::-1], mask[::-1]]
            hb, u = theano.scan(backward_step, seq, [initstate])
            hb = hb[::-1]

            return theano.tensor.concatenate([hf, hb], 2)

        self.name = 'encoder'
        self.option = option
        self.forward = forward
        self.parameter = params

    def __call__(self, x, mask, initstate):
        return self.forward(x, mask, initstate)

class decoder:

    def __init__(self, emb_size, shidden_size, thidden_size, ahidden_size,
                 mhidden_size, maxpart, dhidden_size, voc_size, **option):
        opt = option
        ctx_size = 2 * shidden_size

        iopt = extract_option(opt, 'init-transform')
        aopt = extract_option(opt, 'annotation-transform')
        sopt = extract_option(opt, 'state-transform')
        topt = extract_option(opt, 'context-transform')
        ropt = extract_option(opt, 'rnn')
        mopt = extract_option(opt, 'maxout')
        dopt = extract_option(opt, 'deepout')
        copt = extract_option(opt, 'classify')

        iopt['name'] = 'init-transform'
        aopt['name'] = 'annotation-transform'
        sopt['name'] = 'state-transform'
        topt['name'] = 'context-transform'
        ropt['name'] = 'rnn'
        mopt['name'] = 'maxout'
        dopt['name'] = 'deepout'
        copt['name'] = 'classify'
        iopt['function'] = theano.tensor.tanh
        mopt['maxpart'] = maxpart

        init_transform = feedforward(shidden_size, thidden_size, **iopt)
        # attention
        annotation_transform = linear(ctx_size, ahidden_size, **aopt)
        state_transform = linear(thidden_size, ahidden_size, **sopt)
        context_transform = linear(ahidden_size, 1, **topt)
        # decoder rnn
        rnn = gru([emb_size, ctx_size], thidden_size, **ropt)
        maxout_transform = maxout([thidden_size, emb_size, ctx_size],
                                  mhidden_size, **mopt)
        deepout_transform = linear(mhidden_size, dhidden_size, **dopt)
        classify_transform = linear(dhidden_size, voc_size, **copt)

        params = []
        add_parameters(params, 'decoder', *init_transform.parameter)
        add_parameters(params, 'decoder', *annotation_transform.parameter)
        add_parameters(params, 'decoder', *state_transform.parameter)
        add_parameters(params, 'decoder', *context_transform.parameter)
        add_parameters(params, 'decoder', *rnn.parameter)
        add_parameters(params, 'decoder', *maxout_transform.parameter)
        add_parameters(params, 'decoder', *deepout_transform.parameter)
        add_parameters(params, 'decoder', *classify_transform.parameter)

        def attention(state, xmask, mapped_annotation):
            mapped_state = state_transform(state)
            hidden = theano.tensor.tanh(mapped_state + mapped_annotation)
            score = context_transform(hidden)
            score = score.reshape((score.shape[0], score.shape[1]))
            # softmax over masked batch
            alpha = theano.tensor.exp(score)
            alpha = alpha * xmask
            alpha = alpha / theano.tensor.sum(alpha, 0)
            return alpha

        def compute_initstate(annotation):
            hb = annotation[0, :, -annotation.shape[2] / 2:]
            inis = init_transform(hb)
            mapped_annotation = annotation_transform(annotation)

            return inis, mapped_annotation

        def compute_context(state, xmask, annotation, mapped_annotation):
            alpha = attention(state, xmask, mapped_annotation)
            context = theano.tensor.sum(alpha[:, :, None] * annotation, 0)
            return [alpha, context]

        def compute_probability(yemb, state, context):
            maxhid = maxout_transform([state, yemb, context])
            readout = deepout_transform(maxhid)
            preact = classify_transform(readout)
            prob = theano.tensor.nnet.softmax(preact)

            return prob

        def compute_state(yemb, ymask, state, context):
            new_state = rnn([yemb, context], state)
            ymask = ymask[:, None]
            new_state = (1.0 - ymask) * state + ymask * new_state

            return new_state

        def forward(yseq, xmask, ymask, annotation):
            yshift = theano.tensor.zeros_like(yseq)
            yshift = theano.tensor.set_subtensor(yshift[1:], yseq[:-1])

            initstate, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, state, xmask, annotation, mannotation):
                outs = compute_context(state, xmask, annotation, mannotation)
                alpha, context = outs
                new_state = compute_state(yemb, ymask, state, context)
                return [new_state, context]

            seq = [yseq, ymask]
            oinfo = [initstate, None]
            nonseq = [xmask, annotation, mapped_annotation]
            (states, contexts), updates = theano.scan(step, seq, oinfo, nonseq)

            inis = initstate[None, :, :]
            all_states = theano.tensor.concatenate([inis, states], 0)
            prev_states = all_states[:-1]

            maxhid = maxout_transform([prev_states, yshift, contexts])
            readout = deepout_transform(maxhid)
            preact = classify_transform(readout)
            preact = preact.reshape((preact.shape[0] * preact.shape[1], -1))
            prob = theano.tensor.nnet.softmax(preact)

            return prob

        self.name = 'decoder'
        self.option = opt
        self.forward = forward
        self.parameter = params
        self.compute_initstate = compute_initstate
        self.compute_context = compute_context
        self.compute_probability = compute_probability
        self.compute_state = compute_state

    def __call__(self, yseq, xmask, ymask, annotation):
        return self.forward(yseq, xmask, ymask, annotation)

class rnnsearch:

    def __init__(self, **option):
        opt = rnnsearch_config()

        update_option(opt, option)
        sedim, tedim = option['embdim']
        shdim, thdim, ahdim = option['hidden']
        maxdim = option['maxhid']
        deephid = option['deephid']
        k = option['maxpart']
        svocab, tvocab = option['vocabulary']
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        svsize = len(sid2w)
        tvsize = len(tid2w)

        sopt = extract_option(opt, 'source-embedding')
        topt = extract_option(opt, 'target-embedding')
        eopt = extract_option(opt, 'encoder')
        dopt = extract_option(opt, 'decoder')
        sopt['name'] = 'source-embedding'
        topt['name'] = 'target-embedding'

        source_embedding = embedding(svsize, sedim, **sopt)
        target_embedding = embedding(tvsize, tedim, **topt)
        rnn_encoder = encoder(sedim, shdim, **eopt)
        rnn_decoder = decoder(tedim, shdim, thdim, ahdim, maxdim, k, deephid,
                              tvsize, **dopt)

        params = []
        add_parameters(params, 'rnnsearch', *source_embedding.parameter)
        add_parameters(params, 'rnnsearch', *target_embedding.parameter)
        add_parameters(params, 'rnnsearch', *rnn_encoder.parameter)
        add_parameters(params, 'rnnsearch', *rnn_decoder.parameter)

        def build_training():
            xseq = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()
            yseq = theano.tensor.imatrix()
            ymask = theano.tensor.matrix()

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)
            initstate = theano.tensor.zeros((xemb.shape[1], shdim))

            annotation = rnn_encoder(xemb, xmask, initstate)
            probs = rnn_decoder(yemb, xmask, ymask, annotation)

            idx = theano.tensor.arange(yseq.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, yseq.flatten()])
            cost = cost.reshape(yseq.shape)
            cost = theano.tensor.sum(cost * ymask, 0)
            cost = theano.tensor.mean(cost)

            return [xseq, xmask, yseq, ymask], [cost]

        def build_sampling():

            def encode():
                xseq = theano.tensor.imatrix()
                xmask = theano.tensor.matrix()

                xemb = source_embedding(xseq)
                initstate = theano.tensor.zeros((xseq.shape[1], shdim))
                annotation = rnn_encoder(xemb, xmask, initstate)

                return theano.function([xseq, xmask], annotation)

            def compute_initstate():
                annotation = theano.tensor.tensor3()

                # initstate, mapped_annotation
                outputs = rnn_decoder.compute_initstate(annotation)

                return theano.function([annotation], outputs)

            def compute_context():
                state = theano.tensor.matrix()
                xmask = theano.tensor.matrix()
                annotation = theano.tensor.tensor3()
                mannotation = theano.tensor.tensor3()

                inputs = [state, xmask, annotation, mannotation]
                alpha, context = rnn_decoder.compute_context(*inputs)

                return theano.function(inputs, context)

            def compute_probability():
                y = theano.tensor.ivector()
                state = theano.tensor.matrix()
                context = theano.tensor.matrix()

                # 0 for initial index
                cond = theano.tensor.neq(y, 0)
                yemb = target_embedding(y)
                # zeros out embedding if y is 0
                yemb = yemb * cond[:, None]
                probs = rnn_decoder.compute_probability(yemb, state, context)

                return theano.function([y, state, context], probs)

            def compute_state():
                y = theano.tensor.ivector()
                ymask = theano.tensor.vector()
                state = theano.tensor.matrix()
                context = theano.tensor.matrix()

                yemb = target_embedding(y)
                inputs = [yemb, ymask, state, context]
                new_state = rnn_decoder.compute_state(*inputs)

                return theano.function([y, ymask, state, context], new_state)

            functions = []
            functions.append(encode())
            functions.append(compute_initstate())
            functions.append(compute_context())
            functions.append(compute_probability())
            functions.append(compute_state())

            return functions

        train_inputs, train_outputs = build_training()
        functions = build_sampling()

        self.cost = train_outputs[0]
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.updates = []
        self.parameter = params
        self.sample = functions
        self.option = opt

# based on groundhog's impelmentation
def beamsearch(model, xseq, **option):
    add_if_not_exsit(option, 'beamsize', 10)
    add_if_not_exsit(option, 'normalize', False)
    add_if_not_exsit(option, 'maxlen', None)
    add_if_not_exsit(option, 'minlen', None)

    functions = model.sample

    encode = functions[0]
    compute_istate = functions[1]
    compute_context = functions[2]
    compute_probs = functions[3]
    compute_state = functions[4]

    vocabulary = model.option['vocabulary']
    eos = model.option['eos']
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]

    size = option['beamsize']
    maxlen = option['maxlen']
    minlen = option['minlen']
    normalize = option['normalize']

    if maxlen == None:
        maxlen = len(xseq) * 3

    if minlen == None:
        minlen = len(xseq) / 2

    xmask = numpy.ones(xseq.shape, 'float32')
    annot = encode(xseq, xmask)
    state, mannot = compute_istate(annot)

    hdim = state.shape[1]
    cdim = annot.shape[2]
    states = state

    trans = [[]]
    costs = [0.0]
    final_trans = []
    final_costs = []

    for k in range(maxlen):
        if size == 0:
            break

        num = len(trans)

        if k > 0:
            last_words = numpy.array(map(lambda t: t[-1], trans))
            last_words = last_words.astype('int32')
        else:
            last_words = numpy.zeros(num, 'int32')

        xmasks = numpy.repeat(xmask, num, 1)
        annots = numpy.repeat(annot, num, 1)
        mannots = numpy.repeat(mannot, num, 1)
        contexts = compute_context(states, xmasks, annots, mannots)

        probs = compute_probs(last_words, states, contexts)
        logprobs = numpy.log(probs)

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob

        ncosts = numpy.array(costs)[:, None] - logprobs
        fcosts = ncosts.flatten()
        nbest = numpy.argpartition(fcosts, size)[:size]

        vocsize = logprobs.shape[1]
        tinds = nbest / vocsize
        winds = nbest % vocsize
        costs = fcosts[nbest]

        newtrans = [[]] * size
        newcosts = numpy.zeros(size)
        newstates = numpy.zeros((size, hdim), 'float32')
        newcontexts = numpy.zeros((size, cdim), 'float32')
        inputs = numpy.zeros(size, 'int32')

        for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
            newtrans[i] = trans[idx] + [nword]
            newcosts[i] = ncost
            newstates[i] = states[idx]
            newcontexts[i] = contexts[idx]
            inputs[i] = nword

        ymask = numpy.ones((size,), 'float32')
        newstates = compute_state(inputs, ymask, newstates, newcontexts)

        trans = []
        costs = []
        indices = []

        for i in range(size):
            if newtrans[i][-1] != eosid:
                trans.append(newtrans[i])
                costs.append(newcosts[i])
                indices.append(i)
            else:
                size -= 1
                final_trans.append(newtrans[i])
                final_costs.append(newcosts[i])
        states = newstates[indices]

    if len(final_trans) == 0:
        final_trans = [[]]
        final_costs = [0.0]

    for i, (cost, trans) in enumerate(zip(final_costs, final_trans)):
        count = len(trans)
        if count > 0:
            if normalize:
                final_costs[i] = cost / count
            else:
                final_costs[i] = cost

    final_trans = numpy.array(final_trans)[numpy.argsort(final_costs)]
    final_costs = numpy.array(sorted(final_costs))

    translations = []

    for cost, trans in zip(final_costs, final_trans):
        trans = map(lambda x: vocab[x], trans)
        translations.append((trans, cost))

    return translations
