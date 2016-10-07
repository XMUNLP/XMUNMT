# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano

from nn import gru, gru_config
from nn import linear, linear_config
from nn import maxout, maxout_config
from nn import config, variable_scope
from nn import embedding, embedding_config
from nn import feedforward, feedforward_config
from utils import get_or_default, add_if_not_exsit


class encoder_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "encoder"
    * forward_rnn: gru_config, config behavior of forward rnn
    * backward_rnn: gru_config, config behavior of backward rnn
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "encoder")
        self.forward_rnn = gru_config(dtype=self.dtype, scope="forward_rnn")
        self.backward_rnn = gru_config(dtype=self.dtype, scope="backward_rnn")


class decoder_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "decoder"
    * init_transform: feedforward_config, config initial state transform
    * annotation_transform: linear_config, config annotation transform
    * state_transform: linear_config, config state transform
    * context_transform: linear_config, config context transform
    * rnn: gru_config, config decoder rnn
    * maxout: maxout_config, config maxout unit
    * deepout: linear_config, config deepout transform
    * classify: linear_config, config classify transform
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "decoder")
        self.init_transform = feedforward_config(dtype=self.dtype,
                                                 scope="init_transform",
                                                 activation=theano.tensor.tanh)
        self.annotation_transform = linear_config(dtype=self.dtype,
                                                  scope="annotation_transform")
        self.state_transform = linear_config(dtype=self.dtype,
                                             scope="state_transform")
        self.context_transform = linear_config(dtype=self.dtype,
                                               scope="context_transform")
        self.rnn = gru_config(dtype=self.dtype, scope="rnn")
        self.maxout = maxout_config(dtype=self.dtype, scope="maxout")
        self.deepout = linear_config(dtypde=self.dtype, scope="deepout")
        self.classify = linear_config(dtype=self.dtype, scope="classify")


class rnnsearch_config(config):
    """
    * dtype: str, default theano.config.floatX
    * scope: str, default "rnnsearch"
    * source_embedding: embedding_config, config source side embedding
    * target_embedding: embedding_config, config target side embedding
    * encoder: encoder_config, config encoder
    * decoder: decoder_config, config decoder
    """

    def __init__(self, **kwargs):
        self.dtype = get_or_default(kwargs, "dtype", theano.config.floatX)
        self.scope = get_or_default(kwargs, "scope", "rnnsearch")
        self.source_embedding = embedding_config(dtype=self.dtype,
                                                 scope="source_embedding")
        self.target_embedding = embedding_config(dtype=self.dtype,
                                                 scope="target_embedding")
        self.encoder = encoder_config(dtype=self.dtype)
        self.decoder = decoder_config(dtype=self.dtype)


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


class encoder:

    def __init__(self, input_size, hidden_size, config=encoder_config()):
        scope = config.scope

        with variable_scope(scope):
            forward_encoder = gru(input_size, hidden_size, config.forward_rnn)
            backward_encoder = gru(input_size, hidden_size,
                                   config.backward_rnn)

        params = []
        params.extend(forward_encoder.parameter)
        params.extend(backward_encoder.parameter)

        def forward(x, mask, initstate):
            def forward_step(x, m, h):
                nh, states = forward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            def backward_step(x, m, h):
                nh, states = backward_encoder(x, h)
                nh = (1.0 - m[:, None]) * h + m[:, None] * nh
                return [nh]

            seq = [x, mask]
            hf, u = theano.scan(forward_step, seq, [initstate])

            seq = [x[::-1], mask[::-1]]
            hb, u = theano.scan(backward_step, seq, [initstate])
            hb = hb[::-1]

            return theano.tensor.concatenate([hf, hb], 2)

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = params

    def __call__(self, x, mask, initstate):
        return self.forward(x, mask, initstate)


class decoder:

    def __init__(self, emb_size, shidden_size, thidden_size, ahidden_size,
                 mhidden_size, maxpart, dhidden_size, voc_size,
                 config=decoder_config()):
        scope = config.scope
        ctx_size = 2 * shidden_size

        with variable_scope(scope):
            init_transform = feedforward(shidden_size, thidden_size,
                                         config.init_transform)
            annotation_transform = linear(ctx_size, ahidden_size,
                                          config.annotation_transform)
            state_transform = linear(thidden_size, ahidden_size,
                                     config.state_transform)
            context_transform = linear(ahidden_size, 1,
                                       config.context_transform)
            rnn = gru([emb_size, ctx_size], thidden_size, config.rnn)
            maxout_transform = maxout([thidden_size, emb_size, ctx_size],
                                      mhidden_size, maxpart, config.maxout)
            deepout_transform = linear(mhidden_size, dhidden_size,
                                       config.deepout)
            classify_transform = linear(dhidden_size, voc_size,
                                        config.classify)

        params = []
        params.extend(init_transform.parameter)
        params.extend(annotation_transform.parameter)
        params.extend(state_transform.parameter)
        params.extend(context_transform.parameter)
        params.extend(rnn.parameter)
        params.extend(maxout_transform.parameter)
        params.extend(deepout_transform.parameter)
        params.extend(classify_transform.parameter)

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
            new_state, states = rnn([yemb, context], state)
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

        self.name = scope
        self.config = config
        self.forward = forward
        self.parameter = params
        self.compute_initstate = compute_initstate
        self.compute_context = compute_context
        self.compute_probability = compute_probability
        self.compute_state = compute_state

    def __call__(self, yseq, xmask, ymask, annotation):
        return self.forward(yseq, xmask, ymask, annotation)


class rnnsearch:

    def __init__(self, config=get_config(), **option):
        scope = config.scope

        sedim, tedim = option["embdim"]
        shdim, thdim, ahdim = option["hidden"]
        maxdim = option["maxhid"]
        deephid = option["deephid"]
        k = option["maxpart"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        svsize = len(sid2w)
        tvsize = len(tid2w)

        with variable_scope(scope):
            source_embedding = embedding(svsize, sedim,
                                         config.source_embedding)
            target_embedding = embedding(tvsize, tedim,
                                         config.target_embedding)
            rnn_encoder = encoder(sedim, shdim, config.encoder)
            rnn_decoder = decoder(tedim, shdim, thdim, ahdim, maxdim, k,
                                  deephid, tvsize, config.decoder)

        params = []
        params.extend(source_embedding.parameter)
        params.extend(target_embedding.parameter)
        params.extend(rnn_encoder.parameter)
        params.extend(rnn_decoder.parameter)

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

                return theano.function(inputs, [context, alpha])

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

        self.name = scope
        self.config = config
        self.parameter = params
        self.option = option
        self.cost = train_outputs[0]
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.updates = []
        self.sample = functions


# based on groundhog's impelmentation
def beamsearch(models, xseq, **option):
    add_if_not_exsit(option, "beamsize", 10)
    add_if_not_exsit(option, "normalize", False)
    add_if_not_exsit(option, "maxlen", None)
    add_if_not_exsit(option, "minlen", None)
    add_if_not_exsit(option, "arithmetric", False)

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)
    functions = [model.sample for model in models]

    vocabulary = models[0].option["vocabulary"]
    eos = models[0].option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]

    size = option["beamsize"]
    maxlen = option["maxlen"]
    minlen = option["minlen"]
    normalize = option["normalize"]
    arithmetric = option["arithmetric"]

    if maxlen == None:
        maxlen = len(xseq) * 3

    if minlen == None:
        minlen = len(xseq) / 2

    annot = [None for i in range(num_models)]
    mannot = [None for i in range(num_models)]
    contexts = [None for i in range(num_models)]
    states = [None for i in range(num_models)]
    probs = [None for i in range(num_models)]

    xmask = numpy.ones(xseq.shape, "float32")

    for i in range(num_models):
        encode = functions[i][0]
        compute_istate = functions[i][1]
        annot[i] = encode(xseq, xmask)
        states[i], mannot[i] = compute_istate(annot[i])

    hdim = states[0].shape[1]
    cdim = annot[0].shape[2]
    # [num_models, batch, dim]
    states = numpy.array(states)

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
            last_words = last_words.astype("int32")
        else:
            last_words = numpy.zeros(num, "int32")

        xmasks = numpy.repeat(xmask, num, 1)
        ymask = numpy.ones((num,), "float32")
        annots = [numpy.repeat(annot[i], num, 1) for i in range(num_models)]
        mannots = [numpy.repeat(mannot[i], num, 1) for i in range(num_models)]

        for i in range(num_models):
            compute_context = functions[i][2]
            contexts[i], alpha = compute_context(states[i], xmasks, annots[i],
                                          mannots[i])

        for i in range(num_models):
            compute_probs = functions[i][3]
            probs[i] = compute_probs(last_words, states[i], contexts[i])

        if arithmetric:
            logprobs = numpy.log(sum(probs) / num_models)
        else:
            # geometric mean
            logprobs = sum(numpy.log(probs)) / num_models

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
        newstates = numpy.zeros((num_models, size, hdim), "float32")
        newcontexts = numpy.zeros((num_models, size, cdim), "float32")
        inputs = numpy.zeros(size, "int32")

        for i, (idx, nword, ncost) in enumerate(zip(tinds, winds, costs)):
            newtrans[i] = trans[idx] + [nword]
            newcosts[i] = ncost
            for j in range(num_models):
                newstates[j][i] = states[j][idx]
                newcontexts[j][i] = contexts[j][idx]
            inputs[i] = nword

        ymask = numpy.ones((size,), "float32")

        for i in range(num_models):
            compute_state = functions[i][-1]
            newstates[i] = compute_state(inputs, ymask, newstates[i],
                                         newcontexts[i])

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

        states = newstates[:, indices]

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
