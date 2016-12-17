# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import theano
import theano.sandbox.rng_mrg

from nn import gru, gru_config
from nn import linear, linear_config
from nn import maxout, maxout_config
from nn import config, variable_scope
from nn import embedding, embedding_config
from nn import feedforward, feedforward_config
from utils import get_or_default, add_if_not_exsit
from search import beam, select_nbest


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

        def compute_attention_score(yseq, xmask, ymask, annotation):
            initstate, mapped_annotation = compute_initstate(annotation)

            def step(yemb, ymask, state, xmask, annotation, mannotation):
                outs = compute_context(state, xmask, annotation, mannotation)
                alpha, context = outs
                new_state = compute_state(yemb, ymask, state, context)
                return [new_state, alpha]

            seq = [yseq, ymask]
            oinfo = [initstate, None]
            nonseq = [xmask, annotation, mapped_annotation]
            (states, alpha), updates = theano.scan(step, seq, oinfo, nonseq)

            return alpha

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
        self.compute_attention_score = compute_attention_score

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

        def training_graph():
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

        def attention_graph():
            xseq = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()
            yseq = theano.tensor.imatrix()
            ymask = theano.tensor.matrix()

            xemb = source_embedding(xseq)
            yemb = target_embedding(yseq)
            initstate = theano.tensor.zeros((xemb.shape[1], shdim))

            annotation = rnn_encoder(xemb, xmask, initstate)
            alpha = rnn_decoder.compute_attention_score(yemb, xmask, ymask,
                                                        annotation)

            return [xseq, xmask, yseq, ymask], alpha

        def sampling_graph():
            seed = option["seed"]
            seed_rng = numpy.random.RandomState(numpy.random.randint(seed))
            tseed = seed_rng.randint(numpy.iinfo(numpy.int32).max)
            stream = theano.sandbox.rng_mrg.MRG_RandomStreams(tseed)

            xseq = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()
            maxlen = theano.tensor.iscalar()

            batch = xseq.shape[1]
            xemb = source_embedding(xseq)
            initstate = theano.tensor.zeros([batch, shdim])

            annot = rnn_encoder(xemb, xmask, initstate)

            ymask = theano.tensor.ones([batch])
            istate, mannot = rnn_decoder.compute_initstate(annot)

            def sample_step(pemb, state, xmask, ymask, annot, mannot):
                alpha, context = rnn_decoder.compute_context(state, xmask,
                                                             annot, mannot)
                probs = rnn_decoder.compute_probability(pemb, state, context)
                next_words = stream.multinomial(pvals=probs).argmax(axis=1)
                yemb = target_embedding(next_words)
                next_state = rnn_decoder.compute_state(yemb, ymask, state,
                                                       context)
                return [next_words, yemb, next_state]

            iemb = theano.tensor.zeros([batch, tedim])

            seqs = []
            outputs_info = [None, iemb, istate]
            nonseqs = [xmask, ymask, annot, mannot]

            outputs, u = theano.scan(sample_step, seqs, outputs_info,
                                     nonseqs, n_steps=maxlen)

            return [xseq, xmask, maxlen], outputs[0], u

        # for beamsearch
        def encoding_graph():
            xseq = theano.tensor.imatrix()
            xmask = theano.tensor.matrix()

            xemb = source_embedding(xseq)
            initstate = theano.tensor.zeros((xseq.shape[1], shdim))
            annotation = rnn_encoder(xemb, xmask, initstate)

            return [xseq, xmask], annotation

        def initial_state_graph():
            annotation = theano.tensor.tensor3()

            # initstate, mapped_annotation
            outputs = rnn_decoder.compute_initstate(annotation)

            return [annotation], outputs

        def context_graph():
            state = theano.tensor.matrix()
            xmask = theano.tensor.matrix()
            annotation = theano.tensor.tensor3()
            mannotation = theano.tensor.tensor3()

            inputs = [state, xmask, annotation, mannotation]
            alpha, context = rnn_decoder.compute_context(*inputs)

            return inputs, [context, alpha]

        def probability_graph():
            y = theano.tensor.ivector()
            state = theano.tensor.matrix()
            context = theano.tensor.matrix()

            # 0 for initial index
            cond = theano.tensor.neq(y, 0)
            yemb = target_embedding(y)
            # zeros out embedding if y is 0
            yemb = yemb * cond[:, None]
            probs = rnn_decoder.compute_probability(yemb, state, context)

            return [y, state, context], probs

        def state_graph():
            y = theano.tensor.ivector()
            ymask = theano.tensor.vector()
            state = theano.tensor.matrix()
            context = theano.tensor.matrix()

            yemb = target_embedding(y)
            inputs = [yemb, ymask, state, context]
            new_state = rnn_decoder.compute_state(*inputs)

            return [y, ymask, state, context], new_state

        def compile_function(graph_fn):
            outputs = graph_fn()

            if len(outputs) == 2:
                inputs, outputs = outputs
                return theano.function(inputs, outputs)
            else:
                inputs, outputs, updates = outputs
                return theano.function(inputs, outputs, updates=updates)


        train_inputs, train_outputs = training_graph()

        search_fn = []
        search_fn.append(compile_function(encoding_graph))
        search_fn.append(compile_function(initial_state_graph))
        search_fn.append(compile_function(context_graph))
        search_fn.append(compile_function(probability_graph))
        search_fn.append(compile_function(state_graph))

        self.name = scope
        self.config = config
        self.parameter = params
        self.option = option
        self.cost = train_outputs[0]
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.updates = []
        self.search = search_fn
        self.sampler = compile_function(sampling_graph)
        self.attention = compile_function(attention_graph)


def beamsearch(models, seq, beamsize=10, normalize=False, maxlen=None,
               minlen=None, arithmetic=False):
    size = beamsize

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)

    # get vocabulary from the first model
    vocabulary = models[0].option["vocabulary"]
    eos_symbol = models[0].option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos_symbol]

    if maxlen == None:
        maxlen = seq.shape[0] * 3

    if minlen == None:
        minlen = seq.shape[0] / 2

    # encoding source
    mask = numpy.ones(seq.shape, "float32")
    annotations = [model.search[0](seq, mask) for model in models]
    istates_and_mannots = [model.search[1](annot) for annot, model in
                           zip(annotations, models)]

    # compute initial state and map annotation for fast attention
    states = [item[0] for item in istates_and_mannots]
    mapped_annots = [item[1] for item in istates_and_mannots]

    initial_beam = beam(size)
    # </s>
    initial_beam.candidate = [[0]]
    initial_beam.score = numpy.zeros([1], "float32")

    hypo_list = []
    beam_list = [initial_beam]
    cond = lambda x: x[-1] == eosid

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(candidate)
        last_words = numpy.array(map(lambda t: t[-1], candidate), "int32")

        # compute context first, then compute word distribution
        batch_mask = numpy.repeat(mask, num, 1)
        batch_annots = map(numpy.repeat, annotations, [num] * num_models,
                           [1] * num_models)
        batch_mannots = map(numpy.repeat, mapped_annots, [num] * num_models,
                           [1] * num_models)

        # function[2] returns (context, alpha)
        outputs = [model.search[2](state, batch_mask, annot, mannot)
                   for model, state, annot, mannot in
                   zip(models, states, batch_annots, batch_mannots)]
        contexts = [item[0] for item in outputs]
        prob_dists = [model.search[3](last_words, state, context) for
                      model, state, context in zip(models, states, contexts)]

        # search nbest given word distribution
        if arithmetic:
            logprobs = numpy.log(sum(prob_dists) / num_models)
        else:
            # geometric mean
            logprobs = sum(numpy.log(prob_dists)) / num_models

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob

        next_beam = beam(size)
        outputs = next_beam.prune(logprobs, cond, prev_beam)

        # translation complete
        hypo_list.extend(outputs[0])
        batch_indices, word_indices = outputs[1:]
        size -= len(outputs[0])

        if size == 0:
            break

        # generate next state
        candidate = next_beam.candidate
        num = len(candidate)
        last_words = numpy.array(map(lambda t: t[-1], candidate), "int32")

        states = select_nbest(states, batch_indices)
        contexts = select_nbest(contexts, batch_indices)

        batch_ymask = numpy.ones((num,), "float32")

        states = [model.search[-1](last_words, batch_ymask, state, context)
                  for model, state, context in zip(models, states, contexts)]

        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [["</s>"]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = numpy.array(hypo_list)[numpy.argsort(score_list)]
    score_list = numpy.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output


def batchsample(model, xseq, xmask, **option):
    add_if_not_exsit(option, "maxlen", None)
    maxlen = option["maxlen"]

    sampler = model.sampler

    vocabulary = model.option["vocabulary"]
    eos = model.option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]

    if maxlen == None:
        maxlen = int(len(xseq) * 1.5)

    words = sampler(xseq, xmask, maxlen)
    trans = words.astype("int32")

    samples = []

    for i in range(trans.shape[1]):
        example = trans[:, i]
        # remove eos symbol
        index = -1

        for i in range(len(example)):
            if example[i] == eosid:
                index = i
                break

        if index >= 0:
            example = example[:index]

        example = map(lambda x: vocab[x], example)

        samples.append(example)

    return samples
