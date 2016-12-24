# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import nn
import ops
import numpy
import theano
import theano.sandbox.rng_mrg

from search import beam, select_nbest


def gru_encoder(cell, inputs, mask, initial_state=None, dtype=None):
    if not isinstance(cell, nn.rnn_cell.gru_cell):
        raise ValueError("only gru_cell is supported")

    if isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a tensor, not list or tuple")

    def loop_fn(inputs, mask, state):
        mask = mask[:, None]
        output, next_state = cell(inputs, state)
        next_state = (1.0 - mask) * state + mask * next_state
        return next_state

    if initial_state is None:
        batch = inputs.shape[1]
        state_size = cell.state_size
        initial_state = theano.tensor.zeros([batch, state_size], dtype=dtype)

    seq = [inputs, mask]
    states, updates = theano.scan(loop_fn, seq, [initial_state])

    return states


def encoder(inputs, mask, input_size, output_size, initial_state=None,
            dtype=None, scope=None):
    size = [input_size, output_size]
    cell = nn.rnn_cell.gru_cell(size)

    with ops.variable_scope(scope or "encoder"):
        with ops.variable_scope("forward"):
            fd_states = gru_encoder(cell, inputs, mask, initial_state, dtype)
        with ops.variable_scope("backward"):
            inputs = inputs[::-1]
            mask = mask[::-1]
            bd_states = gru_encoder(cell, inputs, mask, initial_state, dtype)
            bd_states = bd_states[::-1]

    return fd_states, bd_states


# precompute mapped attention states to speed up decoding
# attention_states: [time_steps, batch, input_size]
# outputs: [time_steps, batch, attn_size]
def map_attention_states(attention_states, input_size, attn_size, scope=None):
    with ops.variable_scope(scope or "attention"):
        mapped_states = nn.linear(attention_states, [input_size, attn_size],
                                  False, scope="attention_w")

    return mapped_states


def attention(query, mapped_states, state_size, attn_size, attention_mask=None,
              scope=None):
    with ops.variable_scope(scope or "attention"):
        mapped_query = nn.linear(query, [state_size, attn_size], False,
                                 scope="query_w")

        mapped_query = mapped_query[None, :, :]
        hidden = theano.tensor.tanh(mapped_query + mapped_states)

        score = nn.linear(hidden, [attn_size, 1], False, scope="attention_v")
        score = score.reshape([score.shape[0], score.shape[1]])

        exp_score = theano.tensor.exp(score)

        if attention_mask is not None:
            exp_score = exp_score * attention_mask

        alpha = exp_score / theano.tensor.sum(exp_score, 0)

    return alpha


def decoder(inputs, mask, initial_state, attention_states, attention_mask,
            input_size, output_size, states_size, attn_size, dtype=None,
            scope=None):
    cell = nn.rnn_cell.gru_cell([[input_size, states_size], output_size])

    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    def loop_fn(inputs, mask, state, attn_states, attn_mask, mapped_states):
        mask = mask[:, None]
        alpha = attention(state, mapped_states, output_size, attn_size,
                          attn_mask)
        context = theano.tensor.sum(alpha[:, :, None] * attn_states, 0)
        output, next_state = cell([inputs, context], state)
        next_state = (1.0 - mask) * state +  mask * next_state

        return [next_state, context]

    with ops.variable_scope(scope or "decoder"):
        mapped_states = map_attention_states(attention_states, states_size,
                                             attn_size)
        seq = [inputs, mask]
        outputs_info = [initial_state, None]
        non_seq = [attention_states, attention_mask, mapped_states]
        (states, contexts), updates = theano.scan(loop_fn, seq, outputs_info,
                                                  non_seq)

    return states, contexts


class rnnsearch:

    def __init__(self, **option):
        # source and target embedding dim
        sedim, tedim = option["embdim"]
        # source, target and attention hidden dim
        shdim, thdim, ahdim = option["hidden"]
        # maxout hidden dim
        maxdim = option["maxhid"]
        # maxout part
        maxpart = option["maxpart"]
        # deepout hidden dim
        deephid = option["deephid"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        # source and target vocabulary size
        svsize, tvsize = len(sid2w), len(tid2w)

        if "scope" not in option or option["scope"] is None:
            option["scope"] = "rnnsearch"

        if "initializer" not in option or option["initializer"] is None:
            option["initializer"] = None

        dtype = theano.config.floatX
        scope = option["scope"]
        initializer = option["initializer"]

        def prediction(prev_inputs, prev_state, context):
            features = [prev_state, prev_inputs, context]
            maxhid = nn.maxout(features, [[thdim, tedim, 2 * shdim], maxdim],
                               maxpart, True)
            readout = nn.linear(maxhid, [maxdim, deephid], False,
                                scope="deepout")
            logits = nn.linear(readout, [deephid, tvsize], True,
                               scope="logits")

            if logits.ndim == 3:
                new_shape = [logits.shape[0] * logits.shape[1], -1]
                logits = logits.reshape(new_shape)

            probs = theano.tensor.nnet.softmax(logits)

            return probs

        # training graph
        with ops.variable_scope(scope, initializer=initializer, dtype=dtype):
            src_seq = theano.tensor.imatrix("soruce_sequence")
            src_mask = theano.tensor.matrix("soruce_sequence_mask")
            tgt_seq = theano.tensor.imatrix("target_sequence")
            tgt_mask = theano.tensor.matrix("target_sequence_mask")

            with ops.variable_scope("source_embedding"):
                source_embedding = ops.get_variable("embedding",
                                                    [svsize, sedim])
                source_bias = ops.get_variable("bias", [sedim])

            with ops.variable_scope("target_embedding"):
                target_embedding = ops.get_variable("embedding",
                                                [tvsize, tedim])
                target_bias = ops.get_variable("bias", [tedim])

            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)

            source_inputs = source_inputs + source_bias
            target_inputs = target_inputs + target_bias

            outputs = encoder(source_inputs, src_mask, sedim, shdim)
            annotation = theano.tensor.concatenate(outputs, 2)

            # compute initial state for decoder
            # first state of backward encoder
            final_state = outputs[1][0]
            with ops.variable_scope("decoder"):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=theano.tensor.tanh)

            # run decoder
            decoder_outputs = decoder(target_inputs, tgt_mask, initial_state,
                                      annotation, src_mask, tedim, thdim,
                                      2 * shdim, ahdim)
            all_output, all_context = decoder_outputs

            shift_inputs = theano.tensor.zeros_like(target_inputs)
            shift_inputs = theano.tensor.set_subtensor(shift_inputs[1:],
                                                       target_inputs[:-1])

            init_state = initial_state[None, :, :]
            all_states = theano.tensor.concatenate([init_state, all_output], 0)
            prev_states = all_states[:-1]

            with ops.variable_scope("decoder"):
                probs = prediction(shift_inputs, prev_states, all_context)

            # compute cost
            idx = theano.tensor.arange(tgt_seq.flatten().shape[0])
            cost = -theano.tensor.log(probs[idx, tgt_seq.flatten()])
            cost = cost.reshape(tgt_seq.shape)
            cost = theano.tensor.sum(cost * tgt_mask, 0)
            cost = theano.tensor.mean(cost)

        training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        training_outputs = [cost]
        evaluate = theano.function(training_inputs, training_outputs)

        # encoding
        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [annotation, initial_state]
        encode = theano.function(encoding_inputs, encoding_outputs)

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = theano.tensor.ivector("prev_words")

            inputs = nn.embedding_lookup(target_embedding, prev_words)
            inputs = inputs + target_bias

            cond = theano.tensor.neq(prev_words, 0)
            # zeros out embedding if y is 0
            inputs = inputs * cond[:, None]

            cell = nn.rnn_cell.gru_cell([[tedim, 2 * shdim], thdim])

            with ops.variable_scope("decoder"):
                mapped_states = map_attention_states(annotation, 2 * shdim,
                                                     ahdim)
                alpha = attention(initial_state, mapped_states, thdim, ahdim,
                                  src_mask)
                context = theano.tensor.sum(alpha[:, :, None] * annotation, 0)
                output, next_state = cell([inputs, context], initial_state)
                probs = prediction(inputs, initial_state, context)

        # additional functions for decoding
        precomputation_inputs = [annotation]
        precomputation_outputs = mapped_states
        precompute = theano.function(precomputation_inputs,
                                     precomputation_outputs)

        inference_inputs = [initial_state, annotation, mapped_states, src_mask]
        inference_outputs = [alpha, context]
        infer = theano.function(inference_inputs, inference_outputs)

        prediction_inputs = [prev_words, initial_state, context]
        prediction_outputs = probs
        predict = theano.function(prediction_inputs, prediction_outputs)

        generation_inputs = [prev_words, initial_state, context]
        generation_outputs = next_state
        generate = theano.function(generation_inputs, generation_outputs)

        # sampling graph, this feature is optional
        with ops.variable_scope(scope, reuse=True):
            seed = option["seed"]
            seed_rng = numpy.random.RandomState(numpy.random.randint(seed))
            tseed = seed_rng.randint(numpy.iinfo(numpy.int32).max)
            stream = theano.sandbox.rng_mrg.MRG_RandomStreams(tseed)

            max_len = theano.tensor.iscalar()

            def sampling_loop(inputs, state):
                alpha = attention(state, mapped_states, shdim, ahdim, src_mask)
                context = theano.tensor.sum(alpha[:, :, None] * annotation, 0)
                probs = prediction(inputs, state, context)
                next_words = stream.multinomial(pvals=probs).argmax(axis=1)
                new_inputs = nn.embedding_lookup(target_embedding, next_words)
                new_inputs = new_inputs + target_bias
                output, next_state = cell([new_inputs, context], state)

                return [next_words, new_inputs, next_state]

            with ops.variable_scope("decoder"):
                batch = src_seq.shape[1]
                initial_inputs = theano.tensor.zeros([batch, tedim],
                                                     dtype=dtype)

                outputs_info = [None, initial_inputs, initial_state]
                outputs, updates = theano.scan(sampling_loop, [], outputs_info,
                                              n_steps=max_len)
                sampled_words = outputs[0]

        sampling_inputs = [src_seq, src_mask, max_len]
        sampling_outputs = sampled_words
        sample = theano.function(sampling_inputs, sampling_outputs,
                                 updates=updates)

        # attention graph, this feature is optional
        with ops.variable_scope(scope, reuse=True):
            def attention_loop(inputs, mask, state):
                mask = mask[:, None]
                alpha = attention(state, mapped_states, shdim, ahdim, src_mask)
                context = theano.tensor.sum(alpha[:, :, None] * annotation, 0)
                output, next_state = cell([inputs, context], state)
                next_state = (1.0 - mask) * state + mask * next_state

                return [alpha, next_state]

            with ops.variable_scope("decoder"):
                seq = [target_inputs, tgt_mask]
                outputs_info = [None, initial_state]
                outputs, updates = theano.scan(attention_loop, seq,
                                              outputs_info)
                attention_score = outputs[0]

        alignment_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        alignment_outputs = attention_score
        align = theano.function(alignment_inputs, alignment_outputs)

        self.cost = cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        self.align = align
        self.infer = infer
        self.sample = sample
        self.encode = encode
        self.predict = predict
        self.generate = generate
        self.evaluate = evaluate
        self.precompute = precompute
        self.option = option


# TODO: add batched decoding
def beamsearch(models, seq, mask=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None):
    size = beamsize
    dtype = dtype or theano.config.floatX

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)

    # get vocabulary from the first model
    vocab = models[0].option["vocabulary"][1][1]
    eosid = models[0].option["eosid"]
    bosid = models[0].option["bosid"]

    if maxlen == None:
        maxlen = seq.shape[0] * 3

    if minlen == None:
        minlen = seq.shape[0] / 2

    # encoding source
    if mask is None:
        mask = numpy.ones(seq.shape, dtype)

    annotations_and_istates = [model.encode(seq, mask) for model in models]
    annotations = [item[0] for item in annotations_and_istates]
    states = [item[1] for item in annotations_and_istates]
    mapped_annots = [model.precompute(annot) for annot, model in
                     zip(annotations, models)]

    initial_beam = beam(size)
    # bosid must be 0
    initial_beam.candidate = [[bosid]]
    initial_beam.score = numpy.zeros([1], dtype)

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

        # align returns [alpha, context]
        outputs = [model.infer(state, annot, mannot, batch_mask)
                   for model, state, annot, mannot in
                   zip(models, states, batch_annots, batch_mannots)]
        contexts = [item[1] for item in outputs]
        prob_dists = [model.predict(last_words, state, context) for
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

        states = [model.generate(last_words, state, context)
                  for model, state, context in zip(models, states, contexts)]

        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
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


def batchsample(model, seq, mask, maxlen=None):
    sampler = model.sample

    vocabulary = model.option["vocabulary"]
    eos = model.option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos]

    if maxlen == None:
        maxlen = int(len(seq) * 1.5)

    words = sampler(seq, mask, maxlen)
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
