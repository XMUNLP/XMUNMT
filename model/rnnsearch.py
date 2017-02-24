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
    if not isinstance(cell, nn.rnn_cell.rnn_cell):
        raise ValueError("cell is not an instance of rnn_cell")

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
    # ops.scan is a wrapper of theano.scan, which automatically add updates to
    # optimizer, you can set return_updates=True to behave like Theano's scan
    states = ops.scan(loop_fn, seq, [initial_state])

    return states


def encoder(cell, inputs, mask, initial_state=None, dtype=None, scope=None):
    with ops.variable_scope(scope or "encoder", dtype=dtype):
        with ops.variable_scope("forward"):
            fd_states = gru_encoder(cell, inputs, mask, initial_state, dtype)
        with ops.variable_scope("backward"):
            inputs = inputs[::-1]
            mask = mask[::-1]
            bd_states = gru_encoder(cell, inputs, mask, initial_state, dtype)
            bd_states = bd_states[::-1]

    return fd_states, bd_states


def attention(query, states, mapped_states, attention_mask, size, dtype=None,
              scope=None):
    query_size, states_size, attn_size = size

    with ops.variable_scope(scope or "attention", dtype=dtype):
        if mapped_states is None:
            mapped_states = nn.linear(states, [states_size, attn_size],
                                      False, scope="attention_w")

            if query is None:
                return mapped_states

        mapped_query = nn.linear(query, [query_size, attn_size], False,
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


def decoder(cell, inputs, mask, initial_state, attention_states,
            attention_mask, attn_size, mapped_states=None, dtype=None,
            scope=None):
    input_size, states_size = cell.input_size

    output_size = cell.output_size
    dtype = dtype or inputs.dtype
    att_size = [output_size, states_size, attn_size]

    def loop_fn(inputs, mask, state, attn_states, attn_mask, mapped_states):
        mask = mask[:, None]
        alpha = attention(state, None, mapped_states, attn_mask, att_size)
        context = theano.tensor.sum(alpha[:, :, None] * attn_states, 0)
        output, next_state = cell([inputs, context], state)
        next_state = (1.0 - mask) * state +  mask * next_state

        return [next_state, context]

    with ops.variable_scope(scope or "decoder"):
        if mapped_states is None:
            mapped_states = attention(None, attention_states, None, None,
                                      att_size)
        seq = [inputs, mask]
        outputs_info = [initial_state, None]
        non_seq = [attention_states, attention_mask, mapped_states]
        (states, contexts) = ops.scan(loop_fn, seq, outputs_info, non_seq)

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

        if "initializer" not in option:
            option["initializer"] = None

        if "regularizer" not in option:
            option["regularizer"] = None

        if "criterion" not in option:
            option["criterion"] = "mle"

        if "keep_prob" not in option:
            option["keep_prob"] = 1.0

        dtype = theano.config.floatX
        scope = option["scope"]
        criterion = option["criterion"]
        initializer = option["initializer"]
        regularizer = option["regularizer"]
        keep_prob = option["keep_prob"] or 1.0

        # MRT mode do not use dropout
        if criterion == "mrt":
            keep_prob = 1.0

        def prediction(prev_inputs, prev_state, context, keep_prob=1.0):
            features = [prev_state, prev_inputs, context]
            maxhid = nn.maxout(features, [[thdim, tedim, 2 * shdim], maxdim],
                               maxpart, True)
            readout = nn.linear(maxhid, [maxdim, deephid], False,
                                scope="deepout")

            if keep_prob < 1.0:
                readout = nn.dropout(readout, keep_prob=keep_prob)

            logits = nn.linear(readout, [deephid, tvsize], True,
                               scope="logits")

            if logits.ndim == 3:
                new_shape = [logits.shape[0] * logits.shape[1], -1]
                logits = logits.reshape(new_shape)

            probs = theano.tensor.nnet.softmax(logits)

            return probs

        # training graph
        with ops.variable_scope(scope, initializer=initializer,
                                regularizer=regularizer, dtype=dtype):
            src_seq = theano.tensor.imatrix("soruce_sequence")
            src_mask = theano.tensor.matrix("soruce_sequence_mask")
            tgt_seq = theano.tensor.imatrix("target_sequence")
            tgt_mask = theano.tensor.matrix("target_sequence_mask")

            if criterion == "mrt":
                loss = theano.tensor.vector("loss_score")
                sharp = theano.tensor.scalar("sharpness")

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

            if keep_prob < 1.0:
                source_inputs = nn.dropout(source_inputs, keep_prob=keep_prob)
                target_inputs = nn.dropout(target_inputs, keep_prob=keep_prob)

            cell = nn.rnn_cell.gru_cell([sedim, shdim])

            outputs = encoder(cell, source_inputs, src_mask)
            annotation = theano.tensor.concatenate(outputs, 2)

            annotation = nn.dropout(annotation, keep_prob=keep_prob)

            # compute initial state for decoder
            # first state of backward encoder
            final_state = outputs[1][0]
            with ops.variable_scope("decoder"):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=theano.tensor.tanh)

            # run decoder
            cell = nn.rnn_cell.gru_cell([[tedim, 2 * shdim], thdim])

            if criterion == "mrt":
                # In MRT training, shape of src_seq and src_mask are assumed
                # to have [len, 1]
                batch = tgt_seq.shape[1]
                with ops.variable_scope("decoder"):
                    mapped_states = attention(None, annotation, None, None,
                                              [thdim, 2 * shdim, ahdim])
                b_src_mask = theano.tensor.repeat(src_mask, batch, 1)
                b_annotation = theano.tensor.repeat(annotation, batch, 1)
                b_mapped_states = theano.tensor.repeat(mapped_states, batch, 1)
                b_initial_state = theano.tensor.repeat(initial_state, batch, 0)

                decoder_outputs = decoder(cell, target_inputs, tgt_mask,
                                          b_initial_state, b_annotation,
                                          b_src_mask, ahdim, b_mapped_states)
            else:
                decoder_outputs = decoder(cell, target_inputs, tgt_mask,
                                          initial_state, annotation, src_mask,
                                          ahdim)

            all_output, all_context = decoder_outputs
            shift_inputs = theano.tensor.zeros_like(target_inputs)
            shift_inputs = theano.tensor.set_subtensor(shift_inputs[1:],
                                                       target_inputs[:-1])

            if criterion == "mrt":
                init_state = b_initial_state[None, :, :]
            else:
                init_state = initial_state[None, :, :]

            all_states = theano.tensor.concatenate([init_state, all_output], 0)
            prev_states = all_states[:-1]

            with ops.variable_scope("decoder"):
                probs = prediction(shift_inputs, prev_states, all_context,
                                   keep_prob=keep_prob)

            # compute cost
            idx = theano.tensor.arange(tgt_seq.flatten().shape[0])
            ce = -theano.tensor.log(probs[idx, tgt_seq.flatten()])
            ce = ce.reshape(tgt_seq.shape)
            ce = theano.tensor.sum(ce * tgt_mask, 0)

            if criterion == "mle":
                cost = theano.tensor.mean(ce)
            else:
                # ce is positive here
                logp = -ce
                score = sharp * logp
                # safe softmax
                score = score - theano.tensor.max(score)
                score = theano.tensor.exp(score)
                qprob = score / theano.tensor.sum(score)
                risk = theano.tensor.sum(qprob * loss)
                cost = risk

        if criterion == "mle":
            training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        else:
            training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask, loss,
                               sharp]
        training_outputs = [cost]

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = theano.tensor.ivector("prev_words")

            # disable dropout
            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            source_inputs = source_inputs + source_bias
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)
            target_inputs = target_inputs + target_bias

            cell = nn.rnn_cell.gru_cell([sedim, shdim])
            outputs = encoder(cell, source_inputs, src_mask)
            annotation = theano.tensor.concatenate(outputs, 2)

            # decoder
            final_state = outputs[1][0]
            with ops.variable_scope("decoder"):
                initial_state = nn.feedforward(final_state, [shdim, thdim],
                                               True, scope="initial",
                                               activation=theano.tensor.tanh)

            inputs = nn.embedding_lookup(target_embedding, prev_words)
            inputs = inputs + target_bias

            cond = theano.tensor.neq(prev_words, 0)
            # zeros out embedding if y is 0
            inputs = inputs * cond[:, None]

            cell = nn.rnn_cell.gru_cell([[tedim, 2 * shdim], thdim])

            # encode -> prediction -> generation
            # prediction: prev_word + prev_state => context, next_word
            # generation: curr_word + context + prev_state => next_state
            # here, initial_state is merely a placeholder
            with ops.variable_scope("decoder"):
                # used in encoding
                mapped_states = attention(None, annotation, None, None,
                                          [thdim, 2 * shdim, ahdim])
                # used in prediction
                alpha = attention(initial_state, None, mapped_states, src_mask,
                                  [thdim, 2 * shdim, ahdim])
                context = theano.tensor.sum(alpha[:, :, None] * annotation, 0)
                probs = prediction(inputs, initial_state, context)
                # used in generation
                output, next_state = cell([inputs, context], initial_state)

        # encoding
        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [annotation, initial_state, mapped_states]
        encode = theano.function(encoding_inputs, encoding_outputs)

        prediction_inputs = [prev_words, initial_state, annotation,
                             mapped_states, src_mask]
        prediction_outputs = [probs, context, alpha]
        predict = theano.function(prediction_inputs, prediction_outputs)

        generation_inputs = [prev_words, initial_state, context]
        generation_outputs = next_state
        generate = theano.function(generation_inputs, generation_outputs)

        # sampling graph, this feature is optional
        with ops.variable_scope(scope, reuse=True):
            max_len = theano.tensor.iscalar()

            def sampling_loop(inputs, state, attn_states, attn_mask, m_states):
                alpha = attention(state, None, m_states, attn_mask,
                                  [thdim, 2 * shdim, ahdim])
                context = theano.tensor.sum(alpha[:, :, None] * attn_states, 0)
                probs = prediction(inputs, state, context)
                next_words = ops.random.multinomial(probs).argmax(axis=1)
                new_inputs = nn.embedding_lookup(target_embedding, next_words)
                new_inputs = new_inputs + target_bias
                output, next_state = cell([new_inputs, context], state)

                return [next_words, new_inputs, next_state]

            with ops.variable_scope("decoder"):
                batch = src_seq.shape[1]
                initial_inputs = theano.tensor.zeros([batch, tedim],
                                                     dtype=dtype)

                outputs_info = [None, initial_inputs, initial_state]
                nonseq = [annotation, src_mask, mapped_states]
                outputs, updates = theano.scan(sampling_loop, [], outputs_info,
                                               nonseq, n_steps=max_len)
                sampled_words = outputs[0]

        sampling_inputs = [src_seq, src_mask, max_len]
        sampling_outputs = sampled_words
        sample = theano.function(sampling_inputs, sampling_outputs,
                                 updates=updates)

        # attention graph, this feature is optional
        with ops.variable_scope(scope, reuse=True):
            def attention_loop(inputs, mask, state, attn_states, attn_mask,
                               m_states):
                mask = mask[:, None]
                alpha = attention(state, None, m_states, attn_mask,
                                  [thdim, 2 * shdim, ahdim])
                context = theano.tensor.sum(alpha[:, :, None] * attn_states, 0)
                output, next_state = cell([inputs, context], state)
                next_state = (1.0 - mask) * state + mask * next_state

                return [alpha, next_state]

            with ops.variable_scope("decoder"):
                seq = [target_inputs, tgt_mask]
                outputs_info = [None, initial_state]
                nonseq = [annotation, src_mask, mapped_states]
                (alpha, state), updaptes = theano.scan(attention_loop, seq,
                                                       outputs_info, nonseq)
                attention_score = alpha

        alignment_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        alignment_outputs = attention_score
        align = theano.function(alignment_inputs, alignment_outputs)

        self.cost = cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        self.align = align
        self.sample = sample
        self.encode = encode
        self.predict = predict
        self.generate = generate
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

    outputs = [model.encode(seq, mask) for model in models]
    annotations = [item[0] for item in outputs]
    states = [item[1] for item in outputs]
    mapped_annots = [item[2] for item in outputs]

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

        # predict returns [probs, context, alpha]
        outputs = [model.predict(last_words, state, annot, mannot, batch_mask)
                                 for model, state, annot, mannot in
                                 zip(models, states, batch_annots,
                                     batch_mannots)]
        prob_dists = [item[0] for item in outputs]
        contexts = [item[1] for item in outputs]

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
    eosid = model.option["eosid"]
    vocab = vocabulary[1][1]

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


# used for analysis
def evaluate_model(model, xseq, xmask, yseq, ymask, alignment=None,
                   verbose=False):
    t = yseq.shape[0]
    batch = yseq.shape[1]

    vocab = model.option["vocabulary"][1][1]

    annotation, states, mapped_annot = model.encode(xseq, xmask)

    last_words = numpy.zeros([batch], "int32")
    costs = numpy.zeros([batch], "float32")
    indices = numpy.arange(batch, dtype="int32")

    for i in range(t):
        outputs = model.predict(last_words, states, annotation, mapped_annot,
                                xmask)
        # probs: batch * vocab
        # contexts: batch * hdim
        # alpha: batch * srclen
        probs, contexts, alpha = outputs

        if alignment is not None:
            # alignment tgt * src * batch
            contexts = numpy.sum(alignment[i][:, :, None] * annotation, 0)

        max_prob = probs.argmax(1)
        order = numpy.argsort(-probs)
        label = yseq[i]
        mask = ymask[i]

        if verbose:
            for i, (pred, gold, msk) in enumerate(zip(max_prob, label, mask)):
                if msk and pred != gold:
                    gold_order = None

                    for j in range(len(order[i])):
                        if order[i][j] == gold:
                            gold_order = j
                            break

                    ent = -numpy.sum(probs[i] * numpy.log(probs[i]))
                    pp = probs[i, pred]
                    gp = probs[i, gold]
                    pred = vocab[pred]
                    gold = vocab[gold]
                    print "%d: predication error, %s vs %s" % (i, pred, gold)
                    print "prob: %f vs %f, entropy: %f" % (pp, gp, ent)
                    print "gold is %d-th best" % (gold_order + 1)

        costs -= numpy.log(probs[indices, label]) * mask

        last_words = label
        states = model.generate(last_words, states, contexts)

    return costs
