# nmt.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops
import numpy as np
import tensorflow as tf

from utils import function
from encoder import encoder
from decoder import attention, decoder
from search import beam, select_nbest


class NMT:

    def __init__(self, emb_size, hidden_size, attn_size, svocab_size,
                 tvocab_size, **option):

        if "initializer" in option:
            initializer = option["initializer"]
        else:
            initializer = None

        if "keep_prob" in option:
            keep_prob = option["keep_prob"]
        else:
            keep_prob = 1.0

        def prediction(prev_inputs, states, context, keep_prob=1.0):
            if states.get_shape().ndims == 3:
                states = tf.reshape(states, [-1, hidden_size])

            if prev_inputs.get_shape().ndims == 3:
                prev_inputs = tf.reshape(prev_inputs, [-1, emb_size])

            if context.get_shape().ndims == 3:
                context = tf.reshape(context, [-1, 2 * hidden_size])

            features = [states, prev_inputs, context]
            readout = ops.nn.linear(features, emb_size, True,
                                    multibias=True, scope="deepout")
            readout = tf.tanh(readout)

            if keep_prob < 1.0:
                readout = tf.nn.dropout(readout, keep_prob=keep_prob)

            logits = ops.nn.linear(readout, tvocab_size, True,
                                   scope="logits")

            return logits

        # training graph
        with tf.variable_scope("rnnsearch", initializer=initializer):
            src_seq = tf.placeholder(tf.int32, [None, None], "soruce_sequence")
            src_mask = tf.placeholder(tf.float32, [None, None], "source_mask")
            tgt_seq = tf.placeholder(tf.int32, [None, None], "target_sequence")
            tgt_mask = tf.placeholder(tf.int32, [None, None], "target_mask")

            with tf.device("/cpu:0"):
                source_embedding = tf.get_variable("source_embedding",
                                                   [svocab_size, emb_size],
                                                   tf.float32)
                target_embedding = tf.get_variable("target_embedding",
                                                   [tvocab_size, emb_size],
                                                   tf.float32)
                source_inputs = tf.gather(source_embedding, src_seq)
                target_inputs = tf.gather(target_embedding, tgt_seq)

            if keep_prob < 1.0:
                source_inputs = tf.nn.dropout(source_inputs, keep_prob)
                target_inputs = tf.nn.dropout(target_inputs, keep_prob)

            cell = ops.rnn_cell.GRUCell(hidden_size)
            annotation = encoder(cell, cell, source_inputs, src_mask)

            with tf.variable_scope("decoder"):
                ctx_sum = tf.reduce_sum(annotation, 0)
                initial_state = ops.nn.linear(ctx_sum, hidden_size, True,
                                              scope="initial")
                initial_state = tf.tanh(initial_state)

            zero_embedding = tf.zeros([1, tf.shape(tgt_seq)[1], emb_size])
            shift_inputs = tf.concat(0, [zero_embedding, target_inputs])
            shift_inputs = shift_inputs[:-1, :, :]
            shift_inputs.set_shape([None, None, emb_size])

            cell = ops.rnn_cell.GRUCell(hidden_size)

            decoder_outputs = decoder(cell, shift_inputs, initial_state,
                                      annotation, src_mask, attn_size)
            output, context = decoder_outputs

            with tf.variable_scope("decoder"):
                logits = prediction(shift_inputs, output, context,
                                    keep_prob=keep_prob)

            labels = tf.reshape(tgt_seq, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                      labels)
            crossent = tf.reshape(crossent, tf.shape(tgt_seq))
            cost = tf.reduce_mean(tf.reduce_sum(crossent * tgt_mask, 0))

        train_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        train_outputs = [cost]

        # decoding graph
        with tf.variable_scope("rnnsearch", reuse=True):
            prev_word = tf.placeholder(tf.int32, [None], "prev_token")

            with tf.device("/cpu:0"):
                source_embedding = tf.get_variable("source_embedding",
                                                   [svocab_size, emb_size],
                                                   tf.float32)
                target_embedding = tf.get_variable("target_embedding",
                                                   [tvocab_size, emb_size],
                                                   tf.float32)

                source_inputs = tf.gather(source_embedding, src_seq)
                target_inputs = tf.gather(target_embedding, prev_word)

            cond = tf.equal(prev_word, 0)
            cond = tf.cast(cond, tf.float32)
            target_inputs = target_inputs * (1.0 - tf.expand_dims(cond, 1))

            # encoder
            cell = ops.rnn_cell.GRUCell(hidden_size)
            annotation = encoder(cell, cell, source_inputs, src_mask)

            # decoder
            with tf.variable_scope("decoder"):
                ctx_sum = tf.reduce_sum(annotation, 0)
                initial_state = ops.nn.linear(ctx_sum, hidden_size, True,
                                              scope="initial")
                initial_state = tf.tanh(initial_state)


            with tf.variable_scope("decoder"):
                mapped_states = attention(None, annotation, None, None,
                                          attn_size)

            cell = ops.rnn_cell.GRUCell(hidden_size)

            with tf.variable_scope("decoder"):
                with tf.variable_scope("below"):
                    output, state = cell(target_inputs, initial_state)
                alpha = attention(output, annotation, mapped_states,
                                  src_mask, attn_size)
                context = tf.reduce_sum(alpha * annotation, 0)
                with tf.variable_scope("above"):
                    output, next_state = cell(context, state)
                logits = prediction(target_inputs, next_state, context)
                probs = tf.nn.softmax(logits)

        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [annotation, mapped_states, initial_state]
        encode = function(encoding_inputs, encoding_outputs)

        prediction_inputs = [prev_word, initial_state, annotation,
                             mapped_states, src_mask]
        prediction_outputs = [probs, next_state, alpha]
        predict = function(prediction_inputs, prediction_outputs)

        self.cost = cost
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.encode = encode
        self.predict = predict
        self.option = option


# TODO: add batched decoding
def beamsearch(models, seq, mask=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None):
    size = beamsize

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
        mask = np.ones(seq.shape, dtype)

    outputs = [model.encode(seq, mask) for model in models]
    annotations = [item[0] for item in outputs]
    states = [item[1] for item in outputs]
    mapped_annots = [item[2] for item in outputs]

    initial_beam = beam(size)
    # bosid must be 0
    initial_beam.candidate = [[bosid]]
    initial_beam.score = np.zeros([1], dtype)

    hypo_list = []
    beam_list = [initial_beam]
    cond = lambda x: x[-1] == eosid

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(candidate)
        last_words = np.array(map(lambda t: t[-1], candidate), "int32")

        # compute context first, then compute word distribution
        batch_mask = np.repeat(mask, num, 1)
        batch_annots = map(np.repeat, annotations, [num] * num_models,
                           [1] * num_models)
        batch_mannots = map(np.repeat, mapped_annots, [num] * num_models,
                           [1] * num_models)

        # predict returns [probs, context, alpha]
        outputs = [model.predict(last_words, state, annot, mannot, batch_mask)
                                 for model, state, annot, mannot in
                                 zip(models, states, batch_annots,
                                     batch_mannots)]
        prob_dists = [item[0] for item in outputs]
        states = [item[1] for item in outputs]

        # search nbest given word distribution
        if arithmetic:
            logprobs = np.log(sum(prob_dists) / num_models)
        else:
            # geometric mean
            logprobs = sum(np.log(prob_dists)) / num_models

        if k < minlen:
            logprobs[:, eosid] = -np.inf

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -np.inf
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
        last_words = np.array(map(lambda t: t[-1], candidate), "int32")

        states = select_nbest(states, batch_indices)

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
    hypo_list = np.array(hypo_list)[np.argsort(score_list)]
    score_list = np.array(sorted(score_list))

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
