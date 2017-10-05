# rnnsearch.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops
import numpy as np
import tensorflow as tf

from utils import function
from search import beam, select_nbest


def rnn_encoder(cell, inputs, sequence_length, parallel_iterations=None,
                swap_memory=False, dtype=None):
    parallel_iterations = parallel_iterations or 32

    batch = tf.shape(inputs)[1]
    dtype = dtype or inputs.dtype

    state = cell.zero_state(batch, dtype)

    (outputs, final_state) = ops.rnn.rnn_loop(cell, inputs, state,
                                              parallel_iterations,
                                              swap_memory, sequence_length,
                                              dtype)

    return (outputs, final_state)


def encoder(cell_below, cell_above, inputs, sequence_length,
            parallel_iterations=None, swap_memory=False, dtype=None,
            scope=None):
    time_dim = 0
    batch_dim = 1

    with tf.variable_scope(scope or "encoder"):
        with tf.variable_scope("forward"):
            output_fw, state_fw = rnn_encoder(cell_below, inputs,
                                              sequence_length,
                                              parallel_iterations, swap_memory,
                                              dtype)

        # backward direction
        inputs_reverse = tf.reverse_sequence(inputs, sequence_length, time_dim,
                                             batch_dim)

        with tf.variable_scope("backward"):
            output_bw, state_bw = rnn_encoder(cell_below, inputs_reverse,
                                              sequence_length,
                                              parallel_iterations, swap_memory,
                                              dtype)

            output_bw = tf.reverse_sequence(output_bw, sequence_length,
                                            time_dim, batch_dim)

    return tf.concat([output_fw, output_bw], 2)


def attention(query, attention_states, mapped_states, attention_mask,
              attn_size, dtype=None, scope=None):
    with tf.variable_scope(scope or "attention", dtype=dtype):
        hidden_size = attention_states.get_shape().as_list()[2]
        shape = tf.shape(attention_states)

        if mapped_states is None:
            batched_states = tf.reshape(attention_states, [-1, hidden_size])
            mapped_states = ops.nn.linear(batched_states, attn_size, True,
                                          scope="states")
            mapped_states = tf.reshape(mapped_states,
                                       [shape[0], shape[1], attn_size])

            if query is None:
                return mapped_states

        mapped_query = ops.nn.linear(query, attn_size, False, scope="logits")
        mapped_query = mapped_query[None, :, :]

        hidden = tf.tanh(mapped_query + mapped_states)
        hidden = tf.reshape(hidden, [-1, attn_size])

        score = ops.nn.linear(hidden, 1, True, scope="hidden")
        exp_score = tf.exp(score)
        exp_score = tf.reshape(exp_score, [shape[0], shape[1]])

        if attention_mask is not None:
            exp_score = exp_score * attention_mask

        alpha = exp_score / tf.reduce_sum(exp_score, 0)[None, :]

    return alpha[:, :, None]


def decoder(cell, inputs, initial_state, attention_states,  attention_length,
            attention_size, dtype=None, scope=None):
    if inputs is None:
        raise ValueError("inputs must not be None")

    time_steps = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    output_size = cell.output_size
    dtype = dtype or inputs.dtype
    attention_mask = tf.sequence_mask(attention_length, dtype=dtype)
    attention_mask = tf.transpose(attention_mask)

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    with tf.variable_scope(scope or "decoder", dtype=dtype):
        mapped_states = attention(None, attention_states, None, None,
                                  attention_size)

        input_ta = tf.TensorArray(tf.float32, time_steps,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_steps,
                                   tensor_array_name="output_array")
        context_ta = tf.TensorArray(tf.float32, time_steps,
                                    tensor_array_name="context_array")
        input_ta = input_ta.unstack(inputs)

        def loop(time, output_ta, context_ta, state):
            inputs = input_ta.read(time)

            with tf.variable_scope("below"):
                output, state = cell(inputs, state)

            alpha = attention(output, attention_states, mapped_states,
                              attention_mask, attention_size)
            context = tf.reduce_sum(alpha * attention_states, 0)

            with tf.variable_scope("above"):
                output, new_state = cell(context, state)
            output_ta = output_ta.write(time, output)
            context_ta = context_ta.write(time, context)
            return (time + 1, output_ta, context_ta, new_state)

        time = tf.constant(0, dtype=tf.int32, name="time")
        cond = lambda time, *_: time < time_steps
        loop_vars = (time, output_ta, context_ta, initial_state)

        outputs = tf.while_loop(cond, loop, loop_vars, parallel_iterations=32,
                                swap_memory=True)

        time, output_final_ta, context_final_ta, final_state = outputs

        final_output = output_final_ta.stack()
        final_context = context_final_ta.stack()
        final_output.set_shape([None, None, output_size])
        final_context.set_shape([None, None, 2 * output_size])

    return final_output, final_context


class nmt:

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
            src_len = tf.placeholder(tf.int32, [None], "source_length")
            tgt_seq = tf.placeholder(tf.int32, [None, None], "target_sequence")
            tgt_len = tf.placeholder(tf.int32, [None], "target_length")

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
            annotation = encoder(cell, cell, source_inputs, src_len)

            with tf.variable_scope("decoder"):
                ctx_sum = tf.reduce_sum(annotation, 0)
                initial_state = ops.nn.linear(ctx_sum, hidden_size, True,
                                              scope="initial")
                initial_state = tf.tanh(initial_state)

            zero_embedding = tf.zeros([1, tf.shape(tgt_seq)[1], emb_size])
            shift_inputs = tf.concat([zero_embedding, target_inputs], 0)
            shift_inputs = shift_inputs[:-1, :, :]
            shift_inputs.set_shape([None, None, emb_size])

            cell = ops.rnn_cell.GRUCell(hidden_size)

            decoder_outputs = decoder(cell, shift_inputs, initial_state,
                                      annotation, src_len, attn_size)
            output, context = decoder_outputs

            with tf.variable_scope("decoder"):
                logits = prediction(shift_inputs, output, context,
                                    keep_prob=keep_prob)

            labels = tf.reshape(tgt_seq, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=labels)
            crossent = tf.reshape(crossent, tf.shape(tgt_seq))
            mask = tf.sequence_mask(tgt_len, dtype=tf.float32)
            mask = tf.transpose(mask)
            cost = tf.reduce_mean(tf.reduce_sum(crossent * mask, 0))

        train_inputs = [src_seq, src_len, tgt_seq, tgt_len]
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
            annotation = encoder(cell, cell, source_inputs, src_len)

            # decoder
            with tf.variable_scope("decoder"):
                ctx_sum = tf.reduce_sum(annotation, 0)
                initial_state = ops.nn.linear(ctx_sum, hidden_size, True,
                                              scope="initial")
                initial_state = tf.tanh(initial_state)


            with tf.variable_scope("decoder"):
                mask = tf.sequence_mask(src_len, tf.shape(src_seq)[0],
                                        dtype=tf.float32)
                mask = tf.transpose(mask)
                mapped_states = attention(None, annotation, None, None,
                                          attn_size)

            cell = ops.rnn_cell.GRUCell(hidden_size)

            with tf.variable_scope("decoder"):
                with tf.variable_scope("below"):
                    output, state = cell(target_inputs, initial_state)
                alpha = attention(output, annotation, mapped_states,
                                  mask, attn_size)
                context = tf.reduce_sum(alpha * annotation, 0)
                with tf.variable_scope("above"):
                    output, next_state = cell(context, state)
                logits = prediction(target_inputs, next_state, context)
                probs = tf.nn.softmax(logits)

        encoding_inputs = [src_seq, src_len]
        encoding_outputs = [annotation, mapped_states, initial_state, mask]
        encode = function(encoding_inputs, encoding_outputs)

        prediction_inputs = [prev_word, initial_state, annotation,
                             mapped_states, mask]
        prediction_outputs = [probs, next_state, alpha]
        predict = function(prediction_inputs, prediction_outputs)

        self.cost = cost
        self.inputs = train_inputs
        self.outputs = train_outputs
        self.encode = encode
        self.predict = predict
        self.option = option


def beamsearch(model, seq, seqlen=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None):
    size = beamsize
    encode = model.encode
    predict = model.predict

    vocabulary = model.option["vocabulary"]
    eos_symbol = model.option["eos"]
    vocab = vocabulary[1][1]
    eosid = vocabulary[1][0][eos_symbol]

    time_dim = 0
    batch_dim = 1

    if seqlen is None:
        seq_len = np.array([seq.shape[time_dim]])
    else:
        seq_len = seqlen

    if maxlen == None:
        maxlen = seq_len[0] * 3

    if minlen == None:
        minlen = seq_len[0] / 2

    annotation, mapped_states, initial_state, attn_mask = encode(seq, seq_len)
    state = initial_state

    initial_beam = beam(size)
    initial_beam.candidate = [[eosid]]
    initial_beam.score = np.zeros([1], "float32")

    hypo_list = []
    beam_list = [initial_beam]
    cond = lambda x: x[-1] == eosid

    for k in range(maxlen):
        if size == 0:
            break

        prev_beam = beam_list[-1]
        candidate = prev_beam.candidate
        num = len(prev_beam.candidate)
        last_words = np.array(map(lambda t: t[-1], candidate), "int32")

        batch_annot = np.repeat(annotation, num, batch_dim)
        batch_mannot = np.repeat(mapped_states, num, batch_dim)
        batch_mask = np.repeat(attn_mask, num, batch_dim)

        prob_dist, state, alpha = predict(last_words, state, batch_annot,
                                          batch_mannot, batch_mask)

        logprobs = np.log(prob_dist)

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

        hypo_list.extend(outputs[0])
        batch_indices, word_indices = outputs[1:]
        size -= len(outputs[0])
        state = select_nbest(state, batch_indices)
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
