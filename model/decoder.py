# decoder.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops
import tensorflow as tf


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


def decoder(cell, inputs, initial_state, attention_states,  attention_mask,
            attention_size, dtype=None, scope=None):
    if inputs is None:
        raise ValueError("inputs must not be None")

    time_steps = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    output_size = cell.output_size
    dtype = dtype or inputs.dtype

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
        input_ta = input_ta.unpack(inputs)

        def loop(time, output_ta, context_ta, state):
            inputs = input_ta.read(time)

            with tf.variable_scope("below"):
                query, state = cell(inputs, state)

            alpha = attention(query, attention_states, mapped_states,
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

        final_output = output_final_ta.pack()
        final_context = context_final_ta.pack()
        final_output.set_shape([None, None, output_size])
        final_context.set_shape([None, None, 2 * output_size])

    return final_output, final_context
