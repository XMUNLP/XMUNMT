# encoder.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import ops
import tensorflow as tf


def rnn_encoder(cell, inputs, mask, initial_state, parallel_iterations=None,
                swap_memory=False, dtype=None):
    if not isinstance(cell, ops.nn.RNNCell):
        raise ValueError("only instances of RNNCell are supported")

    if inputs is None:
        raise ValueError("inputs must not be None")

    output_size = cell.output_size
    dtype = dtype or inputs.dtype

    time_steps = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]

    if initial_state is None:
        initial_state = cell.zero_state(batch, dtype)

    input_ta = tf.TensorArray(dtype, time_steps,
                              tensor_array_name="input_array")
    mask_ta = tf.TensorArray(dtype, time_steps,
                             tensor_array_name="mask_array")
    output_ta = tf.TensorArray(dtype, time_steps,
                               tensor_array_name="output_array")
    input_ta = input_ta.unstack(inputs)
    mask_ta = mask_ta.unstack(mask_ta)

    def loop(time, output_ta, state):
        inputs = input_ta.read(time)
        mask = mask_ta.read(time)
        output, new_state = cell(inputs, state)
        output_ta = output_ta.write(time, output)
        new_state = (1.0 - mask[:, None]) * state + mask[:, None] * new_state
        return (time + 1, output_ta, new_state)

    time = tf.constant(0, dtype=tf.int32, name="time")
    cond = lambda time, *_: time < time_steps
    loop_vars = (time, output_ta, initial_state)

    outputs = tf.while_loop(cond, loop, loop_vars,
                            parallel_iterations=parallel_iterations,
                            swap_memory=swap_memory)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])

    return all_output, final_state


def encoder(cell_fw, cell_bw, inputs, mask, parallel_iterations=None,
            swap_memory=False, dtype=None, scope=None):
    with tf.variable_scope(scope or "encoder"):
        with tf.variable_scope("forward"):
            output_fw, state_fw = rnn_encoder(cell_fw, inputs, mask,
                                              parallel_iterations, swap_memory,
                                              dtype)

        with tf.variable_scope("backward"):
            inputs = inputs[::-1]
            mask = mask[::-1]
            output_bw, state_bw = rnn_encoder(cell_bw, inputs, mask,
                                              parallel_iterations, swap_memory,
                                              dtype)

            output_bw = output_bw[::-1]

    return tf.concat(2, [output_fw, output_bw]), (state_fw, state_bw)
