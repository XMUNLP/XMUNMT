# rnn.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn
# rnn_step => rnn_loop => dynamic_rnn => dynamic_bidirectional_rnn

import tensorflow as tf

from util import nest, on_device, infer_state_dtype, state_size_with_prefix


# one step for rnn
def rnn_step(time, sequence_length, min_sequence_length, zero_output, state,
             call_cell):
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def copy_one_through(output, new_output):
        copy_cond = (time >= sequence_length)
        fn = lambda: tf.where(copy_cond, output, new_output)
        return on_device(fn, new_output.op.device)

    # zero out state and output if time >= sequence_length
    def copy_some_through(flat_new_output, flat_new_state):
        new_output = map(copy_one_through, flat_zero_output, flat_new_output)
        new_state = map(copy_one_through, flat_state, flat_new_state)
        return new_output + new_state

    # start copy when time >= min_sequence_length
    def maybe_copy_some_through(flat_new_output, flat_new_state):
        cond = (time < min_sequence_length)
        output = lambda: flat_new_output + flat_new_state
        zero_output = lambda: copy_some_through(flat_new_output,
                                                flat_new_state)

        return tf.cond(cond, output, zero_output)

    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)

    if sequence_length is None:
        final_output_and_state = new_output + new_state
    elif min_sequence_length is None:
        final_output_and_state = copy_some_through(new_output, new_state)
    else:
        final_output_and_state = maybe_copy_some_through(new_output, new_state)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("state and output were not concatenated correctly.")

    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    # restore shape information
    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(zero_output, final_output)
    final_state = nest.pack_sequence_as(state, final_state)

    return final_output, final_state


def rnn_loop(cell, inputs, initial_state, parallel_iterations,
             swap_memory, sequence_length=None, dtype=None):
    state = initial_state

    if not isinstance(parallel_iterations, int):
        raise ValueError("parallel_iterations must be int")

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # construct an initial output
    input_shape = tf.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = map(lambda x: x.get_shape().with_rank_at_least(3),
                           flat_input)

    # an int number or None
    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    # check shape
    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
          raise ValueError("input size must be accessible via shape inference")
        got_time_steps = shape[0]
        got_batch_size = shape[1]
        if const_time_steps != got_time_steps:
            raise ValueError("time steps is not the same")
        if const_batch_size != got_batch_size:
            raise ValueError("batch_size is not the same")

    def create_zero_arrays(size):
        size = state_size_with_prefix(size, prefix=[batch_size])
        return tf.zeros(tf.stack(size), infer_state_dtype(dtype, state))

    flat_zero_output = map(create_zero_arrays, flat_output_size)
    zero_output = nest.pack_sequence_as(cell.output_size, flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = tf.reduce_min(sequence_length)

    time = tf.constant(0, dtype=tf.int32, name="time")

    with tf.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def create_ta(name, dtype):
        name = base_name + name
        return tf.TensorArray(dtype, time_steps, tensor_array_name=name)

    output_ta = tuple(create_ta("output_%d" % i,
                                infer_state_dtype(dtype, state))
                                for i in range(len(flat_output_size)))
    input_ta = tuple(create_ta("input_%d" % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(ip) for ta, ip in zip(input_ta, flat_input))

    def time_step(time, output_ta_t, state):
        input_t = tuple(ta.read(time) for ta in input_ta)
        # restore some shape information
        for ip, shape in zip(input_t, inputs_got_shape):
              ip.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(inputs, input_t)
        call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = rnn_step(time, sequence_length,
                                           min_sequence_length, zero_output,
                                           state, call_cell)
        else:
            (output, new_state) = call_cell()

        # pack state if using state tuples
        output = nest.flatten(output)
        output_ta_t = map(lambda x, y: x.write(time, y), output_ta_t, output)
        output_ta_t = tuple(output_ta_t)

        return (time + 1, output_ta_t, new_state)

    cond = lambda time, *_: time < time_steps
    loop_vars = (time, output_ta, state)

    outputs = tf.while_loop(cond, time_step, loop_vars,
                            parallel_iterations=parallel_iterations,
                            swap_memory=swap_memory)

    time, output_final_ta, final_state = outputs

    # unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = state_size_with_prefix(output_size,
                                       [const_time_steps, const_batch_size])
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(cell.output_size, final_outputs)

    return (final_outputs, final_state)


def dynamic_rnn(cell, inputs, initial_state=None, sequence_length=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                scope=None):
    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of rnn_cell")

    flat_input = nest.flatten(inputs)
    parallel_iterations = parallel_iterations or 32

    # check sequence_length dimension
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError("seq_len must be a vector of length batch_size")
        sequence_length = tf.identity(sequence_length, name="sequence_length")

    with tf.variable_scope(scope or "dynamic_rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        input_shape = map(lambda x: tf.shape(x), flat_input)
        batch_size = input_shape[0][1]

        for shape in input_shape:
            if shape[1].get_shape() != batch_size.get_shape():
                raise ValueError("all inputs should have the same batch size")

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("dtype must be provided if no initial_state")

        state = cell.zero_state(batch_size, dtype)

        def assert_has_shape(x, shape):
            x_shape = tf.shape(x)
            packed_shape = tf.stack(shape)
            return tf.Assert(tf.reduce_all(tf.equal(x_shape, packed_shape)),
                             ["expected shape for Tensor %s is " % x.name,
                              packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # perform some shape validation
            ops = [assert_has_shape(sequence_length, [batch_size])]
            with tf.control_dependencies(ops):
                sequence_length = tf.identity(sequence_length, name="checklen")

        inputs = nest.pack_sequence_as(inputs, flat_input)

        (outputs, final_state) = rnn_loop(cell, inputs, state,
                                          parallel_iterations, swap_memory,
                                          sequence_length, dtype)

    return (outputs, final_state)


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, scope=None):
    if not isinstance(cell_fw, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")

    with tf.variable_scope(scope or "birectional_dynamic_rnn"):
        with tf.variable_scope("forward") as fw_scope:
            output_fw, state_fw = dynamic_rnn(cell_fw, inputs, sequence_length,
                                              initial_state_fw, dtype,
                                              parallel_iterations, swap_memory,
                                              fw_scope)

        # backward direction
        time_dim = 0
        batch_dim = 1

        inputs_reverse = tf.reverse_sequence(inputs, sequence_length, time_dim,
                                             batch_dim)

        with tf.variable_scope("backward") as bw_scope:
            output_bw, state_bw = dynamic_rnn(cell_bw, inputs_reverse,
                                              sequence_length,
                                              initial_state_bw, dtype,
                                              parallel_iterations,
                                              swap_memory,
                                              bw_scope)

        output_bw = tf.reverse_sequence(output_bw, sequence_length, time_dim,
                                        batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (state_fw, state_bw)

    return (outputs, output_states)
