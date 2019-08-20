import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


def attention_decoder(decoder_inputs, initial_state, emb_categoryWord, categoryWord, en_batch, encoder_states,
                      enc_padding_mask, cell,
                      initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None):
    with variable_scope.variable_scope("attention_decoder") as scope:
        batch_size = encoder_states.get_shape()[0].value
        attn_size = encoder_states.get_shape()[2].value
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        attention_vec_size = attn_size
        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
        v = variable_scope.get_variable("v", [attention_vec_size])
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])
        if prev_coverage is not None:
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        def attention(decoder_state, coverage=None):
            """Calculate the context vector and attention distribution from the decoder state.
            Args:
              decoder_state: state of the decoder
              coverage: Optional. Previous timestep's coverage vector, shape (batch_size, attn_len, 1, 1).
            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
              coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
            """
            with variable_scope.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                # shape (batch_size, attention_vec_size)
                decoder_features = linear(decoder_state, attention_vec_size, True)
                # reshape to (batch_size, 1, 1, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    #####################################主题词
                    # eUnstack = tf.unstack(e)
                    # en_batchUnstack = tf.unstack(en_batch)
                    # categoryWordUnstack = tf.unstack(categoryWord)
                    # temp = list()
                    # for index, row in enumerate(eUnstack):
                    #     condition = tf.equal(en_batchUnstack[index], categoryWordUnstack[index])
                    #     temp.append(tf.where(condition, row * 10, row))
                    # e = tf.stack(temp)
                    ########################################
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    res = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                    return res

                if use_coverage and coverage is not None:  # non-first step of coverage
                    # Multiply coverage vector by w_c to get coverage_features.
                    # c has shape (batch_size, attn_length, 1, attention_vec_size)
                    coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME")
                    # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)公式(11)
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                        [2, 3])  # shape (batch_size,attn_length)
                    # Calculate attention distribution
                    attn_dist = masked_attention(e)
                    # Update coverage vector
                    coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    # calculate e
                    e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features),
                                            [2, 3])
                    attn_dist = masked_attention(e)  # 得到的是论文公式(2)
                    if use_coverage:  # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)  # initialize coverage
                # Calculate the context vector from attn_dist and encoder_states
                # shape (batch_size, attn_size).
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])
            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage
        context_vector = array_ops.zeros([batch_size, attn_size])
        context_vector.set_shape([None, attn_size])
        if initial_state_attention:
            context_vector, _, coverage = attention(initial_state, coverage)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            input_size = inp.get_shape().with_rank(2)[1]
            # 计算w*h 和 decodeinput合为一个context_vector是论文的(3)公式
            x = linear([inp] + [context_vector], input_size, True)
            cell_output, state = cell(x, state)
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    context_vector, attn_dist, _ = attention(state, coverage)
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):  # 公式(8)
                    p_gen = linear([context_vector, state.c, state.h, x], 1, True)
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)
            # This is V[s_t, h*_t] + b in the paper 公式(4) 内部分
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)
        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])
        return outputs, state, attn_dists, p_gens, coverage


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        total_arg_size += shape[1]
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term
