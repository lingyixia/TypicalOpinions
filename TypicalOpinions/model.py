import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


class SummarizationModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        hps = self._hps
        # encoder part encode输入的最大长度没准
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        self.categoryWord = tf.placeholder(tf.int32, shape=[hps.batch_size], name='categoryWord')
        self.tagWords = tf.placeholder(tf.int32, shape=[hps.batch_size, None], name='tagWords')
        self.tagWordsLen = tf.placeholder(tf.int32, shape=[hps.batch_size], name='tagWordLen')
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')

        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        feed_dict[self.categoryWord] = batch.categoryWordId
        feed_dict[self.tagWords] = batch.tagWordIds
        feed_dict[self.tagWordsLen] = batch.tagWordLen
        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_topicEncoder(self, emb_categoryWord, emb_tagWords, tagWordsLen):
        with tf.variable_scope('topicEncoder'):
            input = tf.concat([tf.expand_dims(emb_categoryWord, axis=1), emb_tagWords], axis=1)
            tf.add(tagWordsLen, 1)
            cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            _, (state_c, state_h) = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32, sequence_length=tagWordsLen,
                                                      swap_memory=True)
            return state_c, state_h

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st, emb_categoryWord, emb_tagWords, tagWordsLen):
        emb_tagWordsEnv = list()
        # topicWord = tf.concat([tf.expand_dims(emb_categoryWord, axis=1), emb_tagWords], axis=1)
        topicWords = tf.unstack(emb_tagWords, axis=0)
        tagWordsLen = tf.unstack(tagWordsLen, axis=0)
        for topicWords, len in zip(topicWords, tagWordsLen):
            temp = topicWords[:len, :]
            emb_tagWordsEnv.append(tf.reduce_mean(temp, axis=0))
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2 + emb_tagWords.shape[2].value, hidden_dim],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2 + emb_tagWords.shape[2].value, hidden_dim],
                                         dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c, emb_tagWordsEnv])
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h, emb_tagWordsEnv])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, emb_categoryWord, inputs):
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None
        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state,
                                                                             emb_categoryWord, self.categoryWord,
                                                                             self._enc_batch,
                                                                             self._enc_states, self._enc_padding_mask,
                                                                             cell, initial_state_attention=(
                    hps.mode == "decode"), pointer_gen=hps.pointer_gen, use_coverage=hps.coverage,
                                                                             prev_coverage=prev_coverage)
        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]
            extended_vsize = self._vocab.size() + self._max_art_oovs
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                    vocab_dists]

            batch_nums = tf.range(0, limit=self._hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
            batch_nums = tf.tile(batch_nums, [1, attn_len])
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                    attn_dists]
            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path)
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()
        with tf.variable_scope('seq2seq'):
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            with tf.variable_scope('embedding'):
                # embeddingToken = tf.get_variable('embedding', [4, hps.emb_dim], dtype=tf.float32,
                #                                  initializer=self.trunc_norm_init)
                # embedding = tf.Variable(dtype=tf.float32,initial_value=np.array(self._vocab.embedding))
                # embedding=tf.concat([embeddingToken,embedding],axis=0)
                embedding = tf.get_variable('embedding', [self._vocab.size(), hps.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
                if hps.mode == "train": self._add_emb_vis(embedding)
                # (batch_size,max_enc_steps,emb_size)(16,<=150,64)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
                category_embeddings = tf.nn.embedding_lookup(embedding, self.categoryWord)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]
                emb_categoryWord = tf.nn.embedding_lookup(embedding, self.categoryWord)
                emb_tagWords = tf.nn.embedding_lookup(embedding, self.tagWords)
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self._enc_states = enc_outputs  # (16,?,512)
            self._dec_in_state = self._reduce_states(fw_st, bw_st, emb_categoryWord, emb_tagWords, self.tagWordsLen)
            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(
                    emb_categoryWord,
                    emb_dec_inputs)
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                w_t = tf.transpose(w)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))

                vocab_dists = [tf.nn.softmax(s) for s in
                               vocab_scores]

            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:
                final_dists = vocab_dists

            if hps.mode in ['train', 'eval']:
                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        loss_per_step = []
                        batch_nums = tf.range(0, limit=hps.batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:, dec_step]
                            indices = tf.stack((batch_nums, targets), axis=1)
                            gold_probs = tf.gather_nd(dist, indices)
                            losses = -tf.log(gold_probs + 1e-10)
                            # losses = tf.clip_by_value(gold_probs, 1e-10, 1.0)
                            # if not np.isfinite(losses):
                            #     print('fffffffffffffffffffffffffffffffffffffff' * 10, losses)
                            loss_per_step.append(losses)

                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                    else:
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                      self._target_batch,
                                                                      self._dec_padding_mask)
                    tf.summary.scalar('loss', self._loss)
                    if hps.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)
                        self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                        tf.summary.scalar('total_loss', self._total_loss)

        if hps.mode == "decode":
            assert len(final_dists) == 1
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)
            self._topk_log_probs = tf.log(topk_probs)

    def _add_train_op(self):
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        with tf.device("/gpu:1"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        with tf.device("/gpu:1"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')

    def build_graph(self):
        t0 = time.time()
        self._add_placeholders()
        with tf.device("/gpu:1"):
            self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()

    def run_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step],
                                                           feed_dict)
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
        beam_size = len(dec_init_states)
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)
        new_h = np.concatenate(hiddens, axis=0)
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self._enc_batch: batch.enc_batch,
            self.categoryWord: batch.categoryWordId
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()
        if FLAGS.pointer_gen:
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
    dec_lens = tf.reduce_sum(padding_mask, axis=1)
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens
    return tf.reduce_mean(values_per_ex)


def _coverage_loss(attn_dists, padding_mask):
    coverage = tf.zeros_like(attn_dists[0])
    covlosses = []
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)
        coverage += a
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
