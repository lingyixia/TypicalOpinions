import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', 'finished/chunked/train_*',
                           '数据路径')
tf.app.flags.DEFINE_string('vocab_path', 'finished/vocab', '词表路径')
tf.app.flags.DEFINE_string('embedding_path', 'Wordvec/Word2vec_model.pkl',
                           '词向量路径')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'batchsize是否等于1(用于predict阶段)')

# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log', '日志路径')
tf.app.flags.DEFINE_string('exp_name', 'myTest', '保存路径名')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'rnn隐藏层维度')
tf.app.flags.DEFINE_integer('emb_dim', 128, '词向量维度')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batchsize')
tf.app.flags.DEFINE_integer('max_enc_steps', 150, 'encoder最大')
tf.app.flags.DEFINE_integer('max_dec_steps', 35, '解码最大长度')
tf.app.flags.DEFINE_integer('min_dec_steps', 2, '最小解码长度')
tf.app.flags.DEFINE_integer('beam_size', 5, 'beam size')
tf.app.flags.DEFINE_integer('vocab_size', 45000, '词表大小')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'random参数')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'random参数')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, '梯度最大最小')
tf.app.flags.DEFINE_boolean('pointer_gen', True, '是否使用指针网络')
tf.app.flags.DEFINE_boolean('coverage', True, '是否使用coverage机制')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'coverage参数')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False)
tf.app.flags.DEFINE_boolean('restore_best_model', False, '保存位置')

tf.app.flags.DEFINE_boolean('debug', False, '')


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    sess = tf.Session(config=util.get_config())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    curr_ckpt = util.load_ckpt(saver, sess, "eval")

    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    new_saver = tf.train.Saver()
    new_saver.save(sess, new_fname)
    exit()


def convert_to_coverage_model():
    sess = tf.Session(config=util.get_config())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    curr_ckpt = util.load_ckpt(saver, sess)

    new_fname = curr_ckpt + '_cov_init'
    new_saver = tf.train.Saver()
    new_saver.save(sess, new_fname)
    exit()


def setup_training(model, batcher):
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()
    if FLAGS.convert_to_coverage_model:
        convert_to_coverage_model()
    if FLAGS.restore_best_model:
        restore_best_model()
    saver = tf.train.Saver(max_to_keep=3)

    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=60,
                             global_step=model.global_step)
    summary_writer = sv.summary_writer
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    try:
        run_training(model, batcher, sess_context_manager, sv, summary_writer)
    except KeyboardInterrupt:
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    with sess_context_manager as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        while True:
            batch = batcher.next_batch()
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            loss = results['loss']
            if not np.isfinite(loss):
                pass
            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']

            summaries = results['summaries']
            train_step = results['global_step']

            summary_writer.add_summary(summaries, train_step)
            if train_step % 100 == 0:
                summary_writer.flush()


def run_eval(model, batcher, vocab):
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval")
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0
    best_loss = None

    while True:
        _ = util.load_ckpt(saver, sess)
        batch = batcher.next_batch()
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        loss = results['loss']
        if FLAGS.coverage:
            coverage_loss = results['coverage_loss']

        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)
        if best_loss is None or running_avg_loss < best_loss:
            saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        if train_step % 100 == 0:
            summary_writer.flush()


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size, FLAGS.embedding_path)
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size
    if FLAGS.single_pass and FLAGS.mode != 'decode':
        pass

    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    tf.set_random_seed(111)
    if hps.mode == 'train':
        model = SummarizationModel(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        decode_model_hps = hps._replace(max_dec_steps=1)
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder.decode()


if __name__ == '__main__':
    tf.app.run()
