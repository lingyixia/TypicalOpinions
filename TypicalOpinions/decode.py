import os
import time
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
import logging
import numpy as np

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60


class BeamSearchDecoder(object):
    def __init__(self, model, batcher, vocab):
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=util.get_config())

        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if FLAGS.single_pass:
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))

        else:
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)
        if FLAGS.single_pass:
            self._rouge_art_dir = os.path.join(self._decode_dir, "article")
            if not os.path.exists(self._rouge_art_dir): os.mkdir(self._rouge_art_dir)
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
            self._rouge_tag_dir = os.path.join(self._decode_dir, "tag")
            if not os.path.exists(self._rouge_tag_dir): os.mkdir(self._rouge_tag_dir)
            self._rouge_category_dir = os.path.join(self._decode_dir, "category")
            if not os.path.exists(self._rouge_category_dir): os.mkdir(self._rouge_category_dir)

    def decode(self):
        t0 = time.time()
        counter = 0
        while True:
            batch = self._batcher.next_batch()
            if batch is None:
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]
            original_abstract = batch.original_abstracts[0]
            original_abstract_sents = batch.original_abstracts_sents[0]
            original_category = batch.categoryWords[0]
            original_tags = batch.tagWords[0]

            article_withunks = data.show_art_oovs(original_article, self._vocab)
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string
            category_withunks = data.show_art_oovs(original_category, self._vocab)
            best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab,
                                                 (batch.art_oovs[0] if FLAGS.pointer_gen else None))
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words)

            if FLAGS.single_pass:
                self.write_for_rouge(original_article, original_abstract_sents, decoded_words, original_tags,
                                     original_category, counter)
                counter += 1
            else:
                print_results(article_withunks, abstract_withunks, decoded_output,
                              original_category)
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                                       best_hyp.p_gens)
                t1 = time.time()
                if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()

    def write_for_rouge(self, original_article, reference_sents, decoded_words, tags, original_category, ex_index):
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]
            decoded_words = decoded_words[fst_period_idx + 1:]
            decoded_sents.append(' '.join(sent))
        original_article = [make_html_safe(w) for w in [original_article]]
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]
        tags = [make_html_safe(w) for w in tags]
        category = [make_html_safe(w) for w in [original_category]]

        article_file = os.path.join(self._rouge_art_dir, "%06d_article.txt" % ex_index)
        ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)
        tags_file = os.path.join(self._rouge_tag_dir, "%06d_tag_file.txt" % ex_index)
        category_file = os.path.join(self._rouge_category_dir, "%06d_category_file.txt" % ex_index)
        with open(article_file, "w") as f:
            for idx, sent in enumerate(original_article):
                f.write(sent) if idx == len(original_article) - 1 else f.write(sent + "\n")
        with open(ref_file, "w") as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
        with open(decoded_file, "w") as f:
            for idx, sent in enumerate(decoded_sents):
                f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
        with open(tags_file, "w") as f:
            for idx, sent in enumerate(tags):
                f.write(sent) if idx == len(tags) - 1 else f.write(sent + "\n")
        with open(category_file, "w") as f:
            for idx, sent in enumerate(category):
                f.write(sent) if idx == len(tags) - 1 else f.write(sent + "\n")

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        article_lst = article.split()
        decoded_lst = decoded_words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)


def print_results(article, abstract, decoded_output, category):
    pass
    # datas = list()
    # with open('result.txt', 'r', encoding='utf-8') as reader:
    #     try:
    #         datas = json.load(reader)
    #     except Exception as e:
    #         pass
    #     datas.append({'article': article, 'abstract': abstract, 'decoded_output': decoded_output})
    # with open('result.txt', 'w', encoding='utf-8') as writer:
    #     json.dump(datas, writer, ensure_ascii=False)


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    tf.logging.info(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    with open(results_file, "w") as f:
        f.write(log_str)


def get_decode_dir_name(ckpt_name):
    if "train" in FLAGS.data_path:
        dataset = "train"
    elif "val" in FLAGS.data_path:
        dataset = "val"
    elif "test" in FLAGS.data_path:
        dataset = "test"
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (
        dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
