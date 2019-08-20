import data
import jieba
import queue as Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf


class Example(object):
    def __init__(self, article, abstract_sentences, categoryWord, tagWord, vocab, hps):
        self.hps = hps
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words)
        self.enc_input = [vocab.word2id(w) for w in article_words]

        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        self.category_id = vocab.word2id(categoryWord)
        tagWords = jieba.lcut(tagWord)
        self.tag_ids = list()
        for i in tagWords:
            self.tag_ids.append(vocab.word2id(i))
        # categoryWords = list()
        # for word in set(article_words):
        #     if word in vocab._word_to_id.keys():
        #         categoryWords.append((word, vocab.embedding_model.similarity(categoryWord,word)))
        # categoryWords.sort(key=lambda x: x[1],reverse=True)
        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)
        if hps.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding,
                                                        stop_decoding)
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences
        self.original_categoryWord = categoryWord
        self.original_tagWords = tagWords

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:
            inp = inp[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):

    def __init__(self, example_list, hps, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self.init_encoder_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.categoryWordId = np.zeros((hps.batch_size), dtype=np.int32)
        self.tagWordLen = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.categoryWordId[i] = ex.category_id
            self.tagWordLen[i] = len(ex.tag_ids)
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
        maxTagLen = np.max(self.tagWordLen)
        self.tagWordIds = np.zeros((hps.batch_size, maxTagLen), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.tagWordIds[i, 0: self.tagWordLen[i]] = ex.tag_ids
        if hps.pointer_gen:
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            self.art_oovs = [ex.article_oovs for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, hps):
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]
        self.categoryWords = [ex.original_categoryWord for ex in example_list]
        self.tagWords = [ex.original_tagWords for ex in example_list]


class Batcher(object):
    BATCH_QUEUE_MAX = 100

    def __init__(self, data_path, vocab, hps, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
            self._finished_reading = False
        else:
            self._num_example_q_threads = 16
            self._num_batch_q_threads = 4
            self._bucketing_cache_size = 100

        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                return None

        batch = self._batch_queue.get()
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
        while True:
            try:
                (article, abstract, category, tag) = next(input_gen)
            except StopIteration:  # if there are no more examples:
                if self._single_pass:
                    self._finished_reading = True
                    break

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(
                abstract)]
            example = Example(article, abstract_sentences, category, tag, self._vocab, self._hps)
            self._example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if self._hps.mode != 'decode':
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        while True:
            e = next(example_generator)
            try:
                article_text = e.features.feature['article'].bytes_list.value[
                    0].decode()
                abstract_text = e.features.feature['abstract'].bytes_list.value[
                    0].decode()
                category = e.features.feature['category'].bytes_list.value[0].decode()
                tag = e.features.feature['tag'].bytes_list.value[0].decode()
            except ValueError:
                continue
            else:
                yield (article_text, abstract_text, category, tag)
