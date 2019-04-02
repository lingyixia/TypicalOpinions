# -*-coding:utf-8-*-
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/29 15:22
# @Author : 陈飞宇
# @File : make_datafiles.py
# @Software: PyCharm
import os
import struct
import collections
import tensorflow as tf

CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = 'finished/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join('finished/chunked', '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir('finished/chunked'):
        os.mkdir('finished/chunked')
    # Chunk the data
    for set_name in ['train', 'test', 'val']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % ('finished/chunked'))


# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

train_file = "./data/train.txt"
val_file = "./data/eval.txt"
test_file = "./data/test.txt"
finished_files_dir = "./finished"

VOCAB_SIZE = 45000


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(input_file, out_file, makevocab=False):
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        # read the input text file , make even line become article and odd line to be abstract（line number begin with 0）
        lines = read_text_file(input_file)
        for i, new_line in enumerate(lines):
            if i % 2 == 0:
                article = lines[i].split('\t')[0].encode('utf-8')
                category = lines[i].split('\t')[-2].encode('utf-8')
                tag = lines[i].split('\t')[-1].encode('utf-8')
            if i % 2 != 0:
                abstract = ("%s %s %s" % (SENTENCE_START, lines[i], SENTENCE_END)).encode('utf-8')
                # Write to tf.Example
                tf_example = tf.train.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example.features.feature['category'].bytes_list.value.extend([category])
                tf_example.features.feature['tag'].bytes_list.value.extend([tag])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = article.split(' '.encode('utf-8')) + category.split(' '.encode('utf-8'))
                    abs_tokens = abstract.split(' '.encode('utf-8'))
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START.encode('utf-8'),
                                            SENTENCE_END.encode('utf-8')]]  # remove these tags from vocab
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens]  # strip
                    tokens = [t for t in tokens if t != ""]  # remove empty
                    vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word.decode('utf-8') + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Read the text file, do a little postprocessing then write to bin files
    write_to_bin(test_file, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(val_file, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
    chunk_all()
