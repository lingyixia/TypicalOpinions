#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/29 15:57
# @Author : 陈飞宇
# @File : data_handler.py
# @Software: PyCharm
MONGODB_HOST = '172.20.206.28'
MONGODB_PORT = 27017
MONGODB_DBNAME = 'koubei'
MONGODB_AUTOHOME = 'autohome'
MONGODB_AUTOHOMETAGED = 'autohometagedentire'
import pymongo, jieba, re, random, os, collections
from gensim.models.word2vec import Word2Vec


def word2vec_train(sentences):
    model = Word2Vec(size=128, min_count=1)
    model.build_vocab(sentences)  # input: list
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.most_similar()
    model.save('data/Word2vec_model.pkl')
    print('词向量训练完毕')


def save(fileName, data):
    if not os.path.exists('data'): os.makedirs('data')
    articleElimTokenList = data['noTagedContents'].replace('\n', ' ').replace('\t', ' ').strip().split()
    article = ' '.join(articleElimTokenList)  # 多空格合并

    abstractElimTokenList = data['tagedContents'].replace('\n', ' ').replace('\t', ' ').strip().split()
    abstract = ' '.join(abstractElimTokenList)
    with open('{fileName}.txt'.format(fileName=fileName), 'a', encoding='utf-8') as writer:
        articleCut = ' '.join(jieba.lcut(article)) + '\t' + data['category'] + '\t' + re.sub('\(\d+\)', '', data['tag'])
        abstractCut = ' '.join(jieba.lcut(abstract))
        writer.write(articleCut + '\n')
        writer.write(abstractCut + '\n')
        articleList = articleCut.split()
        articleList.insert(0, '<GO>')
        articleList.append('<EOS>')
    return articleList


def judge():
    sentences = list()
    host = MONGODB_HOST
    port = MONGODB_PORT
    dbName = MONGODB_DBNAME
    client = pymongo.MongoClient(host=host, port=port)
    tdb = client[dbName]
    post = tdb[MONGODB_AUTOHOMETAGED]
    datas = post.find({}, {"contents": 1, "category": 1, "tag": 1, "_id": 0})
    shuffleDatas = list()
    tempDic = dict()
    tags = list()
    for data in datas:
        if len(data['contents']['tagedContents']) == 0:
            continue
        tempDic[(re.sub('\(\d+\)', '', data['tag']), data['contents']['commentsId'])] = (
            data['contents']['tagedContents'][0], data['contents']['noTagedContents'], data['category'],
            re.sub('\(\d+\)', '', data['tag']), data['contents']['commentsId'])
    for data in tempDic.values():
        item = {'tagedContents': data[0],
                'noTagedContents': data[1],
                'category': data[2],
                'tag': data[3]}
        shuffleDatas.append(item)
        tags.append(data[3])
    #counter = collections.Counter(tags)
    #tags = counter.most_common(100)
    #tags = [i[0] for i in tags]
    random.shuffle(shuffleDatas)
    trainIndex = int(len(shuffleDatas) * 0.6)
    evalNum = int(trainIndex + len(shuffleDatas) * 0.2)
    num = 0
    for index, data in enumerate(shuffleDatas):
        if data['tag'] not in tags:
            continue
        if index <= trainIndex:
            sentences.append(save('data/train', data))
        elif index <= evalNum:
            sentences.append(save('data/eval', data))
        else:
            sentences.append(save('data/test', data))
        num += 1
    print('一共', num)
    datas.close()
    client.close()
    print('数据保存完毕,开始训练词向量')
    return sentences


if __name__ == '__main__':
    sentences = judge()
    # word2vec_train(sentences)
