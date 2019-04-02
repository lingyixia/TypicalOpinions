#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/12/11 15:52
# @Author : 陈飞宇
# @File : dataAnylise.py
# @Software: PyCharm
import os, json


def longestCommon(strs, reverse=False):
    """
    :type strs: List[str]
    :rtype: str
    """
    # 判断是否为空
    if not strs:
        return ''
    if reverse:
        for i, s in enumerate(strs):
            s = list(s)
            s.reverse()
            s = "".join(s)
            strs[i] = s
    # 在使用max和min的时候已经把字符串比较了一遍
    # 当前列表的字符串中，每个字符串从第一个字母往后比较直至出现ASCII码 最小的字符串
    s1 = min(strs)
    # 当前列表的字符串中，每个字符串从第一个字母往后比较直至出现ASCII码 最大的字符串
    s2 = max(strs)
    # 使用枚举变量s1字符串的每个字母和下标
    for i, c in enumerate(s1):
        # 比较是否相同的字符串，不相同则使用下标截取字符串
        if c != s2[i]:
            return s1[:i]
    return s1


def readToDic(dir):
    print('相应参数', dir.split('/')[-1])
    articlesDir = os.path.join(dir, 'article')
    decodesDir = os.path.join(dir, 'decoded')
    referencesDir = os.path.join(dir, 'reference')
    categorysDir = os.path.join(dir, 'category')
    tagsDir = os.path.join(dir, 'tag')

    articles = os.listdir(articlesDir)
    articles.sort()
    decodes = os.listdir(decodesDir)
    decodes.sort()
    references = os.listdir(referencesDir)
    references.sort()
    categorys = os.listdir(categorysDir)
    categorys.sort()
    tags = os.listdir(tagsDir)
    tags.sort()

    listDecodes = list()
    listReferences = list()
    listArticle = list()
    listCategory = list()
    listTag = list()
    for a in articles:
        with open(os.path.join(articlesDir, a), 'r', encoding='utf-8') as reader:
            listArticle.append(reader.readline())
    for d in decodes:
        with open(os.path.join(decodesDir, d), 'r', encoding='utf-8') as reader:
            listDecodes.append(reader.readline())
    for r in references:
        with open(os.path.join(referencesDir, r), 'r', encoding='utf-8') as reader:
            listReferences.append(reader.readline())
    for c in categorys:
        with open(os.path.join(categorysDir, c), 'r', encoding='utf-8') as reader:
            listCategory.append(reader.readline())
    for t in tags:
        with open(os.path.join(tagsDir, t), 'r', encoding='utf-8') as reader:
            temp = list()
            for l in reader.readlines():
                temp.append(l)
            listTag.append(''.join(temp))
    totalDifferentList = list()
    totalSameList = list()
    decodeInReferenceList = list()
    referenceInDecodeList = list()
    commonPrefixList = list()
    commonTailList = list()
    allList = list()
    for index in range(0, len(listDecodes)):
        allList.append(
            {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
             'catetory': listCategory[index], 'tag': listTag[index]})
        if listDecodes[index] == listReferences[index]:
            totalSameList.append(
                {'decode': listDecodes[index], 'article': listArticle[index], 'catetory': listCategory[index],
                 'tag': listTag[index]})
        elif listDecodes[index] in listReferences[index]:
            decodeInReferenceList.append(
                {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
                 'catetory': listCategory[index], 'tag': listTag[index]})
        elif listReferences[index] in listDecodes[index]:
            referenceInDecodeList.append(
                {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
                 'catetory': listCategory[index], 'tag': listTag[index]})
        elif longestCommon([listReferences[index], listDecodes[index]]) != '':
            commonPrefixList.append(
                {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
                 'catetory': listCategory[index], 'tag': listTag[index]})
        elif longestCommon([listReferences[index], listDecodes[index]], reverse=True) != '':
            commonTailList.append(
                {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
                 'catetory': listCategory[index], 'tag': listTag[index]})
        else:
            totalDifferentList.append(
                {'reference': listReferences[index], 'decode': listDecodes[index], 'article': listArticle[index],
                 'catetory': listCategory[index], 'tag': listTag[index]})
    print('测试集一共', len(listDecodes))
    print('解码完全相同', len(totalSameList), len(totalSameList) / len(listDecodes))
    print('decode in reference:', len(decodeInReferenceList), len(decodeInReferenceList) / len(listDecodes))
    print('referencce in decode', len(referenceInDecodeList), len(referenceInDecodeList) / len(listDecodes))
    print("公共前缀:", len(commonPrefixList), len(commonPrefixList) / len(listDecodes))
    print("公共后缀:", len(commonTailList), len(commonTailList) / len(listDecodes))
    print('合计', (len(totalSameList) + len(decodeInReferenceList) + len(referenceInDecodeList)) / len(listDecodes))

    if not os.path.exists('analyse'): os.mkdir('analyse')
    with open('analyse/all.json', 'w', encoding='utf-8') as writer:
        json.dump(allList, writer, ensure_ascii=False)
    with open('analyse/referenceInDecode.json', 'w', encoding='utf-8') as writer:
        json.dump(referenceInDecodeList, writer, ensure_ascii=False)
    with open('analyse/DecodeInreference.json', 'w', encoding='utf-8') as writer:
        json.dump(decodeInReferenceList, writer, ensure_ascii=False)
    with open('analyse/totalDifferent.json', 'w', encoding='utf-8') as writer:
        json.dump(totalDifferentList, writer, ensure_ascii=False)
    with open('analyse/commonPrefix.json', 'w', encoding='utf-8') as writer:
        json.dump(commonPrefixList, writer, ensure_ascii=False)
    with open('analyse/commonTail.json', 'w', encoding='utf-8') as writer:
        json.dump(commonTailList, writer, ensure_ascii=False)


if __name__ == '__main__':
    # readToDic('log/myChineseTopic/decode_test_150maxenc_4beam_5mindec_35maxdec_ckpt-30565')
    # print(100 * "*")
    # readToDic('log/myChineseTopic2/decode_test_150maxenc_4beam_5mindec_35maxdec_ckpt-28704')
    # print(100 * "*")
    # readToDic('log/myChineseTopic5/decode_test_150maxenc_4beam_5mindec_35maxdec_ckpt-28729')
    # print(100 * "*")
    # readToDic('log/myChineseTopic10/decode_test_150maxenc_4beam_5mindec_35maxdec_ckpt-28085')
    # print(100 * "*")
    # readToDic('log/myChineseTopicNew/decode_test_150maxenc_4beam_2mindec_35maxdec_ckpt-254071')
    # print(100 * "*")
    readToDic('log/myChineseTopicMultiTop100/decode_test_150maxenc_5beam_2mindec_35maxdec_ckpt-241688')
