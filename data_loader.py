# -*- coding: utf-8 -*-

from collections import Counter

import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def load_stackoverflow(data_path='data/stackoverflow', a=0.1):
    # 1. 获取 word -> vector 的映射
    # noinspection PyTypeChecker
    w2i = dict(line.rstrip().split('\t') for line in open(f'{data_path}/vocab_withIdx.dic'))
    i2w = {v: k for k, v in w2i.items()}
    w2v = {i2w[i.strip()]: np.fromstring(v, sep=' ')
           for i, v in zip(open(f'{data_path}/vocab_emb_Word2vec_48_index.dic'),
                           open(f'{data_path}/vocab_emb_Word2vec_48.vec'))}

    # 2.a 获取每句的词向量加权表征
    # nltk.download('punkt')
    with open(f'{data_path}/title_StackOverflow.txt') as f:
        lines = [line.strip() for line in f]  # 载入内存，后面复用
        # 统计每个word的unigram_prob
        word_counter = Counter(word for line in lines for word in nltk.word_tokenize(line))
        n_unigrams = sum(word_counter.values())
        unigram_prob = {w: c / n_unigrams for w, c in word_counter.items()}
        # 用反向unigram概率 加权求和 得到句子向量
        X = []
        for line in lines:
            vecs, wgts = [], []
            for w in nltk.word_tokenize(line):
                if w not in w2v:
                    continue
                vecs.append(w2v[w])
                wgts.append(a / (a + unigram_prob[w]))
            line_vec = np.zeros((48,)) if len(vecs) == 0 else np.average(vecs, weights=wgts, axis=0)
            X.append(line_vec)
    X = np.stack(X)
    # 2.b 减去主特征向量，得到SIF特征
    major = PCA(n_components=1).fit(X).components_  # 主特征向量 shape: (1, 48)
    X_SIF = MinMaxScaler().fit_transform(X - X @ major.T * major)  # 各句去掉自己在主方向上的投影

    # 3. 加载各句的cluster_id (已知的标注)
    y = np.loadtxt(f'{data_path}/label_StackOverflow.txt')
    return X_SIF, y


def load_search_snippet2(data_path='data/SearchSnippets/new/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_biomedical(data_path='data/Biomedical/'):
    mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'Biomedical_vocab2idx.dic', 'r') as inp_indx:
        # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'Biomedical.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_
    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return load_stackoverflow()
    elif dataset_name == 'biomedical':
        return load_biomedical()
    elif dataset_name == 'search_snippets':
        return load_search_snippet2()
    else:
        raise Exception('dataset not found...')
