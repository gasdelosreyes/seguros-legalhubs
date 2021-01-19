import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys

from tqdm import tqdm as tq

import spacy

from cleaner import *
from functions import *
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

nlp = spacy.load("es_core_news_sm")


def getSeedScore(bowCorpus, tokenizedContexts, vector_size=300, seed=None):
    randomSeed = random.randint(1, 2**32 - 1)
    kmeans = KMeans(n_clusters=8)
    model = Doc2Vec(bowCorpus, vector_size=vector_size, seed=randomSeed)
    modelVectors = model.docvecs.vectors_docs
    pca = PCA(n_components=50, svd_solver='full', whiten=True)
    # pca = pca.fit(modelVectors)
    pcaVectors = pca.fit_transform(modelVectors)
    kmeans.fit(pcaVectors)

    return (silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean'), randomSeed)


def maxScore(bowCorpus, tokenizedCorpus, training=100, vector_size=300):
    score = []
    k = 0
    for k in tq(range(100)):
        score.append(getSeedScore(bowCorpus, tokenizedCorpus, vector_size=vector_size))
    return max(score)


def checkWrongContext(string):
    parts = ['parte delantera izquierda', 'parte trasera izquierda', 'parte delantera derecha', 'parte delantera izquierda',
             'parte trasera derecha', 'parte trasera izquierda', 'parte delantera', 'parte trasera', 'parte derecha', 'parte izquierda']
    for value in parts:
        if(re.search(value, string)):
            return True
    return False


def get_concordance(vector, window=2, key='parte'):
    array = []
    # deleteArray = []
    for value in vector:
        index = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(value))
        concordance = index.find_concordance(key)
        for i in range(len(concordance)):
            string = ' '.join(concordance[i][0][-(window):]) + ' ' + concordance[i][1] + ' ' + ' '.join(concordance[i][2][:(window)])
            array.append(string)
    #         if(checkWrongContext(string)):
    #         else:
    #             deleteArray.append(string)
    # # print(deleteArray)
    return array


def get_frequency(word, words):
    suma = 0
    for w in words:
        if w == word:
            suma += 1
    return suma


def get_trainTest(bowCorpus, trainFrac=0.2):
    test = []
    for i in range(round(len(bowCorpus) * trainFrac)):
        ran = random.randint(0, round(len(bowCorpus) * trainFrac))
        if bowCorpus[ran] not in test:
            test.append(bowCorpus[ran])
    train = [ctx for ctx in bowCorpus if ctx not in test]
    return train, test


def partir2parte(words):
    for i in range(len(words)):
        if words[i] == 'partir':
            words[i] = 'parte'
    return words


def lemmatizer(toLem):
    lem = nlp(toLem)
    return partir2parte([word.lemma_ for word in lem])


def list2str(x): return ' '.join(x)


def filterGrams(gram):
    for i in directionWords:
        if i in gram:
            return True
    return False


directionWords = ['izquierda', 'derecha', 'delantera', 'trasera']

"""
(mi, parte, delantera)              500
(colisiona, parte, trasera)         310
(parte, delantera, derecha)         239
(colisiona, parte, delantera)       199
(parte, delantera, izquierda)       187
(su, parte, delantera)              177
(parte, trasera, izquierda)         174
(parte, trasera, derecha)           172
(su, parte, trasera)                156
(parte, delantera, parte)           153
(parte, trasera, vehiculo)          120
(parte, trasera, parte)              96
(trasera, vehiculo, tercero)         95
(delantera, parte, trasera)          94
(parte, delantera, vehiculo)         92
(colisiona, parte, izquierda)        90
(marcha, trasera, colisiona)         87
(colisiona, parte, derecha)          85
(vehiculo, circulaba, delantera)     68
(marcha, trasera, para)              66
            #########
(colisiona, su, parte, delantera)           96
(parte, delantera, parte, trasera)          72
(colisiona, su, parte, trasera)             71
(parte, trasera, vehiculo, tercero)         63
(colisiona, parte, delantera, derecha)      57
(su, parte, delantera, parte)               55
(tercero, colisiona, parte, trasera)        50
(mi, parte, delantera, parte)               47
(colisiona, parte, trasera, vehiculo)       46
(colisiona, parte, delantera, izquierda)    45
(colisiona, parte, trasera, izquierda)      42
(colisiona, mi, parte, delantera)           42
(mi, parte, delantera, auto)                41
(parte, delantera, vehiculo, asegurado)     39
(colisiona, parte, trasera, derecha)        39
(tercero, colisiona, parte, delantera)      38
(colisiona, parte, trasera, parte)          35
(su, parte, delantera, izquierda)           34
(parte, trasera, parte, delantera)          34
(parte, delantera, derecha, vehiculo)       32
            #########
(colisiona, su, parte, delantera, parte)              42
(su, parte, delantera, parte, trasera)                33
(vehiculo, tercero, colisiona, parte, delantera)      25
(vehiculo, tercero, colisiona, parte, trasera)        25
(tercero, colisiona, su, parte, delantera)            22
(mi, parte, delantera, parte, trasera)                20
(parte, delantera, parte, trasera, vehiculo)          19
(colisiona, parte, trasera, vehiculo, tercero)        19
(parte, delantera, vehiculo, asegurado, parte)        17
(parte, delantera, parte, trasera, tercero)           17
(colisiona, su, parte, delantera, derecha)            17
(colisiona, su, parte, delantera, izquierda)          17
(asegurado, parte, trasera, vehiculo, tercero)        16
(parte, delantera, derecha, vehiculo, asegurado)      16
(delantera, vehiculo, asegurado, parte, trasera)      16
(mi, parte, delantera, su, parte)                     15
(colisiona, parte, delantera, vehiculo, asegurado)    15
(colisiona, parte, delantera, parte, trasera)         15
(delantera, parte, trasera, vehiculo, tercero)        15
(trasera, vehiculo, tercero, no, hubo)                15
"""


def analisis(descripciones, n_grams, dic=False):
    grams = []
    dic_grams = {}
    for index, descripcion in enumerate(descripciones):
        aux_gram = list(nltk.ngrams(word_tokenize(descripcion), n_grams))
        aux_gram = [gram for gram in aux_gram if filterGrams(gram)]
        dic_grams[str(index)] = aux_gram
        grams += aux_gram

    if dic:
        return dic_grams
    return grams


def is_in(words, grams):
    tuplas = []
    for tupla in grams:
        for word in words:
            for direccion in directionWords:
                if word == tupla[0] and direccion in tupla:
                    tuplas.append(tupla)
    return tuplas


def is_trivial(descripciones):

    pentaGrams = analisis(descripciones, 6, True)
    lado_aseg, lado_ter = [], []
    for i in range(len(pentaGrams) - 1):
        gram = is_in(['parte'], pentaGrams[str(i)])
        if gram:
            lado_aseg.append([gram, i])
        gram = is_in(['su', 'tercero'], pentaGrams[str(i)])
        if gram:
            lado_ter.append([gram, i])
    print('se encontraron', len(lado_ter), 'descripciones del tercero y ', len(lado_aseg), 'del asegurado')
    return lado_aseg, lado_ter


if __name__ == "__main__":
    df = pd.read_csv('~/Documentos/LegalHub/dataset/casos/auto.csv')
    lado_aseg, lado_ter = is_trivial(df['descripcion'])
    tokenizedCorpus = [i[0] for i in lado_aseg]
    # df_model = get_concordance(df['descripcion'])
    # tokenizedCorpus = [list(nltk.ngrams(word_tokenize(i), 2)) for i in df['descripcion']]
    # tokenizedCorpus += [list(nltk.ngrams(word_tokenize(i), 3)) for i in df['descripcion']]
    # tokenizedCorpus += [list(nltk.ngrams(word_tokenize(i), 4)) for i in df['descripcion']]
    # tokenizedCorpus += [list(nltk.ngrams(word_tokenize(i), 5)) for i in df['descripcion']]
    # tokenizedCorpus += [nltk.tokenize.word_tokenize(i) for i in df_model]
    print(len(tokenizedCorpus))
    bowCorpus = [TaggedDocument([' '.join(tup) for tup in words], [idx_tag]) for idx_tag, words in
                 enumerate(tokenizedCorpus)]
    fixSeed = 774965317
    np.random.seed(fixSeed)
    model = Doc2Vec(bowCorpus, vector_size=300, seed=fixSeed, dm=1, epochs=750)
    modelVectors = model.docvecs.vectors_docs
    pca = PCA(n_components=2, svd_solver='full', whiten=True)
    pcaVectors = pca.fit_transform(modelVectors)

    kmeans = KMeans(n_clusters=5, random_state=fixSeed)
    kmeans.fit(pcaVectors)
    pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    print('dimension reduced= ', silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean'))
    # k = 0
    # rans = []
    # while k < 5:
    #     ran = random.randint(0, len(pcaVectors) - 1)
    #     if ran not in rans:
    #         ax.annotate(' '.join(bowCorpus[ran][0]), (pcaVectors[ran][0], pcaVectors[ran][1]),
    #                     fontsize=12, alpha=0.9)
    #         rans.append(ran)
    #         k += 1
    # ax1 = fig.add_subplot(122)

    # xedges = np.linspace(-2, 2, 10)
    # yedges = np.linspace(-2, 2, 10)
    # x = pcaVectors['x'].to_array()
    # y = pcaVectors['y'].to_array()
    # hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    # xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0] - 1)
    # yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1] - 1)
    # c = hist[xidx, yidx]
    # ax1.scatter(x, y, c=c)

    ax.grid()
    # plt.show()
    ax.scatter(pca_df.x, pca_df.y, s=15)
    plt.savefig('test_plot.png')
    pca_df['descripcion'] = pd.Series(tokenizedCorpus)
    pca_df['cluster'] = pd.Series(kmeans.labels_)
    sns.lmplot('x', 'y', data=pca_df, hue='cluster', fit_reg=False)
    plt.savefig('test_cluster_sns.png')
    pca_df.to_csv('parte_test.csv')
    # direcciones = ['delantera', 'trasera', 'izquierda', 'derecha']
    # for direccion in direcciones:
    #     for j in range(8):
    #         aux = 0
    #         for row in pca_df['descripcion'][pca_df['cluster'] == j]:
    #             for i in row:
    #                 if direccion in i:
    #                     aux += 1
    #                     break
    #         prob = 1.0 * aux / len(pca_df['descripcion'][pca_df['cluster'] == 0]) * 100
    #         print('probabilidad de que el cluster ', j, ' sea de la direccion ', direccion, ': ', prob)
    #         print()
    """plt.clf()
    tokenizedCorpus = [i for i in pca_df['descripcion'][pca_df['cluster'] == 0]]
    print(len(tokenizedCorpus))
    bowCorpus = [TaggedDocument([' '.join(tup) for tup in words], [idx_tag]) for idx_tag, words in
                 enumerate(tokenizedCorpus)]
    model = Doc2Vec(bowCorpus, vector_size=300, seed=fixSeed, min_count=1, dm=1, epochs=1000)
    modelVectors = model.docvecs.vectors_docs
    pca = PCA(n_components=2, svd_solver='full', whiten=True)
    pcaVectors = pca.fit_transform(modelVectors)

    kmeans = KMeans(n_clusters=2, random_state=fixSeed)
    kmeans.fit(pcaVectors)
    pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    print('dimension reduced= ', silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean'))
    ax.grid()
    # plt.show()
    ax.scatter(pca_df.x, pca_df.y, s=15)
    plt.savefig('test_plot_cluster0.png')
    pca_df['descripcion'] = pd.Series(tokenizedCorpus)
    pca_df['cluster'] = pd.Series(kmeans.labels_)
    sns.lmplot('x', 'y', data=pca_df, hue='cluster', fit_reg=False)
    plt.savefig('test_recluster_sns.png')
    pca_df.to_csv('parte_test.csv')"""
