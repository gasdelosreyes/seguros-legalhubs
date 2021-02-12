import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys
import time

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
    """
    verdadero si la cuarta palabra de gram es direccional
    """
    longitud = len(gram)
    # print(longitud)
    for i in directionWords:
        if i in gram:
            if longitud < 5 and i == gram[-1]:
                return True
            elif 5 <= longitud and i == gram[3]:
                return True
    return False


directionWords = ['izquierda', 'derecha', 'delantera', 'trasera']
grlWords = directionWords + ['asegurado', 'colisiona', 'tercero', 'parte', 'detenido']


def reClean(row):
    row = row.split()
    i = 0
    while i < len(row):
        if row[i] not in grlWords:
            del row[i]
            i -= 1
        i += 1
    return ' '.join(row)


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


def direccionInTupla(tupla):
    for direccion in directionWords:
        if direccion in tupla:
            return True
    return False


def is_in(words, grams, pos=0):
    tuplas = []
    for tupla in grams:
        for word in words:
            if word == tupla[pos]:
                tuplas.append(tupla)
    return tuplas


def is_trivial(descripciones, n_grams=3, tercero=0):
    pentaGrams = analisis(descripciones, n_grams, True)
    lado_aseg, lado_ter = [], []
    for i in range(len(pentaGrams) - 1):
        gram = is_in(['asegurado'], pentaGrams[str(i)])
        # gram += is_in(['asegurado'], pentaGrams[str(i)], pos=0)
        if gram:
            lado_aseg.append([gram, i])
        gram = is_in(['su', 'tercero'], pentaGrams[str(i)])
        if gram:
            lado_ter.append([gram, i])
    print('se encontraron', len(lado_ter), 'descripciones del tercero y ', len(lado_aseg), 'del asegurado')
    if tercero:
        return lado_ter
    return lado_aseg


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv('~/Documentos/LegalHub/dataset/casos/auto.csv')
    # df['descripcion'] = pd.Series(list(map(reClean, df['descripcion'])))
    lado_aseg = is_trivial(df['descripcion'], 3)
    # tokenizedCorpus = [i[0] for i in lado_aseg]
    lado_aseg += is_trivial(df['descripcion'], 4)
    # tokenizedCorpus += [i[0] for i in lado_aseg]
    lado_aseg += is_trivial(df['descripcion'], 5)
    tokenizedCorpus = [i[0] for i in lado_aseg]
    file_name = 'lado_aseg'
    # df_model = get_concordance(df['descripcion'])
    # tokenizedCorpus = [list(nltk.ngrams(word_tokenize(i), 2)) for i in df['descripcion']]
    # tokenizedCorpus += [list(nltk.ngrams(word_tokenize(i), 3)) for i in df['descripcion']]
    # tokenizedCorpus += [list(nltk.ngrams(word_tokenize(i), 4)) for i in df['descripcion']]
    # tokenizedCorpus = [list(nltk.ngrams(word_tokenize(i), 5)) for i in df['descripcion']]
    # tokenizedCorpus += [nltk.tokenize.word_tokenize(i) for i in df_model]
    print(len(tokenizedCorpus))
    bowCorpus = [TaggedDocument([' '.join(tup) for tup in words], [idx_tag]) for idx_tag, words in
                 enumerate(tokenizedCorpus)]

    fixSeed = 774965317
    np.random.seed(fixSeed)

    model = Doc2Vec(bowCorpus, vector_size=300, seed=fixSeed, dm=1, epochs=1000)
    modelVectors = model.docvecs.vectors_docs

    pca = PCA(n_components=2, svd_solver='full', whiten=True)
    pcaVectors = pca.fit_transform(modelVectors)

    kmeans = KMeans(n_clusters=8, random_state=fixSeed)
    kmeans.fit(pcaVectors)

    pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])
    pca_df['idx_descripcion'] = pd.Series([i[1] for i in lado_aseg])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    print('silhouette_score = ', silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean'))
    ax.grid()
    ax.scatter(pca_df.x, pca_df.y, s=15)
    plt.savefig(file_name + '.png')
    pca_df['descripcion'] = pd.Series(tokenizedCorpus)
    pca_df['cluster'] = pd.Series(kmeans.labels_)
    sns.lmplot('x', 'y', data=pca_df, hue='cluster', fit_reg=False)
    plt.savefig(file_name + '_sns.png')
    pca_df['responsabilidad'] = df['responsabilidad']
    pca_df.to_csv(file_name + '.csv', index=False)

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
    t = round(time.time() - start)
    print('Tiempo transcurrido :', t // 60, 'min:', t % 60, 'seg')
