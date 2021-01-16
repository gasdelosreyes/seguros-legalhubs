import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm as tq
import random
import os

from functions import *
from cleaner import *
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def getSeedScore(bowCorpus, vector_size=100,seed=None):
    randomSeed = random.randint(1, 2**32 - 1)
    kmeans = KMeans(n_clusters=8)
    model = Doc2Vec(bowCorpus, vector_size=vector_size, seed=randomSeed)
    pca = PCA(n_components=2, svd_solver='full', whiten=True)
    pca = pca.fit(model.docvecs.vectors_docs)
    kmeans.fit(model.docvecs.vectors_docs)
    return (silhouette_score(model.docvecs.vectors_docs, kmeans.labels_, metric='euclidean'), randomSeed)

def maxScore(bowCorpus,tokenizedCorpus,training=100):
    score = []       
    for k in tq(range(training)):
        score.append(getSeedScore(bowCorpus,tokenizedCorpus))
    return max(score)

def checkWrongContext(string):
    parts = ['parte delantera izquierda','parte trasera izquierda','parte delantera derecha','parte delantera izquierda',
            'parte trasera derecha','parte trasera izquierda','parte delantera','parte trasera','parte derecha','parte izquierda']
    for value in parts:
        if(re.search(value, string)):
            return True
    return False
    
def get_concordance(vector,window=2,key='parte'):
    array = []
    deleteArray = []
    for value in vector:
        index = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(value))
        concordance = index.find_concordance(key)
        for i in range(len(concordance)):
            string = ' '.join(concordance[i][0][-(window):]) + ' ' + concordance[i][1] + ' ' + ' '.join(concordance[i][2][:(window)])
            if(checkWrongContext(string)):
                array.append(string)
            else:
                deleteArray.append(string)
    return array
    
def getNGrams(corpus,number):
    fstring = ''
    for row in corpus:
        fstring += row + ' '
    tokens = word_tokenize(fstring)
    array = []
    n_series = pd.Series(nltk.ngrams(tokens, number))
    n_values = n_series.value_counts().index
    n_counts = n_series.value_counts().values
    for ngram in n_values:
        string = ' '.join(i for i in ngram)
        if(checkWrongContext(string)):
            array.append(string)
    print(f'Cantidad TOTAL de NGRAMAS [{len(n_values)}]')
    print(f'Reduccion de NGRAMAS [{len(array)}]')
    return array
df = pd.read_csv('../dataset/casos/auto.csv')

# df_model = get_concordance(df['descripcion'])
# # print(pd.Series(df_model).value_counts())

def searchNGramIndex(corpus,ngrams):
    array = []
    for index, row in enumerate(corpus):
        for ngram in ngrams:
            if(re.search(ngram,row)):
                array.append(TaggedDocument(nltk.tokenize.word_tokenize(ngram),[index]))
    return array


ngramTagged = searchNGramIndex(df['descripcion'],getNGrams(df['descripcion'],3))

tokenizedCorpus = [nltk.tokenize.word_tokenize(i) for i in df['descripcion']].copy()
# model = Doc2Vec(ngramTagged, vector_size=100, window=2, min_count=2, workers=7)
d2v_model = Doc2Vec(ngramTagged, vector_size = 100, window = 10, min_count = 10, workers=7, dm = 1,alpha=0.025, min_alpha=0.001)

d2v_model.train(ngramTagged, total_examples= d2v_model.corpus_count, epochs=20,start_alpha=0.002, end_alpha=0.016)

kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)

X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)
plt.figure()
label1 = ['#FFFF00', '#008000', '#0000FF', '#800080']
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()