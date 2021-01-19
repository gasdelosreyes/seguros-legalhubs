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
    vector = [(n_values[i],n_counts[i]) for i in range(len(n_values))]
    for value in vector:
        if(number == 2):
            if(int(value[1]) >= 200):
                string = ' '.join(i for i in value[0])
                if(checkWrongContext(string)):
                    array.append(string)
        if(number == 3):
            if(int(value[1]) >= 100):
                string = ' '.join(i for i in value[0])
                if(checkWrongContext(string)):
                    array.append(string)
        if(number == 4):
            if(int(value[1]) >= 40):
                string = ' '.join(i for i in value[0])
                if(checkWrongContext(string)):
                    array.append(string)
        if(number == 5):
            if(int(value[1]) >= 20):
                string = ' '.join(i for i in value[0])
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
        string = []
        for ngram in ngrams:
            if(re.search(ngram,row)):
                string.append(ngram)
        if len(string) != 0:
            array.append(TaggedDocument(string,[index]))
    return array
"""
pca = PCA(n_components=2, svd_solver='full', whiten=True)
pcaVectors = pca.fit_transform(modelVectors)

kmeans = KMeans(n_clusters=8, random_state=fixSeed)
kmeans.fit(pcaVectors)
pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])
ax.scatter(pca_df.x, pca_df.y, s=15)

tokenizedCorpus = [list(nltk.ngrams(word_tokenize(i), 3)) for i in df['descripcion']]
"""
ngramTagged = searchNGramIndex(df['descripcion'],getNGrams(df['descripcion'],2))
ngramTagged += searchNGramIndex(df['descripcion'],getNGrams(df['descripcion'],3))
ngramTagged += searchNGramIndex(df['descripcion'],getNGrams(df['descripcion'],4))
ngramTagged += searchNGramIndex(df['descripcion'],getNGrams(df['descripcion'],5))

#Forma del taggedDocument [['ngrama','ngrama',etc],['indice']]

# tokenizedCorpus = [nltk.tokenize.word_tokenize(i) for i in df['descripcion']].copy()

d2v_model = Doc2Vec(ngramTagged, vector_size = 300, window = 10, min_count = 10, workers=7, dm = 1,alpha=0.05, min_alpha=0.001)

d2v_model.train(ngramTagged, total_examples= d2v_model.corpus_count, epochs=1000,start_alpha=0.002, end_alpha=0.016)

kmeans_model = KMeans(n_clusters=8, init='k-means++', max_iter=500)

## d2v_model.docvecs.vectors_docs and d2v_model.docvecs.doctag_syn0 are the same)
## Fitting with all vectors

pca = PCA(n_components=2,svd_solver='full', whiten=True).fit(d2v_model.docvecs.vectors_docs)

datapoint = pca.transform(d2v_model.docvecs.vectors_docs) #Vectores de 2 Dimensiones
X = kmeans_model.fit(datapoint)
pca_df = pd.DataFrame(data=datapoint, columns=['x', 'y'])
labels=kmeans_model.labels_.tolist()
# l = kmeans_model.fit_predict(datapoint)

import seaborn as sns

pca_df['descripcion'] = pd.Series([i[0] for i in ngramTagged])
pca_df['cluster'] = pd.Series(kmeans_model.labels_)

sns.lmplot('x', 'y', data=pca_df, hue='cluster', fit_reg=False)

plt.savefig('test_cluster_sns.png')

pca_df.to_csv('../dataset/pca_cluster.csv', index=False, header=True)