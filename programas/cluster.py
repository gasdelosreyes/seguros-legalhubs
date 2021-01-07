import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm as tq
import random

from functions import *
from cleaner import *
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def getSeedScore(bowCorpus, tokenizedContexts, vector_size=300,seed=None):
    randomSeed = random.randint(1, 2**32 - 1)
    # np.random.seed(randomSeed)
    kmeans = KMeans(n_clusters=8)
    model = Doc2Vec(bowCorpus, vector_size=vector_size, seed=randomSeed)
    pca = PCA(n_components=2, svd_solver='full', whiten=True)

    to_fit = []
    # for value in tokenizedContexts:
    #     # try:
    #     #     if len(model[value]) != vector_size:
    #     #         for j in model[value]:
    #     #             to_fit.append(j)
    #     #     else:
    #     #         to_fit.append(model[value])
    #     # except:
    #     #     pass
    #     for vector in model.docvecs.vectors_docs:
    #         to_fit.append(vector)

    pca = pca.fit(model.docvecs.vectors_docs)
    # mod_vec = [GeoCenter(pca.transform(i)) for i in model.docvecs.vectors_docs]
    for i in tokenizedContexts:
        try:
            mod_vec.append(GeoCenter(pca.transform(model[i])))
        except:
            pass

    kmeans.fit(mod_vec)
    return (silhouette_score(mod_vec, kmeans.labels_, metric='euclidean'), randomSeed)

def maxScore(bowCorpus,tokenizedCorpus,training=100):
    score = []       
    for k in tq(range(training)):
        score.append(getSeedScore(bowCorpus,tokenizedCorpus))
    return max(score)

def get_concordance(vector,window=2,key='parte'):
    array = []
    for value in vector:
        index = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(value))
        concordance = index.find_concordance(key)
        for i in range(len(concordance)):
            array.append(' '.join(concordance[i][0][-(window):]) + ' ' + concordance[i][1] + ' ' + ' '.join(concordance[i][2][:(window)]))
    return array
    
df = pd.read_csv('../dataset/casos/auto.csv')
df_model = get_concordance(df['descripcion'])
print(pd.Series(df_model).value_counts())

tokenizedCorpus = [nltk.tokenize.word_tokenize(i) for i in df_model].copy()
bowCorpus = [TaggedDocument(words, [idx_tag]) for idx_tag, words in enumerate(tokenizedCorpus)]

seed = maxScore(bowCorpus,tokenizedCorpus)[1]

with open('seed.txt','w') as f:
    f.write(str(seed))


np.random.seed(seed)
model = Doc2Vec(bowCorpus, vector_size=300, seed=seed) 
pca = PCA(n_components=2, svd_solver='full', whiten=True) 

to_fit = []
for i in tokenizedCorpus:
    try:
        if len(model[i]) != 300:
            for j in model[i]:
                to_fit.append(j)
        else:
            to_fit.append(model[i])
    except:
        pass
        
pca = pca.fit(to_fit)
mod_vec = []
seguidor = {'contexto': [], 'vector': []}
for i in tokenizedCorpus:
    try:
        mod_vec.append(GeoCenter(pca.transform(model[i])))
        seguidor['contexto'].append(i)
        seguidor['vector'].append(GeoCenter(pca.transform(model[i])))
    except:
        pass
kmeans = KMeans(n_clusters=8)
kmeans.fit(mod_vec)
pca_df = pd.DataFrame(data=mod_vec, columns=['x', 'y'])
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
print(silhouette_score(mod_vec, kmeans.labels_, metric='euclidean'))
#ax.title(silhouette_score(mod_vec, kmeans.labels_, metric='euclidean'))
ax.grid()
ax.scatter(pca_df.x, pca_df.y)
plt.savefig('test_plot.png')

