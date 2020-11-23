
# %%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot
from scipy.sparse import data
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics
import matplotlib.pyplot as plt
import seaborn as sns
from os import sep
from re import split
import spacy
import es_core_news_sm
import pandas as pd
import numpy as np
import gensim
path = '../dataset/'
# %%
bici = pd.read_csv('../dataset/casos/bici_clean.csv', sep=';')
moto = pd.read_csv('../dataset/casos/moto_clean.csv', sep=';')
auto = pd.read_csv('../dataset/casos/auto_clean.csv', sep=';')
peaton = pd.read_csv('../dataset/casos/peaton_clean.csv', sep=';')
dataset = pd.DataFrame()
dataset = pd.concat([bici, moto, auto, peaton])
dataset.pop('Unnamed: 0')
dataset = dataset.rename(columns={'Idenx original': 'Index original'})
dataset = pd.concat([dataset, pd.get_dummies(
    dataset['tipo_de_accidente'])], axis=1)
dataset.sort_index()

# %%
col = 'descripcion_del_hecho - Final'
corpus = dataset[col]
# %%
# Aca vamos a crear una lista de unigramas, osea palabras.
lst_corpus = []
for i in corpus:
    lst_words = str(i).split()
    lst_grams = [' '.join(lst_words[i:i+1]) for i in range(len(lst_words))]
    lst_corpus.append(lst_grams)
# %%
# Aca vamos a crear una lista de bigramas.
bigrams_detector = gensim.models.Phrases(
    lst_corpus, delimiter=" ".encode(), min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
# %%
# trigrama
trigrams_detector = gensim.models.phrases.Phrases(
    bigrams_detector[lst_corpus], delimiter=" ".encode(), min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

# %%
# Para el fitting debemos especificar la longitud del vector objetivo, la distancia maxima entre la palabra actual
# y la palabra deredicha y el algoritmo de entrenamiento en este caso es skip-gram.

nlp = gensim.models.word2vec.Word2Vec(
    lst_corpus, size=300, window=8, min_count=1, sg=1, iter=30)

# %%
# ahora debemos aplicar un algoritmo de reducción de la dimensión para poder observar las palabras.

# %%
word = 'colisiona'

tot_words = [word] + [tupla[0] for tupla in nlp.wv.most_similar(word, topn=20)]
X = nlp[tot_words]

# %%
# pca to reduce dimensionality from 300 to 3
pca = manifold.TSNE(perplexity=10, n_components=3, init='pca')
X = pca.fit_transform(X)  # create dtf
dtf_ = pd.DataFrame(X, index=tot_words, columns=["x", "y", "z"])
dtf_["input"] = 0
dtf_.loc[:1, 'input'] = 1
# plot 3d

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(dtf_[dtf_["input"] == 0]['x'], dtf_[dtf_["input"]
                                               == 0]['y'], dtf_[dtf_["input"] == 0]['z'], 'b')
ax.scatter(dtf_[dtf_["input"] == 1]['x'], dtf_[dtf_["input"]
                                               == 1]['y'], dtf_[dtf_["input"] == 1]['z'], 'r')
ax.set(xlabel=None, ylabel=None, zlabel=None,
       xticklabels=[], yticklabels=[], zticklabels=[])
for label, row in dtf_[["x", "y", "z"]].iterrows():
    x, y, z = row
    ax.text(x, y, z, s=label)

# plt.savefig('w2v_test.png')
# %%
# %%
