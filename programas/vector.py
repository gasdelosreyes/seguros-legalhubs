# %%
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from functions import *
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
path_dat = '../dat/'

# no pude importar el modulo de funciones :(
# ::::SOLVED::::
# es que para importar el modulo de funciones tenes que iniciar el kernel
# en la misma carpeta donde esta tu modulo y tu programa.


df = pd.read_csv('../dat/versiones.csv')
df_model = pd.read_csv('../dat/auto.csv')
df_model['descripcion_del_hecho - Final'] = df_model['descripcion_del_hecho - Final'].apply(cleaner)
for i, w in enumerate(df_model['descripcion_del_hecho - Final']):
    w = w.replace('izquierdo', 'izquierda')
    w = w.replace('derecho', 'derecha')
    w = w.replace('delantero', 'delantera')
    w = w.replace('trasero', 'trasera')
    df_model.loc[i, 'descripcion_del_hecho - Final'] = w
# %%
fstring = ''
for i in df_model['descripcion_del_hecho - Final']:
    fstring += i + ' '

tokens = nltk.tokenize.word_tokenize(fstring)

uniGrams = pd.Series(nltk.ngrams(tokens, 1))

biGrams = pd.Series(nltk.ngrams(tokens, 2))
biGrams = [' '.join(i) for i in biGrams]

triGrams = pd.Series(nltk.ngrams(tokens, 3))
triGrams = [' '.join(i) for i in triGrams]

cuartiGrams = pd.Series(nltk.ngrams(tokens, 4))


corpus = [biGrams] + [triGrams] + [nltk.tokenize.word_tokenize(i) for i in df_model['descripcion_del_hecho - Final']]

# %%
dic = gensim.corpora.Dictionary(corpus)
tfidf = gensim.models.TfidfModel(dictionary=dic, normalize=True)  # normaliza por defecto?
corpus_vectors = [tfidf[dic.doc2bow(i)] for i in corpus]

corpus = [
    TaggedDocument(words, [idx])
    for idx, words in enumerate(corpus)
]

model = Doc2Vec(corpus, vector_size=300, min_count=3, seed=1)

target = Labeler()

########################################
# esto me permite graficar los diferentes vectores
########################################

# target_vec = []
# for i, v in enumerate(target):
#   try:
#       print(v)
#       target_vec.append(model.infer_vector([v], steps=500))
#   except KeyError:
#       pass

# pca = PCA(n_components=2)
# pca_tokens = pca.fit_transform(target_vec)
# pca_df = pd.DataFrame(data=pca_tokens, columns=['x', 'y'])
# fig = plt.figure(figsize=(10, 6))
# plt.grid()
# ax = fig.add_subplot(111)
# for i in range(len(target_vec)):
#   ax.annotate(target[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
# ax.scatter(pca_df.x, pca_df.y)
#fig.savefig(path_dat + 'posicion_dist.png')

########################################
# esto me permitió elegir los steps#####
########################################
# cada coordenada varía aproximadamente 2% respecto al promedio con steps=1000
# casi un 3% con 500 steps, con varios epochs se obtiene un resultado similar peor tarda más.
# for i in range(100, 500, 100):
#   suma = 0
#   infer_list = []
#   for j in range(i):
#       infer = model.infer_vector([target[0]],steps=500)
#       #pca_tokens = pca.fit_transform(infer)
#       suma += infer[0]
#       infer_list.append(infer[0])
#   prom = suma / i
#   suma_dif = 0
#   for j in infer_list:
#       suma_dif += abs(prom - j)
#   prom_dif = suma_dif / i
#   print("en promedio se aleja del valor promedio un ", prom_dif / prom * 100.0, "%", " con ", i, ' repeticiones')


#####################################################
#                       NEW CORPUS                  #
#####################################################

# %%
index = pd.read_csv(path_dat + 'index.csv')

fstring = ''
for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']:
    fstring += i + ' '

tokens = nltk.tokenize.word_tokenize(fstring)
new_biGrams = pd.Series(nltk.ngrams(tokens, 2))
new_biGrams = [' '.join(i) for i in new_biGrams]
new_triGrams = pd.Series(nltk.ngrams(tokens, 3))
new_triGrams = [' '.join(i) for i in new_triGrams]

# Solo descripciones estructuradas
new_corpus = [nltk.tokenize.word_tokenize(i) for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']] + [new_biGrams] + [new_triGrams]

corpus1 = new_corpus.copy()

dic = gensim.corpora.Dictionary(new_corpus)

tfidf = gensim.models.TfidfModel(dictionary=dic, normalize=True, smartirs='lnc')
# Al tfidf le damos los tokens, bigramas y trigramas como si fueran docs
corpus_vectors = [tfidf[dic.doc2bow(i)] for i in new_corpus]
diccionario = dic.id2token

new_corpus = [
    TaggedDocument(words, [idx])
    for idx, words in enumerate(new_corpus)
]

model = Doc2Vec(new_corpus, vector_size=300, min_count=5, seed=1)


vec = []
for i in corpus1:
    try:
        vec.append(model[i])
    except:
        pass
vec1 = [j for i in vec for j in i]


pca = PCA(n_components=2)
# Fitteamos el PCA con tokens,bi y triGrams de las descripiones estructuradas
# vamos a usar este mismo algortimo entrenado para todos.
pca = pca.fit(vec1)
pca_tokens = pca.transform(vec1)
pca_df = pd.DataFrame(data=pca_tokens, columns=['x', 'y'])
fig = plt.figure(figsize=(10, 6))
plt.grid()
ax = fig.add_subplot(111)
# ax.set_title('vectors crudos')
# for i in range(len(target_vec)):
# ax.annotate(target[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
ax.scatter(pca_df.x, pca_df.y)

# %%

######################################################################################################################
# esto solo toma los contextos de parte y lateral y a
# diferencia de conIdxPart no solapa las oraciones

new_df_model = []
for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']:
    part = ['parte', 'lateral']
    idx = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(i))
    for j in part:
        concor = idx.find_concordance(j)
        for i in range(len(concor)):
            new_df_model.append(' '.join(concor[i][0][-2:]) + ' ' + concor[i][1] + ' ' + ' '.join(concor[i][2][:2]))
            #i += 2

#new_df_model = df_model.loc[index['0'], 'descripcion_del_hecho - Final'].apply(conIdxPart)

print(new_df_model[:5])

fstring = ''
for i in new_df_model:
    fstring += i + ' '

tokens = nltk.tokenize.word_tokenize(fstring)

new_biGrams = pd.Series(nltk.ngrams(tokens, 2))
new_biGrams = [' '.join(i) for i in new_biGrams]

new_triGrams = pd.Series(nltk.ngrams(tokens, 3))
new_triGrams = [' '.join(i) for i in new_triGrams]

new_corpus = [nltk.tokenize.word_tokenize(i) for i in new_df_model]

print(new_corpus[:5])

# Cada elemento de test_cor tiene una lista con las palabras del contexto alrededor de parte
test_cor = new_corpus.copy()
# random.shuffle(new_corpus)
# Se genera el diccionario con el que se configuran los paramtros del modelo
dic = gensim.corpora.Dictionary(new_corpus)

# instanciamos el modelo
# https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
tfidf = gensim.models.TfidfModel(dictionary=dic, normalize=True, smartirs='lnc')

# convert to bow format, par odenado (indice de aparicion,frecuencia)
corpus_vectors = [tfidf[dic.doc2bow(i)] for i in new_corpus]
new_corpus = [TaggedDocument(words, [idx_tag]) for idx_tag, words in enumerate(new_corpus)]

print(new_corpus[:5])

# entrenamos el modelo con el new_corpus etiquetado, cada 'doc' es en realidad un contexto.
vector_size = 300

model = Doc2Vec(new_corpus, vector_size=vector_size)
pca = PCA(n_components=2, svd_solver='full', whiten=True)

to_fit = []
for i in test_cor:
    try:
        if len(model[i]) != 300:
            for j in model[i]:
                to_fit.append(j)
        else:
            to_fit.append(model[i])
    except:
        to_fit.append(model.infer_vector(i))

pca = pca.fit(to_fit)


# vector de etiquetas de categoria, nos da una idea de donde se encuentran las palabras que
# buscamos en el espacio vectorial

target_vec = []
target2 = ['trasera', 'delantera', 'izquierda', 'derecha']
target3 = ['parte', 'lateral', 'parte lateral']

for i in range(len(target2)):
    try:
        target_vec.append(model[target2[i]])
        # print(target2[i])
    except KeyError:
        pass

# aca se guardan los vectores del modelo, es decir cada vector correspondiente
# a cada contexto con la que se entreno el modelo
mod_vec = []
seguidor = {'contexto': [], 'vector': []}


for i in test_cor:
    try:
        mod_vec.append(GeoCenter(pca.transform(model[i])))
        seguidor['contexto'].append(i)
        seguidor['vector'].append(GeoCenter(pca.transform(model[i])))
    except:
        pass
        # mod_vec.append(GeoCenter(pca.transform([model.infer_vector(i)])))
        # seguidor['contexto'].append(i)
        # seguidor['vector'].append(GeoCenter(pca.transform([model.infer_vector(i)])))

pca_df = pd.DataFrame(data=mod_vec, columns=['x', 'y'])
fig = plt.figure(figsize=(15, 10))
plt.grid()
ax = fig.add_subplot(111)

# for i in range(len(target_vec)):
#    ax.annotate(target2[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
import random

ax.scatter(pca_df.x, pca_df.y)
for i in range(10):
    ran = random.randint(0, len(seguidor['contexto']))
    ax.annotate(' '.join(seguidor['contexto'][ran]), (seguidor['vector'][ran][0], seguidor['vector'][ran][1]), fontsize=12, alpha=0.9)

# pca_tokens = pca.transform(target_vec)
# pca_df = pd.DataFrame(data=pca_tokens, columns=['x', 'y'])
# for i in range(len(target_vec)):
#     ax.annotate(target2[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
# ax.scatter(pca_df.x, pca_df.y)
# plt.savefig(path_dat + 'its_beautiful.png')

# Aca empieza el kmeans
# %%
sil = {'sil': [], 'idx': []}
for i in range(2, 15):

    kmeans = KMeans(n_clusters=i)
    # se lo entrena
    kmeans.fit(mod_vec)
    sil['sil'].append(silhouette_score(mod_vec, kmeans.labels_, metric='euclidean'))
    sil['idx'].append(i)
# plt.plot(sil['idx'],sil['sil'])
# plt.savefig(path_dat+'sil.png')
kmeans = KMeans(n_clusters=8)
# se lo entrena
kmeans.fit(mod_vec)

df_pca = pd.DataFrame(data=mod_vec, columns=['x', 'y'])

df_pca['cluster'] = pd.Series(kmeans.labels_)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(111)

color = np.array(['red', 'green', 'blue', 'orange'])
plt.title('silhouette_score ' + str(silhouette_score(mod_vec, kmeans.labels_, metric='euclidean')))
#ax.scatter(x=df_pca.x, y=df_pca.y, c=color[df_pca.cluster])

sns.lmplot('x', 'y', data=df_pca, hue='cluster', fit_reg=False)
# plt.savefig(path_dat + 'sns_more_beautiful.png')

# df_pca.to_csv(path_dat + 'context_cluster_8.csv')

################################################################################
# Continuamos el análisis con los clusters para ver de que descripción vienen
################################################################################
# %%
for i in range(len(seguidor['vector'])):
    if seguidor['vector'][i][0] == seguidor['vector'][0][0] and seguidor['vector'][i][1] == seguidor['vector'][0][1]:
        print(seguidor['contexto'][i], i)
# %%
df_pca['contexto'] = pd.Series()
for i in range(len(df_pca['cluster'])):
    for j in range(len(seguidor['vector'])):
        if df_pca['x'][i] == seguidor['vector'][j][0] and df_pca['y'][i] == seguidor['vector'][j][1]:
            df_pca['contexto'][i] = ' '.join(seguidor['contexto'][j])
            break
#df_pca = df_pca.sort_values('cluster')

# df_pca.to_csv(path_dat + 'cluster_final.csv')
# %%


# aca se van a guardar pares ordenados [idx de contexto, idx de la descripción]
linker = []

for i, c in enumerate(df_pca['contexto']):
    for j, v in enumerate(df_model['descripcion_del_hecho - Final']):
        if c in v:
            linker.append([i, j])
            break
# vamos a formar ahora una lista de contextos por cada descripción
ctxs = []
desc = []
k = 0

for i in range(len(linker)):
    k = linker[i][1]
    desc.append(df_model['descripcion_del_hecho - Final'][k])
    aux = []
    for j in linker[i:]:
        if j[1] == k:
            aux.append(j[0])
    ctxs.append(list(set(aux)))
desc = list(set(desc))
# %%
# al parecer el cluster 7 corresponde con la parte delantera
target = ['trasera', 'delantera', 'izquierda', 'derecha', 'frontal', 'parte', 'lateral', 'atras', 'adelante']
categorias = []
for i in range(8):
    dic_cluster = {}
    for k in df_pca['contexto'][df_pca['cluster'] == i]:
        for j in k.split():
            try:
                # dic_cluster[j]+=1
                if 3 < len(j) and j != 'parte' and j != 'lateral':
                    dic_cluster[j] += 1
            except KeyError:
                dic_cluster[j] = 0
    # print('Cluster número',i,'\n',dic_cluster)
    tot = np.sum(list(dic_cluster.values()))
    for i in dic_cluster.keys():
        dic_cluster[i] = round(dic_cluster[i] / tot * 100.0, 2)
    categorias.append(dic_cluster)

categorias_final = []
for idx, i in enumerate(categorias):
    dic_final = {}
    suma = 0
    print('Cluster ', idx)
    i = sorted(i.items(), key=lambda x: x[1], reverse=True)

    for j in i:
        if j[0] in target:
            dic_final[j[0]] = j[1]
            print("{} {}%    ".format(j[0], j[1]), end="\t\t")
            suma += j[1]
    categorias_final.append(dic_final)
    print('---', round(suma, 2), '%')
    print()

# %%
fig = plt.figure(figsize=(15, 10))
fig.add_subplot(241)
plt.bar(categorias_final[0].keys(), categorias_final[0].values())
plt.title('Cluster 0')
fig.add_subplot(242)
plt.bar(categorias_final[1].keys(), categorias_final[1].values())
plt.title('Cluster 1')
fig.add_subplot(243)
plt.bar(categorias_final[2].keys(), categorias_final[2].values())
plt.title('Cluster 2')
fig.add_subplot(244)
plt.bar(categorias_final[3].keys(), categorias_final[3].values())
plt.title('Cluster 3')
fig.add_subplot(245)
plt.bar(categorias_final[4].keys(), categorias_final[4].values())
plt.title('Cluster 4')
fig.add_subplot(246)
plt.bar(categorias_final[5].keys(), categorias_final[5].values())
plt.title('Cluster 5')
fig.add_subplot(247)
plt.bar(categorias_final[6].keys(), categorias_final[6].values())
plt.title('Cluster 6')
fig.add_subplot(248)
plt.bar(categorias_final[7].keys(), categorias_final[7].values())
plt.title('Cluster 7')
plt.suptitle('Distribución de clusters',fontsize=20)
# plt.savefig(path_plots+'clusters.jpg')