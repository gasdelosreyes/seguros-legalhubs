
import pandas as pd
import nltk
import gensim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import random
from functions import *
path_dat = '../dat/'

# no pude importar el modulo de funciones :(
# ::::SOLVED::::
# es que para importar el modulo de funciones tenes que iniciar el kernel
# en la misma carpeta donde esta tu modulo y tu programa.


df = pd.read_csv('../dat/versiones.csv')
df_model = pd.read_csv('../dat/auto.csv')


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


index = pd.read_csv(path_dat + 'index.csv')

fstring = ''
for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']:
    fstring += i + ' '

tokens = nltk.tokenize.word_tokenize(fstring)
new_biGrams = pd.Series(nltk.ngrams(tokens, 2))
new_biGrams = [' '.join(i) for i in new_biGrams]
new_triGrams = pd.Series(nltk.ngrams(tokens, 3))
new_triGrams = [' '.join(i) for i in new_triGrams]

new_corpus = [nltk.tokenize.word_tokenize(i) for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']]

dic = gensim.corpora.Dictionary(new_corpus)
tfidf = gensim.models.TfidfModel(dictionary=dic, normalize=True)
corpus_vectors = [tfidf[dic.doc2bow(i)] for i in new_corpus]

new_corpus = [
    TaggedDocument(words, [idx])
    for idx, words in enumerate(new_corpus)
]

model = Doc2Vec(new_corpus, vector_size=300, min_count=3, seed=1)
print(len(model.wv.vectors), len(new_corpus))

len(corpus_vectors)

pca = PCA(n_components=2)
pca_tokens = pca.fit_transform(model.wv.vectors)  # esto tenes que arreglar porque no esta bien
pca_df = pd.DataFrame(data=pca_tokens, columns=['x', 'y'])
fig = plt.figure(figsize=(10, 6))
plt.grid()
ax = fig.add_subplot(111)
ax.set_title('vectors crudos')
# for i in range(len(target_vec)):
# ax.annotate(target[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
ax.scatter(pca_df.x, pca_df.y)

######################################################################################################################


def conIdxPart(w):
    part = ['parte', 'lateral']
    idx = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(w))
    aux = ''
    for j in part:
        concor = idx.find_concordance(j)
        for i in range(len(concor)):
            aux += ' '.join(concor[i][0][-2:]) + ' ' + concor[i][1] + ' ' + ' '.join(concor[i][2][:2])
            aux += ', '
            i += 2
            print(aux)

    return aux.split(', ')


new_df_model = []
for i in df_model.loc[index['0'], 'descripcion_del_hecho - Final']:
    part = ['parte', 'lateral']
    idx = nltk.ConcordanceIndex(nltk.tokenize.word_tokenize(i))
    for j in part:
        concor = idx.find_concordance(j)
        for i in range(len(concor)):
            new_df_model.append(' '.join(concor[i][0][-2:]) + ' ' + concor[i][1] + ' ' + ' '.join(concor[i][2][:2]))
            i += 2

#new_df_model = df_model.loc[index['0'], 'descripcion_del_hecho - Final'].apply(conIdxPart)

new_df_model[:5]


fstring = ''
for i in new_df_model:
    fstring += i + ' '

tokens = nltk.tokenize.word_tokenize(fstring)
# random.shuffle(tokens)
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

model = Doc2Vec(new_corpus, vector_size=vector_size, seed=1)
pca = PCA(n_components=2)

# vector de etiquetas de categoria, nos da una idea de donde se encuentran las palabras que
# buscamos en el espacio vectorial

target_vec = []
target2 = ['izquierdo', 'derecho', 'trasera', 'delantera', 'izquierda', 'derecha', 'atras', 'adelante']
target3 = ['parte', 'lateral', 'parte lateral']

for i in range(len(target2)):
    try:
        target_vec.append(model[target2[i]])
        print(target2[i])
    except KeyError:
        pass
# aca se guardan los vectores del modelo, es decir cada vector correspondiente
# a cada contexto con la que se entreno el modelo
mod_vec = []
for i in test_cor:
    try:
        mod_vec.append(pca.fit_transform(model[i]))

    except:
        pass

aux = []
for i in mod_vec:
    for j in i:
        aux.append(j)


# print(pca.fit(target_vec).noise_variance_)
# pca_tokens = pca.fit_transform(model.wv.vectors)

pca_df = pd.DataFrame(data=aux, columns=['x', 'y'])
fig = plt.figure(figsize=(12, 8))
plt.grid()
ax = fig.add_subplot(111)

# for i in range(len(target_vec)):
#    ax.annotate(target2[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
ax.scatter(pca_df.x, pca_df.y)
pca_tokens = pca.fit_transform(target_vec)
pca_df = pd.DataFrame(data=pca_tokens, columns=['x', 'y'])
for i in range(len(target_vec)):
    ax.annotate(target2[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))
ax.scatter(pca_df.x, pca_df.y)
#plt.savefig(path_dat + 'good_dist_today.png')

#######
# *Resta dividir el data set en train-test para poder comparar mejor los resultados

# *Guardar el modelo con los parametros entrenados para poder utilizarlo en otro
# programa.

#######

# Aca empieza el kmeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

for i in range(2, 11):

    kmeans = KMeans(n_clusters=i)
    # se lo entrena
    kmeans.fit(aux)
    print(silhouette_score(aux, kmeans.labels_, metric='euclidean'))

kmeans = KMeans(n_clusters=4)
# se lo entrena
kmeans.fit(aux)

pca = PCA(n_components=2)

df_pca = pd.DataFrame(data=aux, columns=['x', 'y'])

df_pca['cluster'] = pd.Series(kmeans.labels_)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(111)

color = np.array(['red', 'green', 'blue', 'orange'])
plt.title('silhouette_score ' + str(silhouette_score(aux, kmeans.labels_, metric='euclidean')))
ax.scatter(x=df_pca.x, y=df_pca.y, c=color[df_pca.cluster])
