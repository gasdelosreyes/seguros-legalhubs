import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
import nltk as nk
import gensim
import random
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from functions import *
src_path = '/home/lobon/Documentos/LegalHub/Notas/src/'
dat_path = '/home/lobon/Documentos/LegalHub/Notas/dat/'
df = pd.read_csv(dat_path + 'versiones.csv')
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 120

# el contexto de "del"  es 3 a izq 1 a derecha -- -- -- del --
# por defecto el impacto se produce con la parte delantera del asegurado
# buscar colisiona .*? tercero siempre que no siga un contexto con la palabra parte
# porque puede ser "colisiona al tercero en parte trasera"
# buscar colisiona .*? tercero .*? [parte,derecha,izquierda,delantera]

# lo que voy a hacer despues es formar todos los contextos de interes para cada des-
# cripcion, vectorizar los contetos y tomar la media entre contextos de una misma des-
# cripcion para poder clasificar la descripción. (por que no hice esto antes?)
 
def clean(serie):
    """Limpia la columna donde estan las descripciones 

    :function: estructura todas las descripciones
    :returns: void

    """
    to_rep = [
    'izq.*? ', 'derecho', 'delantero', 'trasero', 'frontal', 'lat.*? ', 'lado',
    'frente', 'de atras', 'puerta', 'parte lateral', 'choque',
    r'tercero .* impacta', 'desde izq.*? ', 'vh.*? ', 'colis.*? ', 'choc.*? ',
    'impac.*? ', 'su delat.*? ', ' p ', 'aseg.*? ', 'emb.*? ', 'redg.*? ',
    'parte parte', 'roza', 'golp.*? ', 'contra\s', 'por detras', 'detras',
    ' ro '
    ]  # tentativo ,'precede']

    for_rep = [
    'izquierda ', 'derecha', 'delantera', 'trasera', 'parte delantera',
    'parte ', 'parte', 'parte delantera', 'en parte trasera', 'parte', 'parte',
    'mi parte delantera', 'tercero impacta', 'en parte izquierda ',
    'vehiculo ', 'colisiona ', 'colisiona ', 'colisiona ',
    'su parte delantera ', ' parte ', 'asegurado ', 'colisiona ', '', 'parte',
    'colisiona', 'colisiona ', 'con ', 'en parte trasera', 'trasera', 'tercero'
    ]  #,'con parte delantera']
    
    for i, w in enumerate(serie):
        # w=re.sub('desde izquierda','en parte izquierda',w)
        for j in range(len(to_rep) - 1):
            w = re.sub(to_rep[j], for_rep[j], w)
        w = re.sub(to_rep[len(to_rep) - 1], for_rep[len(for_rep) - 1],
               w)  #esto se puso muy extraño
        serie[i]= w
   
clean(df[no_estruc])
df = df.drop(columns=df.columns[0])
aux = []
for w in df['no_estruc']:
    if 'aseg' in w:
        aux.append(w)

_df = pd.DataFrame()
_df['aseg_no_estruc'] = pd.Series(aux)

aux1 = []
for w in df['no_estruc']:
    if w not in aux:
        aux1.append(w)

_df['no_aseg_no_estruc'] = pd.Series(aux1)
# print(_df['aseg_no_estruc'][32])
# _df.to_csv(dat_path + 'no_estruc_aseg_noAseg.csv')

aux, aux1, aux2 = [], [], []
for i, w in enumerate(_df['aseg_no_estruc']):
    mat = re.search(
        'colisiona .*? parte .*? tercero|derecha|izquierda|delantera|trasera',
        w)
    mat1 = re.search('tercero .*? colisiona', w)
    mat2 = re.search('tercero .*? parte .*? colisiona', w)
    if mat:
        aux.append(mat.group())
    # DESCOMENTAR ESTO SI ES QUE QUEREMOS GUARDAR LA REALACIÓN DE INDICES CON LA DESCRIPCIÓN
    else:
        aux.append('')
    if mat1:
        aux1.append(mat1.group())
    else:
        aux1.append('')
    if mat2:
        aux2.append(mat2.group())
    else:
        aux2.append('')
# print(_df['no_aseg_no_estruc'][5])
_df['colisiona_parte_tercero'] = pd.Series(aux)
_df['tercero_colisiona'] = pd.Series(aux1)
_df['tercero_parte_colisiona'] = pd.Series(aux2)

aux, aux1, aux2 = [], [], []
for i, w in enumerate(_df['aseg_no_estruc'][:10]):
    if type(w) == type('s'):
        mat = re.search(
            'colisiona .*? parte .*? [tercero|derecha|izquierda|delantera|trasera]',
            w)
        mat1 = re.search('tercero .*? colisiona', w)
        mat2 = re.search('tercero .*? parte .*? colisiona', w)
    if mat:
        aux.append(mat.group())
    # DESCOMENTAR ESTO SI ES QUE QUEREMOS GUARDAR LA REALACIÓN DE INDICES CON LA DESCRIPCIÓN
    else:
        aux.append('')
    if mat1:
        aux1.append(mat1.group())
    else:
        aux1.append('')
    if mat2:
        aux2.append(mat2.group())
    else:
        aux2.append('')
# print(_df['no_aseg_no_estruc'][5])
_df['colisiona_parte_tercero_noAseg'] = pd.Series(aux)
_df['tercero_colisiona_noAseg'] = pd.Series(aux1)
_df['tercero_parte_colisiona_noAseg'] = pd.Series(aux2)
# print(_df.columns)
# _df.to_csv(dat_path + 'df-test.csv')

# obtengamos ahora los contextos de parte y vinculemoslos con las descripciones

print('obtenemos los contextos')

df_model = {'descr': [], 'index': []}
for j, i in enumerate(df['no_estruc']):
    idx = nk.ConcordanceIndex(nk.tokenize.word_tokenize(i))
    concor = idx.find_concordance('parte')
    df_model['index'].append(j)
    aux = []
    for i in range(len(concor)):
        aux.append(' '.join(concor[i][0][-2:]) + ' ' + concor[i][1] + ' ' +
                   ' '.join(concor[i][2][:2]))
    df_model['descr'].append(aux)

# formamos el corpus=[[contexto,spliteado],[no,vacio]]
print('fomramos el corpus')

corpus = []
for j in df_model['descr']:
    corpus += [nk.word_tokenize(i) for i in j if 0 < len(i)]

# Formamos el diccionario
print('formamos el dic')

dic = gensim.corpora.Dictionary(corpus)

test_cor = corpus.copy()
# Pasamos a formato bow
corpus = [
    TaggedDocument(words, [idx_tag]) for idx_tag, words in enumerate(corpus)
]

# elegimos la mejor semilla
# la seed es 2201433561
vector_size = 300
print('empezó')
scores = []
import tqdm
for k in tqdm.tqdm(range(500)):
    seed = random.randint(1, 2**32 - 1)
    np.random.seed(seed)
    model = Doc2Vec(corpus, vector_size=vector_size, seed=seed)
    pca = PCA(n_components=2, svd_solver='full', whiten=True)

    mod_vec = []
    to_fit = []
    for i in test_cor:
        try:
            if len(model[i]) != 300:
                for j in model[i]:
                    to_fit.append(j)
            else:
                to_fit.append(model[i])
        except:
            pass
    pca = pca.fit(to_fit)
    # hay palabras que no toma el modelo como 'me' o 'ipc'
    for i in test_cor:
        try:
            mod_vec.append(GeoCenter(pca.transform(model[i])))
        except:
            pass
    # print('ultimo paso',k)

    kmeans = KMeans()
    kmeans.fit(mod_vec)
    scores.append((silhouette_score(mod_vec,
                                    kmeans.labels_,
                                    metric='euclidean'), seed))

print(max(scores))
seed = max(scores)[1]


# seed = 2201433561
seed = max(scores)[1]
np.random.seed(seed)
model = Doc2Vec(corpus, vector_size=vector_size, seed=seed)
pca = PCA(n_components=2, svd_solver='randomized', whiten=True,random_state=seed)
mod_vec = []
to_fit = []
seguidor = {'contexto': [], 'vector': []}
for i in test_cor:
    try:
        if len(model[i]) != 300:
            for j in model[i]:
                to_fit.append(j)
        else:
            to_fit.append(model[i])
    except:
        pass

pca.fit(to_fit)

for i in test_cor:
    try:
        mod_vec.append(GeoCenter(pca.transform(model[i])))
        seguidor['contexto'].append(i)
        seguidor['vector'].append(GeoCenter(pca.transform(model[i])))
    except:
        pass

kmeans = KMeans()
kmeans.fit(mod_vec)
print(silhouette_score(mod_vec, kmeans.labels_))

# with open(dat_path + 'pca_params_note', 'wb') as f:
#     pickle.dump(pca, f)

pca_df = pd.DataFrame(data=mod_vec, columns=['x', 'y'])
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111)

# for i in range(len(target_vec)):
#    ax.annotate(target2[i], (pca_df.loc[i, 'x'], pca_df.loc[i, 'y']))

ax.grid()
ax.scatter(pca_df.x, pca_df.y)
# plt.grid()
for i in range(10):
    ran = random.randint(0, len(seguidor['contexto']))
    ax.annotate(' '.join(seguidor['contexto'][ran]),
                (seguidor['vector'][ran][0], seguidor['vector'][ran][1]),
                fontsize=12,
                alpha=0.9)

# plt.savefig(dat_path + 'plots/temp3.png')

df_pca = pd.DataFrame(data=mod_vec, columns=['x', 'y'])
df_pca['cluster'] = pd.Series(kmeans.labels_)
df_pca['contexto'] = pd.Series()
for i in range(len(df_pca['cluster'])):
    for j in range(len(seguidor['vector'])):
        if df_pca['x'][i] == seguidor['vector'][j][0] and df_pca['y'][i] == seguidor['vector'][j][1]:
            df_pca['contexto'][i] = ' '.join(seguidor['contexto'][j])
            break

# aca se van a guardar pares ordenados [idx de contexto, idx de la descripción]
linker = []
for i, c in enumerate(df_pca['contexto']):
    for j, v in enumerate(df['no_estruc']):
        if c in v:
            linker.append([i, j])
            break

# vamos a formar ahora una lista de contextos por cada descripción
ctxs = []
desc = []
k = 0
for i in range(len(linker)):
    k = linker[i][1]
    desc.append(df['no_estruc'][k])
    aux = []
    for j in linker[i:]:
        if j[1] == k:
            aux.append(j[0])
    ctxs.append(list(set(aux)))
desc = list(set(desc))

target = ['trasera', 'delantera', 'izquierda', 'derecha']
          
categorias = []
for i in range(8):
    dic_cluster = {}
    for k in df_pca['contexto'][df_pca['cluster'] == i]:
        for j in k.split():
            try:
                # dic_cluster[j]+=1
                if 3 < len(j) and j != 'parte' :
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
    i = sorted(i.items(), key=lambda x: x[1], reverse=True)
    for j in i:
        if j[0] in target:
            dic_final[j[0]] = j[1]
            #print("{} {} %    ".format(j[0], j[1]), end="\t\t")
            suma += j[1]
    categorias_final.append(dic_final)
print(categorias_final)
descripciones = df['no_estruc']

df_pca['descripcion'] = pd.Series(dtype=str)
df_pca['ubicacion'] = pd.Series(dtype=str)

for i, contexto in enumerate(df_pca['contexto']):
    des = []
    idxs = []

    for j, descr1 in enumerate(descripciones):
        if contexto in descr1:
            des.append(descr1)
            idxs.append(j)
    df_pca['descripcion'][i] = list(set(des))
    df_pca['ubicacion'][i] = idxs


df_pca.to_csv(dat_path + 'no_estruc_cluster_contex_descrip.csv')
