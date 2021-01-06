#!/usr/bin/env python
# coding: utf-8

# #### Leyendo descripciones
# Al parecer cuando la descripción la hace la operadora, habla en tercera persona utilizando las palabras "asegurado" "aseg" "vh" "vhl"
# Gramas que me llaman la atención y estructuras interesantes:
# * soy embestido en mi parte delantera derecha por parte delantera izquierda de un b
# * asegurado circulando por beiro embiste con su parte delantera a un tercero en su parte trasera derecha
# ### * me detengo completamente siento embestida de un gol blanco patente ipc desde atras
# * siento impacto del vh de un tercero en mi parte delantera derecha con su parte delantera
# * asegurado en calle gral hornos no alcanza a frenar y colisiona con parte delantera parte trasera de un terceropatrullero
# * vh aseg circulaba por calle zorzal impacta con su parte delantera izquierda en lateral izquierdo de un tercero
# * asegurado embiste con parte delantera derecha a parte trasera derecha del tercero
# * vhl asegurado impacta con su parte delantera parte trasera del primer tercero
# * me embiste con su parte delantera derecha contra mi parte delantera izquierda
# * asegurada circulando en av boyaca caba y un tercero detenido repentinamente por misma calle adelante de asegurada es colisionado por esta con su parte delantera a parte trasera del tercero
#
# #### Posible caso especial:
# * asegurado circulando por aut gral paz a altura de av cabildo al querer cambiar de carril por piso humedo vhl gira en trompo perdiendo control y colisiona su delantera izquierda con parte trasera derecha del tercero e impacta contra un guard raid con su lateral izquierdo vhl sin movilidad sin lesiones intervino policia venia por gral paz sentido norte
# * colisona a un vh se encontraba circulando por misma con parte delantera izquierda con parte delantera izquierda
# * conductor asegurado por av independencia cuando un tercero a su izquierda se cruza de carril colisionando en parte delantera izquierda cuando se va a detener vuelve a colisionar en frente sin lesionados y desplazamientos denuncia segun tercero circulaba por mi carril y vehiculo del tercero me choco en mi parte trasera derecha me provoc un semitrompo y vuelve a chocarme en mi lateral derecho recib atencin medica por mis propios medios danos guardabarro trasero farol trasero tapa de baul trasera puertas laterales derechas tercero fiat siena sedan fire aa modelo patente mbu cond jose jonas velasco mejia velasco dni cel asegurado pulido velasco edicson edir compania

# ### N-grams
# Los n-grams van a ser utilizados para confeccionar un diccionario puesto que si utilizamos solo palabras, cuando convirtamos las oraciones en vectores, si el contenido en palabras es el mismo pero el orden es distinto entre dos oraciones nos dará el mismo vector, en cambio con los tri o bi es más difícil que eso ocurra.
# The n-gram model lets you take into account the sequences of words in contrast to what just using singular words (unigrams) will allow you to do.
# ![tf y idf terms frequency](tfidf.png)
# the number of times certain words (w) appear in a document (d), indexes corresponding to the vocabulary set y la frecuencia tf-idf es el producto entre ambos.
# ‘D’ represents the entire text corpus (fstring).  ‘df(d,w)’ , represents how many documents the word appears in.
# So a word that is prevalent throughout the entire corpus will have a lesser TFIDF value because the IDF value it would multiply with the TF value will be smaller than others.
# #### Ejemplo tf-idf:
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#
# arr = ["Car was cleaned by Jack",
# 	"Jack was cleaned by Car."]
#
#  If you want to take into account just term frequencies:
# vectorizer = CountVectorizer(ngram_range=(2,2))
#  The ngram range specifies your ngram configuration.
#
# X = vectorizer.fit_transform(arr)
#  Testing the ngram generation:
# print(vectorizer.get_feature_names())
#  This will print: ['by car', 'by jack', 'car was', 'cleaned by', 'jack was', 'was cleaned']
#
# print(X.toarray())
#  This will print:[[0 1 1 1 0 1], [1 0 0 1 1 1]]
#
#  And now testing TFIDF vectorizer:
# vectorizer = TfidfVectorizer(ngram_range=(2,2)) # You can still specify n-grams here.
# X = vectorizer.fit_transform(arr)
#
#
#  Testing the TFIDF value + ngrams:
# print(X.toarray())
#  This will print:  [[ 0.          0.57615236  0.57615236  0.40993715  0.          0.40993715  ]
#  [ 0.57615236  0.          0.          0.40993715  0.57615236  0.40993715]]
#
#
#  Testing TFIDF vectorizer without normalization:
# vectorizer = TfidfVectorizer(ngram_range=(2,2), norm=None) # You can still specify n-grams h  ere.
# X = vectorizer.fit_transform(arr)
#
#  Testing TFIDF value before normalization:
# print(X.toarray())
#  This will print: [[ 0.          1.40546511  1.40546511  1.          0.          1.        ]
#  [ 1.40546511  0.          0.          1.          1.40546511  1.        ]]
#
#

# ### Posible Work-Flow
# ![](workflow.png)
# El perfil no es mas que un set con las ocurrencias de cada n-grama.

import functions
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import gensim
import sklearn
import nltk
pd.options.display.max_colwidth = None
pd.options.display.max_rows = 8
import spacy


def findStruct(to_find, where):
    words = []
    for i in where:
        for j in range(len(i)):
            if i[j].pos_ == to_find[0] and j + 2 < len(i):

                if i[j + 1].pos_ == to_find[1]:

                    if i[j + 2].pos_ == to_find[2]:
                        words.append(i[j].text + ' ' + i[j + 1].text + ' ' + i[j + 2].text)
    return words


sp = spacy.load('es_core_news_lg')

a = pd.read_csv('/home/lobon/Documentos/LegalHub/Notas/dat/auto.csv')

a['triGrams'] = a['descripcion_del_hecho - Final'].apply(lambda x: [i for i in nltk.ngrams(x.split(), 3)])

a['biGrams'] = a['descripcion_del_hecho - Final'].apply(lambda x: [i for i in nltk.ngrams(x.split(), 2)])

a.head()

fstring = ''
for row in a['descripcion_del_hecho - Final']:
    fstring += row + ' '

tokens = nltk.tokenize.word_tokenize(fstring)

uniGrams = pd.Series(nltk.ngrams(tokens, 1))

biGrams = pd.Series(nltk.ngrams(tokens, 2))

triGrams = pd.Series(nltk.ngrams(tokens, 3))

cuartiGrams = pd.Series(nltk.ngrams(tokens, 4))

quintiGrams = pd.Series(nltk.ngrams(tokens, 5))


# El trigrama (con, su, parte) y (su, parte, delantera) identifica al tercero?

grams = pd.DataFrame(index=None)
grams['uniGrams'] = uniGrams
grams['biGrams'] = biGrams
grams['triGrams'] = triGrams
grams['cuartiGrams'] = cuartiGrams
grams['quintiGrams'] = quintiGrams


# grams.to_csv('../dat/grams.csv',index=False)

a['not_Aseg'] = a['descripcion_del_hecho - Final'].apply(functions.Aseg)

a['yes_Aseg'] = a['descripcion_del_hecho - Final'][a['not_Aseg'] == '']

a['not_Aseg'] = a['descripcion_del_hecho - Final'][a['not_Aseg'] != '']


notAseg = a['not_Aseg'].dropna().reset_index().drop(columns='index')
yesAseg = a['yes_Aseg'].dropna().reset_index().drop(columns='index')


# ### Análisis para descripciones del asegurado

# A quellos trigramas o bigramas que indican propiedad como 'mi' indican que con un 80% de probabilidad el asegurado escribió la descripción, además permiten reconocer también que es el asegurado el que se refiere a ellos.

index = []

test = [('en', 'mi', 'lateral'), ('en', 'mi', 'parte')]
k = 0
good = 0
for i, v in enumerate(a['triGrams']):
    for j, v1 in enumerate(v):
        if v1 in test:
            if type(a.loc[i, 'yes_Aseg']) == float:
                good += 1
            # print(i,k,a.loc[i,['not_Aseg','yes_Aseg']])
            k += 1
print(k, good / k * 100)

aseg, tero = [], []
for i, v in enumerate(a['descripcion_del_hecho - Final']):
    if 'en mi parte' in v or 'en mi lateral' in v or 'en mi' in v:
        aux = v.split()
        for j, v1 in enumerate(aux):
            if v1 == 'parte' or v1 == 'lateral':
                index.append(i)
                aseg.append(' '.join(aux[:j + 3]))
                tero.append(' '.join(aux[j + 3:]))
                break


# aseg-analisis para descripciones del operador

test = [('con', 'su', 'parte'), ('en', 'su', 'parte')]
k = 0
good = 0
for i, v in enumerate(a['triGrams']):
    for j, v1 in enumerate(v):
        if v1 in test:
            if type(a.loc[i, 'not_Aseg']) == float:  # significa que no esta en las desc del asegurado
                good += 1
            # print(i,k,a.loc[i,['not_Aseg','yes_Aseg']])
            k += 1
print(k, good / k * 100)

# aseg,tero=[],[]
for i, v in enumerate(a['descripcion_del_hecho - Final']):
    if 'con su parte' in v or 'en su parte' in v or 'su parte' in v:
        aux = v.split()
        for j, v1 in enumerate(aux):
            if v1 == 'parte' or v1 == 'lateral':
                index.append(i)
                tero.append(' '.join(aux[:j + 3]))
                aseg.append(' '.join(aux[j + 3:]))
                break


versiones = pd.DataFrame()
versiones['no_estruc'] = a.loc[[i for i in range(len(a['descripcion_del_hecho - Final'])) if i not in index], 'descripcion_del_hecho - Final']
versiones['Lado_asegurado'] = pd.Series(aseg)
versiones['Lado_tercero'] = pd.Series(tero)

index = pd.Series(index)

index.to_csv('/home/lobon/Documentos/LegalHub/Notas/dat/index.csv')

# versiones.to_csv('../dat/versiones.csv')


# ADP: adposition =  is a cover term for prepositions and postpositions.
# DET: determiner = are words that modify nouns or noun phrases and express the reference of the noun phrase in context.
# AUX: auxiliary = is a function word that accompanies the lexical verb of a verb phrase and expresses grammatical distinctions not carried by the lexical verb, such as person, number, tense, mood, aspect, voice or evidentiality.
# NOUN: noun = person, place, thing, animal or idea.


pos = []

for i, v in enumerate(a['descripcion_del_hecho - Final']):
    aux = v.split()
    if ('de tercero' in v or 'de ero' in v or 'de un tercero' in v) and i not in index:
        for j, w in enumerate(aux):
            if w == 'tercero':
                doc = sp(' '.join(aux[j - 3:j + 3]))
                # i== indice en a, j== indice en a[i]
                pos.append([i, j, ' '.join(aux[j - 3:j + 3]), [i.pos_ for i in doc]])

doc = pd.DataFrame({'doc': pd.Series([sp(i) for i in a['descripcion_del_hecho - Final']])})


pos_lis = []
for i, row in enumerate(doc['doc']):
    pos_lis.append([i.pos_ for i in row])


pos_tot = []
for i in pos_lis:
	for j in i:
		pos_tot.append(j)

triPos = pd.Series(nltk.ngrams(pos_tot, 3))

dic_triPos = triPos.value_counts().to_dict()
val = list(dic_triPos.values())
key = list(str(i) for i in dic_triPos.keys())

# lim = 20
# plt.figure(figsize=(8, 4))
# plt.barh(key[:lim], val[:lim])
# plt.savefig('../dat/stat_triPosTags.png')


#%%
# path = "/home/lobon/Documentos/LegalHub/Notas/dat/"
# for i in [0, 1, 2, 3, 4, 5]:
# 	to_find = triPos[i]
# 	# Las ternas de la lista words son la estructura más común encontrada en el corpus

# 	words = pd.Series(findStruct(to_find, doc['doc']))
# 	# words.value_counts()

# 	dic_triPos = words.value_counts().to_dict()
# 	val = list(dic_triPos.values())j
# 	key = list(str(i) for i in dic_triPos.keys())
# 	lim = 25
# 	plt.figure(figsize=(15, 10))
# 	plt.barh(key[:lim], val[:lim])
# 	plt.savefig(path + 'triPos_' + str(i) + '.png')

"""
