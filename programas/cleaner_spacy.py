# %%[markdown]
# Tokenizamos y luego tag
# %%
from os import sep
from re import split
import spacy
import es_core_news_sm
import pandas as pd
path = '../dataset/'
# %%
nlp = es_core_news_sm.load()
# %%
bici = pd.read_csv('../dataset/bici_clean.csv', sep=';')
moto = pd.read_csv('../dataset/moto_clean.csv', sep=';')
auto = pd.read_csv('../dataset/auto_clean.csv', sep=';')
peaton = pd.read_csv('../dataset/peaton_clean.csv', sep=';')

bici['descripcion_del_hecho - Final'] = bici['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
peaton['descripcion_del_hecho - Final'] = peaton['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
moto['descripcion_del_hecho - Final'] = moto['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
auto['descripcion_del_hecho - Final'] = auto['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)

bici.to_csv(path + 'token/bici_token.csv')
moto.to_csv(path + 'token/moto_token.csv')
auto.to_csv(path + 'token/auto_token.csv')
peaton.to_csv(path + 'token/peaton_token.csv')
# %%
# %%
#Tokenization - Etiquetado - Lemmatization
# La función nlp recibe un string
# El objeto Doc es una secuencia de objetos Token, osea cada fila de la columnda es un Doc.
bici = pd.read_csv('../dataset/bici_clean.csv', sep=';')
moto = pd.read_csv('../dataset/moto_clean.csv', sep=';')
auto = pd.read_csv('../dataset/auto_clean.csv', sep=';')
peaton = pd.read_csv('../dataset/peaton_clean.csv', sep=';')

bici['descripcion_del_hecho - Final'] = bici['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
peaton['descripcion_del_hecho - Final'] = peaton['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
moto['descripcion_del_hecho - Final'] = moto['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
auto['descripcion_del_hecho - Final'] = auto['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)

# %%
# Funcion que recibe la fila con cada palabra tipo Doc y devuelve cada palabra en par ordenado (Lemma,Pos)


def Doc2Par(line):
    aux = []
    for i in line:
        aux.append((i.lemma_, i.pos_))
    return aux

# %%


bici['Spacy Lemma-Pos'] = bici['descripcion_del_hecho - Final'].apply(Doc2Par)
moto['Spacy Lemma-Pos'] = moto['descripcion_del_hecho - Final'].apply(Doc2Par)
auto['Spacy Lemma-Pos'] = auto['descripcion_del_hecho - Final'].apply(Doc2Par)
peaton['Spacy Lemma-Pos'] = peaton['descripcion_del_hecho - Final'].apply(
    Doc2Par)

# %%

bici.to_csv(path + 'lemma-pos/bici_lemma-pos.csv', sep=';')
moto.to_csv(path + 'lemma-pos/moto_lemma-pos.csv', sep=';')
auto.to_csv(path + 'lemma-pos/auto_lemma-pos.csv', sep=';')
peaton.to_csv(path + 'lemma-pos/peaton_lemma-pos.csv', sep=';')

# %%
# %%
