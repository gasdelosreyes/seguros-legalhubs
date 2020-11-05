# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
from nltk.corpus import wordnet,stopwords,swadesh
import re


# %%
def Dictionary():
    dic = ['vh','amb','av','avenida','ambulancia','cruza','calle','policia','impacto','observo','conductor','rotonda','casco']
    for i in stopwords.words('spanish'):
        if(i != 'derecha' and i != 'izquierda'):
            dic.append(i)
    for i in swadesh.words('es'):
        if(i != 'derecha' and i != 'izquierda'):
            dic.append(i)
    return dic


# %%
def deleteExtraData(text):
    words = ['intervino','interviene','ampliacion','formalizo']
    for w in words:
        if(w in text):
            #print(text.split(w)[1])
            text = text.split(w)[0]
    return text


# %%
def cleaner(dataset):
    dataset = dataset.apply(lambda x: x.astype(str).str.lower())
    dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
        lambda x: re.sub('[^\w\s]','',x)
    )
    dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
        lambda x: re.sub('(\r\n?|\n)+','',x)
    )
    dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
        lambda x: re.sub('\d+','',x)
    )
    dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
        lambda x: " ".join(x for x in x.split() if x not in Dictionary()))
    for row in dataset['descripcion_del_hecho - Final']:
        dataset.loc[dataset['descripcion_del_hecho - Final'] == row, 'descripcion_del_hecho - Final'] = deleteExtraData(row)
    return dataset


# %%
path = '../dataset/'
moto = pd.read_csv( path+'moto.csv')
moto.drop('descripcion_del_hecho',1,inplace=True)
peaton = pd.read_csv( path+'peaton.csv')
peaton.drop('descripcion_del_hecho',1,inplace=True)
bici = pd.read_csv( path+'bici.csv')
bici.drop('descripcion_del_hecho',1,inplace=True)
auto = pd.read_csv( path+'auto.csv')
auto.drop('descripcion_del_hecho',1,inplace=True)


# %%
moto = cleaner(moto)
peaton = cleaner(peaton)
bici = cleaner(bici)
auto = cleaner(auto)


# %%


# %% [markdown]
# moto['word_tokenize'] = moto['descripcion_del_hecho - Final']
# moto['word_tokenize']=moto['word_tokenize'].apply(word_tokenize)
# moto = moto.rename(columns={'descripcion_del_hecho - Final':'descripcion'})
# 
# 
# %% [markdown]
# moto['pos_tag'] = moto['word_tokenize']
# moto['pos_tag'] = moto['pos_tag'].apply(pos_tag)
# %% [markdown]
# def get_words(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN
# 
# %% [markdown]
# moto['pos_tag']= moto['pos_tag'].apply(lambda x: [(word, get_words(pos_tag)) for (word, pos_tag) in x])
# 
# %% [markdown]
# moto.head()

