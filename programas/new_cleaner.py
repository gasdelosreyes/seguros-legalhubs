#%%
from nltk import data
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

dataset = pd.read_excel('../dataset/casos_universidad.xlsx')

#dataset = ds
#automatizar esta limpieza
dataset["descripcion_del_hecho - Final"] = dataset["descripcion_del_hecho - Final"].apply(lambda x: re.sub('[^\w\s]','',str(x)))
dataset["descripcion_del_hecho - Final"] = dataset["descripcion_del_hecho - Final"].apply(lambda x: re.sub('\d+','',str(x))) 
dataset=dataset.apply(lambda x : x.astype(str).str.lower())
dataset["descripcion_del_hecho - Final"]
# %%
for i in range(len(dataset['descripcion_del_hecho - Final'])):
    aux = dataset['descripcion_del_hecho - Final'][i].split()
    k=0
    for j in range(len(aux)):
        if aux[j-k] == 'aseg':
            aux[j-k] = 'asegurado'
        if len(aux[j-k])<3:
            k+=1
            aux.pop(j-k)
    dataset['descripcion_del_hecho - Final'][i] = " ".join(aux)
 
ds_df = pd.DataFrame(dataset)

# Con esto obtenemos la columna de "descripcion_del_hecho - Final"
#ds_df.iloc[:,[11]].to_csv('../dataset/descripciones.csv',sep=' ',index=False,header=False)

# Strings a limpiar fechas,caracteres especiales ":", números especiales, ";\red0\green0lue0;", nombres de personas, cosas entre cochetes "[mailto:jorgejonitz@hotmail.com]", arreglar ascentos.
# %%
stop = nltk.corpus.stopwords.words('spanish')
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop)
)
# %%
ds_df = pd.DataFrame(dataset)
#ds_df.iloc[:,[11]].to_csv('../dataset/descripciones.csv',sep=' ',index=False,header=False)
# %%
with open("../dataset/words_deleted.txt","w") as f:
    f.truncate()
    for i in stop:
        f.write(i+' ')
# %%
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(word_tokenize)
# %%
dataset['descripcion_del_hecho - Final']= dataset['descripcion_del_hecho - Final'].apply(nltk.tag.pos_tag)


dataset['descripcion_del_hecho - Final'].head(5)

# %%
#dataset.iloc[:].to_csv('../dataset/testing.csv')
# %%
#devuelve una lista de palabras que no estaban en el diccionario.
#recibe un iterable text que si es un texto tenes que usar text.split
def unusual_words(text):
    text_vocab = set(w.lower() for w in text.split() if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words()) #esto sería el spanish_vocab
    unusual = text_vocab.difference(english_vocab) #acá en vez de tomar la diferencia
    #text_vocab = " ".join(text_vocab.difference(english_vocab))
    return sorted(unusual)

# %%
nltk.FreqDist('asegurado')
# %%
palabras=list()
for i in dataset['descripcion_del_hecho - Final']:
    desc = i.split()
    desc = set(desc)
    desc = desc.difference(set(palabras))
    for j in desc:
        palabras.append(j)  
palabras.sort()
# %%
len(palabras)
# %%
dataset['descripcion_del_hecho - Final'].head(5)
# %%
dataset['descripcion_del_hecho - Final'].to_csv('../dataset/testing.csv')
# %%
from nltk.corpus import wordnet
def get_words(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

datatest['descripcion_del_hecho - Final']= datatest['descripcion_del_hecho - Final'].apply(lambda x: [(word, get_words(pos_tag)) for (word, pos_tag) in x])

#%%
