#%%
from nltk import data
import pandas as pd
import re
import nltk
nltk.download('stopwords')

dataset = pd.read_excel('../dataset/casos_universidad.xlsx')

#dataset = ds
#automatizar esta limpieza
dataset["descripcion_del_hecho - Final"] = dataset["descripcion_del_hecho - Final"].apply(lambda x: re.sub('[^\w\s]','',str(x)))
dataset["descripcion_del_hecho - Final"] = dataset["descripcion_del_hecho - Final"].apply(lambda x: re.sub('\d+','',str(x))) 
dataset=dataset.apply(lambda x : x.astype(str).str.lower())
dataset["descripcion_del_hecho - Final"]
ds_df = pd.DataFrame(dataset)

# Con esto obtenemos la columna de "descripcion_del_hecho - Final"
ds_df.iloc[:,[11]].to_csv('../dataset/descripciones.csv',sep=' ',index=False,header=False)
"""
calles = pd.read_csv('../dataset/calles_caba.csv')
calles.dtypes
calles_df = pd.DataFrame(calles)
calles_df.iloc[:,[3,8,9]]
"""
# Strings a limpiar fechas,caracteres especiales ":", números especiales, ";\red0\green0lue0;", nombres de personas, cosas entre cochetes "[mailto:jorgejonitz@hotmail.com]", arreglar ascentos.
# %%
stop = nltk.corpus.stopwords.words('spanish')
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop)
)
# %%
ds_df = pd.DataFrame(dataset)
ds_df.iloc[:,[11]].to_csv('../dataset/descripciones.csv',sep=' ',index=False,header=False)
# %%
with open("../dataset/words_deleted.txt","w") as f:
    f.truncate()
    for i in stop:
        f.write(i+' ')
# %%
from nltk.tokenize import word_tokenize
nltk.download('punkt')
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(word_tokenize)
# %%
nltk.download('averaged_perceptron_tagger')
dataset['descripcion_del_hecho - Final']= dataset['descripcion_del_hecho - Final'].apply(nltk.tag.pos_tag)

# %%

dataset['descripcion_del_hecho - Final'].head(5)
# %%
dataset.iloc[:].to_csv('../dataset/testing.csv')
# %%
