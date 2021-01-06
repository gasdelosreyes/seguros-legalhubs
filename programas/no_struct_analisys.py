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
seed = 4196538107 # es la que generó el csv
df = pd.read_csv(dat_path + 'no_estruc_cluster_contex_descrip.csv')
df1 = pd.read_csv(dat_path + 'no_estruc_aseg_noAseg.csv')
# print(df.head())
clean(df1['aseg_no_estruc'])
clean(df1['no_aseg_no_estruc'])
# Ahora para cada descripción quiero saber su cluster
tmp=[]
df1['cluster_aseg'] = pd.Series()
for i,rowc in enumerate(df1['aseg_no_estruc']):
    for j,row in enumerate(df['descripcion']):
        if rowc in row:
            df1.loc[i,'cluster_aseg'] = df.loc[j,'cluster']

df1['cluster_no_aseg'] = pd.Series()
for i,rowc in enumerate(df1['no_aseg_no_estruc']):
    for j,row in enumerate(df['descripcion']):
        try:
            if rowc in row:
                df1.loc[i,'cluster_no_aseg'] = df.loc[j,'cluster']
        except:
            pass
# Hay que agregar un a regla dura que cuando diga "choco" o "choque" directamente
# nos mande a que choco con la parte delantera, si solamente se nombra una parte 
# del rodado esa es la parte donde ocurrio el impacto.
print(df1.head())
# df1.to_csv(dat_path + 'versiones_cluster.csv')
print(round(len(df1['cluster_aseg'].dropna())/len(df1['cluster_aseg']),2)*100,'% casos clasificados aseg')
print()
print(round(len(df1['cluster_no_aseg'].dropna())/len(df1['cluster_no_aseg'])*100,2),'% casos clasificados no_aseg')



