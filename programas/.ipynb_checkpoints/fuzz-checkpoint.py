#!/usr/bin/env python
# coding: utf-8

# In[79]:


#import spellchecker as sp
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd


# In[77]:


bad = ['izq','derechacolisiona','izqueirda','dericha']
good = ['izquierda','derecha']


# In[78]:


for i in bad:
    for j in good:
        print(i,j)
        print(fuzz.ratio(i,j))


# In[174]:


bici = pd.read_csv('../lobon/Documentos/LegalHub/dataset/bici_clean.csv',sep=';')

col = 'descripcion_del_hecho - Final'

for i in range(len(bici[col])):
    if type(bici[col][i]) == str:
        aux = bici[col][i].split()
    else:
        continue
    for j in range(len(aux)):
        if 70<=fuzz.ratio(aux[j],'izquierda') or aux[j] == 'izq':        
            bici[col][i] = bici[col][i].replace(aux[j],'izquierda')
        if 70<=fuzz.ratio(aux[j],'derecha'):
            bici[col][i] = bici[col][i].replace(aux[j],'derecha')

bici.to_csv('Documentos/LegalHub/dataset/bici_fuzz.csv')


# In[ ]:




