import pandas as pd
from objetos import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load
import matplotlib.pyplot as plt
# import seaborn as sns
import random
import numpy as np
import sys
import re
np.random.seed(7)

"""
Preparacion del dataset
"""
df = pd.read_csv('tmp/Tabla1.csv')

# df = df.dropna()
dummie_df = pd.DataFrame()
# dummie_df = pd.get_dummies(df[['quien']])

mov = []

for row in df['movimiento']:
    if row == 'si':
        mov.append(1)
    else:
        mov.append(0)

dummie_df['movimiento'] = pd.Series(mov)

impac_position = []
for row in df['impac_position']:
    if row == 'delantera':
        impac_position.append(2)
    if row == 'delantera izquierda':
        impac_position.append(4)
    if row == 'delantera derecha':
        impac_position.append(6)
    if row == 'trasera':
        impac_position.append(8)
    if row == 'trasera izquierda':
        impac_position.append(10)
    if row == 'trasera derecha':
        impac_position.append(12)

dummie_df['impac_position'] = pd.Series(impac_position)

aux = []
for i in df['responsabilidad']:
    if i == 'COMPROMETIDA' or i == 'DISCUTIDA':
        aux.append(1)
    else:
        aux.append(0)
dummie_df['responsabilidad'] = pd.Series(aux)

quien = [] 
for row in df['quien']:
    if row == 'asegurado':
        quien.append(1)
    else:
        quien.append(0)

dummie_df['quien'] = pd.Series(quien)

dummie_df = pd.concat([dummie_df[dummie_df['responsabilidad'] == 0].dropna(), dummie_df[dummie_df['responsabilidad'] == 1].dropna().sample(300)])

print(len(dummie_df),len(dummie_df[dummie_df['responsabilidad'] == 0]),len(dummie_df[dummie_df['responsabilidad'] == 1]))
# dummie_df = dummie_df.dropna()
location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']

for loc in location:
    aux = []
    for i in range(len(df['ubicacion_vial'])):
        try:
            if re.search(loc, df.loc[i, 'ubicacion_vial']):
                aux.append(1)
            else:
                aux.append(0)
        except:
            aux.append(0)
    dummie_df[loc] = pd.Series(aux)

clf=KNeighborsClassifier(n_neighbors=8,weights='distance')

x_train, x_test, y_train, y_test = train_test_split(dummie_df[[col for col in dummie_df.columns if col != 'responsabilidad']], dummie_df['responsabilidad'], test_size=0.2, random_state=7)
print(x_test.shape,y_test.shape)

clf.fit(x_train, y_train)
print('Precisión con un KNeighborsClassifier: ', clf.score(x_test, y_test))
dump(clf,'model_kneighbors.pkl')
# x, y = [], []
# for n in range(2, 17):
#     x.append(n)
#     print('n = ', n)
#     clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
#     x_train, x_test, y_train, y_test = train_test_split(dummie_df[[col for col in dummie_df.columns if col != 'responsabilidad']], dummie_df['responsabilidad'], test_size=0.2, random_state=7)
#     clf.fit(x_train, y_train)
#     print('Precisión con un KNeighborsClassifier: ', clf.score(x_test, y_test))
#     y.append(clf.score(x_test, y_test))

# plt.plot(x, y, '-')

# plt.title('n_neighbors vs. score')
# plt.savefig('tmp/neighbors_8.png')


dummie_df.to_csv('tmp/Tabla1_dummie.csv')
