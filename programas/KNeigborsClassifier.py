import pandas as pd
from objetos import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
np.random.seed(7)

"""
Preparacion del dataset
"""
df = pd.read_csv('tmp/Tabla1.csv')
dummie_df = pd.get_dummies(df[['impac_position', 'quien', 'movimiento']])
aux = []
for i in df['responsabilidad']:
    if i == 'COMPROMETIDA' or i == 'DISCUTIDA':
        aux.append(1)
    else:
        aux.append(0)
dummie_df['responsabilidad'] = pd.Series(aux)
dummie_df = pd.concat([dummie_df[dummie_df['responsabilidad'] == 0], dummie_df[dummie_df['responsabilidad'] == 1].sample(1100)])
location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']

for loc in location:
    aux = []
    for i in range(len(df['ubicacion_vial'])):
        if re.search(loc, df.loc[i, 'ubicacion_vial']):
            aux.append(1)
        else:
            aux.append(0)
    dummie_df[loc] = pd.Series(aux)
print(dummie_df.columns)

x, y = [], []
for n in range(2, 17):
    x.append(n)
    print('n = ', n)
    clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
    x_train, x_test, y_train, y_test = train_test_split(dummie_df[[col for col in dummie_df.columns if col != 'responsabilidad']], dummie_df['responsabilidad'], test_size=0.2, random_state=7)
    clf.fit(x_train, y_train)
    print('Precisión con un KNeighborsClassifier: ', clf.score(x_test, y_test))
    y.append(clf.score(x_test, y_test))

plt.plot(x, y, '-')

plt.title('n_neighbors vs. score')
plt.savefig('tmp/best_n.png')


dummie_df.to_csv('tmp/Tabla1_dummie.csv')
