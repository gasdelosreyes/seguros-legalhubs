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
dummie_df = pd.DataFrame()
# dummie_df = pd.get_dummies(df[['quien']])

mov = []

for row in df['movimiento']:
    if row == 'si':
        mov.append(1)
    else:
        mov.append(0)
print(len(mov))

dummie_df['movimiento'] = pd.Series(mov)

delantera,delantera_derecha,delantera_izquierda,trasera,trasera_derecha,trasera_izquierda = [],[],[],[],[],[]
for row in df['impac_position']:
    if row == 'delantera':
        delantera.append(1)
    if row == 'delantera izquierda':
        delantera.append(2)
    if row == 'delantera derecha':
        delantera.append(3)
    if row == 'trasera':
        delantera.append(4)
    if row == 'trasera izquierda':
        delantera.append(5)
    if row == 'trasera derecha':
        delantera.append(6)

dummie_df['delantera'] = pd.Series(delantera)
# dummie_df['delantera_izquierda'] = pd.Series(delantera_izquierda)
# dummie_df['delantera_derecha'] = pd.Series(delantera_derecha)
# dummie_df['trasera'] = pd.Series(trasera)
# dummie_df['trasera_izquierda'] = pd.Series(trasera_izquierda)
# dummie_df['trasera_derecha'] = pd.Series(trasera_derecha)
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
dummie_df = dummie_df.dropna()
print(dummie_df.head())

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
