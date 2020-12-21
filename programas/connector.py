# %%
import numpy as np
import pandas as pd

from functions import *
from matplotlib import pyplot as plt

# %%
df = pd.read_csv('../dat/cluste_contex_descrip.csv')

# ahora para cada descrićion vamos a busacar los clusters a los que pertenence
clus = []
for i, desc in enumerate(df['descripcion']):
	clusters = []
	clusters.append(df.loc[i, 'cluster'])
	for j, desc2 in enumerate(df['descripcion']):
		if desc == desc2 and j != i:
			clusters.append(df.loc[j, 'cluster'])
	clus.append(list(set(clusters)))
df['campos'] = clus
df.to_csv('../dat/calsificacion_final.csv')

# ahora vamos a agreagar la responsabilidad a cada set de descripciones

responsabilidad = pd.read_csv('../dat/auto.csv')
responsabilidad['descripcion_del_hecho - Final'] = responsabilidad['descripcion_del_hecho - Final'].apply(
    cleaner)
for i, w in enumerate(responsabilidad['descripcion_del_hecho - Final']):
    w = w.replace('izquierdo', 'izquierda')
    w = w.replace('derecho', 'derecha')
    w = w.replace('delantero', 'delantera')
    w = w.replace('trasero', 'trasera')
    responsabilidad.loc[i, 'descripcion_del_hecho - Final'] = w
resp = responsabilidad.drop(columns='cod_accidente')

# convertimos la lista de descripciones en list()
for i in range(len(df['descripcion'])):
	df['descripcion'][i] = df['descripcion'][i].replace('[', '')
	df['descripcion'][i] = df['descripcion'][i].replace(']', '')
	df['descripcion'][i] = df['descripcion'][i].replace('\'', '')
	df['descripcion'][i] = df['descripcion'][i].replace('\'', '')
	df['descripcion'][i] = df['descripcion'][i].split(', ')

# para cada descripcion buscamoms la responsabilidad
res_tot = []
for i, d in enumerate(df['descripcion'][:]):
	# d = lista de descripciones
	set_res = []
	for j, d1 in enumerate(d):
		# d1 = descripcion j-esima de d
		for k, r in enumerate(resp['descripcion_del_hecho - Final']):
			# r = descripcion con responsabilidad
			if d1 == r:
				# print(resp.loc[k, 'responsabilidad'],k)
				set_res.append(resp.loc[k, 'responsabilidad'])
				# print(set_res,j)
	res_tot.append(','.join(list(set(set_res))))

df['responsabilidad'] = res_tot

#df = pd.concat([df, pd.get_dummies(df['responsabilidad'])])

COMPROMETIDA_x, COMPROMETIDA_y = df['x'][df['responsabilidad'] == 'COMPROMETIDA'], df['y'][df['responsabilidad'] == 'COMPROMETIDA']

CONCURRENTE_x, CONCURRENTE_y = df['x'][df['responsabilidad'] == 'CONCURRENTE'], df['y'][df['responsabilidad'] == 'CONCURRENTE']

DISCUTIDA_x, DISCUTIDA_y = df['x'][df['responsabilidad'] == 'DISCUTIDA'], df['y'][df['responsabilidad'] == 'DISCUTIDA']

# otros_x, otros_y = df['x'][df['responsabilidad'] == df.columns[10]], df['y'][df['responsabilidad'] == df.columns[10]]
# otros_x1, otros_y1 = df['x'][df['responsabilidad'] == df.columns[13]], df['y'][df['responsabilidad'] == df.columns[13]]
# otros_x2, otros_y2 = df['x'][df['responsabilidad'] == df.columns[14]], df['y'][df['responsabilidad'] == df.columns[14]]
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.grid()

ax.scatter(COMPROMETIDA_x, COMPROMETIDA_y, color='xkcd:sky blue', label='COMPROMETIDA')
ax.scatter(CONCURRENTE_x, CONCURRENTE_y, color='r', label='CONCURRENTE')
ax.scatter(DISCUTIDA_x, DISCUTIDA_y, color='g', label='DISCUTIDA')
# ax.scatter(otros_x, otros_y, color='black', alpha=0.3, label='COMBINACION')
# ax.scatter(otros_x1, otros_y1, color='black', alpha=0.3)
# ax.scatter(otros_x2, otros_y2, color='black', alpha=0.3)
ax.legend()
# plt.savefig('../dat/plots/cluster_resp.png')

df = df.drop(columns='Unnamed: 0')
# %%
# realizamos ahora un claculo estadístico para saber que porcentaje de responsabilidad tenemos por cluster
cluster_stat = []
for cluster in range(8):
	percent = {}
	val = df['responsabilidad'][df['cluster'] == cluster].values
	percent['cluster'] = cluster
	percent['stats'] = {'responsabilidad': [], 'percent': []}
	for i in list(set(val)):
		count = 0
		for j in range(len(val)):
			if i == val[j]:
				count += 1
		percent['stats']['responsabilidad'].append(i)
		percent['stats']['percent'].append(round(count * 100 / len(val), 2))
	cluster_stat.append(percent)
# %%
fig = plt.figure(figsize=(15, 15))
for i in range(len(cluster_stat)):
	fig.add_subplot(240 + i + 1)
	plt.bar(cluster_stat[i]['stats']['responsabilidad'], cluster_stat[i]['stats']['percent'])
	plt.title('Cluster: ' + str(cluster_stat[i]['cluster']))
	plt.xticks(rotation=20)

# plt.savefig('../dat/plots/clusters_resp_stats.png')
