# %%
from cluster import *
from cleaner import *
from stats import *


def compareLists(a, b):
	"""
	retorna 1 si son iguales, sino 0
	"""
	if (len(a) and not len(b)) or (len(b) and not len(a)):
		return False
	elif not len(a) and not len(b):
		return True
	elif a[0] == b[0]:
		return compareLists(a[1:], b[1:])
	else:
		return False


def isInside(a, b):
	"""
	retorna 1 si a esta dentro de b
	retorna 0 si no es igual o no esta dentro de b
	"""
	if not len(b) and len(a):
		return 0
	elif len(b) and not len(a):
		return 1
	elif not len(a) and not len(b):
		return 1
	elif a[0] == b[0]:
		return isInside(a[1:], b[1:])
	else:
		return 0


def maxLen(a, b):
	max_len = None
	b_max = None
	if not len(a):
		return
	for i in range(len(a)):
		if type(a[i][0]) == list:
			return maxLen(a[i], b)

	if len(a) == 1 and type(a[0]) == list:  # [[as,asd,asd]]
		return a[0], b[0]
	for i in range(len(a) - 1):
		if len(a[i]) < len(a[i + 1]):
			try:
				max_len = a[i + 1]
				b_max = b[i + 1]
			except:
				max_len = a[i + 1]
				b_max = b[-1]
		if len(a[i]) == len(a[i + 1]):
			max_len = a[-1]
			b_max = b[-1]
	return max_len, b_max


def resolveLists(l):
	"""
	recibe listas sub anidadas y devuelve lista de listas
	[[[1, 2, 3], [2, 3, 4]], [4,5,6]] == [ [1,2,3], [2,3,4], [4,5,6] ]
	"""

	if type(l[0]) != list:
		return [l]
	aux = []
	for i in range(len(l)):
		aux += resolveLists(l[i])
	return aux


def solveOverlap(df, cluster_trash):
	"""
	df: DF de entrada que corresponde a los casos clasificados
	final: DF de salida que corresponde a los gramas no solapados y clusterizado
	"""
	columns = list(df.columns)
	x = columns[0]
	y = columns[1]
	idx_descripcion = columns[2]
	descripcion = columns[3]
	cluster = columns[4]
	responsabilidad = columns[5]

	df[descripcion] = df[descripcion].apply(gram2string)  # el nombre gram2string esta al reves
	idx_desc = df[idx_descripcion].sort_values().unique()

	final = {x: [], y: [], idx_descripcion: [], descripcion: [], cluster: [], responsabilidad: []}

	for j in idx_desc:
		pares = {x: [], y: [], idx_descripcion: [], descripcion: [], cluster: [], responsabilidad: []}

		for i in range(len(df)):
			if df.loc[i, idx_descripcion] == j and df.loc[i, cluster] != cluster_trash:
				pares[x].append(df.loc[i, x])
				pares[y].append(df.loc[i, y])
				pares[descripcion].append(df.loc[i, descripcion])
				pares[cluster].append(df.loc[i, cluster])
				pares[responsabilidad].append(df.loc[i, responsabilidad])

		if len(pares[descripcion]):
			pares[descripcion] = resolveLists(pares[descripcion])
			max_desc = maxLen(pares[descripcion], pares[cluster])[0]
			for i in range(len(pares[descripcion])):
				if compareLists(pares[descripcion][i], max_desc):
					try:
						final[x].append(pares[x][i])
						final[y].append(pares[y][i])
						final[idx_descripcion].append(j)
						final[descripcion].append(pares[descripcion][i])
						final[cluster].append(pares[cluster][i])
						final[responsabilidad].append(pares[responsabilidad][i])
					except IndexError:
						final[x].append(pares[x][-1])
						final[y].append(pares[y][-1])
						final[idx_descripcion].append(j)
						final[descripcion].append(pares[descripcion][i])
						final[cluster].append(pares[cluster][-1])
						final[responsabilidad].append(pares[responsabilidad][-1])

	return pd.DataFrame(final)


df = pd.read_csv('lado_aseg.csv')
df_max = solveOverlap(df, cluster_trash=0)
df_max.to_csv('lado_aseg_no-overlap.csv')


file_name = 'recluster_lado_aseg_no-overlap'
tokenizedCorpus = df_max['descripcion']
bowCorpus = [TaggedDocument([' '.join(tup) for tup in words], [idx_tag]) for idx_tag, words in
             enumerate(tokenizedCorpus)]

fixSeed = 774965317
np.random.seed(fixSeed)

model = Doc2Vec(bowCorpus, vector_size=300, seed=fixSeed, dm=1, epochs=1000)
modelVectors = model.docvecs.vectors_docs

pca = PCA(n_components=2, svd_solver='full', whiten=True)
pcaVectors = pca.fit_transform(modelVectors)


n = 1
n_clusters = 0
aux = []
while n_clusters < 0.8:
	n += 1
	kmeans = KMeans(n_clusters=n, random_state=fixSeed)
	kmeans.fit(pcaVectors)
	n_clusters = silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean')
	aux.append([n_clusters, n])
	if 16 <= n:
		break

pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])


kmeans = KMeans(n_clusters=max(aux)[1], random_state=fixSeed)
kmeans.fit(pcaVectors)

pca_df = pd.DataFrame(data=pcaVectors, columns=['x', 'y'])
pca_df['idx_descripcion'] = pd.Series([i for i in df_max['idx_descripcion']])

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
print('silhouette_score = ', silhouette_score(pcaVectors, kmeans.labels_, metric='euclidean'))
ax.grid()
ax.scatter(pca_df.x, pca_df.y, s=15)
plt.savefig(file_name + '.png')
pca_df['descripcion'] = pd.Series(tokenizedCorpus)
pca_df['cluster'] = pd.Series(kmeans.labels_)
sns.lmplot('x', 'y', data=pca_df, hue='cluster', fit_reg=False)
plt.savefig(file_name + '_sns.png')
pca_df['responsabilidad'] = df['responsabilidad']
pca_df.to_csv(file_name + '.csv', index=False)

# plotCar(pca_df, 'recluster_colision_distribution', clusters_list=range(10), show=1)
