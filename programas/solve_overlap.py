from cluster import *
from cleaner import *
from stats import *


def compareLists(a: list, b: list) -> int:
	"""
	retorna 1 si son iguales, sino 0
	"""
	if (len(a) and not len(b)) or (len(b) and not len(a)):
		return 0
	elif not len(a) and not len(b):
		return 1
	elif a[0] == b[0]:
		return compareLists(a[1:], b[1:])
	else:
		return 0


def isInside(a: list, b: list) -> int:
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

	if len(a) == 1 and type(a[0]) == list:
		return a[0], b[0]
	for i in range(len(a) - 1):
		if len(a[i]) < len(a[i + 1]):
			max_len = a[i + 1]
			b_max = b[i + 1]
		if len(a[i]) == len(a[i + 1]):
			max_len = a[-1]
			b_max = b[-1]
	return max_len, b_max


def solveOverlap(df):
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
			if df.loc[i, idx_descripcion] == j and df.loc[i, cluster] != 0:
				pares[x].append(df.loc[i, x])
				pares[y].append(df.loc[i, y])
				pares[descripcion].append(df.loc[i, descripcion])
				pares[cluster].append(df.loc[i, cluster])
				pares[responsabilidad].append(df.loc[i, responsabilidad])

		if len(pares[descripcion]):
			max_desc = maxLen(pares[descripcion], pares[cluster])[0]
			max_clust = maxLen(pares[descripcion], pares[cluster])[1]
			for i in range(len(pares[descripcion])):
				if compareLists(pares[descripcion][i], max_desc):
					final[x].append(pares[x][i])
					final[y].append(pares[y][i])
					final[idx_descripcion].append(j)
					final[descripcion].append(pares[descripcion][i])
					final[cluster].append(pares[cluster][i])
					final[responsabilidad].append(pares[responsabilidad][i])

	return pd.DataFrame(final)


df = pd.read_csv('lado_aseg_dataset-final2.csv')
df_max = solveOverlap(df)
df_max.to_csv('lado_aseg_dataset-final2_no-overlap.csv')
