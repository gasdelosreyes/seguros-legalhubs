# estadistica de palabras por palabra
# funcion para graficar nube de palabras
#
# %%
from cleaner import *
from cluster import *
from wordcloud import WordCloud
import collections


def plotHist(dic, nombre, sub_folder=''):
	"dic = { clave:str, valor:int }"
	plt.clf()
	plt.title(nombre)
	plt.xlabel('%')
	plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
	plt.barh(list(dic.keys())[:20], [i * 100 for i in list(dic.values())[:20]])
	plt.savefig(sub_folder + 'hist_' + nombre + '.png')


def plotWordCloud(text, nombre, sub_folder=''):
	wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(nombre)
	plt.savefig(sub_folder + 'wordcloud_' + nombre + '.png')


def calcWeight(descripcion, x, y):
	if 'delantera' in descripcion:
		y += 4
	if 'trasera' in descripcion:
		y -= 4
	if 'izquierda' in descripcion:
		x -= 2
	if 'derecho' in descripcion or 'derecha' in descripcion:
		x += 2
	return x, y


def set_pie(responsabilidad):
	labels = list(collections.Counter(responsabilidad).keys())
	sizes = list(collections.Counter(responsabilidad).values())
	colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
	return labels, sizes, colors


def plotCar(df, file_name, clusters_list=range(8), show=1):
	plt.clf()
	plt.figure(figsize=(40, 40))
	x_final, y_final = np.array([]), np.array([])
	new_df = {'x': [], 'y': [], 'descripcion': [], 'cluster': []}
	resp = []
	delantera, trasera, izquierda, derecha = [], [], [], []
	delantera_derecha, delantera_izquierda = [], []
	trasera_derecha, trasera_izquierda = [], []

	for j in clusters_list:
		x, y = [], []
		for i in range(len(df)):
			if df.loc[i, 'cluster'] in [j]:
				x.append(df.loc[i, 'x'])
				y.append(df.loc[i, 'y'])

		x = np.array(x)
		y = np.array(y)
		cm_x = np.mean(x)
		cm_y = np.mean(y)
		for i in range(len(df)):
			if df.loc[i, 'cluster'] in [j]:
				xw, yw = calcWeight(df.loc[i, 'descripcion'], df.loc[i, 'x'] - cm_x, df.loc[i, 'y'] - cm_y)

				if -1 < xw < 1:
					if 2 < yw:
						delantera.append(df.loc[i, 'responsabilidad'])
					elif yw < -2:
						trasera.append(df.loc[i, 'responsabilidad'])
				elif xw < -1:
					if yw < -2:
						trasera_izquierda.append(df.loc[i, 'responsabilidad'])
					elif 2 < yw:
						delantera_izquierda.append(df.loc[i, 'responsabilidad'])
					elif -2 < yw < 2:
						izquierda.append(df.loc[i, 'responsabilidad'])
				elif 1 < xw:
					if yw < -2:
						trasera_derecha.append(df.loc[i, 'responsabilidad'])
					elif 2 < yw:
						delantera_derecha.append(df.loc[i, 'responsabilidad'])
					elif -2 < yw < 2:
						derecha.append(df.loc[i, 'responsabilidad'])

				new_df['descripcion'].append(df.loc[i, 'descripcion'])
				new_df['cluster'].append(j)
				new_df['x'].append(xw)
				new_df['y'].append(yw)

				x_final = np.append(x_final, xw)
				y_final = np.append(y_final, yw)
				resp.append(df.loc[i, 'responsabilidad'])

	img = plt.imread('auto.png')
	dimension = (13, 9)
	centro = plt.subplot2grid(dimension, (3, 2), colspan=4, rowspan=7)
	centro.axis('off')
	centro.imshow(img, extent=[-2.25, 2.25, -4.5, 4.5])
	# centro.figure(figsize=(20, 20))
	for i in range(len(resp)):
		if resp[i] == 'COMPROMETIDA' or resp[i] == 'CONCURRENTE':
			centro.plot(x_final[i], y_final[i], '.', alpha=0.2, markersize=15, color='#4D4CD0')
		elif resp[i] == 'SIN RESPONSABILIDAD' or resp[i] == 'DISCUTIDA':
			centro.plot(x_final[i], y_final[i], 'rx', markersize=18)

	labels, sizes, colors = set_pie(delantera_izquierda)
	delantera_izquierda = plt.subplot2grid(dimension, (0, 0), colspan=2, rowspan=3)
	delantera_izquierda.axis('off')
	delantera_izquierda.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = delantera_izquierda.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Delantera izquierda', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(delantera)
	delantera = plt.subplot2grid(dimension, (0, 3), colspan=2, rowspan=3)
	delantera.axis('off')
	delantera.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = delantera.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Delantera', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(delantera_derecha)
	delantera_derecha = plt.subplot2grid(dimension, (0, 6), colspan=2, rowspan=3)
	delantera_derecha.axis('off')
	delantera_derecha.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = delantera_derecha.pie(sizes, colors=colors, shadow=True, startangle=180, textprops={'size': 30})
	plt.title('Delantera derecha', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(izquierda)
	izquierda = plt.subplot2grid(dimension, (5, 0), colspan=2, rowspan=3)
	izquierda.axis('off')
	izquierda.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = izquierda.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Izquierda', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(derecha)
	derecha = plt.subplot2grid(dimension, (5, 6), colspan=2, rowspan=3)
	derecha.axis('off')
	derecha.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = derecha.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Derecha', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(trasera_izquierda)
	trasera_izquierda = plt.subplot2grid(dimension, (10, 0), colspan=2, rowspan=3)
	trasera_izquierda.axis('off')
	trasera_izquierda.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = trasera_izquierda.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Trasera izquierda', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(trasera)
	trasera = plt.subplot2grid(dimension, (10, 3), colspan=2, rowspan=3)
	trasera.axis('off')
	trasera.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = trasera.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Trasera', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	labels, sizes, colors = set_pie(trasera_derecha)
	trasera_derecha = plt.subplot2grid(dimension, (10, 6), colspan=2, rowspan=3)
	trasera_derecha.axis('off')
	trasera_derecha.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, textprops={'size': 40})
	# patches, texts = trasera_derecha.pie(sizes, colors=colors, shadow=True, startangle=180)
	plt.title('Trasera derecha', fontsize=30)
	# plt.legend(patches, labels, loc="best", fontsize=20)

	out_df = pd.DataFrame(new_df)

	if show:
		plt.show()
		return out_df
	else:
		plt.savefig(file_name + '.png')
		return out_df


def gram2string(gram):
	if type(gram) == list:
		return gram
	if re.search(r'\), \(', gram):
		gram = gram.split('), (')
		for i in range(len(gram)):
			gram[i] = re.sub(r'\[', '', gram[i])
			gram[i] = re.sub(r'\(', '', gram[i])
			gram[i] = re.sub(r'\]', '', gram[i])
			gram[i] = re.sub(r'\)', '', gram[i])
			gram[i] = re.sub(' ', '', gram[i])
			gram[i] = re.sub('\'', '', gram[i])
			gram[i] = gram[i].strip()
		gram = [i.split(',') for i in gram]
		return gram
	else:
		gram = re.sub(r'\[', '', gram)
		gram = re.sub(r'\(', '', gram)
		gram = re.sub(r'\]', '', gram)
		gram = re.sub(r'\)', '', gram)
		gram = re.sub(' ', '', gram)
		gram = re.sub('\'', '', gram)
		gram = gram.strip()
	return gram.split(',')


def identLocation(descripcion):
	location = ['calle', r'gara\w*', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze',
             r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*']
	aux = []
	for loc in location:
		st = re.search(loc, descripcion)
		if st:
			aux.append(st.group())
			# print(st.group())
	if aux:
		return ' '.join(set(aux))
	return 'desconocido'


def conectOrigen(df, i):
	return df.loc[i, 'descripcion']


"""
df = pd.read_csv('../dataset/casos/auto.csv')
# for i in range(len(df['cluster'])):
# 	if df.loc[i, 'cluster'] == 7:
# 		df.loc[i, 'cluster'] = 3
# 	if df.loc[i, 'cluster'] == 5:
# 		df.loc[i, 'cluster'] = 2

# for i in range(len(df['cluster'])):
# 	for j in df['cluster'].unique():
# 		if df.loc[i, 'cluster'] == j:
# 			df.loc[i, 'label'] = df['descripcion'][df['cluster'] == j].value_counts().keys()[0]
"""


# ['asegurado,parte,delantera'], ['asegurado,parte,trasera']  hay que
# alargarlo en ambos lados y reclusterizar para asegurarse que
# se refiere al asegurado y no al tercero idx:599,103
# capaz sea mejor tomar un regex para el recluster
# r'asegurado .*? tercero'
# r'colisiona .*? asegurado .*? tercero'
# r'colisiona .*? tercero .*? asegurado'
# en los casos de arriba "vehiculo asegurado colisiona parte delantera izquierda vehiculo tercero"
# hay que alargar todo los trigramas y reclusterizar porque cortan mucha información
# %%
if __name__ == '__main__':

	df = pd.read_csv('recluster_lado_aseg_no-overlap.csv')
	plotCar(df, 'colision_distribution', clusters_list=range(15), show=0)
	# df2 = pd.read_csv('../dataset/casos/auto.csv')
	# df['descripcion'] = df['descripcion'].apply(gram2string)
	# df['origen'] = df['idx_descripcion'].apply(lambda x: conectOrigen(df2, x))
	# df['ubicacion_vial'] = df['origen'].apply(identLocation)
	# df.to_csv('recluster_lado_aseg_no-overlap_vial.csv')
	# df
	# # %%
	# if '__main__' == __name__:
	# 	df = pd.read_csv('recluster_no-overlap.csv')
	# 	df2 = pd.read_csv('../dataset/casos/auto.csv')
	# 	df['descripcion'] = df['descripcion'].apply(gram2string)
	# 	df['ubicacion_vial'] = df['origen'].apply(identLocation)
	# 	df.to_csv('cluster_dataset-final_con-orig.csv')
	# 	df
	# original = []

	# for i in range(len(df)):
	# 	original.append(df2.loc[df.loc[i, 'idx_descripcion'], 'descripcion'])
	# df['original'] = pd.Series(original)
	# df.to_csv('cluster_dataset-final_con-orig.csv')
