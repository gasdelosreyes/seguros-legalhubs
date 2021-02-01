# estadistica de palabras por palabra
# funcion para graficar nube de palabras
#

from cleaner import *
from cluster import *
from wordcloud import WordCloud


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


def plotCar(df, file_name, clusters_list=range(8), show=1):
	plt.clf()
	x_final, y_final = np.array([]), np.array([])
	new_df = {'x': [], 'y': [], 'descripcion': [], 'cluster': []}
	resp = []

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

				new_df['descripcion'].append(df.loc[i, 'descripcion'])
				new_df['cluster'].append(j)
				new_df['x'].append(xw)
				new_df['y'].append(yw)

				x_final = np.append(x_final, xw)
				y_final = np.append(y_final, yw)
				resp.append(df.loc[i, 'responsabilidad'])
	img = plt.imread('auto.png')
	fig, ax = plt.subplots()
	ax.imshow(img, extent=[-2.25, 2.25, -4.5, 4.5])
	for i in range(len(resp)):
		if resp[i] == 'COMPROMETIDA' or resp[i] == 'CONCURRENTE':
			ax.plot(x_final[i], y_final[i], 'b.', alpha=0.2)
		elif resp[i] == 'SIN RESPONSABILIDAD' or resp[i] == 'DISCUTIDA':
			ax.plot(x_final[i], y_final[i], 'rx')

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
	location = ['calle', r'gara.*?', 'rotonda', 'autopista', 'au', 'avenida', 'cruce', 'cruze', 'esquina', r'estaciona.*?', 'carril', 'ruta']
	aux = []
	for loc in location:
		if re.search(loc, descripcion):
			aux.append(loc)
	if aux:
		return aux
	return 'desconocido'


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
if '__main__' == __name__:
	plotCar(pd.read_csv('lado_aseg_dataset-final_filtro-nuevo.csv'), 'colision_distribution_filtro-nuevo', clusters_list=[0, 2, 3, 4, 5, 6, 7], show=0)
	# df = pd.read_csv('lado_aseg_dataset-final.csv')
	# df2 = pd.read_csv('../dataset/casos/auto.csv')
	# df = plotCar(df, 'colision_distribution2', clusters_list=range(1, 8), show=1)
	# sys.exit()
	# ubicacion = []
	# lateral = []
	# for i in range(len(df3)):
	# 	if df3.loc[i, 'y'] < 0 and round(df3.loc[i, 'y']) != 0:
	# 		ubicacion.append('trasera')
	# 	elif round(df3.loc[i, 'y']) == 0:
	# 		ubicacion.append('centro')
	# 	else:
	# 		ubicacion.append('delantera')

	# 	if df3.loc[i, 'x'] < 0 and round(df3.loc[i, 'x']):
	# 		lateral.append('izquierda')
	# 	elif round(df3.loc[i, 'x']) == 0:
	# 		lateral.append('centro')
	# 	else:
	# 		lateral.append('derecha')
	# df3['ubicacion'] = pd.Series(ubicacion)
	# df3['lateral'] = pd.Series(lateral)
	# df['descripcion'] = df['descripcion'].apply(gram2string)
	# df['location'] = df2['descripcion'].apply(identLocation)
	# df.to_csv('cluster_dataset-final_con-orig.csv')

	# original = []

	# for i in range(len(df)):
	# 	original.append(df2.loc[df.loc[i, 'idx_descripcion'], 'descripcion'])
	# df['original'] = pd.Series(original)
	# df.to_csv('cluster_dataset-final_con-orig.csv')
