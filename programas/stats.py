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
	plt.barh(list(dic.keys())[:20], [i * 100 for i in list(dic.values())[:20]])
	plt.savefig(sub_folder + 'hist_' + nombre + '.png')


def plotWordCloud(text, nombre, sub_folder=''):
	wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(nombre)
	plt.savefig(sub_folder + 'wordcloud_' + nombre + '.png')


df = pd.read_csv('../dataset/casos/auto.csv')
responsabilidades = ['COMPROMETIDA', 'CONCURRENTE', 'SIN RESPONSABILIDAD', 'DISCUTIDA']
for responsabilidad in responsabilidades:
	resp = list(map(word_tokenize, df['descripcion'][df['responsabilidad'] == responsabilidad]))
	fstring = []
	for i in resp:
		fstring += i
	resp = pd.Series(fstring)
	plotHist(dict(resp.value_counts(normalize=True)), responsabilidad + '1', sub_folder='plots/')
	plotWordCloud(' '.join(fstring), responsabilidad + '1', sub_folder='plots/')
