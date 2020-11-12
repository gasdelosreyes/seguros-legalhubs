import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import swadesh
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

'''
    Create a dictionary
'''


# Elimina palabras que no sirven y las stopwords.


def Dictionary():
    dic = ['vh', 'amb', 'av', 'avenida', 'ambulancia', 'cruza', 'calle',
           'policia', 'impacto', 'observo', 'conductor', 'rotonda', 'casco']
    for i in stopwords.words('spanish'):
        if i != 'derecha' and i != 'izquierda':
            dic.append(i)
    for i in swadesh.words('es'):
        if i != 'derecha' and i != 'izquierda':
            dic.append(i)
    return dic


# Elimina las palabras del words, text es un string. Falta que busque 'interv'


def deleteExtraData(text):
    words = ['interv', 'amplia', 'formal']
    for w in words:
        if w in text:
            text = text.split(w)[0]
    return text


# Retorna una lista de las palabras cuya frecuencia es menor a 'value'
# si value=20 retorna la frecuencia de las de 10 dentro de las de 20.


def deleteFrequence(freq, tokens, value):
    deletetokens = []
    for w in tokens:
        if freq[w] <= value:
            # text = re.sub('\bw\b', '',text)
            deletetokens.append(w)
    # print(deletetokens)
    return deletetokens


# Tokeniza y después te muestra las 30 más frecuentes.


def tokenization(text):
    words = word_tokenize(text)
    freq = nltk.FreqDist(words)
    print(freq.most_common(30))


# Devuelve la palabra más cercana según el algoritmo de levenshtein


def max_ratio(w):
    try:
        dic = ['parte', 'circulaba', 'delantera', 'delantero', 'delante', 'enfrente', 'izquierda', 'izquierdo',
               'derecha', 'lateral',
               'colisiona', 'colision', 'colisiono', 'colisionada', 'colisionado', 'colisionando', 'trasero', 'trasera',
               'impacta',
               'impacto', 'impactado', 'impactada', 'embiste', 'embistio', 'embestido', 'iba', 'venia', 'frente',
               'lado', 'detras',
               'atras', 'costado', 'choca', 'choco', 'chocando', 'choque', 'costado', 'frontal']

        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.ratio(w, i) and 80 <= fuzz.ratio(w, i)) or 90 <= fuzz.partial_ratio(w, i):
                aux = max(fuzz.ratio(w, i), fuzz.partial_ratio(w, i))
                word = i
        # print(w,word)
        if word != '':
            return word
        return w
        # return w hay que poner para usar, mientras me sirve para evaluar
    except TypeError:
        return w


def max_ratio_anti(w):
    try:
        dic = ['asegurado', 'cae', 'venia', 'piso', 'aseg', 'lesiones', 'medios', 'vehiculo', 'ocupante', 'persona',
               'datos', 'propios', 'desplazamientos', 'solo', 'llegar', 'puerta', 'cayo', 'ultima', 'hecho', 'caen',
               'maniobra', 'acompanante', 'segun', 'pavimento', 'hospital', 'espejo', 'ambos', 'habia', 'suelo',
               'tenia', 'frena', 'mecanica', 'levanta', 'ocupantes', 'momento', 'dolor', 'velocidad', 'version',
               'personas', 'san', 'asfalto', 'marcha', 'llevaba', 'retira', 'mismo', 'sola', 'produce', 'ingresar',
               'puesto', 'trasladado', 'luz', 'presentaba', 'retiro', 'maniobro', 'tomar', 'asfalto']

        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.ratio(w, i) and 80 <= fuzz.ratio(w, i)) or 90 <= fuzz.partial_ratio(w, i):
                aux = max(fuzz.ratio(w, i), fuzz.partial_ratio(w, i))
                word = i
        # print(w,word)
        if word != '':
            return word
        return w
        # return w hay que poner para usar, mientras me sirve para evaluar
    except TypeError:
        return w


col = 'descripcion_del_hecho - Final'

dataframe = pd.read_excel('../dataset/casos_universidad.xlsx')
# len = 1171 rows x 12 columns
# dataframe = pd.DataFrame(openfile)


'''
    Getting the columns that needs to be processed
'''
dataframe = dataframe.iloc[:, [5, 11, 7]]
# len = 1171 rows x 3 columns

# guardamos solo las columnas de interes
dataframe.to_csv('../dataset/casos_filtrados.csv', index=False, header=True)

dataset = pd.read_csv('../dataset/casos_filtrados.csv')
# len = 1249 rows x 3 columns PRLOBLEMS!!
# Algo esta pasando en el to_csv porque esta metiendo comprometida en la columna
# de descripcion del hecho.

# esto lo soluciona, en principio
del_row = []
for i, row in enumerate(dataset['descripcion_del_hecho - Final']):
    if type(row) != str or row.lower() == 'comprometida' or row == '' or row == ' ':
        del_row.append(i)

dataset = dataset.drop(del_row)

# todo a minuscula
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

# countedWords = []

# for desc in dataset['descripcion_del_hecho - Final']:
#     countedWords.append(len(desc))

'''
    Apply regular expression
       ' [^\w\s] '-> Start with alphanumeric and allow whitespace
        '(\r\r?|\n) '-> Delete line breaks
        \d+ -> Delete all the numbers
'''
# Este segmento limpia las palabras de los hexadecimales y borra las palabras con Dictionary().
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('[^\w\s]', '', x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('(\r\n?|\n)+', '', x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('\d+', '', x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in Dictionary())
)
# le cambié el separador por una ',' simple :)
dataset.to_csv('../dataset/casos_filtrados_regex.csv',
               index=False, header=True)

for row in dataset['descripcion_del_hecho - Final']:
    dataset.loc[dataset['descripcion_del_hecho - Final'] == row,
                'descripcion_del_hecho - Final'] = deleteExtraData(row)

# le cambié el separador por una ',' simple :)
dataset.to_csv('../dataset/casos_filtrados_cut.csv',
               index=False, header=True)

# si vamos a hacer estadística sobre todas las palabras y tokenizarlas el ; no es necesario.11/11/2020
# estoy teniendo problemas porque en el csv me esta agregando lineas en la columna de descripcion del hecho
fstring = ''
for row in dataset['descripcion_del_hecho - Final']:
    fstring += row + ';'

tokens = word_tokenize(fstring)

nltkText = Text(tokens)

f = FreqDist(nltkText)

# fstring = deleteFrequence(f, nltkText, fstring, 30)


# '''
#     Frequences less than 10
#     Frequences less than 20
# '''
# deletetokens = []
# twenty = []
# for w in nltkText:
#     if(f[w] <= 10):
#         deletetokens.append(w)
#     if(f[w] > 10 and f[w] <= 20):
#         twenty.append(w)
# print(deletetokens)
# print(twenty)
dataset.insert(2, 'frecuencias-10', '')
dataset.insert(3, 'frecuencias-20', '')
dataset.insert(4, 'frecuencias-30', '')
dataset.insert(5, 'frecuencias-40', '')
dataset.insert(6, 'frecuencias-50', '')

f10 = deleteFrequence(f, nltkText, 10)
f20 = deleteFrequence(f, nltkText, 20)
f30 = deleteFrequence(f, nltkText, 30)
f40 = deleteFrequence(f, nltkText, 40)
f50 = deleteFrequence(f, nltkText, 50)

dataset['frecuencias-10'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in f10)
)

dataset['frecuencias-20'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in f20)
)

dataset['frecuencias-30'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in f30)
)

dataset['frecuencias-40'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in f40)
)

dataset['frecuencias-50'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in f50)
)

dataset.to_csv('../dataset/casos_filtrados_frequences.csv',
               index=False, header=True)

dataset = pd.read_csv('../dataset/casos_filtrados_frequences.csv')
col = 'descripcion_del_hecho - Final'
for row1 in dataset[col]:
    try:
        row = row1.split()
        row2 = []
        for w in row:
            row2.append(max_ratio(w))
        row = ' '.join(row2)
        dataset.replace(row1, row)
    except TypeError:
        continue

fstring = ''
for row in dataset[col]:
    fstring += str(row) + ' '

tokens = word_tokenize(fstring)
new_tok = []
for w in tokens:
    new_tok.append(max_ratio(w))

nltkText = Text(tokens)
f = FreqDist(nltkText)
common1 = f.most_common(100)

tokens = new_tok
new_tok = []

for w in tokens:
    new_tok.append(max_ratio_anti(w))
tokens = new_tok

nltkText = Text(tokens)
f = FreqDist(nltkText)
common2 = f.most_common(100)

# Comparar los hapaxes con las palabras positivas del dic con un ratio mayor a 70 para extraer las que esten mal
# escritas
dic = ['parte', 'circulaba', 'delantera', 'delantero', 'delante', 'enfrente', 'izquierda', 'derecha', 'lateral',
       'colisiona',
       'colision', 'colisiono', 'colisionada', 'colisionado', 'colisionando', 'trasero', 'trasera', 'impacta',
       'impacto',
       'impactado', 'impactada', 'embiste', 'embistio', 'embestido', 'iba', 'venia', 'frente', 'lado', 'detras',
       'atras', 'costado', 'choca', 'choco', 'chocando', 'choque', 'costado', 'frontal', 'parte']

#     tokenization(row)

# countedWordsDeleted = []
# for desc in dataset['descripcion_del_hecho - Final']:
#     countedWordsDeleted.append(len(desc))

# plt.plot(countedWords)
# plt.plot(countedWordsDeleted)
# plt.ylabel('Words per Description')
# plt.show()
