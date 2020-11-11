import nltk
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import swadesh
from nltk.tag import pos_tag

'''
    Create a dictionary
'''
# Elimina palabras que no sirven y las stopwords.
def Dictionary():
    dic = ['vh','amb','av','avenida','ambulancia','cruza','calle','policia','impacto','observo','conductor','rotonda','casco']
    for i in stopwords.words('spanish'):
        if(i != 'derecha' and i != 'izquierda'):
            dic.append(i)
    for i in swadesh.words('es'):
        if(i != 'derecha' and i != 'izquierda'):
            dic.append(i)
    return dic

#Elimina las palabras del words, text es un string. Falta que busque 'interv'
def deleteExtraData(text):
    words = ['interv','amplia','formal']
    for w in words:
        if(w in text):
            text = text.split(w)[0]
    return text

def deleteFrequence(freq, tokens, value):
    deletetokens = []
    for w in tokens:
        if(freq[w] <= value):
            # text = re.sub('\bw\b', '',text)
            deletetokens.append(w)
    print(deletetokens)
    return deletetokens

#Tokeniza y después te muestra las 30 más frecuentes.
def tokenization(text):
    words = word_tokenize(text)
    freq = nltk.FreqDist(words)
    print(freq.most_common(30))

openfile = pd.read_excel('../dataset/casos_universidad.xlsx')

dataframe = pd.DataFrame(openfile)

'''
    Getting the columns that needs to be processed
'''
dataframe = dataframe.iloc[:,[5,11,7]]

dataframe.to_csv('../dataset/casos_filtrados.csv',index=False,header=True)

dataset = pd.read_csv('../dataset/casos_filtrados.csv')

dataset = dataset.apply(lambda x: x.astype(str).str.lower())

# countedWords = []

# for desc in dataset['descripcion_del_hecho - Final']:
#     countedWords.append(len(desc))

'''
    Apply regular expression
        [^\w\s] -> Start with alphanumeric and allow whitespace
        (\r\r?|\n) -> Delete line breaks
        \d+ -> Delete all the numbers
'''
#Este segmento limpia las palabras de los hexadecimales y borra las palabras con Dictionary().
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('[^\w\s]','',x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('(\r\n?|\n)+','',x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: re.sub('\d+','',x)
)
dataset['descripcion_del_hecho - Final'] = dataset['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in Dictionary())
)

dataset.to_csv('../dataset/casos_filtrados_regex.csv',sep=';',index=False,header=True)

for row in dataset['descripcion_del_hecho - Final']:
    dataset.loc[dataset['descripcion_del_hecho - Final'] == row, 'descripcion_del_hecho - Final'] = deleteExtraData(row)

dataset.to_csv('../dataset/casos_filtrados_cut.csv',sep=';',index=False,header=True)

fstring = ''
for row in dataset['descripcion_del_hecho - Final']:
    fstring += row + ';'

from nltk.text import Text
from nltk.probability import FreqDist

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
dataset.insert(2,'frecuencias-10','')
dataset.insert(3,'frecuencias-20','')
dataset.insert(4,'frecuencias-30','')
dataset.insert(5,'frecuencias-40','')
dataset.insert(6,'frecuencias-50','')

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

dataset.to_csv('../dataset/casos_filtrados_frequences.csv',sep=';',index=False,header=True)

print(f.most_common(100))
print(f.hapaxes())


#     tokenization(row)

# countedWordsDeleted = []
# for desc in dataset['descripcion_del_hecho - Final']:
#     countedWordsDeleted.append(len(desc))

# plt.plot(countedWords)
# plt.plot(countedWordsDeleted)
# plt.ylabel('Words per Description')
# plt.show()