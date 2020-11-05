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
def Dictionary():
    dic = ['vh','amb','av','avenida','ambulancia','cruza','calle','policia','impacto','observo','conductor','rotonda','casco']
    # print(stopwords.words('spanish'))
    for i in stopwords.words('spanish'):
        if(i is not 'derecha' or i is not 'izquierda'):
            dic.append(i)
    # print(swadesh.words('es'))
    for i in swadesh.words('es'):
        if(i is not 'derecha' or i is not 'izquierda'):
            dic.append(i)
    return dic

def deleteExtraData(text):
    words = ['intervino','interviene','ampliacion','formalizo']
    for w in words:
        if(w in text):
            print(text.split(w)[1])
            text = text.split(w)[0]
    return text

def tokenization(text):
    words = word_tokenize(text)
    freq = nltk.FreqDist(words)
    print(freq.most_common(30))
    # tagged = pos_tag(words)
    # print(tagged)

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

#     tokenization(row)

# countedWordsDeleted = []
# for desc in dataset['descripcion_del_hecho - Final']:
#     countedWordsDeleted.append(len(desc))

# plt.plot(countedWords)
# plt.plot(countedWordsDeleted)
# plt.ylabel('Words per Description')
# plt.show()