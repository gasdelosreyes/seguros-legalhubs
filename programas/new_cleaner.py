import pandas as pd

dataset = pd.read_excel('../dataset/casos_universidad.xlsx')
ds_df = pd.DataFrame(dataset)

'''
Create a dataset for testing
'''

ds_df.iloc[0:500,[11]].to_csv('../dataset/descripciones-test.csv',sep=' ', index=False, header=True)

datatest = pd.read_csv('../dataset/descripciones-test.csv')

'''
    Convert the data in String and lowercase
'''
datatest = datatest.apply(lambda x: x.astype(str).str.lower())

'''
    Apply regular expression 
        [^\w\s] -> Start with alphanumeric and allow whitespace
        \d+ -> Delete all the numbers
'''
import re
datatest['descripcion_del_hecho - Final'] = datatest['descripcion_del_hecho - Final'].apply(lambda x: re.sub('[^\w\s]','',x))
datatest['descripcion_del_hecho - Final'] = datatest['descripcion_del_hecho - Final'].apply(lambda x: re.sub('\d+','',x))

countedWords = []

for desc in datatest['descripcion_del_hecho - Final']:
    countedWords.append(len(desc))
#Save
#datatest.iloc[:].to_csv('../dataset/descripciones-test.csv',sep=' ', index=False, header=True)

'''
    StopWords
'''
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import swadesh

dictionary = ['vh','amb','av','avenida','ambulancia','cruza','calle','policia','impacto','observo','conductor','rotonda','casco']
#Added stopword dictionary

for word in swadesh.words('es'):
    dictionary.append(word)

for word in stopwords.words('spanish'):
    dictionary.append(word)

datatest['descripcion_del_hecho - Final'] = datatest['descripcion_del_hecho - Final'].apply(
    lambda x: " ".join(x for x in x.split() if x not in dictionary)
)

countedWordsDeleted = []
for desc in datatest['descripcion_del_hecho - Final']:
    countedWordsDeleted.append(len(desc))

print(countedWords)
print(countedWordsDeleted)

#print(datatest)
'''
    Tokenize
'''
#import nltk
#nltk.download('punkt')

from nltk.tokenize import word_tokenize
datatest['descripcion_del_hecho - Final'] = datatest['descripcion_del_hecho - Final'].apply(word_tokenize)

#print(datatest)

'''
    Pos tagger
'''

#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.tag import pos_tag

datatest['descripcion_del_hecho - Final']= datatest['descripcion_del_hecho - Final'].apply(pos_tag)

#datatest.iloc[:].to_csv('../dataset/descripciones-test.csv',sep=' ', index=False, header=True)

from nltk.corpus import wordnet
def get_words(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

datatest['descripcion_del_hecho - Final']= datatest['descripcion_del_hecho - Final'].apply(lambda x: [(word, get_words(pos_tag)) for (word, pos_tag) in x])

for desc in datatest['descripcion_del_hecho - Final']:
    for value in desc:
        if(value[1] is 'v' or value[1] is 'r'):
            desc.remove(value)

countedWordsDeleted2 = []
for desc in datatest['descripcion_del_hecho - Final']:
    countedWordsDeleted2.append(len(desc))

print(countedWordsDeleted2)
'''
   Lemmatization
'''
from nltk.stem import WordNetLemmatizer
datatest['descripcion_del_hecho - Final']= datatest['descripcion_del_hecho - Final'].apply(lambda x: [WordNetLemmatizer().lemmatize(word,tag) for word, tag in x])
#print(datatest)

from nltk.probability import FreqDist
for value in datatest['descripcion_del_hecho - Final']:
    frecuency = FreqDist(value)

datatest.iloc[:].to_csv('../dataset/descripciones-test.csv',sep=' ', index=False, header=True)