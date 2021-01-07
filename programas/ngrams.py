import nltk
import pandas as pd
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize

df = pd.read_csv('../dataset/casos/auto.csv')

fstring = ''
for row in df['descripcion']:
    fstring += row + ' '

tokens = word_tokenize(fstring)

nltkText = Text(tokens)

fileCreated = open("bigrams.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 2))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values

for i in range(len(trigrams_values)):
    fileCreated.write('BIGRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("trigramas.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 3))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values

for i in range(len(trigrams_values)):
    fileCreated.write('TRIGRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("5grama.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 5))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values

for i in range(len(trigrams_values)):
    fileCreated.write('5GRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("6grama.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 6))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values

for i in range(len(trigrams_values)):
    fileCreated.write('6GRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("10grama.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 10))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values

for i in range(len(trigrams_values)):
    fileCreated.write('10GRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()