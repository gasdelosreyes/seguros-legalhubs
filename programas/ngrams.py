import nltk
import pandas as pd
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/casos/auto.csv')

fstring = ''
for row in df['descripcion']:
    fstring += row + ' '

tokens = word_tokenize(fstring)

nltkText = Text(tokens)

fileCreated = open("bigrams.txt","w")
bigrams_series = pd.Series(nltk.ngrams(tokens, 2))
bigrams_values = bigrams_series.value_counts().index
bigrams_counts = bigrams_series.value_counts().values
plot_bigram = (bigrams_series.value_counts())[:20]
plot_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(15, 15))
plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.title('20 Bigramas con mayor frecuencia')
plt.ylabel('Bigramas')
plt.xlabel('N de ocurrencias')
plt.show('bigrams.png')
plt.clf()
for i in range(len(bigrams_values)):
    fileCreated.write('BIGRAMA: ' + str(bigrams_values[i]) + ' - CANTIDAD: ' + str(bigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("trigramas.txt","w")
trigrams_series = pd.Series(nltk.ngrams(tokens, 3))
trigrams_values = trigrams_series.value_counts().index
trigrams_counts = trigrams_series.value_counts().values
plot_trigram = (trigrams_series.value_counts())[:20]
plot_trigram.sort_values().plot.barh(color='blue', width=.9, figsize=(15, 15))
plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.title('20 Trigramas con mayor frecuencia')
plt.ylabel('Trigramas')
plt.xlabel('N de ocurrencias')
plt.show('trigrams.png')
plt.clf()
for i in range(len(trigrams_values)):
    fileCreated.write('TRIGRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

fileCreated.close()

fileCreated = open("5grama.txt","w")
pentagram_series = pd.Series(nltk.ngrams(tokens, 5))
pentagram_values = pentagram_series.value_counts().index
pentagram_counts = pentagram_series.value_counts().values
plot_pentagram = (pentagram_series.value_counts())[:20]
plot_pentagram.sort_values().plot.barh(color='blue', width=.9, figsize=(20, 15))
plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.title('20 5gramas con mayor frecuencia')
plt.ylabel('5GRAMAS',labelpad=2)
plt.xlabel('N de ocurrencias')
plt.show('5grams.png')
plt.clf()
for i in range(len(pentagram_values)):
    fileCreated.write('5GRAMA: ' + str(pentagram_values[i]) + ' - CANTIDAD: ' + str(pentagram_counts[i]) + '\n')

fileCreated.close()

# fileCreated = open("6grama.txt","w")
# trigrams_series = pd.Series(nltk.ngrams(tokens, 6))
# trigrams_values = trigrams_series.value_counts().index
# trigrams_counts = trigrams_series.value_counts().values

# for i in range(len(trigrams_values)):
#     fileCreated.write('6GRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

# fileCreated.close()

# fileCreated = open("10grama.txt","w")
# trigrams_series = pd.Series(nltk.ngrams(tokens, 10))
# trigrams_values = trigrams_series.value_counts().index
# trigrams_counts = trigrams_series.value_counts().values

# for i in range(len(trigrams_values)):
#     fileCreated.write('10GRAMA: ' + str(trigrams_values[i]) + ' - CANTIDAD: ' + str(trigrams_counts[i]) + '\n')

# fileCreated.close()