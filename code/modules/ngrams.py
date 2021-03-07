import nltk
import pandas as pd
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

def getNGram(dataframe, number, max_value=20):
    fstring = ''
    for row in dataframe:
        fstring += row + ' '
    tokens = word_tokenize(fstring)
    ngram_series = pd.Series(nltk.ngrams(tokens, int(number)))
    # ngram_values = ngram_series.value_counts().index
    # ngram_counts = ngram_series.value_counts().values
    plot_bigram = (ngram_series.value_counts())[:int(max_value)]
    plot_bigram.sort_values().plot.barh(color='blue', width=.9, figsize=(15, 15))
    plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.title(str(max_value) + '_' + str(number) + '_grama con mayor frecuencia')
    plt.ylabel(str(number) + '_grama')
    plt.xlabel('N de ocurrencias')
    plt.savefig(str(number) + '_grama.png')
    plt.clf()