import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize


df = pd.read_csv('../dataset/casos/auto.csv')

fstring = ''
for row in df['descripcion']:
    fstring += row + ' '

tokens = word_tokenize(fstring)
fdist = FreqDist(tokens)

print(len(list(filter(lambda x: x[1]>=1, fdist.items()))))